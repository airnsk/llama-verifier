// llama-verifier — внешний верификатор спекулятивных цепочек (проект verifier-llamacpp).
//
// Протокол (docs/SPEC.md): POST /vverify
//   {"prefix_tokens":[...], "chains":[{"tokens":[...]}], "mode":"greedy", "temperature":0}
//   -> {"chain_id","accepted","bonus_token","tokens_out","prefill_ms","verify_ms","round_ms"}
//
// Дизайн (docs/DECISIONS.md D-004/D-006): отдельный бинарь на ЧИСТОМ master libllama
// (73a43d1f), llama-server не патчим; база без dp4a (D-005).
//   - Ствол KV = seq 0 ("trunk"): принятые прошлых раундов + весь префикс текущего.
//   - Stateless: клиент шлёт полный префикс; longest match с cached; рассогласование
//     снимается ЧАСТИЧНЫМ seq_rm с ПРОВЕРКОЙ возврата (у гибрида qwen35 частичный rm
//     может молча не удаться — замерено 06.09 13:23: проигнорированный false留下了
//     хвост и ломал все последующие раунды). Не удалось -> полный сброс и досчёт.
//   - ЗАПРЕЩЕНО (hybrid qwen35, M-RoPE + GDN): переоценка позиции, уже стоящей в KV
//     (требование строго X < Y, где X=последняя позиция seq в памяти). Поэтому p0 =
//     argmax после конца префикса КЭШИРУЕТСЯ (p0_top/p0_len) и при повторном запросе
//     на том же префиксе не пересчитывается; при мисте — откат хвоста (или последнего
//     токена префикса с проверкой seq_rm) и досчёт.
//   - Цепочки: ВСЕ на ветках seq 1..N (ствол в decode не участвует): seq_cp ствола
//     [0,L) на ветку, K токенов на позициях L.., logits=1 под каждым; один
//     llama_decode на сумму K×N. Стили веток — ТОЛЬКО полным удалением seq (rm(-1,-1),
//     гарантированно succeeds), частичных rm на ветках нет.
//   - Коммит ЛЕНИВЫЙ (V1): принятые токены в ствол НЕ пишутся; в следующем раунде
//     клиент пришлёт их в составе префикса, и они встанят хвостом (K+1 токенов pp,
//     копейки против отдельного commit-прохода). Обоснование — D-006.
//   - bonus (SPEC "следующий токен таргета") = argmax строки ПОСЛЕДНЕГО ПРИНЯТОГО
//     токена выбранной цепочки; при j=0 — argmax p0.
//
// Границы v1 (V1=greedy): mode!=greedy и temperature!=0 — 400; q[] (exact) — 400.
// ⚠ Риск T3c: seq_cp на гибрид копирует ли рекурентное состояние — проверить на
//   таргете Qwen3.8-27B (гибрид! GDN-слои есть) равством accepted против оффлайн-
//   логитов этапа 0. Если ветки гибрида врóт — план Б: последовательные раунды на
//   стволе с полным сбросом хвоста между цепочками (documented in RESULTS).

#include "llama.h"
#include "ggml.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <random>
#include <signal.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;
using tokens_t = std::vector<llama_token>;

static void on_signal(int) { exit(0); }  // v1: грубая остановка процессом допустима

struct vparams {
    std::string model_path;
    std::string host      = "127.0.0.1";
    int         port      = 8088;
    int32_t     n_gpu     = 999;
    int32_t     n_ctx     = 8192;
    int32_t     n_seq_max = 8;      // 1 ствол + до 7 веток
    int32_t     n_threads = 0;
    int32_t     n_batch   = 2048;
    int32_t     n_ubatch  = 512;
    // V2 (план V2.1): окно склейки запросов сессий в один батч (SPEC ~5–10 мс)
    int         window_ms = 8;
    int         max_queue = 64;   // глубина pending-очереди, выше — 429
    // снапшоты рекурентного состояния на seq: частичный rollback на гибриде qwen35
    // возможен только в пределах n_rs_seq (COMMON_CONTEXT_SEQ_RM_TYPE_RS).
    // ПАМЯТЬ: ~96 MiB на (seq × snapshot) на этом стенде (замер OOM 13:55: 8seq×16rs
    // = 12.3 GiB не влез). Поэтому:
    //   режим ствола (chains==1):  --np 1 --rs 18  (K<=16 + запас, ~1.7 GiB)
    //   режим веток (chains>1):   --np 8 --rs 0   (ствол не трогаем, rs не нужен)
    int32_t     n_rs_seq  = 0;
    bool        flash_attn = true;
    bool        no_mmap    = false;
    // Блок 0.1 плана V2: переключатель kv_unified для замера equivalence
    // (unified = мета-шаринг ячеек seq_cp; non-unified = cross-stream cp).
    bool        kv_unified = true;
};

// ================== V2: сессии + очередь-склейка (план V2.1) ==================
// Слот сессии = 2 seq: trunk (префикс, живёт между раундами) + branch (цепочка,
// сноситсЯ полным rm). seq 0 зарезервирован за stateless /vverify (не смешивать).
struct vsession {
    int32_t   slot = -1;              // trunk=2*slot+1, branch=2*slot+2
    tokens_t  cached;                 // == принятая история сессии (логический ствол)
    int64_t   kv_len = 0;             // ячейки ствола [0, kv_len); delta = cached[kv_len..)
    int32_t   p0_top = -1;            // валиден при p0_len == cached.size()
    int64_t   p0_len = -1;
    void invalidate() { p0_top = -1; p0_len = -1; }
};

struct Pending {
    int64_t  sid = -1;
    tokens_t prefix, chain;
    std::chrono::steady_clock::time_point t_enq{};
    double   queue_ms = 0.0;
    std::promise<std::pair<int, json>> pr;  // {http_code, payload}
};

struct verifier {
    vparams p;

    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    llama_memory_t  mem   = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t       n_vocab = 0;

    tokens_t      cached;   // == ствол seq0 (принятые + префикс последнего раунда)
    int32_t       p0_top = -1;   // argmax p0 после cached.back(); валиден при p0_len
    int64_t       p0_len = -1;
    std::mutex    mu;            // сериализация раундов (/vverify и worker)

    // V2-состояние (sessions — под mu; очередь — под qmu)
    std::unordered_map<int64_t, vsession>   sessions;
    std::unordered_map<int32_t, int64_t>    slot_owner;
    std::deque<std::shared_ptr<Pending>>    queue;
    std::unordered_set<int64_t>             qbusy;
    std::mutex              qmu;
    std::condition_variable qcv;
    std::atomic<bool>       stop_flag{false};
    std::atomic<uint64_t>   enq{0}, rejected{0}, batches{0}, sess_rounds{0};
    std::atomic<int64_t>    sid_next{0};

    std::atomic<uint64_t> rounds{0};
    std::atomic<uint64_t> round_tokens{0};
    std::atomic<uint64_t> accepted_tokens{0};

    void p0_invalidate() { p0_top = -1; p0_len = -1; }
};

static int32_t argmax_row(const float * row, int32_t n) {
    int32_t best = 0;
    float bv = row[0];
    for (int32_t i = 1; i < n; i++) {
        if (row[i] > bv) { bv = row[i]; best = i; }
    }
    return best;
}

static bool decode_full(verifier & v, const llama_batch & batch, std::string & err) {
    const int rc = llama_decode(v.ctx, batch);
    if (rc != 0) {
        err = "llama_decode rc=" + std::to_string(rc);
        return false;
    }
    return true;
}

static json round_verify(verifier & v, const json & req, std::string & err) {
    std::lock_guard<std::mutex> lock(v.mu);
    if (!v.sessions.empty()) { err = "stateless /vverify cannot run while sessions are active (/vclose them first)"; return {}; }

    // ---------- разбор/валидация ----------
    const std::string mode = req.value("mode", std::string("greedy"));
    if (mode != "greedy" && mode != "exact" && mode != "threshold") {
        err = "mode must be greedy|exact|threshold"; return {};
    }
    const bool sampled_mode = (mode != "greedy");
    if (sampled_mode && (!req.contains("temperature") || !req["temperature"].is_number() || req["temperature"].get<double>() <= 0.0)) {
        err = "mode=" + mode + " requires temperature > 0"; return {};
    }
    const double temp = sampled_mode ? req["temperature"].get<double>() : 0.0;
    const double tau  = req.value("tau", 0.3);
    if (mode == "threshold" && (tau <= 0.0 || tau >= 1.0)) { err = "tau must be in (0,1)"; return {}; }
    const unsigned vseed = req.value("seed", (unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 rng(vseed);
    if (!req.contains("prefix_tokens") || !req["prefix_tokens"].is_array() || req["prefix_tokens"].empty()) {
        err = "prefix_tokens must be a non-empty array"; return {};
    }
    tokens_t prefix;
    for (const auto & t : req["prefix_tokens"]) {
        if (!t.is_number_integer()) { err = "prefix_tokens must be integers"; return {}; }
        llama_token id = t.get<llama_token>();
        if (id < 0 || id >= v.n_vocab) { err = "prefix token id out of vocab"; return {}; }
        prefix.push_back(id);
    }
    if (!req.contains("chains") || !req["chains"].is_array() || req["chains"].empty()) {
        err = "chains must be a non-empty array"; return {};
    }
    const int n_chains = (int) req["chains"].size();
    const bool single_chain_early = (n_chains == 1); // chains==1 -> ствол, seq не тратятся на ветки
    if ((single_chain_early ? 1 : n_chains + 1) > v.p.n_seq_max) {
        err = "too many chains for n_seq_max"; return {};
    }
    std::vector<tokens_t> chains(n_chains);
    int64_t maxK = 0, sumK = 0;
    for (int i = 0; i < n_chains; i++) {
        const auto & jc = req["chains"][i];
        if (!jc.is_object() || !jc.contains("tokens") || !jc["tokens"].is_array()) {
            err = "chains[i].tokens must be array"; return {};
        }
        if (jc.contains("q") && mode != "exact") { err = "q[] is only valid for mode=exact"; return {}; }
        for (const auto & t : jc["tokens"]) {
            if (!t.is_number_integer()) { err = "chain tokens must be integers"; return {}; }
            llama_token id = t.get<llama_token>();
            if (id < 0 || id >= v.n_vocab) { err = "chain token id out of vocab"; return {}; }
            chains[i].push_back(id);
        }
        if (chains[i].empty()) { err = "chains[i].tokens must be non-empty"; return {}; }
        maxK = std::max(maxK, (int64_t) chains[i].size());
        sumK += (int64_t) chains[i].size();
    }
    std::vector<std::vector<double>> qs(n_chains);
    if (mode == "exact") {
        for (int i = 0; i < n_chains; i++) {
            const auto & jc = req["chains"][i];
            if (!jc.contains("q") || !jc["q"].is_array() || (int64_t) jc["q"].size() != (int64_t) chains[i].size()) {
                err = "mode=exact requires chains[i].q with exactly K elements"; return {};
            }
            for (const auto & qv : jc["q"]) {
                if (!qv.is_number()) { err = "q values must be numbers"; return {}; }
                const double q = qv.get<double>();
                if (!(q > 0.0) || q > 1.0) { err = "q values must be in (0,1]"; return {}; }
                qs[i].push_back(q);
            }
        }
    }
    const int64_t L = (int64_t) prefix.size();
    if (L + std::max(maxK, (int64_t) 1) > (int64_t) llama_n_ctx(v.ctx)) {
        err = "prefix + max chain exceeds n_ctx"; return {};
    }
    if (sumK > (int64_t) v.p.n_batch) { err = "sum(chain) exceeds n_batch"; return {}; }

    if (req.value("clear_cache", false)) {
        llama_memory_seq_rm(v.mem, 0, -1, -1);
        for (llama_seq_id s = 1; s < v.p.n_seq_max; s++) { llama_memory_seq_rm(v.mem, s, -1, -1); }
        v.cached.clear();
        v.p0_invalidate();
    }

    const int64_t t0 = ggml_time_us();

    // ---------- longest match + откат ствола ----------
    int64_t m = 0;
    const int64_t cmax = std::min((int64_t) v.cached.size(), L);
    while (m < cmax && v.cached[m] == prefix[m]) { m++; }
    if (m < (int64_t) v.cached.size()) {
        bool ok = llama_memory_seq_rm(v.mem, 0, (llama_pos) m, -1);
        if (ok) {
            v.cached.resize(m);
        } else { // гибрид отказал в частичном rm — полный сброс, досчёт с нуля
            llama_memory_seq_rm(v.mem, 0, -1, -1);
            v.cached.clear();
            m = 0;
        }
        v.p0_invalidate();
    }

    // ---------- p0 / хвост префикса ----------
    // m == L: ствол == prefix. p0 или из кэша, или откат 1 токена + досчёт.
    if (sampled_mode) v.p0_invalidate(); // кэш хранит только argmax — exact/threshold нужен полный дистрибутив p0
    if (m == L && !(v.p0_len == L && v.p0_top >= 0)) {
        bool ok = llama_memory_seq_rm(v.mem, 0, (llama_pos) (L - 1), -1);
        if (ok) {
            v.cached.resize(L - 1);
            m = L - 1;
        } else {
            llama_memory_seq_rm(v.mem, 0, -1, -1);
            v.cached.clear();
            m = 0;
        }
    }
    int32_t p0_top;
    std::vector<float> p0_logits; // полный вектор логитов p0 (sampled_mode)
    tokens_t tail;
    if (m < L) {
        tail.assign(prefix.begin() + m, prefix.end());
        llama_pos tail_pos = (llama_pos) m;
        size_t off = 0;
        int p0_idx = 0; // индекс tokena p0 в последнем чанке (get_logits_ith — индекс в батче!)
        while (off < tail.size()) {
            const size_t cnt = std::min((size_t) v.p.n_batch, tail.size() - off);
            const bool last_chunk = (off + cnt == tail.size());
            p0_idx = (int) cnt - 1;
            llama_batch b = llama_batch_init((int) cnt, 0, 1);
            for (size_t i = 0; i < cnt; i++) {
                const int k = b.n_tokens;
                b.token[k]     = tail[off + i];
                b.pos[k]       = tail_pos + (llama_pos) (off + i);
                b.n_seq_id[k]  = 1;
                b.seq_id[k][0] = 0;
                b.logits[k]    = (last_chunk && i + 1 == cnt) ? 1 : 0;
                b.n_tokens++;
            }
            if (!decode_full(v, b, err)) { llama_batch_free(b); return {}; }
            llama_batch_free(b);
            off += cnt;
        }
        const float * row = llama_get_logits_ith(v.ctx, p0_idx);
        if (!row) { err = "no p0 logits"; return {}; }
        if (sampled_mode) p0_logits.assign(row, row + v.n_vocab);
        p0_top = argmax_row(row, v.n_vocab);
        v.cached.insert(v.cached.end(), tail.begin(), tail.end()); // cached == prefix
    } else {
        p0_top = v.p0_top; // кэш
    }
    v.p0_top = p0_top;
    v.p0_len = L;
    const int64_t t_a = ggml_time_us();

    // ---------- режим батча цепочек: ТОЛЬКО ветки (решение D-007, 14:30) ----------
    // Изначально chains==1 гонялась на стволе seq 0 батчем [L, L+K). ЗАМЕРЕНО 14:30:
    //   пайплайн 16 раундов по K=1 на стволе (rs-rollback) дал self16; повторная
    //   верификация ЭТОЙ ЖЕ цепочки батчем K=16 на стволе приняла 1 токен вместо 16,
    //   в веточном режиме (seq 1) та же цепочка приняла ровно 16. Т.е. батч из
    //   нескольких токенов на seq 0 гибрида qwen35 даёт НЕВЕРНЫЕ позиции/GDN-состояние
    //   относительно реплейса. Ствол используется ТОЛЬКО для sequential K=1 отката
    //   (rs=18) — см. p0/rollback выше. Все батчи цепочек: seq 1..N, rs=0, --np >= N+1.
    const bool single_chain = false;
    // rs-снапшоты: ~96 MiB на (seq × snapshot). Для теста границы отката (V2-план 0.2)
    // разрешаем np<=2 с rs>0 (ствол + 1 ветка); больше — по-прежнему ОМВ-риск.
    if (v.p.n_rs_seq > 0 && v.p.n_seq_max > 2) {
        err = "rs snapshots consume ~96MiB per (seq x snapshot): launch branches with --rs 0 (rollback-depth probe needs --np 2 --rs N)"; return {};
    }
    if (n_chains + 1 > v.p.n_seq_max) {
        err = "not enough seqs for branch mode: need --np >= chains+1"; return {};
    }

    // ВАЖНО (замерено 13:34): llama_kv_cache::seq_cp с частичным диапазоном GGML_ABORT'ит
    //   'seq_cp() is only supported for full KV buffers' — разрешён только полный буфер
    //   (p0<0 && p1<0), т.е. перенос всех ячеек ствола со всеми их позициями.
    // Бюджет ячеек: ветки занимают собственные ячейки на весь префикс => требуется
    //   (n_chains + 1) * L <= n_ctx, иначе 400 (V2 решит дроблением раундов).
    if (!single_chain) {
        // unified KV (n_stream=1): seq_cp диапазона = мета-операция (seq_add ячеек),
        // ячейки веток = только их собственные токены => L + sumK + запас <= n_ctx
        if (L + sumK + 16 > (int64_t) llama_n_ctx(v.ctx)) {
            err = "cells budget exceeded: prefix+chains+16 > n_ctx"; return {};
        }
        for (int i = 0; i < n_chains; i++) {
            const llama_seq_id br = (llama_seq_id) (i + 1);
            llama_memory_seq_rm(v.mem, br, -1, -1);        // целая секвенция — всегда ок
            if (!chains[i].empty()) {
                llama_memory_seq_cp(v.mem, 0, br, 0, (llama_pos) L); // share prefix cells
            }
        }
    }
    std::vector<tokens_t> targ(n_chains); // argmax строки под каждый токен цепочки
    std::vector<std::vector<float>> rows_logits; // полные строки логитов (sampled_mode)
    if (sumK > 0) {
        llama_batch b = llama_batch_init((int) sumK, 0, (int32_t) v.p.n_seq_max);
        std::vector<int> chain_bidx; chain_bidx.reserve((size_t) sumK);
        for (int i = 0; i < n_chains; i++) {
            for (size_t k = 0; k < chains[i].size(); k++) {
                const int t = b.n_tokens;
                chain_bidx.push_back(t);
                b.token[t]     = chains[i][k];
                b.pos[t]       = (llama_pos) (L + k);
                b.n_seq_id[t]  = 1;
                b.seq_id[t][0] = single_chain ? 0 : (llama_seq_id) (i + 1);
                b.logits[t]    = 1;
                b.n_tokens++;
            }
        }
        if (!decode_full(v, b, err)) { llama_batch_free(b); return {}; }
        llama_batch_free(b);
        int row = 0;
        for (int i = 0; i < n_chains; i++) {
            targ[i].reserve(chains[i].size());
            for (size_t k = 0; k < chains[i].size(); k++) {
                const float * rp = llama_get_logits_ith(v.ctx, chain_bidx[row++]);
                if (!rp) { err = "missing chain logits row"; return {}; }
                if (sampled_mode) rows_logits.emplace_back(rp, rp + v.n_vocab);
                targ[i].push_back(argmax_row(rp, v.n_vocab));
            }
        }
    }
    const int64_t t1 = ggml_time_us();

    // ---------- приёмка: greedy-сличение или exact/threshold (V3, SPEC min(1,p/q)) ----------
    int     win   = -1;
    int64_t win_j = 0;
    int32_t bonus = p0_top;
    if (!sampled_mode) {
        for (int i = 0; i < n_chains && win < 0; i++) {
            int64_t j = 0;
            while (j < (int64_t) chains[i].size()) {
                const int32_t want = (j == 0) ? p0_top : targ[i][(size_t) j - 1];
                if (chains[i][j] != want) { break; }
                j++;
            }
            if (j > 0) { win = i; win_j = j; }
        }
        if (win < 0) { win = 0; win_j = 0; }
        else { bonus = targ[win][(size_t) win_j - 1]; }
    } else {
        // V3 exact/threshold: независимая верификация каждой цепочки (SPEC: accept
        // min(1,p(t)/q(t)); reject -> bonus из norm(max(0,p-q)); threshold: p(t)>tau,
        // bonus из p; полный приём -> bonus из p). Победитель — наибольший принятый
        // префикс. Дистрибутивная точность гарантируется для single-chain (тест V3);
        // multi-chain — практическое расширение (D-016).
        if ((int) rows_logits.size() != sumK) { err = "internal: rows_logits missing"; return {}; }
        auto mkdist = [&](const std::vector<float> & lg) {
            std::vector<double> w(lg.size());
            double mx = -1e30;
            for (float x : lg) mx = std::max(mx, (double) x);
            double ssum = 0;
            for (size_t i = 0; i < lg.size(); i++) { w[i] = std::exp(((double) lg[i] - mx) / temp); ssum += w[i]; }
            for (auto & x : w) x /= ssum;
            return w;
        };
        int off = 0;
        int64_t best_j = -1; int best_i = 0; int32_t best_bonus = p0_top;
        for (int i = 0; i < n_chains; i++) {
            const auto & ch = chains[i];
            int64_t j = 0; int32_t bns = p0_top;
            for (size_t k = 0; k < ch.size(); k++) {
                const auto w = mkdist(rows_logits[off + (int) k]);
                const double p_t = w[ch[k]];
                bool ok;
                if (mode == "exact") {
                    std::uniform_real_distribution<double> ud(0.0, 1.0);
                    ok = ud(rng) < std::min(1.0, p_t / qs[i][k]);
                } else {
                    ok = p_t > tau;
                }
                if (!ok) {
                    if (mode == "exact") { // bonus из norm(max(0, p-q))
                        std::vector<double> r(w.size()); double ssum = 0;
                        for (size_t z = 0; z < w.size(); z++) { r[z] = std::max(0.0, w[z] - qs[i][k]); ssum += r[z]; }
                        if (ssum > 0) { std::discrete_distribution<int32_t> dd(r.begin(), r.end()); bns = dd(rng); }
                        else { bns = (int32_t) (std::max_element(w.begin(), w.end()) - w.begin()); }
                    } else { // bonus из p
                        std::discrete_distribution<int32_t> dd(w.begin(), w.end()); bns = dd(rng);
                    }
                    break;
                }
                j++;
            }
            if (j == (int64_t) ch.size()) { // полный приём: bonus из p на последней строке
                const auto w = mkdist(rows_logits[off + (int) ch.size() - 1]);
                std::discrete_distribution<int32_t> dd(w.begin(), w.end()); bns = dd(rng);
            }
            if (best_j < 0 || j > best_j) { best_j = j; best_i = i; best_bonus = bns; }
            off += (int) ch.size();
        }
        win = best_i; win_j = std::max<int64_t>(best_j, 0); bonus = best_bonus;
    }

    // ---------- коммит ----------
    if (single_chain) {
        // ствол сейчас = prefix + chain0[0..K). Оставляем prefix + chain0[0..win_j).
        const int64_t keep = L + win_j;
        bool ok = llama_memory_seq_rm(v.mem, 0, (llama_pos) keep, -1);
        // САМОПРОВЕРКА (замер 13:53 S3: partial rm вернул true, но ствол остался
        // с хвостом — следующий запрос упёрся в 'inconsistent positions'). Если
        // seq_pos_max всё ещё >= keep — хвост физически не снят, откат полный.
        if (ok) {
            const llama_pos pmax = llama_memory_seq_pos_max(v.mem, 0);
            if (pmax >= (llama_pos) keep) {
                fprintf(stderr, "%s: partial rm no-op (pos_max=%d, keep=%" PRId64
                        ") — full reset fallback\n", __func__, (int) pmax, keep);
                ok = false;
            }
        }
        if (!ok) { // rs-снапшотов не хватило или гибрид не умеет частичный rm —
                   // полный сброс, следующий раунд досчитает префикс заново
            llama_memory_seq_rm(v.mem, 0, -1, -1);
            v.cached.clear();
            v.p0_invalidate();
        } else {
            // cached после добора хвоста == prefix (L шт.). Добавляем РЕАЛЬНО
            // принятые токены победителя. resize(keep) до size<L+win_j заполнял
            // нулями — раунд-2 терял синхрон и откатывался в rs-состояние
            // (замер 14:26: SELF round1 accepted=0).
            v.cached.resize(L);
            for (int64_t k = 0; k < win_j; k++) { v.cached.push_back(chains[win][k]); }
            v.p0_top = bonus;
            v.p0_len = (int64_t) v.cached.size();
        }
    } else {
        // ленивый (D-006, проверено 1000 раундов 0.2): ствол остаётся == prefix
        // (в KV ровно [0,L)); accepted досчитает следующий раунд как delta-pp.
        // НЕ пишем cached=prefix+accepted: ячеек для них в стволе нет — позиция
        // встанет встык к мнимому концу и тихо рассинхронит GDN (проверено на
        // стенде в ранней версии этого коммита, откачено).
        for (int i = 0; i < n_chains; i++) {
            llama_memory_seq_rm(v.mem, (llama_seq_id) (i + 1), -1, -1);
        }
    }

    const int64_t t2 = ggml_time_us();

    json out;
    json tokens_out = json::array();
    for (int64_t k = 0; k < win_j; k++) { tokens_out.push_back(chains[win][k]); }
    tokens_out.push_back(bonus);
    out["mode"]         = mode;
    out["chain_id"]     = win;
    out["accepted"]     = win_j;
    out["bonus_token"]  = bonus;
    out["tokens_out"]   = tokens_out;
    out["prefill_ms"]   = (double) (t_a - t0) / 1000.0; // досчёт хвоста префикса (0 в steady)
    out["verify_ms"]    = (double) (t1 - t_a) / 1000.0; // батч K*N — цена верификации
    out["commit_ms"]    = 0.0;                          // lazy: только rm веток (сливается с t2-t1)
    out["round_ms"]     = (double) (t2 - t0) / 1000.0;
    out["round_tokens"] = (int64_t) (tail.size() + sumK);
    out["kv_cached"]    = (int64_t) v.cached.size();

    v.rounds++;
    v.round_tokens += (uint64_t) out["round_tokens"].get<int64_t>();
    v.accepted_tokens += (uint64_t) win_j;
    return out;
}


// ================== /dbg_batch — эксперимент "маска vs GDN-состояние" (06.09 15:00) ==================
//   H1: в нашем батче позиции цепочки не аттендят друг друга (argmax p2 == argmax p1).
//   H2: GDN-состояние не продвигается внутри multi-token ubatch (p2 != p1 и != seq).
// Запрос: {"chain":[...K],"probe":"A"|"B"|"AB"} — ствол должен быть ровно на len(prefix).
//   A: батч [L..L+K) на seq 1 (как в /vverify) -> p_a[j] = argmax под tok_j (для p_a[0]
//      нужен p0_top: caller передаёт prefix_len, мы сами p0-досчёт не делаем — ствол warm).
//   B: батч [L..L+1) на seq 1 для tok_j с ПРЕДВАРИТЕЛЬНЫМ досчётом tok_0..tok_{j-1} на
//      стволе seq 0 и полным откатом после — это и есть ground truth (как в генерации).
// Ответ: p_a[], p_b[] (j=0 -> -1, берётся из кэша p0 caller'ом), tail-логиты строк.
static json dbg_batch(verifier & v, const json & req, std::string & err) {
    if (!req.contains("chain") || !req["chain"].is_array() || req["chain"].empty()) {
        err = "chain required"; return {};
    }
    tokens_t chain;
    for (const auto & t : req["chain"]) {
        if (!t.is_number_integer()) { err = "chain must be ints"; return {}; }
        llama_token id = t.get<llama_token>();
        if (id < 0 || id >= v.n_vocab) { err = "token out of vocab"; return {}; }
        chain.push_back(id);
    }
    const int64_t K = (int64_t) chain.size();
    const int64_t L = (int64_t) v.cached.size();
    const std::string probe = req.value("probe", std::string("AB"));
    json out;

    auto decode_rows = [&](const llama_seq_id seq, std::vector<tokens_t> & p_rows, std::vector<int64_t> & ms) -> bool {
        llama_batch b = llama_batch_init((int) K, 0, 1);
        for (int64_t k = 0; k < K; k++) {
            const int t = b.n_tokens;
            b.token[t] = chain[k];
            b.pos[t]   = (llama_pos) (L + k);
            b.n_seq_id[t] = 1;
            b.seq_id[t][0] = seq;
            b.logits[t] = 1;
            b.n_tokens++;
        }
        const int64_t t0 = ggml_time_us();
        if (!decode_full(v, b, err)) { llama_batch_free(b); return false; }
        ms.push_back((double) (ggml_time_us() - t0) / 1000.0);
        llama_batch_free(b);
        for (int64_t k = 0; k < K; k++) {
            const float * rp = llama_get_logits_ith(v.ctx, (int) k);
            if (!rp) { err = "no row"; return false; }
            p_rows.push_back({ argmax_row(rp, v.n_vocab) });
        }
        return true;
    };

    if (probe == "A" || probe == "AB") {
        // батч на seq 1, ствол не трогаем; ветку стереть полным rm
        std::vector<tokens_t> rows; std::vector<int64_t> ms;
        llama_memory_seq_rm(v.mem, 1, -1, -1);
        llama_memory_seq_cp(v.mem, 0, 1, 0, (llama_pos) L);
        bool ok = decode_rows(1, rows, ms); // v.mu уже удержан хендлером
        llama_memory_seq_rm(v.mem, 1, -1, -1);
        if (!ok) return {};
        out["batch_rows"] = json::array();
        for (auto & r : rows) out["batch_rows"].push_back(r[0]);
        out["batch_ms"] = ms[0];
    }
    if (probe == "B" || probe == "AB") {
        // ground truth: каждый tok_j досчитан ПОСЛЕДЕОВАТЕЛЬНЫМ батчем всех предыдущих
        // на ветке seq 1 (позиции [L, L+j)), затем строка j. Ствол не трогаем вообще.
        out["seq_rows"] = json::array();
        std::vector<int64_t> ms;
        for (int64_t j = 0; j < K; j++) {
            llama_batch b = llama_batch_init((int) j + 1, 0, 1);
            for (int64_t k = 0; k <= j; k++) {
                const int t = b.n_tokens;
                b.token[t] = chain[k];
                b.pos[t]   = (llama_pos) (L + k);
                b.n_seq_id[t] = 1;
                b.seq_id[t][0] = 1;
                b.logits[t] = (k == j) ? 1 : 0;
                b.n_tokens++;
            }
            llama_memory_seq_rm(v.mem, 1, -1, -1);   // полный rm ветки (rs=0!)
            llama_memory_seq_cp(v.mem, 0, 1, 0, (llama_pos) L); // заново шарить префикс
            std::string e2;
            const int64_t t0 = ggml_time_us();
            if (!decode_full(v, b, e2)) { err = "B decode j=" + std::to_string(j) + ": " + e2; llama_batch_free(b); return {}; }
            ms.push_back((ggml_time_us() - t0) / 1000);
            const float * rp = llama_get_logits_ith(v.ctx, (int) j);
            if (!rp) { err = "B no row"; llama_batch_free(b); return {}; }
            out["seq_rows"].push_back(argmax_row(rp, v.n_vocab));
            llama_batch_free(b);
        }
        llama_memory_seq_rm(v.mem, 1, -1, -1);
        double tot = 0; for (auto m : ms) tot += m;
        out["seq_ms_total"] = tot;
    }
    out["prefix_len"] = L;
    return out;
}

// ================== V2: сессии + очередь-склейка (план V2.1) ==================
// Все функции ниже требуют v.mu (кроме очереди — v.qmu).
// Ствол сессии: seq 2*slot+1, ветка: 2*slot+2. seq 0 — только stateless /vverify.
// Тёплый раунд: дельта-pp (принятые+bonus прошлого раунда, 1..K+1 токенов) одним общим
// батчем по всем сессиям (разные seq — разные стволы, маски не смешиваются), затем
// seq_cp ствол->ветка (мета-шаринг, unified KV, D-006) и ОДИН батч цепочек на ветках.

static int32_t find_free_slot(verifier & v) {
    const int32_t smax = (v.p.n_seq_max - 1) / 2;
    for (int32_t s = 0; s < smax; s++) {
        if (!v.slot_owner.count(2 * s + 1) && !v.slot_owner.count(2 * s + 2)) return s;
    }
    return -1;
}

// Протянуть ствол сессии до == prefix; вернуть p0 (argmax под последним токеном).
// Возвращает false + err при провале decode. Вызывается с удержанным v.mu.
static bool trunk_extend(verifier & v, vsession & s, const tokens_t & prefix,
                         int32_t & p0, std::string & err) {
    const int64_t L = (int64_t) prefix.size();
    const llama_seq_id tr = (llama_seq_id) (2 * s.slot + 1);
    int64_t m = 0;
    const int64_t cmax = std::min((int64_t) s.cached.size(), L);
    while (m < cmax && s.cached[m] == prefix[m]) m++;
    if (m < (int64_t) s.cached.size()) {   // ретракция/рассогласование — полный reset
        llama_memory_seq_rm(v.mem, tr, -1, -1);   // целая секвенция — гарантированно ок
        s.cached.clear(); s.kv_len = 0; s.invalidate();
        m = 0;
    }
    p0 = -1;
    if (m == L && s.p0_len == L && s.p0_top >= 0) { p0 = s.p0_top; return true; }

    tokens_t delta;
    llama_pos dpos = 0;
    if (m == L) { // ствол==prefix, p0 нет: откат 1 токена + досчёт (путь V1, b02d d=1 ок)
        bool ok = llama_memory_seq_rm(v.mem, tr, (llama_pos) (L - 1), -1);
        if (ok && llama_memory_seq_pos_max(v.mem, tr) >= (llama_pos) (L - 1)) ok = false;
        if (ok) { delta.push_back(prefix[L - 1]); dpos = (llama_pos) (L - 1); s.kv_len = L - 1; }
        else { // не вышло — полный reset, досчёт всего
            llama_memory_seq_rm(v.mem, tr, -1, -1);
            s.cached.clear(); s.kv_len = 0; s.invalidate();
            m = 0;
        }
    }
    if (p0 < 0) {
        if (m < L) { delta.assign(prefix.begin() + m, prefix.end()); dpos = (llama_pos) m; }
        size_t off = 0;
        int p0_idx = 0;
        while (off < delta.size()) {
            const size_t cnt = std::min((size_t) v.p.n_batch, delta.size() - off);
            const bool last_chunk = (off + cnt == delta.size());
            p0_idx = (int) cnt - 1;
            llama_batch b = llama_batch_init((int) cnt, 0, 1);
            for (size_t i = 0; i < cnt; i++) {
                const int t = b.n_tokens;
                b.token[t] = delta[off + i];
                b.pos[t] = (llama_pos) (dpos + off + i);
                b.n_seq_id[t] = 1; b.seq_id[t][0] = tr; b.logits[t] = 1; // p0 нужен с каждой
                b.n_tokens++;                                             // чанки — см. V1
            }
            if (!decode_full(v, b, err)) { llama_batch_free(b); return false; }
            const float * rp = llama_get_logits_ith(v.ctx, p0_idx);
            if (!rp) { err = "delta row missing"; llama_batch_free(b); return false; }
            p0 = argmax_row(rp, v.n_vocab);
            llama_batch_free(b);
            off += cnt;
        }
        s.cached = prefix;
        s.kv_len = L;
        s.p0_top = p0; s.p0_len = L;
    }
    return true;
}

// Обработать батч запросов сессий одним (или двумя при переполнении) декодами.
// Каждый Pending = одна цепочка одной сессии. Возвращает true, если батч валиден
// по бюджетам (иначе заполняет promises ошибкой).
// Собрать дельту сессии в общий батч невозможно динамически — plan phase считает
// заранее: RETR/DELTA/P0OK/MISS по текущему состоянию кэша.
enum plan_mode { PM_P0OK, PM_MISS, PM_DELTA, PM_RETR };
struct sitem {
    std::shared_ptr<Pending> r;
    vsession * s;
    int mode = PM_DELTA;
    llama_pos dpos = 0;        // начало дельты в позициях
    int64_t dcnt = 0;          // токенов дельты (RETR -> вся длина префикса)
    int last_row = -1;         // индекс токена в общем батче (логиты p0)
    int32_t p0 = -1;
    bool ok = true;            // false -> ответ уже выставлен
};

static void reset_session(verifier & v, vsession & s) {
    const llama_seq_id tr = (llama_seq_id) (2 * s.slot + 1);
    const llama_seq_id br = (llama_seq_id) (2 * s.slot + 2);
    llama_memory_seq_rm(v.mem, tr, -1, -1);
    llama_memory_seq_rm(v.mem, br, -1, -1);
    s.cached.clear(); s.kv_len = 0; s.invalidate();
}

static void session_batch(verifier & v, std::vector<std::shared_ptr<Pending>> & reqs) {
    const int64_t t_beg = ggml_time_us();
    // stateless-остатки seq0: режимы взаимоисключающие, чистим перед батчем
    if (!v.cached.empty() || v.p0_len >= 0) {
        llama_memory_seq_rm(v.mem, 0, -1, -1);
        v.cached.clear(); v.p0_invalidate();
    }

    std::vector<sitem> items;
    int64_t cells = 0, sumK = 0;
    for (auto & r : reqs) {
        auto it = v.sessions.find(r->sid);
        if (it == v.sessions.end()) { r->pr.set_value({404, json{{"error", "no such session"}}}); continue; }
        if (r->prefix.empty() || r->chain.empty()) {
            r->pr.set_value({400, json{{"error", "prefix_tokens/chain non-empty required"}}}); continue;
        }
        bool bad = false;
        for (auto t : r->prefix) if (t < 0 || t >= v.n_vocab) { bad = true; break; }
        if (!bad) for (auto t : r->chain) if (t < 0 || t >= v.n_vocab) { bad = true; break; }
        if (bad) { r->pr.set_value({400, json{{"error", "token id out of vocab"}}}); continue; }
        sitem x; x.r = r; x.s = &it->second;
        const int64_t L = (int64_t) r->prefix.size();
        int64_t m = 0;
        const int64_t cmax = std::min((int64_t) x.s->cached.size(), L);
        while (m < cmax && x.s->cached[m] == r->prefix[m]) m++;
        if (m < (int64_t) x.s->cached.size()) { x.mode = PM_RETR; x.dcnt = L; }
        else if (m < L)                        { x.mode = PM_DELTA; x.dpos = (llama_pos) m; x.dcnt = L - m; }
        else if (x.s->p0_len == L && x.s->p0_top >= 0) { x.mode = PM_P0OK; x.p0 = x.s->p0_top; x.dcnt = 0; }
        else                                   { x.mode = PM_MISS; x.dcnt = L; }
        // PM_MISS при m==L (частый случай: клиент вернул accepted+bonus, дублик
        // последнего в стволе не валиден) НЕ трогаем ствол: p0 досчитывается на
        // ВЕТКЕ (seq_cp ствола + decode 1 токена). Замер 06.09 18:40: частичный rm
        // +1-токен на ствол = полный reset 879 мс (V1-путь, для сессий неприемлем).
        cells += L + (int64_t) r->chain.size();
        sumK  += (int64_t) r->chain.size();
        items.push_back(std::move(x));
    }
    if (items.empty()) return;
    // Паддинг цепочек до K_max (D-015): equal_seqs-резка рекуррентной памяти делит
    // смешанный батч на несколько ubatch (фикс ~86 мс на каждый); добиваем короткие
    // цепочки последним реальным токеном — один ubatch, один фикс.
    int64_t Kmax = 0;
    for (auto & x : items) Kmax = std::max(Kmax, (int64_t) x.r->chain.size());
    const int64_t sumKpad = Kmax * (int64_t) items.size();
    if (cells + 16 > (int64_t) llama_n_ctx(v.ctx)) {
        for (auto & x : items) x.r->pr.set_value({429, json{{"error", "kv cell budget exceeded"}, {"retry_after", 1}}});
        return;
    }
    if (sumKpad > (int64_t) v.p.n_batch) {
        for (auto & x : items) x.r->pr.set_value({429, json{{"error", "chain tokens over n_batch"}, {"retry_after", 1}}});
        return;
    }
    // ветки несут собственные ячейки цепочки ПОВЕРХ ствола (с pad-строками): cells + sumKpad + 16
    if (sumKpad + 16 > (int64_t) llama_n_ctx(v.ctx) - cells) {
        for (auto & x : items) x.r->pr.set_value({429, json{{"error", "kv cell budget exceeded (branches)"}, {"retry_after", 1}}});
        return;
    }

    // ---------- фаза 1: дельта-pp ----------
    const int64_t t_p0 = ggml_time_us();
    int64_t batch_delta = 0;
    for (auto & x : items) if (x.mode == PM_RETR || x.mode == PM_DELTA) batch_delta += x.dcnt;
    std::vector<sitem *> miss;
    for (auto & x : items) if (x.mode == PM_MISS) miss.push_back(&x);

    // PM_MISS с полным совпадением кэша (m==L): p0 на ветке, ствол не трогаем.
    // Позиция L-1 на ветке валидна: ветка ≠ ствол, повтор позиции внутри другой
    // секвенции — разрешено (пересчёт позиции ВНУТРИ seq запрещён).
    {
        llama_batch bp = llama_batch_init((int) miss.size(), 0, 1);
        for (auto * x : miss) {
            const int64_t L = (int64_t) x->r->prefix.size();
            if (x->s->kv_len != L) continue; // ретракция/рассогласование — в seq-путь
            const llama_seq_id br = (llama_seq_id) (2 * x->s->slot + 2);
            if (bp.n_tokens + 1 > v.p.n_batch) continue;
            llama_memory_seq_rm(v.mem, br, -1, -1);
            // копия ТОЛЬКО [0, L-1): токен на L-1 досчитывается на ветке ПЕРВЫЙ раз
            // (cp всего [0,L) + decode на L-1 = двойной advance GDN — строгое правило
            //  X>max внутри seq; замер V1 14:30). Ветка ≠ ствол — позиция валидна.
            llama_memory_seq_cp(v.mem, (llama_seq_id) (2 * x->s->slot + 1), br, 0, (llama_pos) (L - 1));
            const int t = bp.n_tokens;
            bp.token[t] = x->r->prefix[(size_t) (L - 1)];
            bp.pos[t] = (llama_pos) (L - 1);
            bp.n_seq_id[t] = 1; bp.seq_id[t][0] = br; bp.logits[t] = 1;
            bp.n_tokens++;
            x->last_row = t; // маркер: обработана здесь
        }
        if (bp.n_tokens > 0) {
            std::string e;
            if (!decode_full(v, bp, e)) {
                llama_batch_free(bp);
                for (auto * x : miss) if (x->last_row >= 0) {
                    x->ok = false; x->p0 = -1;
                    x->r->pr.set_value({500, json{{"error", "p0 branch: " + e}}});
                    x->last_row = -2; // не обрабатывать ниже
                }
            } else {
                for (auto * x : miss) {
                    if (x->last_row < 0) continue;
                    const float * rp = llama_get_logits_ith(v.ctx, x->last_row);
                    if (!rp) { x->ok = false; x->p0 = -1; x->last_row = -2;
                               x->r->pr.set_value({500, json{{"error", "p0 branch row missing"}}}); continue; }
                    x->p0 = argmax_row(rp, v.n_vocab);
                    x->s->p0_top = x->p0; x->s->p0_len = (int64_t) x->s->cached.size();
                    x->last_row = -2;
                }
            }
        }
        llama_batch_free(bp);
    }
    for (auto & x : items) if (x.mode == PM_MISS) miss.push_back(&x);

    if (batch_delta > 0 && batch_delta <= (int64_t) v.p.n_batch) {
        llama_batch b = llama_batch_init((int) batch_delta, 0, 1);
        for (auto & x : items) {
            if (x.mode != PM_RETR && x.mode != PM_DELTA) continue;
            const int64_t L = (int64_t) x.r->prefix.size();
            const llama_seq_id tr = (llama_seq_id) (2 * x.s->slot + 1);
            if (x.mode == PM_RETR) { // хвост невалиден — полный сброс ствола, pp всего префикса
                llama_memory_seq_rm(v.mem, tr, -1, -1);
                x.s->cached.clear(); x.s->kv_len = 0; x.s->invalidate();
            }
            for (int64_t i = 0; i < x.dcnt; i++) {
                const int t = b.n_tokens;
                b.token[t] = x.r->prefix[(size_t) (L - x.dcnt + i)]; // RETR: с 0, DELTA: с m
                b.pos[t] = (llama_pos) (x.mode == PM_RETR ? i : x.dpos + i);
                b.n_seq_id[t] = 1; b.seq_id[t][0] = tr;
                b.logits[t] = (i + 1 == x.dcnt) ? 1 : 0; // строка p0 — только последняя
                b.n_tokens++;
                if (i + 1 == x.dcnt) x.last_row = t;
            }
            x.s->cached = x.r->prefix; x.s->kv_len = L; // подтверждается успехом decode
        }
        std::string e;
        if (!decode_full(v, b, e)) {
            llama_batch_free(b);
            for (auto & x : items) {
                if (x.mode == PM_RETR || x.mode == PM_DELTA) { reset_session(v, *x.s); x.ok = false; }
                x.r->pr.set_value({500, json{{"error", "delta decode: " + e}}});
            }
            return;
        }
        llama_batch_free(b);
        for (auto & x : items) {
            if (x.mode == PM_RETR || x.mode == PM_DELTA) {
                const float * rp = llama_get_logits_ith(v.ctx, x.last_row);
                if (!rp) { reset_session(v, *x.s); x.ok = false;
                           x.r->pr.set_value({500, json{{"error", "delta row missing"}}}); continue; }
                x.p0 = argmax_row(rp, v.n_vocab);
                x.s->p0_top = x.p0; x.s->p0_len = (int64_t) x.r->prefix.size();
            }
        }
    } else if (batch_delta > 0) { // всё последовательно (крупный re-prefill, редкий путь)
        for (auto & x : items) {
            if (x.mode != PM_RETR && x.mode != PM_DELTA) continue;
            std::string e;
            if (!trunk_extend(v, *x.s, x.r->prefix, x.p0, e) || x.p0 < 0) {
                reset_session(v, *x.s); x.ok = false;
                x.r->pr.set_value({500, json{{"error", "delta seq: " + e}}});
            }
        }
    }
    for (auto * x : miss) { // оставшиеся (реальные ретракции/рассогласования, редкие)
        if (!x->ok || x->p0 >= 0) continue;
        std::string e;
        if (!trunk_extend(v, *x->s, x->r->prefix, x->p0, e) || x->p0 < 0) {
            reset_session(v, *x->s); x->p0 = -1; x->ok = false;
            x->r->pr.set_value({500, json{{"error", "p0 miss: " + (e.empty() ? std::string("no p0") : e)}}});
        }
    }
    items.erase(std::remove_if(items.begin(), items.end(), [](sitem & y){ return !y.ok || y.p0 < 0; }), items.end());
    if (items.empty()) return;
    const int64_t t_delta = ggml_time_us();

    // ---------- фаза 2: ветки + общий батч цепочек (паддинг до K_max, D-015) ----------
    std::vector<int> tok2item;
    llama_batch b2 = llama_batch_init((int) sumKpad, 0, 1);
    for (size_t ii = 0; ii < items.size(); ii++) {
        auto & x = items[ii];
        const int64_t L = (int64_t) x.r->prefix.size();
        const llama_seq_id tr = (llama_seq_id) (2 * x.s->slot + 1);
        const llama_seq_id br = (llama_seq_id) (2 * x.s->slot + 2);
        llama_memory_seq_rm(v.mem, br, -1, -1);
        llama_memory_seq_cp(v.mem, tr, br, 0, (llama_pos) L); // мета-шаринг ствола (D-006)
        const llama_token pad_tok = x.r->chain.back();
        for (int64_t k = 0; k < Kmax; k++) {
            const int t = b2.n_tokens;
            b2.token[t] = (k < (int64_t) x.r->chain.size()) ? x.r->chain[(size_t) k] : pad_tok;
            b2.pos[t] = (llama_pos) (L + k);
            b2.n_seq_id[t] = 1; b2.seq_id[t][0] = br; b2.logits[t] = 1;
            b2.n_tokens++;
            tok2item.push_back((int) ii);
        }
    }
    std::string e2;
    if (!decode_full(v, b2, e2)) {
        llama_batch_free(b2);
        for (auto & x : items) x.r->pr.set_value({500, json{{"error", "chain decode: " + e2}}});
        return;
    }
    std::vector<tokens_t> targ(items.size());
    for (int t = 0; t < b2.n_tokens; t++) {
        const float * rp = llama_get_logits_ith(v.ctx, t);
        if (!rp) { llama_batch_free(b2);
                   for (auto & x : items) x.r->pr.set_value({500, json{{"error", "chain row missing"}}}); return; }
        targ[(size_t) tok2item[t]].push_back(argmax_row(rp, v.n_vocab));
    }
    llama_batch_free(b2);
    const int64_t t_chain = ggml_time_us();

    // ---------- фаза 3: greedy-сличение, ответы, снятие веток ----------
    for (size_t ii = 0; ii < items.size(); ii++) {
        auto & x = items[ii];
        const llama_seq_id br = (llama_seq_id) (2 * x.s->slot + 2);
        int64_t j = 0;
        while (j < (int64_t) x.r->chain.size()) {
            const int32_t want = (j == 0) ? x.p0 : targ[ii][(size_t) j - 1];
            if (x.r->chain[j] != want) break;
            j++;
        }
        int32_t bonus = x.p0;
        if (j > 0) bonus = targ[ii][(size_t) j - 1];
        json out;
        json tout = json::array();
        for (int64_t k = 0; k < j; k++) tout.push_back(x.r->chain[k]);
        tout.push_back(bonus);
        out["chain_id"]       = 0;
        out["accepted"]       = j;
        out["bonus_token"]    = bonus;
        out["tokens_out"]     = tout;
        out["queue_ms"]       = x.r->queue_ms;
        out["prefill_ms"]     = (double) (t_delta - t_p0) / 1000.0 / (double) items.size(); // доля на сессию
        out["verify_ms"]      = (double) (t_chain - t_delta) / 1000.0;                      // общий на батч
        out["round_ms"]       = (double) (ggml_time_us() - t_beg) / 1000.0 + x.r->queue_ms;
        out["batch_sessions"] = (int64_t) items.size();
        x.r->pr.set_value({200, out});
        llama_memory_seq_rm(v.mem, br, -1, -1); // ветка — полный rm (всегда ок);
        // ленивый коммит: cached остаётся == prefix (фаза 1), accepted досчитает
        // следующий раунд дельтой (D-006)
    }
    v.batches++;
    v.sess_rounds += (uint64_t) items.size();
    v.round_tokens += (uint64_t) sumKpad; // pad-строки — реальная работа декода
}

static void queue_worker(verifier & v) {
    while (true) {
        std::vector<std::shared_ptr<Pending>> batch;
        {
            std::unique_lock<std::mutex> lk(v.qmu);
            v.qcv.wait(lk, [&] { return v.stop_flag.load() || !v.queue.empty(); });
            if (v.stop_flag.load() && v.queue.empty()) break;
            const auto due = v.queue.front()->t_enq + std::chrono::milliseconds(v.p.window_ms);
            while (std::chrono::steady_clock::now() < due && !v.stop_flag.load()) {
                v.qcv.wait_until(lk, due);
            }
            if (v.stop_flag.load()) break;
            while (!v.queue.empty()) { batch.push_back(std::move(v.queue.front())); v.queue.pop_front(); }
            for (auto & r : batch) v.qbusy.erase(r->sid);
        }
        const auto t_out = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(v.mu);
            for (auto & r : batch) r->queue_ms = std::chrono::duration<double, std::milli>(t_out - r->t_enq).count();
            session_batch(v, batch);
        }
    }
}

int main(int argc, char ** argv) {
    vparams pp;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](void) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")           pp.model_path = next();
        else if (a == "--host")       pp.host = next();
        else if (a == "--port")       pp.port = std::stoi(next());
        else if (a == "-ngl")         pp.n_gpu = std::stoi(next());
        else if (a == "-c")           pp.n_ctx = std::stoi(next());
        else if (a == "-np")          pp.n_seq_max = std::stoi(next());
        else if (a == "-t")           pp.n_threads = std::stoi(next());
        else if (a == "-b")           pp.n_batch = std::stoi(next());
        else if (a == "-ub")          pp.n_ubatch = std::stoi(next());
        else if (a == "-fa")          pp.flash_attn = next() == "on";
        else if (a == "--rs")         pp.n_rs_seq = std::stoi(next());
        else if (a == "--no-mmap")    pp.no_mmap = true;
        else if (a == "--kvu")        pp.kv_unified = (next() == "on");
        else if (a == "--window")     pp.window_ms = std::stoi(next());
        else if (a == "-h" || a == "--help") {
            printf("llama-verifier -m <gguf> [--host h] [--port p] [-ngl n] [-c ctx]"
                   " [-np n_seqs] [-t n] [-b n] [-ub n] [-fa on|off] [--no-mmap]"
                   " [--rs n] [--kvu on|off] [--window ms]\n"
                   "  endpoints: /vverify (stateless), /vopen /vround /vclose /vstats (V2 sessions),\n"
                   "             /tokenize /detokenize /dbg_batch /health /props\n");
            return 0;
        } else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (pp.model_path.empty()) { fprintf(stderr, "-m required\n"); return 1; }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    verifier v;
    v.p = pp;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = pp.n_gpu;
    mp.load_mode    = pp.no_mmap ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;
    v.model = llama_model_load_from_file(pp.model_path.c_str(), mp);
    if (!v.model) { fprintf(stderr, "failed to load model %s\n", pp.model_path.c_str()); return 2; }
    v.vocab   = llama_model_get_vocab(v.model);
    v.n_vocab = llama_vocab_n_tokens(v.vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = (uint32_t) pp.n_ctx;
    cp.n_seq_max  = (uint32_t) pp.n_seq_max;
    cp.n_threads  = pp.n_threads;
    cp.n_batch    = (uint32_t) pp.n_batch;
    cp.n_ubatch   = (uint32_t) pp.n_ubatch;
    cp.n_outputs_max = (uint32_t) pp.n_batch; // батч цепочек: до n_batch output-строк
    cp.n_rs_seq   = (uint32_t) pp.n_rs_seq;   // rs-rollback для частичного seq_rm на гибриде
    cp.kv_unified = pp.kv_unified;          // ОДИН стрим: seq_cp = мета-шаринг ячеек
                                              // (cross-stream cp GGML_ABORT'ит на частичных
                                              // диапазонах и копирует весь буфер — 0.8-3.4с/раунд,
                                              // замерено 06.09 13:34; D-006)
                                              // --kvu off: эксперимент Блок 0.1 (V2-план)
    cp.flash_attn_type = pp.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    v.ctx = llama_init_from_model(v.model, cp);
    if (!v.ctx) { fprintf(stderr, "failed to init context\n"); llama_model_free(v.model); return 2; }
    v.mem = llama_get_memory(v.ctx);

    printf("verifier: model=%s n_vocab=%d n_ctx=%u n_seq_max=%d n_rs_seq=%u kv_unified=%d\n",
           pp.model_path.c_str(), v.n_vocab, llama_n_ctx(v.ctx), pp.n_seq_max,
           cp.n_rs_seq, (int) cp.kv_unified);
    fflush(stdout);

    httplib::Server srv;
    srv.set_payload_max_length((size_t) 256 * 1024 * 1024);
    // V2: /vround блокируется до ответа воркера — нужен конкурентный обработчик,
    // иначе очередь физически невозможна (httplib по умолчанию one-thread-per-connection=0)
    srv.new_task_queue = [] { return new httplib::ThreadPool(16); };

    srv.Get("/health", [&](const httplib::Request & /*req*/, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    srv.Get("/props", [&](const httplib::Request & /*req*/, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        json j;
        j["model"]           = v.p.model_path;
        j["n_ctx"]           = (int64_t) llama_n_ctx(v.ctx);
        j["kv_cached"]       = (int64_t) v.cached.size();
        j["rounds"]          = v.rounds.load();
        j["session_rounds"]  = v.sess_rounds.load();
        j["batches"]         = v.batches.load();
        j["n_sessions"]      = (int64_t) v.sessions.size();
        j["window_ms"]       = (int64_t) v.p.window_ms;
        j["round_tokens"]    = v.round_tokens.load();
        j["accepted_tokens"] = v.accepted_tokens.load();
        res.set_content(j.dump(), "application/json");
    });
    srv.Post("/tokenize", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return;
        }
        const std::string text = body.value("text", std::string());
        std::vector<llama_token> ids((size_t) std::max(8, (int) (text.size() + 16)));
        int32_t n = llama_tokenize(v.vocab, text.c_str(), (int32_t) text.size(),
                                   ids.data(), (int32_t) ids.size(),
                                   /*add_special*/ true, /*parse_special*/ true);
        if (n < 0) { // буфер мал — удваиваем (API: негатив = нужный размер)
            ids.resize((size_t) (-n));
            n = llama_tokenize(v.vocab, text.c_str(), (int32_t) text.size(),
                               ids.data(), (int32_t) ids.size(), true, true);
        }
        if (n < 0) { res.status = 500; res.set_content("{\"error\":\"tokenize failed\"}", "application/json"); return; }
        json out = json::array();
        for (int32_t i = 0; i < n; i++) out.push_back(ids[(size_t) i]);
        res.set_content(json{{"ids", out}}.dump(), "application/json");
    });
    srv.Post("/detokenize", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return;
        }
        if (!body.contains("ids") || !body["ids"].is_array()) {
            res.status = 400; res.set_content("{\"error\":\"ids required\"}", "application/json"); return;
        }
        std::vector<llama_token> ids;
        for (const auto & t : body["ids"]) ids.push_back(t.get<llama_token>());
        std::string buf((size_t) 32 * ids.size() + 64, '\0');
        int32_t n = llama_detokenize(v.vocab, ids.data(), (int32_t) ids.size(),
                                     buf.data(), (int32_t) buf.size(),
                                     /*remove_special*/ false, /*unparse_special*/ true);
        if (n < 0) {
            buf.resize((size_t) (-n));
            n = llama_detokenize(v.vocab, ids.data(), (int32_t) ids.size(),
                                 buf.data(), (int32_t) buf.size(), false, true);
        }
        if (n < 0) { res.status = 500; res.set_content("{\"error\":\"detok overflow\"}", "application/json"); return; }
        buf.resize((size_t) n);
        res.set_content(json{{"text", buf}}.dump(), "application/json");
    });
    srv.Post("/dbg_batch", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        std::string err;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return;
        }
        json out = dbg_batch(v, body, err);
        if (out.is_null()) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + err + "\"}", "application/json");
            return;
        }
        res.set_content(out.dump(), "application/json");
    });
    srv.Post("/vverify", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"invalid json\"}", "application/json"); return; }
        std::string err;
        json out = round_verify(v, body, err);
        if (!err.empty()) {
            json e; e["error"] = err;
            res.status = (err.find("not implemented") != std::string::npos ||
                          err.find("requires") != std::string::npos ||
                          err.find("must") != std::string::npos ||
                          err.find("exceeds") != std::string::npos ||
                          err.find("out of vocab") != std::string::npos ||
                          err.find("too many") != std::string::npos ||
                          err.find("sessions are active") != std::string::npos) ? 400 : 500;
            res.set_content(e.dump(), "application/json");
            return;
        }
        res.set_content(out.dump(), "application/json");
    });

    // ================== V2: эндпоинты сессий ==================
    srv.Post("/vopen", [&](const httplib::Request & /*req*/, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        if (v.p.n_seq_max < 3) { res.status = 500;
            res.set_content("{\"error\":\"need --np >= 3 for session mode\"}", "application/json"); return; }
        int32_t slot = find_free_slot(v);
        if (slot < 0) { res.status = 429; res.set_header("Retry-After", "1");
            res.set_content("{\"error\":\"no free slots (np/2 sessions in use)\"}", "application/json"); return; }
        vsession s; s.slot = slot;
        int64_t sid = v.sid_next++;
        v.sessions[sid] = s;
        v.slot_owner[2 * slot + 1] = sid;
        json out; out["session_id"] = sid; out["slot"] = slot;
        out["max_sessions"] = (v.p.n_seq_max - 1) / 2;
        res.set_content(out.dump(), "application/json");
    });
    srv.Post("/vclose", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        const int64_t sid = body.value("session_id", (int64_t) -1);
        std::lock_guard<std::mutex> lock(v.mu);
        auto it = v.sessions.find(sid);
        if (it == v.sessions.end()) { res.status = 404;
            res.set_content("{\"error\":\"no such session\"}", "application/json"); return; }
        v.slot_owner.erase(2 * it->second.slot + 1);
        v.slot_owner.erase(2 * it->second.slot + 2);
        llama_memory_seq_rm(v.mem, (llama_seq_id) (2 * it->second.slot + 1), -1, -1);
        llama_memory_seq_rm(v.mem, (llama_seq_id) (2 * it->second.slot + 2), -1, -1);
        v.sessions.erase(it);
        res.set_content("{\"status\":\"closed\"}", "application/json");
    });
    srv.Post("/vround", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return;
        }
        if (!body.contains("session_id") || !body["session_id"].is_number_integer()) {
            res.status = 400; res.set_content("{\"error\":\"session_id required\"}", "application/json"); return;
        }
        const int64_t sid = body["session_id"].get<int64_t>();
        if (!body.contains("prefix_tokens") || !body["prefix_tokens"].is_array() || body["prefix_tokens"].empty()
            || !body.contains("chain") || !body["chain"].is_array() || body["chain"].empty()) {
            res.status = 400; res.set_content("{\"error\":\"prefix_tokens and chain (non-empty arrays) required\"}", "application/json"); return;
        }
        if (body.contains("mode") && body["mode"].get<std::string>() != "greedy") {
            res.status = 400; res.set_content("{\"error\":\"mode not implemented in v2\"}", "application/json"); return;
        }
        auto vround = std::make_shared<Pending>();
        vround->sid = sid;
        for (const auto & t : body["prefix_tokens"]) {
            if (!t.is_number_integer()) { res.status = 400; res.set_content("{\"error\":\"prefix must be ints\"}", "application/json"); return; }
            vround->prefix.push_back(t.get<llama_token>());
        }
        for (const auto & t : body["chain"]) {
            if (!t.is_number_integer()) { res.status = 400; res.set_content("{\"error\":\"chain must be ints\"}", "application/json"); return; }
            vround->chain.push_back(t.get<llama_token>());
        }
        std::future<std::pair<int, json>> fut = vround->pr.get_future();
        {
            std::lock_guard<std::mutex> sm(v.mu); // sessions под v.mu (/vclose гонка)
            if (!v.sessions.count(sid)) { res.status = 404;
                res.set_content("{\"error\":\"no such session\"}", "application/json"); return; }
        }
        {
            std::lock_guard<std::mutex> lk(v.qmu);
            // справедливость V2.4: очередь/слот заняты -> честный 429, не деградация
            if ((int) v.queue.size() >= v.p.max_queue || v.qbusy.count(sid)) {
                v.rejected++;
                res.status = 429; res.set_header("Retry-After", "1");
                res.set_content("{\"error\":\"queue busy\"}", "application/json"); return;
            }
            if (!v.sessions.count(sid)) { res.status = 404;
                res.set_content("{\"error\":\"no such session\"}", "application/json"); return; }
            v.qbusy.insert(sid);
            vround->t_enq = std::chrono::steady_clock::now();
            v.queue.push_back(vround);
            v.enq++;
            v.qcv.notify_one();
        }
        auto [code, payload] = fut.get();
        res.status = code;
        res.set_content(payload.dump(), "application/json");
    });
    // V3: stateless-сэмплер «драфтера» для статистического теста exact-режима:
    // 1 токен из softmax(logits/temp); возвращает токен и его вероятность q(t).
    srv.Post("/vsample", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        if (!v.sessions.empty()) { res.status = 409; res.set_content("{\"error\":\"sessions active; /vsample is stateless-only\"}", "application/json"); return; }
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        const double temp = body.value("temp", 1.0);
        if (!(temp > 0.0)) { res.status = 400; res.set_content("{\"error\":\"temp must be > 0\"}", "application/json"); return; }
        if (!body.contains("prefix_tokens") || !body["prefix_tokens"].is_array() || body["prefix_tokens"].empty()) {
            res.status = 400; res.set_content("{\"error\":\"prefix_tokens must be non-empty array\"}", "application/json"); return; }
        tokens_t prefix;
        for (const auto & t : body["prefix_tokens"]) {
            if (!t.is_number_integer()) { res.status = 400; res.set_content("{\"error\":\"prefix must be ints\"}", "application/json"); return; }
            llama_token id = t.get<llama_token>();
            if (id < 0 || id >= v.n_vocab) { res.status = 400; res.set_content("{\"error\":\"token out of vocab\"}", "application/json"); return; }
            prefix.push_back(id);
        }
        const unsigned vseed = body.value("seed", (unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
        std::mt19937 rng(vseed);
        std::string err;
        int64_t m = 0;
        {
            const int64_t cmax = std::min((int64_t) v.cached.size(), (int64_t) prefix.size());
            while (m < cmax && v.cached[m] == prefix[m]) { m++; }
            if (m < (int64_t) v.cached.size()) {
                if (llama_memory_seq_rm(v.mem, 0, (llama_pos) m, -1)) { v.cached.resize(m); }
                else { llama_memory_seq_rm(v.mem, 0, -1, -1); v.cached.clear(); m = 0; }
                v.p0_invalidate();
            }
        }
        const int64_t L = (int64_t) prefix.size();
        tokens_t tail;
        if (m == L) { // полный дистрибутив p0: кэш argmax недостаточен — откат 1 токена
            if (llama_memory_seq_rm(v.mem, 0, (llama_pos) (L - 1), -1)) { v.cached.resize((size_t)(L - 1)); m = L - 1; }
            else { llama_memory_seq_rm(v.mem, 0, -1, -1); v.cached.clear(); m = 0; }
        }
        if (m < L) tail.assign(prefix.begin() + m, prefix.end());
        if (tail.empty()) { res.status = 409; res.set_content("{\"error\":\"empty tail\"}", "application/json"); return; }
        llama_batch b = llama_batch_init((int) tail.size(), 0, 1);
        for (size_t i = 0; i < tail.size(); i++) {
            const int k = b.n_tokens;
            b.token[k] = tail[i]; b.pos[k] = (llama_pos) (m + (llama_pos) i);
            b.n_seq_id[k] = 1; b.seq_id[k][0] = 0;
            b.logits[k] = (i + 1 == tail.size()) ? 1 : 0;
            b.n_tokens++;
        }
        if (!decode_full(v, b, err)) { llama_batch_free(b); res.status = 500; res.set_content(json{{"error", "decode: " + err}}.dump(), "application/json"); return; }
        llama_batch_free(b);
        v.cached.insert(v.cached.end(), tail.begin(), tail.end());
        v.p0_len = L; v.p0_top = -1;
        const float * row = llama_get_logits_ith(v.ctx, (int) tail.size() - 1);
        if (!row) { res.status = 500; res.set_content("{\"error\":\"no logits\"}", "application/json"); return; }
        std::vector<double> w(v.n_vocab);
        double mx = -1e30;
        for (int i = 0; i < v.n_vocab; i++) mx = std::max(mx, (double) row[i]);
        double ssum = 0;
        for (int i = 0; i < v.n_vocab; i++) { w[i] = std::exp(((double) row[i] - mx) / temp); ssum += w[i]; }
        for (auto & x : w) x /= ssum;
        std::discrete_distribution<int32_t> dd(w.begin(), w.end());
        const int32_t tok = dd(rng);
        json out; out["token"] = tok; out["prob"] = w[tok];
        res.set_content(out.dump(), "application/json");
    });

    // V3: совмещённый сэмпл+exact для статтеста: 1 декод на токен (та же математика
    // min(1,p/q) + residual-bonus, что и mode=exact).
    srv.Post("/vexact1", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(v.mu);
        if (!v.sessions.empty()) { res.status = 409; res.set_content("{\"error\":\"sessions active\"}", "application/json"); return; }
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        const double ts = body.value("temp_sample", 1.3);
        const double tt = body.value("temp_target", 0.7);
        if (!(ts > 0.0) || !(tt > 0.0)) { res.status = 400; res.set_content("{\"error\":\"temps must be > 0\"}", "application/json"); return; }
        if (!body.contains("prefix_tokens") || !body["prefix_tokens"].is_array() || body["prefix_tokens"].empty()) {
            res.status = 400; res.set_content("{\"error\":\"prefix_tokens required\"}", "application/json"); return; }
        tokens_t prefix;
        for (const auto & t : body["prefix_tokens"]) {
            if (!t.is_number_integer()) { res.status = 400; res.set_content("{\"error\":\"prefix must be ints\"}", "application/json"); return; }
            llama_token id = t.get<llama_token>();
            if (id < 0 || id >= v.n_vocab) { res.status = 400; res.set_content("{\"error\":\"token out of vocab\"}", "application/json"); return; }
            prefix.push_back(id);
        }
        const unsigned vseed = body.value("seed", (unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
        std::mt19937 rng(vseed);
        std::string err;
        int64_t m = 0;
        { const int64_t cmax = std::min((int64_t) v.cached.size(), (int64_t) prefix.size());
          while (m < cmax && v.cached[m] == prefix[m]) { m++; }
          if (m < (int64_t) v.cached.size()) {
              if (llama_memory_seq_rm(v.mem, 0, (llama_pos) m, -1)) v.cached.resize(m);
              else { llama_memory_seq_rm(v.mem, 0, -1, -1); v.cached.clear(); m = 0; } } }
        const int64_t L = (int64_t) prefix.size();
        tokens_t tail;
        if (m == L) { if (llama_memory_seq_rm(v.mem, 0, (llama_pos)(L-1), -1)) { v.cached.resize((size_t)(L-1)); m = L-1; }
                      else { llama_memory_seq_rm(v.mem, 0, -1, -1); v.cached.clear(); m = 0; } }
        if (m < L) tail.assign(prefix.begin()+m, prefix.end());
        if (tail.empty()) { res.status = 409; res.set_content("{\"error\":\"empty tail\"}", "application/json"); return; }
        llama_batch b = llama_batch_init((int) tail.size(), 0, 1);
        for (size_t i = 0; i < tail.size(); i++) { const int k = b.n_tokens;
            b.token[k]=tail[i]; b.pos[k]=(llama_pos)(m+(llama_pos)i); b.n_seq_id[k]=1; b.seq_id[k][0]=0;
            b.logits[k]=(i+1==tail.size())?1:0; b.n_tokens++; }
        if (!decode_full(v, b, err)) { llama_batch_free(b); res.status = 500; res.set_content(json{{"error","decode: "+err}}.dump(), "application/json"); return; }
        llama_batch_free(b);
        v.cached.insert(v.cached.end(), tail.begin(), tail.end());
        const float * row = llama_get_logits_ith(v.ctx, (int) tail.size()-1);
        if (!row) { res.status = 500; res.set_content("{\"error\":\"no logits\"}", "application/json"); return; }
        auto mkdist = [&](double T) { std::vector<double> w(v.n_vocab); double mx=-1e30;
            for (int i=0;i<v.n_vocab;i++) mx=std::max(mx,(double)row[i]);
            double s=0; for (int i=0;i<v.n_vocab;i++){ w[i]=std::exp(((double)row[i]-mx)/T); s+=w[i]; }
            for (auto & x : w) x/=s; return w; };
        const auto q = mkdist(ts);
        std::discrete_distribution<int32_t> dq(q.begin(), q.end());
        const int32_t t = dq(rng);
        const auto p = mkdist(tt);
        std::uniform_real_distribution<double> ud(0.0,1.0);
        int32_t out_tok; int accepted;
        if (ud(rng) < std::min(1.0, p[t]/q[t])) { out_tok = t; accepted = 1; }
        else { std::vector<double> r(v.n_vocab); double s2=0;
            for (int i=0;i<v.n_vocab;i++){ r[i]=std::max(0.0, p[i]-q[i]); s2+=r[i]; }
            if (s2>0){ std::discrete_distribution<int32_t> dd(r.begin(), r.end()); out_tok = dd(rng); }
            else out_tok = (int32_t)(std::max_element(p.begin(), p.end()) - p.begin());
            accepted = 0; }
        json o; o["token"]=out_tok; o["accepted"]=accepted; o["sampled"]=t;
        res.set_content(o.dump(), "application/json");
    });

    srv.Get("/vstats", [&](const httplib::Request & /*req*/, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(v.qmu);
        json j;
        j["queue"]    = (int64_t) v.queue.size();
        j["enq"]      = v.enq.load();
        j["rejected"] = v.rejected.load();
        j["batches"]  = v.batches.load();
        j["rounds"]   = v.sess_rounds.load();
        res.set_content(j.dump(), "application/json");
    });

    std::thread worker([&v] { queue_worker(v); });

    printf("verifier: listening on http://%s:%d\n", pp.host.c_str(), pp.port);
    fflush(stdout);
    if (!srv.listen(pp.host.c_str(), pp.port)) {
        v.stop_flag = true; v.qcv.notify_all();
        worker.join();
        llama_free(v.ctx);
        llama_model_free(v.model);
        return 3;
    }

    llama_free(v.ctx);
    llama_model_free(v.model);
    return 0;
}
