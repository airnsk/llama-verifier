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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <string>
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
    std::mutex    mu;            // сериализация раундов (v1: одна очередь)

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

    // ---------- разбор/валидация ----------
    const std::string mode = req.value("mode", std::string("greedy"));
    if (mode != "greedy") { err = "mode=" + mode + " not implemented in v1"; return {}; }
    if (req.contains("temperature") && req["temperature"].get<double>() != 0.0) {
        err = "temperature>0 not implemented in v1"; return {};
    }
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
        if (jc.contains("q")) { err = "q[] (exact) not implemented in v1"; return {}; }
        for (const auto & t : jc["tokens"]) {
            if (!t.is_number_integer()) { err = "chain tokens must be integers"; return {}; }
            llama_token id = t.get<llama_token>();
            if (id < 0 || id >= v.n_vocab) { err = "chain token id out of vocab"; return {}; }
            chains[i].push_back(id);
        }
        maxK = std::max(maxK, (int64_t) chains[i].size());
        sumK += (int64_t) chains[i].size();
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
                targ[i].push_back(argmax_row(rp, v.n_vocab));
            }
        }
    }
    const int64_t t1 = ggml_time_us();

    // ---------- greedy-сличение, фиксированный порядок ----------
    int     win   = -1;
    int64_t win_j = 0;
    for (int i = 0; i < n_chains && win < 0; i++) {
        int64_t j = 0;
        while (j < (int64_t) chains[i].size()) {
            const int32_t want = (j == 0) ? p0_top : targ[i][(size_t) j - 1];
            if (chains[i][j] != want) { break; }
            j++;
        }
        if (j > 0) { win = i; win_j = j; }
    }
    int32_t bonus = p0_top;
    if (win < 0) { win = 0; win_j = 0; }
    else { bonus = targ[win][(size_t) win_j - 1]; }

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
        // ленивый: ствол = prefix (в KV уже ровно [0,L)); ветки — полным rm
        for (int i = 0; i < n_chains; i++) {
            llama_memory_seq_rm(v.mem, (llama_seq_id) (i + 1), -1, -1);
        }
    }

    const int64_t t2 = ggml_time_us();

    json out;
    json tokens_out = json::array();
    for (int64_t k = 0; k < win_j; k++) { tokens_out.push_back(chains[win][k]); }
    tokens_out.push_back(bonus);
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
// Сигнатура 14:30: батч K=16 принимает 1 токен, последовательно 16/16.
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
        else if (a == "-h" || a == "--help") {
            printf("llama-verifier -m <gguf> [--host h] [--port p] [-ngl n] [-c ctx]"
                   " [-np n_branches] [-t n] [-b n] [-ub n] [-fa on|off] [--no-mmap]\n");
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
                          err.find("must") != std::string::npos ||
                          err.find("exceeds") != std::string::npos ||
                          err.find("out of vocab") != std::string::npos ||
                          err.find("too many") != std::string::npos) ? 400 : 500;
            res.set_content(e.dump(), "application/json");
            return;
        }
        res.set_content(out.dump(), "application/json");
    });

    printf("verifier: listening on http://%s:%d\n", pp.host.c_str(), pp.port);
    fflush(stdout);
    if (!srv.listen(pp.host.c_str(), pp.port)) {
        fprintf(stderr, "listen failed\n");
        llama_free(v.ctx);
        llama_model_free(v.model);
        return 3;
    }

    llama_free(v.ctx);
    llama_model_free(v.model);
    return 0;
}
