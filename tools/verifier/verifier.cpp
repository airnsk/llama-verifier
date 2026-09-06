// llama-verifier — внешний верификатор спекулятивных цепочек (проект verifier-llamacpp).
//
// Протокол (docs/SPEC.md): POST /vverify
//   {"prefix_tokens":[...], "chains":[{"tokens":[...]}], "mode":"greedy", "temperature":0}
//   -> {"chain_id","accepted","bonus_token","tokens_out","verify_ms", ...}
//
// Дизайн (docs/DECISIONS.md D-004): отдельный бинарь на libllama, llama-server не патчим.
//   - Свол KV = seq 0 ("trunk"), кэш `cached` (токены, уже в KV; позиции 0..len-1).
//   - Stateless-протокол: клиент шлёт полный префикс; сервер делает longest match с
//     trunk, откатывает рассогласованный хвост (seq_rm), недостающий хвост досчитывает
//     пачкой (это "prefill_ms" в ответе — в steady state ≈ 1 бонус-токен).
//   - p0 = распределение после последнего токена префикса. Добывается ВСЕГДА прогоном
//     prefix[L-1] с logits=1: при непустом хвосте это его последняя output-строка;
//     при пустом — идемпотентная переоценка той же позиции/токена (K/V перезаписываются
//     теми же значениями; поведение откатов llama-server, см. server-context rollback).
//   - Цепочка 0 проверяется на стволе (seq 0, позиции L..); цепочки i>0 — на ветках:
//     seq_cp(trunk [0,L) -> i), свои токены на позициях L.., затем выбранная ветка
//     переносится в ствол seq_add(i, [L, L+j) -> позиции те же, seq 0), ветки rm.
//   - bonus НЕ дописывается в KV: клиент пришлёт его в префиксе следующего раунда.
//
// Границы v1 (V1=greedy): mode!=greedy и temperature!=0 — 400; q[] (exact) — 400.
//
// ⚠ РИСК, требует теста T3c (V1 выход): seq_cp веток для гибридных моделей
//   (GDN/recurrent state, Qwen3.5/3.6) может не копировать рекурентное состояние —
//   для dense-таргета (Qwen3.8-27B) не применимо, но проверить обязаны.

#include "llama.h"
#include "ggml.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::atomic<bool> g_abort{false};
static void on_signal(int) { g_abort.store(true); }

struct vparams {
    std::string model_path;
    std::string host      = "127.0.0.1";
    int         port      = 8088;
    int32_t     n_gpu     = 999;
    int32_t     n_ctx     = 8192;
    int32_t     n_seq_max = 8;      // trunk + ветки; цепочек за раунд <= n_seq_max-1+... (chain0 на стволе)
    int32_t     n_threads = 0;
    int32_t     n_batch   = 2048;
    int32_t     n_ubatch  = 512;
    bool        flash_attn = true;
    bool        no_mmap    = false;
};

struct verifier {
    vparams p;

    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    llama_memory_t  mem   = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t       n_vocab = 0;

    std::vector<llama_token> cached; // == ствол seq0
    std::mutex    mu;                // сериализация раундов (v1)

    std::atomic<uint64_t> rounds{0};
    std::atomic<uint64_t> round_tokens{0};
    std::atomic<uint64_t> accepted_tokens{0};
};

static int32_t argmax_row(const float * row, int32_t n) {
    int32_t best = 0;
    float bv = row[0];
    for (int32_t i = 1; i < n; i++) {
        if (row[i] > bv) { bv = row[i]; best = i; }
    }
    return best;
}

static bool token_ok(const verifier & v, llama_token t) {
    return t >= 0 && t < v.n_vocab;
}

// один llama_decode с проверкой полного потребления батча
static bool decode_full(verifier & v, const llama_batch & batch, std::string & err) {
    const int rc = llama_decode(v.ctx, batch);
    if (rc != 0) {
        err = "llama_decode rc=" + std::to_string(rc);
        return false;
    }
    return true;
}

// Раунд верификации. err!="" -> 400/500 по тексту.
static json round_verify(verifier & v, const json & req, std::string & err) {
    std::lock_guard<std::mutex> lock(v.mu);

    // --- разбор запроса ---
    const std::string mode = req.value("mode", std::string("greedy"));
    if (mode != "greedy") { err = "mode=" + mode + " not implemented in v1"; return {}; }
    if (req.contains("temperature") && req["temperature"].get<double>() != 0.0) {
        err = "temperature>0 not implemented in v1"; return {};
    }
    if (!req.contains("prefix_tokens") || !req["prefix_tokens"].is_array() || req["prefix_tokens"].empty()) {
        err = "prefix_tokens must be a non-empty array"; return {};
    }
    std::vector<llama_token> prefix;
    for (const auto & t : req["prefix_tokens"]) {
        if (!t.is_number_integer()) { err = "prefix_tokens must be integers"; return {}; }
        llama_token id = t.get<llama_token>();
        if (!token_ok(v, id)) { err = "prefix token id out of vocab"; return {}; }
        prefix.push_back(id);
    }
    if (!req.contains("chains") || !req["chains"].is_array() || req["chains"].empty()) {
        err = "chains must be a non-empty array"; return {};
    }
    const int n_chains = (int) req["chains"].size();
    if (n_chains > v.p.n_seq_max - 1) { // chain0 на стволе, остальные на ветках 1..n_seq_max-1
        err = "too many chains for n_seq_max"; return {};
    }
    std::vector<std::vector<llama_token>> chains(n_chains);
    int64_t maxK = 0;
    int64_t sumK = 0;
    for (int i = 0; i < n_chains; i++) {
        const auto & jc = req["chains"][i];
        if (!jc.is_object() || !jc.contains("tokens") || !jc["tokens"].is_array()) {
            err = "chains[i].tokens must be array"; return {};
        }
        if (jc.contains("q")) { err = "q[] (exact) not implemented in v1"; return {}; }
        for (const auto & t : jc["tokens"]) {
            if (!t.is_number_integer()) { err = "chain tokens must be integers"; return {}; }
            llama_token id = t.get<llama_token>();
            if (!token_ok(v, id)) { err = "chain token id out of vocab"; return {}; }
            chains[i].push_back(id);
        }
        maxK = std::max(maxK, (int64_t) chains[i].size());
        sumK += (int64_t) chains[i].size();
    }

    const int64_t L = (int64_t) prefix.size();
    const bool clear_cache = req.value("clear_cache", false);
    if (clear_cache) {
        llama_memory_seq_rm(v.mem, 0, -1, -1); // удалить seq0 целиком (p0<0,p1<0)
        for (llama_seq_id s = 1; s < v.p.n_seq_max; s++) {
            llama_memory_seq_rm(v.mem, s, -1, -1);
        }
        v.cached.clear();
    }
    if (L + std::max(maxK, (int64_t) 1) > (int64_t) llama_n_ctx(v.ctx)) {
        err = "prefix + max chain exceeds n_ctx"; return {};
    }
    if (sumK > (int64_t) v.p.n_batch) { err = "sum(chain) exceeds n_batch"; return {}; }

    // --- longest match и откат ---
    int64_t m = 0;
    const int64_t cmax = std::min((int64_t) v.cached.size(), L);
    while (m < cmax && v.cached[m] == prefix[m]) { m++; }
    if (m < (int64_t) v.cached.size()) {
        // обрезаем ствол и кэш-вектор до m
        llama_memory_seq_rm(v.mem, 0, (llama_pos) m, -1);
        v.cached.resize(m);
    }
    // m == L && cached.size() == L -> хвост пустой (переоценка последнего токена)
    // m == cached.size() < L       -> досчитываем prefix[m..L)

    const int64_t t0 = ggml_time_us();

    // --- хвост префикса (+ идемпотентная переоценка при пустом хвосте) ---
    std::vector<llama_token> tail;
    llama_pos    tail_pos;
    if (m < L) {
        tail.assign(prefix.begin() + m, prefix.end());
        tail_pos = (llama_pos) m;
    } else {
        tail.push_back(prefix.back()); // m == L == cached.size()
        tail_pos = (llama_pos) (L - 1);
    }
    int32_t bonus = -1; // argmax p0 — строка валидна до след. decode
    {
        // хвост может быть длиннее n_batch — режем на чанки; logits=1 только на
        // ПОСЛЕДНЕМ токене последнего чанка (p0), промежуточные чанки без output
        size_t off = 0;
        int  p0_idx = 0;                       // индекс токена p0 в ПОСЛЕДНЕМ чанке (для get_logits_ith)
        while (off < tail.size()) {
            const size_t cnt = std::min((size_t) v.p.n_batch, tail.size() - off);
            const bool   last_chunk = (off + cnt == tail.size());
            p0_idx = (int) cnt - 1;            // logits=1 ставим на последнем токене последнего чанка
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
        // ВАЖНО: llama_get_logits_ith(i) принимает ИНДЕКС ТОКЕНА В БАТЧЕ последнего
        // llama_decode, output_ids транслирует в строку сам (замерено: row-семантика
        // даёт ошибку 'batch.logits[0] != true').
        const float * row = llama_get_logits_ith(v.ctx, p0_idx);
        if (!row) { err = "no p0 logits"; return {}; }
        bonus = argmax_row(row, v.n_vocab);
        if (m < L) { v.cached.insert(v.cached.end(), tail.begin(), tail.end()); }
    }
    const int64_t t_a = ggml_time_us(); // конец досчёта хвоста префикса

    // --- батч цепочек: chain0 на стволе, chain i>0 на ветке i ---
    for (int i = 1; i < n_chains; i++) {
        if (!chains[i].empty()) {
            llama_memory_seq_rm(v.mem, (llama_seq_id) i, -1, -1); // подчистить ветку
            llama_memory_seq_cp(v.mem, 0, (llama_seq_id) i, 0, (llama_pos) L);
        }
    }
    // таргеты (argmax строки под каждый токен каждой цепочки); строки в порядке батча
    std::vector<std::vector<int32_t>> targ(n_chains);
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
                b.seq_id[t][0] = (llama_seq_id) i;
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
    const int64_t t1 = ggml_time_us(); // конец собственно верификации (K*N)

    // --- greedy-сличение, фиксированный порядок ---
    int     win   = -1;
    int64_t win_j = 0;
    for (int i = 0; i < n_chains && win < 0; i++) {
        int64_t j = 0;
        while (j < (int64_t) chains[i].size()) {
            const int32_t want = (j == 0) ? bonus : targ[i][j - 1];
            if (chains[i][j] != want) { break; }
            j++;
        }
        if (j > 0) { win = i; win_j = j; }
    }
    if (win < 0) { win = 0; win_j = 0; } // все j=0: бонус из p0, цепочка 0 «формальная»

    // --- коммит ---
    // Ствол сейчас содержит chain0[0..] на позициях [L, L+len0). Нужно [0, L+win_j).
    // ВНИМАНИЕ: seq_cp(src,dst,p0,p1) копирует ячейки src в dst поверх существующих
    // (как в llama-server rollback), поэтому ветку можно копировать в ствол сразу —
    // предварительный rm диапазона не обязателен, но делаем rm для ясной семантики.
    {
        const int64_t have = L + (int64_t) chains[0].size();
        const int64_t keep = L + ((win == 0) ? win_j : 0);
        if (keep < have) {
            llama_memory_seq_rm(v.mem, 0, (llama_pos) keep, -1);
        }
        if (win > 0 && win_j > 0) {
            llama_memory_seq_cp(v.mem, (llama_seq_id) win, 0,
                                (llama_pos) L, (llama_pos) (L + win_j));
        }
        for (int i = 1; i < n_chains; i++) {
            llama_memory_seq_rm(v.mem, (llama_seq_id) i, -1, -1);
        }
        // cached = prefix (длина L) + принятые токены выбранной цепочки
        v.cached.resize(L);
        v.cached.insert(v.cached.end(), chains[win].begin(),
                        chains[win].begin() + win_j);
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
    out["prefill_ms"]   = (double) (t_a - t0) / 1000.0; // досчёт хвоста префикса (0 в steady state)
    out["verify_ms"]    = (double) (t1 - t_a) / 1000.0; // батч цепочек K*N
    out["round_ms"]     = (double) (t2 - t0) / 1000.0;  // весь раунд
    out["commit_ms"]    = (double) (t2 - t1) / 1000.0;
    out["round_tokens"] = (int64_t) (tail.size() + sumK);
    out["kv_cached"]    = (int64_t) v.cached.size();

    v.rounds++;
    v.round_tokens += (uint64_t) out["round_tokens"].get<int64_t>();
    v.accepted_tokens += (uint64_t) win_j;
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
        else if (a == "--no-mmap")    pp.no_mmap = true;
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
    cp.n_outputs_max = (uint32_t) pp.n_batch; // батч цепочек: до n_batch output-строк за раунд
    cp.flash_attn_type = pp.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    v.ctx = llama_init_from_model(v.model, cp);
    if (!v.ctx) { fprintf(stderr, "failed to init context\n"); llama_model_free(v.model); return 2; }
    v.mem = llama_get_memory(v.ctx);

    printf("verifier: model=%s n_vocab=%d n_ctx=%u n_seq_max=%d\n",
           pp.model_path.c_str(), v.n_vocab, llama_n_ctx(v.ctx), pp.n_seq_max);
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
