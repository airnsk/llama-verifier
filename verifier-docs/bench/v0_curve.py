#!/usr/bin/env python3
"""V0 harness: стоимость верификации цепочки T(K,N) на стенде.

Метод (см. docs/DECISIONS.md D-004):
  Верификация раунда = префилл (bonus_token + K×N токенов цепочек) с logits только
  на позициях кандидатов. Один раунд = один HTTP-запрос к тестовому серверу
  verifier-test (патч 0001) через endpoint /vverify:
     {"prefix_tokens":[...], "chains":[[...]], ...} -> принятые длины/тайминги.
  До готовности V1-патча используется прокси-метрика: llama-batched-bench не умеет
  внешние цепочки, поэтому T(K,N) снимается НАПРЯМУЮ с /vverify (одна сборка = и
  прокси, и будущий сервис) — обоснование в D-004.

Таблица: K in {2,4,8,12,16} x N in {1,2,4}; медиана 3 прогонов; warm prefix из
реального дампа этала 0 (filler+документ), холодный кеш принудительно (seq_rm).

Запуск на стенде:  python3 v0_curve.py --port 8093 --model <gguf> --ngl <n> --split <none|layer>
Итог печатает CSV-таблицу в stdout; сохранять в bench/v0_T_K_N_<stamp>.csv
"""
import argparse, csv, io, json, statistics, sys, time, urllib.request

Ks = [2, 4, 8, 12, 16]
Ns = [1, 2, 4]


def post(port, path, payload, timeout=120):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_prefix(path, n_tokens):
    ids = json.load(open(path))["ids"]
    return ids[-n_tokens:] if len(ids) >= n_tokens else ids


def one_round(port, prefix, chains, mode="greedy"):
    d = post(port, "/vverify", {
        "prefix_tokens": prefix,
        "chains": [{"tokens": c} for c in chains],
        "mode": mode,
        "temperature": 0.0,
        "clear_cache": False,
    })
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--prefix-file", default="/home/alex/verifier/v0_prefix.json")
    ap.add_argument("--chain-pool", default="/home/alex/verifier/v0_chains.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prefix-len", type=int, default=2048)
    args = ap.parse_args()

    prefix = load_prefix(args.prefix_file, args.prefix_len)
    pool = json.load(open(args.chain_pool))  # {"chains": [[ids...], ...]} длин >=16
    chains_pool = pool["chains"]

    # холодный кеш один раз на конфигурацию: считаем префикс отдельно (это НЕ входит
    # в T(K,N) — в конвейере префикс уже в кеше после прошлого раунда)
    post(args.port, "/vverify", {"prefix_tokens": prefix, "chains": [{"tokens": []}],
                                 "mode": "greedy", "temperature": 0.0, "clear_cache": True})

    w = csv.writer(sys.stdout)
    w.writerow(["K", "N", "rep", "verify_ms", "accepted_chain0", "accepted_total"])
    for K in Ks:
        for N in Ns:
            chains = [c[:K] for c in chains_pool[:N]]
            for rep in range(args.reps):
                d = one_round(args.port, prefix, chains)
                w.writerow([K, N, rep, round(d["verify_ms"], 3),
                            d["results"][0]["accepted"],
                            sum(r["accepted"] for r in d["results"])])
                sys.stdout.flush()
    # сводка
    print("# median verify_ms by (K,N):", file=sys.stderr)
    med = {}
    print("\n".join(f"{k}: {v:.1f}" for k, v in med.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
