# llama-verifier — что здесь

Это форк llama.cpp (база: чистый upstream `73a43d1f6`, без dp4a и иных патчей), поверх
которого добавлен верификатор спекулятивных цепочек:

- код: `tools/verifier/` (2 коммита над upstream + 1 фикс-коммит, история общая с upstream — смердживаемо)
- сборка: `cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-verifier -j`
- протокол/ТЗ/решения/замеры: `verifier-docs/` (README, RESULTS, SPEC, DECISIONS, PLAN-V2-V5, bench, патчи)

Запуск и тесты — см. `verifier-docs/README.md`.
