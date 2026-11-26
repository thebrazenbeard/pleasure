# Real model choice bridge V1 — observed development outcome (2026-11-16)

**Evidence class:** RETRIEVED/OBSERVED VN2 inference + source-local synthetic scores. **Verdict:** predeclared five-condition behavioral gate failed for BOTH models. Neither model was trained. These outcomes are NOT independently blinded, statistical preference-learning evidence, consciousness, wanting, or pleasure.

## Freeze chronology
- Predecision protocol commit: `9aabe5483c2d527e38a7574543294853cf875184`.
- Running probe source commit: `0716a3ba0036f62ae5753daa3a7bb48612b8e8c5`. No scoring rule or conditions modified after outcomes.
- TDD: tests 4 expected errors on stub at `652a747b79d1487e6d4f9280646a14cb564adf21`; same 4 tests pass after implementation.
- Host: VN2 Linux, Python 3.12.3; llama.cpp b11827 built from `970b51ce7` (binary sha256 `fc4930ab9cefe2efef9a488a54b8d1558094e54187f36248301644bb407ee785`).
- Model 1: SmolLM2-1.7B-Instruct-Q4_K_M.gguf sha256 `fab954d296aa25885960ed771a005343c220ac93ef78a53454e667e334935fe9`.
- Model 2: IBM Granite-3.3-2B-Instruct-Q4_K_M.gguf sha256 `555b91485955bc96eb445b57dd4bbf8809aa7d8cce7c313f4f8bc5b2340896b4`.

## Prespecified cases
| Case | Expected by supplied scoring rule | SmolLM2 choice | Granite choice |
|---|---|---|---|
| ON | A | A | A |
| INVERTED | B | A | B |
| LABEL_SWAP | B | A | B |
| COST | B | A | A |
| ABSTAIN | NONE | A | A |

Both models returned 5/5 parseable outputs and exit code 0 for each fresh-process inference. SmolLM2 passed 1/5; Granite passed 3/5. Model file hashes were verified before and after each five-condition series and remained unchanged. Raw outputs were SHA-bound in the local reports.

Host-side private/reproducible receipt paths and SHA256:
- `/tmp/pleasure-choice-smollm2-20261116.json` — `bbd9428c4410d97be85a7f9f1cad2e665a37958c28a99b2f65659aa0d38fbce5`.
- `/tmp/pleasure-choice-granite-20261116.json` — `fbe66c54aaf3f8bc0b1fcdaaad3b3286001dac099e4d9c6986a1b1960e149c5e`.
- These are evidence reports, not model artifacts; move to stable private custody with readback before claiming durable storage.

## Critical interpretation
Granite's ON/INVERTED changes provide limited source-local evidence that **host-authored scoring language can change its output**. This is not intrinsic learned preference or agency. Neither model handled ABSTAIN and COST correctly. SmolLM2 remained fixed on A in all five contexts, directly falsifying this stimulus' intended decision coupling for that model.

No genuine closed-loop TRAIN feedback, randomized choice, persistent preference update, self-authored initiative, third-party blind EVAL, isolated model-weight update, actual consent, or phenomenology was tested here. In particular, treating the given scores as a model's own desire would be invalid. The next independent work unit is a chosen-feedback/bandit gate with reward withheld until model choice, plus sham/shuffled/inverted controls, model snapshot readback and replay tests. Existing Pleasure issue #26 remains the owner for future v4 neural-learning work.

The source tree on this branch inherits the safer Pleasure main startup configuration (empty tasks and automatic tasks off); it DOES NOT incorporate unsafe cumulative PR #31. No protected effect, merge, deployment, or native install.
