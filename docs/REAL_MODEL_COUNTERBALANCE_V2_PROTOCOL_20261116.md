# Real-model choice counterbalance v2 — source-frozen protocol
Status: EXPLORATORY predecision protocol; neither confirmatory nor independent desire learning.

## Motivation
V1's INVERTED and LABEL_SWAP cases share the same numerical A/B scores, so five cases cannot disentangle position bias, content-name bias and score-following. V2 explicitly crosses both activity positions and score/cost direction; V1's frozen five-trial results and gate are unchanged.

## Frozen host, instruments and execution
Execute isolated on VN2 Linux, without credential, network or training access. Use the same local llama.cpp b11827 binary (SHA256 fc4930ab9cefe2efef9a488a54b8d1558094e54187f36248301644bb407ee785), models SmolLM2-1.7B-Instruct-Q4_K_M SHA256 fab954d296aa25885960ed771a005343c220ac93ef78a53454e667e334935fe9 and Granite-3.3-2B-Instruct-Q4_K_M SHA256 555b91485955bc96eb445b57dd4bbf8809aa7d8cce7c313f4f8bc5b2340896b4. Verify SHA256 before first inference; recheck model afterward.

V1 prompt template, deterministic seed 17, temperature 0, 4 threads, context 512, 12 generated tokens and 75-second timeout per fresh process. No adaptation between trials. Required result output consists of a whole-token A, B, or NONE from post-prompt generated text; otherwise invalid.

## Exactly ten case order and expected responses
- SCORE_A_ARCHIVE: archive A score .85, garden B .25 => A.
- SCORE_B_ARCHIVE: archive A .25, garden B .85 => B.
- SCORE_A_GARDEN: garden A .85, archive B .25 => A.
- SCORE_B_GARDEN: garden A .25, archive B .85 => B.
- COST_A_ARCHIVE: archive A score .60 cost .05, garden B .85 cost .65 => A.
- COST_B_ARCHIVE: archive A .85 cost .65, garden B .60 cost .05 => B.
- COST_A_GARDEN: garden A .60 cost .05, archive B .85 cost .65 => A.
- COST_B_GARDEN: garden A .85 cost .65, archive B .60 cost .05 => B.
- TIE: archive A .70, garden B .70 => NONE.
- BELOW_THRESHOLD: archive A .30, garden B .20 => NONE.

Source representation is experiments/model_choice_counterbalance_probe.py, pinned in the same predecision commit as this protocol. No postdecision condition or scoring change is permitted.

## Interpretation gates and falsification
Strict exploratory score-use gate: all 10 parsed and correct, including 8 fully counterbalanced score/cost trials and two abstention trials. Report both partial successes and all failures, per model; use no posthoc rescue metric. A/B or activity-name bias, failure to consider costs, tie/threshold refusal failure and invalid output falsify strict score-use. N=10 deterministic prompts per model: no statistical reliability, no withheld-feedback learning or persistence. Even 10/10 would prove ONLY following supplied prompt evidence, not internally generated preference, attraction, wanting, pleasure, consent, consciousness or native Vera qualification. Untrained model SHA stays unchanged.
