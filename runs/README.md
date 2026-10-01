# Archived runs

Every complete run that still had its artifacts on disk when this repository was assembled. Each directory holds `config.json` (the exact profile), `analysis.json` (the harness's end-of-run summary, where it was written), `training.log`, `metrics.jsonl.gz` (every logged step) and `eval.csv` (the validation records extracted from it).

| run | status | evals | best val PPL | final val PPL | params | tokens | hours |
|---|---|---:|---:|---:|---:|---:|---:|
| `gpt2_162m_pile_2b` | complete | 30 | 14.45 | 15.26 | 162,322,944 | 2,000,027,648 | 10.96 |
| `gpt2_162m_pile_250m` | complete | 19 | 27.57 | 28.39 | 162,322,944 | 250,019,840 | 1.40 |
| `aenca_v7.1_162m_pile_2b` | complete | 30 | 15.68 | 19.84 | 161,994,241 | 1,999,962,112 | 24.69 |
| `aenca_v13_162m_pile_250m` | complete (resumed from step 200) | 18 | 56.98 | 52.88 | 161,999,873 | 250,019,840 | 3.23 |
| `aenca_v15_160m_pile_250m` | complete | 19 | 522.49 | 599.75 | 159,994,661 | 250,019,840 | 7.72 |
| `aenca_v15_160m_tinystories` | complete (2,000 steps, TinyStories) | 10 | 48.54 | — | — | — | — |
| `aenca_v14.1_162m_smoke_20min` | smoke run (20 min) | 8 | 46.96 | 48.23 | 161,993,329 | 16,384,000 | 0.33 |
| `partial/aenca_v7_pile_250m_a` | aborted after 5 steps | 0 | — | — | — | — | — |
| `partial/aenca_v7_pile_250m_b` | aborted at step ~530 | 2 | 550.75 | — | — | — | — |
| `partial/aenca_v7.1_pile_250m` | aborted at step ~750 (superseded by the 2B run) | 3 | 262.42 | — | — | — | — |
| `partial/aenca_v7.1_pile_2b_a` | aborted at step 1,000 (restarted) | 1 | 284.38 | — | — | — | — |
| `partial/aenca_v7.1_pile_2b_b` | aborted at step 1 (restarted) | 0 | — | — | — | — | — |

`runs/partial/` holds restarts that were aborted within the first few hundred steps (hardware or configuration restarts); they are included for completeness and carry no result.

Not retained (quoted in the README from documents only): the v8–v12 250M runs, the v10 run (PPL 437.1), and the v5.x SmolTalk2 5B-token campaign.
