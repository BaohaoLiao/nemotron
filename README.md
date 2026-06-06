# NVIDIA Progress Prize submission

This is the Github repository to the Progress Prize winning submission for NVIDIA Nemotron Model Reasoning Challenge.

Resources on Kaggle

- [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)
- [Notebook](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85)


## Tabs on nemotron.huikang.dev

- **[Base](https://nemotron.huikang.dev/base.html)** — Grid of competition problems colored by how the **base model** (pre-fine-tuning) does on each: solved / partially solved / unsolved across its generation runs. Click a problem for its prompt, parsed transformation table, answer, per-run extracted answer, and the token-level generation trace colored by logprob.
- **[Synthetic](https://nemotron.huikang.dev/synthetic.html)** — Same problem set as Base, but colored by investigation status (rule found / hypothesis formed / rule unknown). Click a problem for its prompt, parsed transformation, answer, submission, reasoning text, and investigation notes.
- **[Corpus](https://nemotron.huikang.dev/corpus.html)** — Sortable table of training corpus entries with masked, unmasked, and total token counts per row. Filter by category or problem ID; open a row to see the token-level trace with masking highlighted.
- **[Training](https://nemotron.huikang.dev/training.html)** — Per-problem table of step, loss-token count, and minimum logprob across training epochs. Select an epoch and a row to see token-level logprob changes against the base model.
- **[Metrics](https://nemotron.huikang.dev/metrics.html)** — Index of training runs (LR, backend, epochs, batch, LoRA rank, examples, tokens, steps). Click a run to see its per-step charts: loss per token (overall and by category), min logprob by category, gradient norm, learning rate, and step time. Cmd+click a legend entry to isolate that category.


Running the webpage locally

```sh
./serve.sh
```

Serves the static site at `http://localhost:33304/`.


## Executing training


```
uv run python3 reasoning.py
uv run python3 augmentation.py
uv run python3 corpus.py
uv run python3 train_sft.py
uv run modal run upload_adapter.py
```


## Solver accuracy (v1 → v2)

Per-category rule-recovery accuracy of the original solvers (v1) versus the
improved solvers (v2). Categories marked `*` are the only ones whose solver
changed between v1 and v2; all other categories are identical.

### test_500

| Category | n | v1 | v2 | Δ |
| --- | ---: | ---: | ---: | ---: |
| bit_manipulation * | 84 | 90.5% (76) | 98.8% (83) | +8.3% |
| cipher | 83 | 100.0% (83) | 100.0% (83) | +0.0% |
| cryptarithm_deduce * | 35 | 14.3% (5) | 28.6% (10) | +14.3% |
| cryptarithm_guess * | 9 | 11.1% (1) | 11.1% (1) | +0.0% |
| equation_numeric_deduce * | 31 | 96.8% (30) | 93.5% (29) | −3.2% |
| equation_numeric_guess * | 7 | 14.3% (1) | 14.3% (1) | +0.0% |
| gravity | 84 | 100.0% (84) | 100.0% (84) | +0.0% |
| numeral | 83 | 100.0% (83) | 100.0% (83) | +0.0% |
| unit_conversion | 84 | 100.0% (84) | 100.0% (84) | +0.0% |
| **Overall** | **500** | **89.4% (447)** | **91.6% (458)** | **+2.2%** |

### train_9000

| Category | n | v1 | v2 | Δ |
| --- | ---: | ---: | ---: | ---: |
| bit_manipulation * | 1518 | 84.8% (1288) | 97.9% (1486) | +13.0% |
| cipher | 1493 | 100.0% (1493) | 100.0% (1493) | +0.0% |
| cryptarithm_deduce * | 624 | 7.9% (49) | 24.8% (155) | +17.0% |
| cryptarithm_guess * | 155 | 6.5% (10) | 6.5% (10) | +0.0% |
| equation_numeric_deduce * | 565 | 90.3% (510) | 92.7% (524) | +2.5% |
| equation_numeric_guess * | 129 | 15.5% (20) | 15.5% (20) | +0.0% |
| gravity | 1513 | 100.0% (1513) | 100.0% (1513) | +0.0% |
| numeral | 1493 | 100.0% (1493) | 100.0% (1493) | +0.0% |
| unit_conversion | 1510 | 100.0% (1510) | 100.0% (1510) | +0.0% |
| **Overall** | **9000** | **87.6% (7886)** | **91.2% (8204)** | **+3.5%** |
