# Real-8B track — rare-subpopulation synthetic control on SMB-v1

Reproduces, **on the real SMB-v1 8B encoders**, the results behind the pitch: the
encoder ablation, the Flow-Expander trust check, the arch decision, and the honest
causal findings. The 2D toy (mechanism-only, laptop-fast) lives one directory up;
this track needs the 8B weights + GPUs and is designed to be run **on your own
infrastructure with your own models**.

> Research use only (models are cc-by-nc-4.0). Oncology substrate is tactical — the
> strategic target is cardiology, contingent on a cardiology encoder.

## Environment

- Python 3.12, `torch==2.6.0+cu124`, `transformers`, `pandas`, `pyarrow`,
  `scikit-learn`, `scipy`.
- `smb_utils` (your MEDS→text formatter): `pip install git+https://github.com/standardmodelbio/smb-utils.git`
- A CUDA GPU (each 8B encoder ≈ 16 GB in bf16; an A100-40GB is comfortable, an
  **H100-80GB more so** — `torch 2.6+cu124` supports Hopper/sm_90, no changes needed).
- `causal_bench` importable for the generative stack (the Flow Expander):
  run with `PYTHONPATH=/path/to/causal_bench`.

## Model paths

The scripts point at box paths (e.g. `/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective`).
**Edit the `QWEN` / model-path constants** at the top of `encode.py`, `fe_trust.py`,
`payoff_v6.py`, `payoff_v7.py`, `meds_gen_test.py` to your local weights. The tokenizer
must be loaded with `fix_mistral_regex=True` (already set); embeddings are the
**mean-pooled** `hidden_states[-1]`, encoded at the **`baseline_time`** anchor for the
causal covariate (pre-treatment — see the findings).

## Run order

```bash
cd demos/smb_rare_subpop/real8b
export PYTHONPATH=/path/to/causal_bench

# 1. Encoder ablation (multi-obj vs curriculum × cross-backbone), swept over n:
python run.py --n 1200 --seed 0        # encodes 4 encoders in parallel, computes metrics
python summarize.py out/results.jsonl  # decision table + ablation_summary.png

# 2. Flow-Expander trust on real 8B embeddings (targets rare / on-manifold / not memorized):
python fe_trust.py

# 3. Arch-3 check — does the CausalLM generate coherent MEDS? (spoiler: encode-only):
python meds_gen_test.py --gen_model <E_gen> --eval_model <E_eval>

# 4. Causal panels (honest): support-vs-confounding two-panel, and the real-machinery probe:
python payoff_v6.py   # Panel A augmentation (support) + Panel B QBA (confounding)
python payoff_v7.py   # cross-fit SuperLearner AIPW/TMLE on the embedding
```

## What each piece shows

| script | finding |
|---|---|
| `run.py` + `metrics.py` + `summarize.py` | regime is a **wash** (multi-obj ≈ curriculum, robust to difficulty); **mean-pool ≫ last-token**; guard is **cross-backbone** |
| `fe_trust.py` | steered generation **holds on 8B**: P(rare) 0.07→0.80, typicality 1.06×, memorize-ratio 1.14× (interpolated, not copied); PCA-128 latent preserves the rare signal |
| `meds_gen_test.py` | the CausalLM is **encode-only** — direct MEDS generation degenerates; generation lives in the Flow Expander |
| `payoff_v6.py` | **generation fixes *support*** (augmentation cuts sparse-region bias); **QBA bounds *confounding*** (mechanism) |
| `payoff_v7.py` | the **adjustment-set frontier**: adjusting for the raw embedding attenuates the effect toward null even with cross-fit doubly-robust estimators + the confounder measured |

## The causal frontier (self-contained, no 8B needed)

The `payoff_v7` finding is reproduced and *scoped* in a controlled synthetic
experiment that runs anywhere:

```bash
python -m experiments.exp50_embedding_adjustment
python -m pytest tests/test_embedding_positivity.py
```

Read-out: a high-fidelity embedding induces a **positivity trap**; a flexible outcome
model then attenuates the effect. **Fixes:** augmented DR-ATO ~halves the bias
(positivity-robust, ATO≠ATE); the **prognostic-score reduction largely recovers** the
effect (the prognostic score is a balancing score that isn't treatment-degenerate,
unlike the propensity score). Tracked in issue #206.
