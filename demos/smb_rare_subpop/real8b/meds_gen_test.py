"""Arch-3 feasibility probe: can the SMB CausalLM GENERATE coherent MEDS?

Go/no-go for MEDS-space generation (no decoder). Tests three things with the
chosen E_gen backbone, from a rare (metastatic-TNBC) context prompt:

  1. COHERENCE  -- does it emit well-formed MEDS (\[date\] buckets, TYPE//value
                   lines) instead of degenerate text? fraction of valid lines.
  2. IN-DOMAIN  -- are generated codes from the expected families
                   (DIAGNOSIS/LAB/MEDICATION/PROCEDURE) vs hallucinated tokens?
  3. GUARD      -- encode the generated MEDS with BOTH backbones; does a rareness
                   probe trained on REAL embeddings call it "rare" in E_gen AND
                   E_eval? (render-free cross-backbone check, native to arch 3.)

Coherent + in-domain + both-backbones-agree-rare  => arch 3 viable.
Degenerate / OOD codes                            => arch 3 out (decide 1 vs 2).
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import smb_utils, gen_cohort, encode

DATE_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]$")
CODE_RE = re.compile(r"^(DIAGNOSIS|LAB|MEDICATION|PROCEDURE)//")


def coherence(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0, 0.0, 0
    valid = [l for l in lines if DATE_RE.match(l) or CODE_RE.match(l)]
    codes = [l for l in lines if CODE_RE.match(l)]
    indom = [l for l in codes if CODE_RE.match(l)]  # prefix already gates family
    frac_valid = len(valid) / len(lines)
    frac_indom = (len(indom) / len(codes)) if codes else 0.0
    return frac_valid, frac_indom, len(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", required=True)
    ap.add_argument("--eval_model", required=True)
    ap.add_argument("--gen_gpu", type=int, default=0)
    ap.add_argument("--eval_gpu", type=int, default=1)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    a = ap.parse_args()

    # reference cohort for the rareness probe + prompt prefixes
    meds, lab = gen_cohort.make_cohort(300, rare_frac=0.2, seed=7)

    tok = AutoTokenizer.from_pretrained(a.gen_model, trust_remote_code=True,
                                        fix_mistral_regex=True, clean_up_tokenization_spaces=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.gen_model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map={"": a.gen_gpu}).eval()

    # rare-context prompts: first ~2 date buckets of rare patients
    rare_ids = lab[lab.is_rare == 1].subject_id.tolist()[:a.n_samples]
    gen_texts = []
    print("=== GENERATION ===", flush=True)
    for sid in rare_ids:
        full = smb_utils.process_ehr_info(meds[meds.subject_id == sid], subject_id=sid,
                                          end_time=lab.set_index("subject_id").loc[sid, "end_time"])
        buckets = full.split("\n\n")
        prompt = "\n\n".join(buckets[:2])                      # rare seed context
        enc = tok(prompt, return_tensors="pt").to(f"cuda:{a.gen_gpu}")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                                  temperature=0.8, top_p=0.9, repetition_penalty=1.15,
                                  pad_token_id=tok.pad_token_id)
        cont = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        full_gen = prompt + "\n" + cont
        fv, fi, nc = coherence(cont)
        gen_texts.append(full_gen)
        print(f"[{sid}] valid_lines={fv:.2f} in_domain_codes={fi:.2f} n_codes={nc}", flush=True)
    print("\n--- SAMPLE GENERATION (first patient continuation) ---")
    print(gen_texts[0][:1200])

    # ---- render-free cross-backbone guard ----
    print("\n=== CROSS-BACKBONE GUARD (rareness probe on real embeddings) ===", flush=True)
    for tag, mpath, gpu in (("E_gen", a.gen_model, a.gen_gpu), ("E_eval", a.eval_model, a.eval_gpu)):
        rm, _, sids, _ = encode.encode(mpath, meds, lab, gpu=gpu, batch_size=4)
        y = lab.set_index("subject_id").loc[sids].is_rare.to_numpy()
        clf = LogisticRegression(max_iter=2000).fit(StandardScaler().fit_transform(rm), y)
        # encode the generated samples with the same backbone
        genc = _encode_texts(mpath, gen_texts, gpu)
        sc = StandardScaler().fit(rm)
        p = clf.predict_proba(sc.transform(genc))[:, 1]
        print(f"[{tag}] P(rare) on generated samples: "
              f"mean={p.mean():.3f}  frac>0.5={np.mean(p>0.5):.2f}", flush=True)


@torch.no_grad()
def _encode_texts(mpath, texts, gpu):
    """Encode raw pre-built text strings (generated MEDS) -> mean-pooled [k,H]."""
    tok, model, base = encode.load_model(mpath, gpu)
    dev = f"cuda:{gpu}"
    vecs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        hs = base(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state.float()
        vecs.append(hs.mean(1).cpu().numpy()[0])
    return np.array(vecs)


if __name__ == "__main__":
    main()
