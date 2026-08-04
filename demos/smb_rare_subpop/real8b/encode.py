"""Encode a MEDS cohort through one SMB encoder; cache pooled patient embeddings.

Caches BOTH poolings from the same forward pass:
  * mean  -> attention-masked mean over tokens of hidden_states[-1]
  * last  -> final real (non-pad) token of hidden_states[-1]  (causal-LM natural pool)

Output: <out_prefix>.npz  with keys mean[n,H], last[n,H], subject_id[n], seq_len[n].

Loading matches the box's known-good test_gpu_load.py:
  fix_mistral_regex=True, clean_up_tokenization_spaces=False, dtype=bf16.
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import smb_utils


def load_model(model_path, gpu):
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        fix_mistral_regex=True, clean_up_tokenization_spaces=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": gpu},
    ).eval()
    # base transformer: last_hidden_state == hidden_states[-1], without retaining
    # the full per-layer tuple (the source of the OOM).
    base = getattr(model, "model", None)
    if base is None:
        base = model.get_decoder()
    return tok, model, base


def build_texts(meds, labels, end_time_col="end_time"):
    texts, sids = [], []
    for _, row in labels.iterrows():
        sid = row["subject_id"]
        txt = smb_utils.process_ehr_info(
            meds[meds.subject_id == sid], subject_id=sid, end_time=row[end_time_col],
        )
        texts.append(txt if txt else "[EMPTY]")
        sids.append(sid)
    return sids, texts


@torch.no_grad()
def encode(model_path, meds, labels, gpu=0, batch_size=4, max_len=2048, end_time_col="end_time"):
    tok, model, base = load_model(model_path, gpu)
    dev = f"cuda:{gpu}"
    sids, texts = build_texts(meds, labels, end_time_col=end_time_col)
    # length-sort for efficient padding; restore original order at the end
    tok_lens = [len(tok(t, truncation=True, max_length=max_len).input_ids) for t in texts]
    order = np.argsort(tok_lens)
    mean_all = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    last_all = np.zeros_like(mean_all)
    seq_all = np.zeros(len(texts), dtype=np.int32)
    t0 = time.time()
    for bi in range(0, len(order), batch_size):
        idx = order[bi:bi + batch_size]
        chunk = [texts[k] for k in idx]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(dev)
        hs = base(input_ids=enc.input_ids, attention_mask=enc.attention_mask
                  ).last_hidden_state.float()             # [b, s, H]
        mask = enc.attention_mask.unsqueeze(-1).float()
        mean_pool = ((hs * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        lengths = enc.attention_mask.sum(1) - 1
        last_pool = hs[torch.arange(hs.size(0)), lengths].cpu().numpy()
        for j, k in enumerate(idx):
            mean_all[k] = mean_pool[j]; last_all[k] = last_pool[j]
            seq_all[k] = int(enc.attention_mask[j].sum())
        del hs, enc
        if (bi // batch_size) % 10 == 0:
            print(f"  {bi+len(idx)}/{len(texts)}  ({time.time()-t0:.0f}s)", flush=True)
    return mean_all, last_all, np.array(sids), seq_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--meds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    a = ap.parse_args()
    meds = pd.read_parquet(a.meds); labels = pd.read_parquet(a.labels)
    mean_v, last_v, sids, seq = encode(a.model_path, meds, labels, a.gpu, a.batch_size)
    np.savez(a.out_prefix + ".npz", mean=mean_v, last=last_v, subject_id=sids, seq_len=seq)
    print(f"saved {a.out_prefix}.npz  mean{mean_v.shape} last{last_v.shape}  "
          f"median_seq={int(np.median(seq))} max_seq={int(seq.max())}")


if __name__ == "__main__":
    main()
