"""Score the primary transcripts with FinBERT and write data/text_features.csv.

    pip install torch transformers
    python scripts/score_finbert.py --events data/events.csv --transcripts data/transcripts_text

FinBERT reads at most 512 tokens, so each call is cut into overlapping windows
(512 tokens, 64 tokens of overlap). For every window we keep the three tone
probabilities and the [CLS] vector of the last layer, then average over windows
to get one row per call.

Each window is also tagged as prepared remarks or Q&A, using the transcript's
"Questions And Answers" / "Q&A" heading (fallbacks are recorded in
`qa_split_method`). That gives the separate tone of each section.

Outputs (all derived from licensed transcripts, never commit them):
  data/text_features.csv   one row per call: tone, section tones, 768-d embedding
  data/finbert_chunks.csv  one row per window: position, section, probabilities

Runs in a few minutes on a GPU; on a laptop CPU allow one to two hours. Progress
is cached in data/finbert_cache/ so an interrupted run resumes where it stopped.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.text import load_transcripts  # noqa: E402

MODEL = "ProsusAI/finbert"
MAX_LEN, STRIDE = 512, 64

_QA_HEADING = re.compile(r"^\s*(questions? and answers?|q\s*&\s*a|\(question and answer\))\s*$",
                         re.I | re.M)
_QA_FALLBACK = re.compile(r"(first question|your question|open (it )?(up )?for questions)", re.I)


def qa_start(text):
    """Character offset where the Q&A starts, and how it was found."""
    m = _QA_HEADING.search(text)
    if m:
        return m.start(), "heading"
    # skip the operator's opening instructions, which mention questions in advance
    m = _QA_FALLBACK.search(text, int(len(text) * 0.15))
    if m:
        return m.start(), "operator_cue"
    return len(text), "not_found"


def score_call(text, tok, model, device, batch, label):
    enc = tok(text, truncation=True, max_length=MAX_LEN, stride=STRIDE,
              return_overflowing_tokens=True, return_offsets_mapping=True,
              padding="max_length", return_tensors="pt")
    probs, vecs = [], []
    with torch.no_grad():
        for start in range(0, enc["input_ids"].shape[0], batch):
            out = model(input_ids=enc["input_ids"][start:start + batch].to(device),
                        attention_mask=enc["attention_mask"][start:start + batch].to(device))
            probs.append(out.logits.softmax(-1).cpu().numpy())
            vecs.append(out.hidden_states[-1][:, 0].cpu().numpy())
    probs, vecs = np.concatenate(probs), np.concatenate(vecs)

    # character span of each window (ignoring special and padding tokens)
    offsets = enc["offset_mapping"].numpy()
    mask = enc["attention_mask"].numpy().astype(bool)
    spans = []
    for o, mk in zip(offsets, mask):
        real = o[mk & (o[:, 1] > 0)]
        spans.append((int(real[0, 0]), int(real[-1, 1])) if len(real) else (0, 0))

    qa_char, method = qa_start(text)
    chunks = pd.DataFrame({
        "chunk": np.arange(len(probs)),
        "char_start": [s for s, _ in spans], "char_end": [e for _, e in spans],
        "pos": probs[:, label["positive"]], "neg": probs[:, label["negative"]],
        "neu": probs[:, label["neutral"]]})
    mid = (chunks["char_start"] + chunks["char_end"]) / 2
    chunks["section"] = np.where(mid >= qa_char, "qa", "remarks")
    return chunks, vecs, qa_char / max(len(text), 1), method


def summarise(call_id, chunks, vecs, qa_share, method):
    row = {"call_id": call_id, "n_chunks": len(chunks),
           "tone_pos": chunks["pos"].mean(), "tone_neg": chunks["neg"].mean(),
           "tone_neu": chunks["neu"].mean()}
    row["tone_net"] = row["tone_pos"] - row["tone_neg"]
    for sec in ("remarks", "qa"):
        part = chunks[chunks["section"] == sec]
        row[f"n_chunks_{sec}"] = len(part)
        row[f"tone_net_{sec}"] = (part["pos"] - part["neg"]).mean() if len(part) else np.nan
    row["qa_start_share"] = qa_share
    row["qa_split_method"] = method
    row.update({f"emb_{j:03d}": v for j, v in enumerate(vecs.mean(0))})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--transcripts", default="data/transcripts_text")
    ap.add_argument("--out", default="data/text_features.csv")
    ap.add_argument("--chunks-out", default="data/finbert_chunks.csv")
    ap.add_argument("--cache", default="data/finbert_cache")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    events = pd.read_csv(args.events)
    ids = events.loc[events["use"] == 1, "primary_call_id"].tolist()
    texts = load_transcripts(args.transcripts, ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, output_hidden_states=True).to(device).eval()
    label = {v.lower(): k for k, v in model.config.id2label.items()}   # positive/negative/neutral
    print("FinBERT labels:", model.config.id2label, "| device:", device)

    rows, all_chunks = [], []
    for i, call_id in enumerate(ids, 1):
        cache = os.path.join(args.cache, f"{call_id}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            chunks = pd.DataFrame(z["chunks"].tolist(), columns=list(z["chunk_cols"]))
            vecs, qa_share, method = z["vecs"], float(z["qa_share"]), str(z["method"])
        else:
            chunks, vecs, qa_share, method = score_call(texts[call_id], tok, model, device,
                                                        args.batch, label)
            np.savez(cache, chunks=chunks.to_numpy(dtype=object), chunk_cols=np.array(chunks.columns),
                     vecs=vecs, qa_share=qa_share, method=method)
        rows.append(summarise(call_id, chunks, vecs, qa_share, method))
        all_chunks.append(chunks.assign(call_id=call_id))
        if i % 10 == 0 or i == len(ids):
            print(f"{i}/{len(ids)} calls scored", flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.6f")
    pd.concat(all_chunks).to_csv(args.chunks_out, index=False, float_format="%.6f")
    print(f"wrote {args.out} and {args.chunks_out}")


if __name__ == "__main__":
    main()
