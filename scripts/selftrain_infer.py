#!/usr/bin/env python3
"""Batch inference with ProtBERT + Transformer classifier on FASTA peptide sequences."""

import os, argparse, csv, json, sys
from typing import List, Tuple, Dict, Iterator, Optional
import torch
import torch.nn as nn


FIXED_ID2LABEL = {0: "class_0", 1: "class_1", 2: "class_3", 3: "class_5"}


def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    cur_id, buf = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if cur_id is not None:
                    yield cur_id, "".join(buf)
                cur_id, buf = s[1:].strip(), []
            else:
                buf.append(s)
    if cur_id is not None:
        yield cur_id, "".join(buf)


def stream_batches(fasta_path: str, batch_size: int, skip: int = 0) -> Iterator[Tuple[List[str], List[str]]]:
    batch_seqs, batch_ids, skipped = [], [], 0
    for sid, seq in iter_fasta(fasta_path):
        if skipped < skip:
            skipped += 1
            continue
        batch_seqs.append(seq); batch_ids.append(sid)
        if len(batch_seqs) == batch_size:
            yield batch_seqs, batch_ids
            batch_seqs, batch_ids = [], []
    if batch_seqs:
        yield batch_seqs, batch_ids


def preprocess_seq(seq: str) -> str:
    seq = seq.strip().upper()
    seq = seq.translate(str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"}))
    return " ".join(list(seq))


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1024, d_model=320, n_heads=8, num_layers=3,
                 ff_dim=1280, max_len=32, dropout=0.4, num_classes=4):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, attn_mask):
        B, L, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, 1, self.d_model)
        x = torch.cat([cls, x], dim=1)
        pos_ids = torch.arange(0, L + 1, device=x.device).unsqueeze(0).expand(B, L + 1)
        x = x + self.pos_embed(pos_ids)
        key_padding = torch.ones((B, L + 1), dtype=torch.bool, device=x.device)
        key_padding[:, 0] = False
        key_padding[:, 1:] = ~(attn_mask.bool())
        x = self.encoder(x, src_key_padding_mask=key_padding)
        cls_out = x[:, 0, :]
        return self.head(self.norm(cls_out))


def build_id2label(ckpt: Dict, fixed_map: Dict[int, str], num_classes: int) -> Dict[int, str]:
    id2label = None
    if "id2label" in ckpt:
        raw = ckpt["id2label"]
        id2label = {int(k): v for k, v in raw.items()}
    elif "label2id" in ckpt:
        raw = ckpt["label2id"]
        id2label = {int(v): k for k, v in raw.items()}
    if id2label is None:
        id2label = {i: fixed_map.get(i, f"class_{i}") for i in range(num_classes)}
    if len(id2label) != num_classes:
        id2label = {i: id2label.get(i, fixed_map.get(i, f"class_{i}")) for i in range(num_classes)}
    return id2label


def embed_with_protbert(tokenizer, protbert, seqs, device, amp: str, max_tok_len=1024):
    proc = [preprocess_seq(s) for s in seqs]
    enc = tokenizer(proc, add_special_tokens=True, padding=True, truncation=True,
                    max_length=max_tok_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    with torch.inference_mode():
        if device.type == "cuda" and dtype is not None:
            with torch.cuda.amp.autocast(dtype=dtype):
                out = protbert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = protbert(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = out.last_hidden_state
    pad_id = tokenizer.pad_token_id; cls_id = tokenizer.cls_token_id; sep_id = tokenizer.sep_token_id
    embs = []
    for j in range(input_ids.size(0)):
        valid = (attention_mask[j] == 1)
        if pad_id is not None: valid &= (input_ids[j] != pad_id)
        if cls_id is not None: valid &= (input_ids[j] != cls_id)
        if sep_id is not None: valid &= (input_ids[j] != sep_id)
        embs.append(last_hidden[j][valid])
    return embs


def pad_for_classifier(token_emb_list, max_tokens_for_cls):
    H = token_emb_list[0].size(-1)
    clipped = [t[:max_tokens_for_cls, :] if t.size(0) > max_tokens_for_cls else t for t in token_emb_list]
    lengths = [t.size(0) for t in clipped]
    Lmax = max(lengths) if lengths else 1
    B = len(clipped); device = clipped[0].device
    x_pad = torch.zeros((B, Lmax, H), dtype=clipped[0].dtype, device=device)
    attn = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, t in enumerate(clipped):
        L = t.size(0); x_pad[i, :L, :] = t; attn[i, :L] = 1
    return x_pad, attn


def resolve_device(device_arg: str, require_gpu: bool) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if require_gpu:
            raise RuntimeError("GPU required but not available")
        return torch.device("cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU required but not available")
    if dev.type == "cpu" and require_gpu:
        raise RuntimeError("GPU required but device set to cpu")
    return dev


def count_finished_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    n = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


@torch.inference_mode()
def run(ckpt_path, protbert_dir, fasta_path, out_csv,
        batch_size=64, device_str="auto", require_gpu=False, amp="fp16",
        protbert_max_len=1024, probs_format="json", precision=6, resume=False):

    device = resolve_device(device_str, require_gpu=require_gpu)
    if device.type == "cuda":
        idx = device.index or 0
        prop = torch.cuda.get_device_properties(idx)
        print(f"[GPU] cuda:{idx} - {prop.name}, {prop.total_memory/1024**3:.1f} GB")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    num_classes_in_head = ckpt["state_dict"]["head.weight"].shape[0]
    model = TransformerClassifier(
        input_dim=args.get("input_dim", 1024), d_model=args.get("d_model", 128),
        n_heads=args.get("n_heads", 4), num_layers=args.get("layers", 2),
        ff_dim=args.get("ff_dim", 512), max_len=args.get("max_len", 16),
        dropout=args.get("dropout", 0.3), num_classes=num_classes_in_head
    ).to(device).eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)

    id2label = build_id2label(ckpt, FIXED_ID2LABEL, num_classes_in_head)
    label_list = [id2label[i] for i in range(num_classes_in_head)]

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(protbert_dir, do_lower_case=False)
    protbert = AutoModel.from_pretrained(protbert_dir).to(device).eval()

    max_tokens_for_cls = args.get("max_len", 16) - 1
    if max_tokens_for_cls <= 0:
        raise ValueError("max_len in checkpoint must be >= 2")

    skip_n = 0
    write_header = True
    if resume and os.path.exists(out_csv):
        skip_n = count_finished_rows(out_csv)
        write_header = (skip_n == 0)
        print(f"[RESUME] Existing output: {out_csv}, {skip_n} done, resuming.")
        mode = "a"
    else:
        mode = "w"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    base_fields = ["id", "seq", "pred_id", "pred_label", "pred_conf"]
    if probs_format == "json":
        fieldnames = base_fields + ["probs_json"]
    elif probs_format == "wide":
        fieldnames = base_fields + [f"P[{lab}]" for lab in label_list]
    else:
        fieldnames = base_fields

    total = skip_n
    amp = amp.lower()
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp, None)

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for step, (seqs, ids) in enumerate(stream_batches(fasta_path, batch_size, skip=skip_n if resume else 0), 1):
            embs = embed_with_protbert(tokenizer, protbert, seqs, device=device, amp=amp, max_tok_len=protbert_max_len)
            x_pad, attn = pad_for_classifier(embs, max_tokens_for_cls=max_tokens_for_cls)
            if device.type == "cuda" and amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(x_pad, attn)
            else:
                logits = model(x_pad, attn)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            probs_np = probs.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            confs_np = confs.detach().cpu().numpy()
            for i in range(len(ids)):
                cid = int(preds_np[i])
                label = id2label.get(cid, f"class_{cid}")
                row = {"id": ids[i], "seq": seqs[i], "pred_id": cid, "pred_label": label,
                       "pred_conf": round(float(confs_np[i]), precision)}
                if probs_format == "json":
                    prob_map = {label_list[j]: round(float(probs_np[i, j]), precision) for j in range(len(label_list))}
                    row["probs_json"] = json.dumps(prob_map)
                elif probs_format == "wide":
                    for j, lab in enumerate(label_list):
                        row[f"P[{lab}]"] = round(float(probs_np[i, j]), precision)
                writer.writerow(row)
            total += len(ids)
            del embs, x_pad, attn, logits, probs, confs, preds
            if device.type == "cuda" and step % 50 == 0:
                torch.cuda.empty_cache()
            if step % 100 == 0:
                f.flush()
                print(f"[Progress] {total} sequences written", flush=True)

    print(f"Saved: {out_csv} ({total} sequences)")


def main():
    ap = argparse.ArgumentParser(description="Batch inference: ProtBERT + Transformer on FASTA")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint (.pt)")
    ap.add_argument("--protbert_dir", required=True, help="ProtBERT model directory")
    ap.add_argument("--input_fasta", required=True, help="Input FASTA file")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--protbert_max_len", type=int, default=1024)
    ap.add_argument("--probs_format", choices=["json", "wide", "none"], default="json")
    ap.add_argument("--precision", type=int, default=6, help="Decimal places for probabilities")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--require_gpu", action="store_true")
    ap.add_argument("--amp", default="fp16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = ap.parse_args()
    run(args.ckpt, args.protbert_dir, args.input_fasta, args.out_csv,
        batch_size=args.batch_size, device_str=args.device, require_gpu=args.require_gpu,
        amp=args.amp, protbert_max_len=args.protbert_max_len,
        probs_format=args.probs_format, precision=args.precision, resume=args.resume)


if __name__ == "__main__":
    main()
