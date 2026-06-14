"""One-off diagnostic: common-support CE + vocab inventory comparison.

Tests whether Finnish's competitive in_vocab_ce is selection bias from the
per-source in-vocab mask. Scores every source model on the identical set of
positions: those whose true next char is in ALL seven source vocabs.
"""

from pathlib import Path

import torch
from scripts.zero_shot_eval import LANGS, load_model_with_vocab
from torch.nn import functional

from char_lstm.data import UNK, encode

CKPT = Path("checkpoints")
SNIP = Path("data/perception")
TARGETS = [lang for lang in LANGS if lang != "fi"]

loaded = {
    s: load_model_with_vocab(CKPT / f"{s}_best.pt", CKPT / f"{s}_vocab.json")
    for s in LANGS
}

print("=== Vocab sizes ===")
for s in LANGS:
    print(f"  {s}: {loaded[s]['vocab_size']}")

print("\n=== Chars in each Turkic vocab that Finnish LACKS ===")
fi_chars = set(loaded["fi"]["stoi"]) - {UNK}
for s in TARGETS:
    missing = sorted((set(loaded[s]["stoi"]) - {UNK}) - fi_chars)
    print(f"  {s}: {''.join(missing)}")

print("\n=== Chars in ug snippet NOT in kk vocab (coverage 0.9997 check) ===")
ug_text = (SNIP / "perception_ug.txt").read_text(encoding="utf-8")
kk_stoi = loaded["kk"]["stoi"]
not_in_kk = sorted({c for c in ug_text[1:] if c not in kk_stoi})
print(f"  {not_in_kk}")

print("\n=== Common-support CE matrix ===")
print("(every model scored on the SAME positions: next char in all 7 vocabs)")
header = "src\\tgt " + "  ".join(f"{t:>6}" for t in TARGETS)
rows: dict[str, dict[str, float]] = {s: {} for s in LANGS}
support: dict[str, float] = {}
for t in TARGETS:
    text = (SNIP / f"perception_{t}.txt").read_text(encoding="utf-8")
    next_chars = list(text)[1:]
    mask_list = [all(c in loaded[s]["stoi"] for s in LANGS) for c in next_chars]
    mask = torch.tensor(mask_list)
    support[t] = sum(mask_list) / len(mask_list)
    for s in LANGS:
        stoi = loaded[s]["stoi"]
        vocab_size = loaded[s]["vocab_size"]
        indices = encode(text, stoi)
        inputs = torch.tensor([indices[:-1]], dtype=torch.long)
        targets_t = torch.tensor(indices[1:], dtype=torch.long)
        with torch.no_grad():
            logits, _ = loaded[s]["model"](inputs)
        losses = functional.cross_entropy(
            logits.view(-1, vocab_size), targets_t, reduction="none"
        )
        rows[s][t] = float(losses[mask].mean())

print(header)
for s in LANGS:
    print(f"  {s}    " + "  ".join(f"{rows[s][t]:6.3f}" for t in TARGETS))
print("common support fraction per target:")
print("        " + "  ".join(f"{support[t]:6.3f}" for t in TARGETS))
