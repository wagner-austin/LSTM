"""One-off diagnostic: is kk's high self-CE the model or the snippet?

(a) Score each model on the last 5000 chars of its own training corpus and
    compare to its CE on its own perception snippet.
(b) Compare char unigram distributions: snippet vs corpus (Jensen-Shannon-ish
    total variation over shared frequent chars), and list snippet chars that
    are rare in the corpus.
"""

from collections import Counter
from pathlib import Path

import torch
from scripts.zero_shot_eval import load_model_with_vocab
from torch.nn import functional

from char_lstm.data import encode

CKPT = Path("checkpoints")
SNIP = Path("data/perception")
CORP = Path("09_Downloaded_Corpora_2")
LANGS = ("az", "kk", "ky", "tr", "ug", "uz")
TAIL = 5000


def ce_on(text: str, lang: str) -> float:
    loaded = load_model_with_vocab(CKPT / f"{lang}_best.pt", CKPT / f"{lang}_vocab.json")
    indices = encode(text, loaded["stoi"])
    inputs = torch.tensor([indices[:-1]], dtype=torch.long)
    targets = torch.tensor(indices[1:], dtype=torch.long)
    with torch.no_grad():
        logits, _ = loaded["model"](inputs)
    losses = functional.cross_entropy(
        logits.view(-1, loaded["vocab_size"]), targets, reduction="none"
    )
    return float(losses.mean())


print("=== Self-CE: own corpus tail vs own perception snippet ===")
print(f"{'lang':>4}  {'corpus_tail':>11}  {'snippet':>8}  {'gap':>6}")
for lang in LANGS:
    corpus_tail = (CORP / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")[-TAIL:]
    snippet = (SNIP / f"perception_{lang}.txt").read_text(encoding="utf-8")
    ce_corpus = ce_on(corpus_tail, lang)
    ce_snip = ce_on(snippet, lang)
    print(f"{lang:>4}  {ce_corpus:11.3f}  {ce_snip:8.3f}  {ce_snip - ce_corpus:6.3f}")

print("\n=== Snippet vs corpus char distribution (total variation distance) ===")
for lang in LANGS:
    corpus_text = (CORP / f"oscar_{lang}_ipa.txt").read_text(encoding="utf-8")
    snippet = (SNIP / f"perception_{lang}.txt").read_text(encoding="utf-8")
    c_corp = Counter(corpus_text)
    c_snip = Counter(snippet)
    n_corp = sum(c_corp.values())
    n_snip = sum(c_snip.values())
    chars = set(c_corp) | set(c_snip)
    tv = 0.5 * sum(
        abs(c_corp.get(ch, 0) / n_corp - c_snip.get(ch, 0) / n_snip) for ch in chars
    )
    # snippet chars that are rare (<0.01%) or absent in corpus
    rare = [
        ch
        for ch, cnt in c_snip.most_common()
        if cnt >= 5 and c_corp.get(ch, 0) / n_corp < 0.0001
    ]
    print(f"{lang:>4}  TV={tv:.4f}  snippet-frequent-but-corpus-rare: {rare!r}")
