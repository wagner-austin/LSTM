"""Cross-module pipeline and contract tests.

No fakes: these tests run the real cleaner, the real training loop (wandb in
its built-in disabled mode), and the real eval against each other's actual
file outputs, verifying the seams between modules:

- clean_corpus output -> char_lstm.train input (corpus contract)
- char_lstm.train output -> zero_shot_eval input (checkpoint/vocab contract)
- build_assimilation output -> zero_shot_eval.load_assimilation_map
  (producer/consumer CSV contract)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from scripts.build_assimilation import BuildArgs
from scripts.build_assimilation import run as build_assimilation_run
from scripts.clean_corpus import CORPUS_TEMPLATE, LANGS, CleanArgs, clean_corpora, clean_snippets
from scripts.zero_shot_eval import EvalArgs, load_assimilation_map
from scripts.zero_shot_eval import run as eval_run

from char_lstm import train as train_module
from char_lstm.data import UNK, load_vocab_json, save_vocab_json

REPO_SYMBOL_MAP = Path("data/symbol_map.csv")


def test_assimilation_csv_contract_with_eval_loader(tmp_path: Path) -> None:
    """The generator's CSV must parse through the eval's loader unchanged."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    itos: dict[int, str] = dict(enumerate(["a", "k", "u", UNK]))
    save_vocab_json(itos, ckpt / "az_vocab.json")
    snip = tmp_path / "snip"
    snip.mkdir()
    (snip / "perception_az.txt").write_text("qqqqqq ɯɯɯ\n", encoding="utf-8")
    build_args: BuildArgs = {
        "checkpoint_dir": ckpt,
        "snippet_dir": snip,
        "output_csv": tmp_path / "assimilation.csv",
        "snippet_template": "perception_{lang}.txt",
        "min_count": 2,
    }

    rows = build_assimilation_run(build_args)

    mapping = load_assimilation_map(build_args["output_csv"])
    assert mapping["az"] == {"q": "k", "ɯ": "u"}
    assert mapping["tr"] == {}
    assert [r["listener"] for r in rows] == ["az", "az"]


@pytest.mark.timeout(180)
def test_clean_train_eval_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleaned corpus -> real training run -> eval consumes the checkpoint."""
    # --- clean: raw corpora with duplicates and junk for all 7 languages ---
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = [(f"ʒɑŋɑləqtɑr{i:02d}" * 5)[:60] for i in range(15)]
    for lang in LANGS:
        content = list(lines)
        if lang == "az":
            content = [lines[0], "微" * 40, "short", *lines]  # dup + junk + short
        (raw / CORPUS_TEMPLATE.format(lang=lang)).write_text(
            "\n".join(content) + "\n", encoding="utf-8"
        )
    raw_snip = tmp_path / "raw_snip"
    raw_snip.mkdir()
    section = ("ʒɑŋɑləq" * 6)[:40]
    (raw_snip / "perception_az.txt").write_text(f"AZERI\n1\n{section}\n", encoding="utf-8")
    clean_args: CleanArgs = {
        "input_dir": raw,
        "output_dir": tmp_path / "cleaned",
        "symbol_map": REPO_SYMBOL_MAP,
        "snippet_dir": raw_snip,
        "snippet_output_dir": tmp_path / "snip_clean",
        "min_line_chars": 30,
        "min_ipa_ratio": 0.95,
    }
    stats = clean_corpora(clean_args)
    assert clean_snippets(clean_args) == ["az"]
    assert stats["az"]["dropped_duplicate"] == 1
    assert stats["az"]["dropped_low_ipa"] == 1

    cleaned_corpus = clean_args["output_dir"] / CORPUS_TEMPLATE.format(lang="az")
    corpus_text = cleaned_corpus.read_text(encoding="utf-8")
    assert len(corpus_text) > 700  # enough for seq_len=100 splits

    # --- train: the real training loop on the cleaned corpus ---
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(train_module, "LANGUAGES", {"az": ("Azerbaijani", str(cleaned_corpus))})
    monkeypatch.setattr(sys, "argv", ["train", "--lang", "az", "--epochs", "1", "--device", "cpu"])
    train_module.main()

    checkpoint_dir = tmp_path / "checkpoints"
    vocab_path = checkpoint_dir / "az_vocab.json"
    stoi, _itos, vocab_size, _unk = load_vocab_json(vocab_path)
    # Corpus contract: training vocab is exactly the cleaned corpus chars + UNK.
    assert set(stoi) == set(corpus_text) | {UNK}
    assert vocab_size == len(set(corpus_text)) + 1

    # --- eval: consumes the trained checkpoint directly ---
    eval_args: EvalArgs = {
        "checkpoint_dir": checkpoint_dir,
        "snippet_dir": clean_args["snippet_output_dir"],
        "output_csv": tmp_path / "results" / "pipeline.csv",
        "snippet_template": "perception_{lang}.txt",
        "oov_mode": "unk",
        "assimilation_csv": tmp_path / "unused.csv",
        "n_boot": 5,
        "seed": 0,
    }
    results = eval_run(eval_args)

    assert [(r["src"], r["tgt"]) for r in results] == [("az", "az")]
    assert results[0]["excess_ce"] == 0.0
    assert results[0]["ce"] > 0.0
    assert eval_args["output_csv"].read_text(encoding="utf-8").count("\n") == 2
