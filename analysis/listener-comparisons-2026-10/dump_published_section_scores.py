"""Export the per-section scores behind results/zero_shot_excess_ce_skip.csv.

The published skip-mode matrix (2026-07-01, commit 2fac83a) was scored by a
pipeline the current tree no longer contains: section headers matched by the
July pattern, and common support taken from vocabulary membership rather than
corpus attestation (8802e06, 2026-08-13). Its intervals can only be re-tested
on the section scores that pipeline produced, so this runs the July code
itself and writes those scores out as JSON for
``python -m scripts.compare_listeners``, which refuses them unless they
re-render the published CSV byte for byte.

Rebuild the July tree first, with only git diff against the empty tree:

    E=$(git hash-object -t tree /dev/null)
    git diff --no-color $E 2fac83a -- scripts/zero_shot_eval.py \\
        scripts/clean_corpus.py scripts/__init__.py data/perception_clean \\
        data/assimilation.csv data/symbol_map.csv > july.patch
    mkdir july && cd july && patch -p1 < ../july.patch

then, from that directory, with the LSTM virtualenv's python:

    python <LSTM>/analysis/listener-comparisons-2026-10/dump_published_section_scores.py \\
        --checkpoint-dir <LSTM>/checkpoints_2026-02 \\
        --output-json <LSTM>/results/zero_shot_excess_ce_skip_section_scores.json

The July tree goes first on ``sys.path``, so ``scripts`` here is the July
package, not the current one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from scripts.zero_shot_eval import (
    _build_masks,
    _load_targets,
    load_sources,
    score_section,
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
cli = parser.parse_args()

args = {
    "checkpoint_dir": cli.checkpoint_dir,
    "snippet_dir": Path("data/perception_clean"),
    "output_csv": Path("unused.csv"),
    "snippet_template": "perception_{lang}.txt",
    "oov_mode": "skip",
    "assimilation_csv": Path("data/assimilation.csv"),
    "n_boot": 2000,
    "seed": 0,
}
sources = load_sources(args["checkpoint_dir"])
targets = _load_targets(args, sources)
masks = _build_masks(targets, sources)

pairs = []
for src, loaded in sources.items():
    for tgt, sections in targets.items():
        scores = [
            score_section(loaded, section, mask)
            for section, mask in zip(sections, masks[tgt], strict=True)
        ]
        pairs.append({"listener": src, "text": tgt, "sections": scores})

document = {"mode": "skip", "pairs": pairs}
cli.output_json.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {len(pairs)} pair(s) to {cli.output_json}")
