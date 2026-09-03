"""What produced a checkpoint, written beside the checkpoint.

``provenance`` holds the primitives every record shares -- how a file is
digested, where a sidecar goes, how one is written. This module holds the
training concern specifically, for two reasons.

The first is that training and scoring disagree about a fingerprint axis,
and the disagreement is the point. ``scoring_fingerprint`` states the card
and driver as absent because the scoring path genuinely uses neither.
Training uses both, so reusing that fingerprint would record something
false, and a false axis is worse than a missing record: a reader has no way
to tell it is wrong.

The second is that this is a separate concern from the training loop and
belongs beside it rather than inside it. ``train.py`` is already far past
the size a module should be, and the way that happens is one useful block
at a time.

Why it exists at all: a trained model with no record of what produced it
cannot be distinguished from one trained on a superseded corpus. That is
not hypothetical here. The v3 corpora wrote shcha as a segment the Kazakh
source excludes, the checkpoints already on the cluster were trained under
it, and nothing beside them says so.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from platform_core.comparability import NO_VALUE, RunFingerprint, image_digest_from_env
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.environment_record import (
    HostProbe,
    VersionReader,
    capture_host_record,
    capture_package_versions,
    installed_version,
    stdlib_host_probe,
)
from platform_core.run_record import Observation

from char_lstm.provenance import SCORING_DISTRIBUTIONS, digest_of, write_run_record

TRAINING_EXPERIMENT = "turkic-char-lstm-base-training"
"""Stable across every language and corpus generation; the arm is the label."""


def training_fingerprint(
    gpu_model: str,
    driver_version: str,
    get_env: Callable[[str], str | None] = os.environ.get,
    probe: HostProbe | None = None,
    read_version: VersionReader = installed_version,
) -> RunFingerprint:
    """Describe the configuration this process is training under.

    Args:
        gpu_model: The card this run trains on, or :data:`NO_VALUE` when it
            is on CPU. Read by the caller, which already holds the torch
            handle, so this module needs no torch import.
        driver_version: The CUDA version behind that card, same rule.
        get_env: Reader for a process environment variable, for the image
            digest a launcher exports.
        probe: Reader for the host facts. Defaults to the stdlib probe.
        read_version: Reader for one distribution's installed version.

    Returns:
        The fingerprint, carrying the card and driver when there was one.
    """
    return RunFingerprint(
        image_digest=image_digest_from_env(get_env),
        gpu_model=gpu_model,
        driver_version=driver_version,
        # UNPINNED_STACK, and honestly. This trainer seeds Python, numpy and
        # torch but sets no deterministic-algorithm flags, and its own
        # seed_everything docstring says two seeded CUDA runs agree closely
        # rather than exactly. Claiming a pinned stack would make a re-run's
        # disagreement read as a defect rather than as the expected result.
        determinism=determinism_record(UNPINNED_STACK, {}),
        host=capture_host_record(stdlib_host_probe(os.cpu_count) if probe is None else probe),
        packages=capture_package_versions(SCORING_DISTRIBUTIONS, read_version),
    )


def corpus_label(corpus_path: Path) -> str:
    """Name the corpus a checkpoint was trained on, generation and content.

    The label is the arm discriminator on a record, and for a training run
    the arm IS the corpus: the same code over v3 and over v4 produces two
    models that must not be compared as though one experiment ran twice.
    The directory name alone would not settle it -- v3 and v4 differ for
    Kazakh and Kyrgyz and are byte-identical for the other five -- so the
    file's digest goes in beside the generation.

    Args:
        corpus_path: The corpus file this run reads.

    Returns:
        The generation directory's name and the digest, colon-joined.
    """
    return f"{corpus_path.parent.name}:{digest_of(corpus_path)[:12]}"


def record_training_run(
    checkpoint: Path,
    corpus_path: Path,
    best_val_loss: float,
    vocab_size: int,
    epochs_run: int,
    gpu_model: str,
    driver_version: str,
) -> Path:
    """Write the record describing what produced ``checkpoint``.

    Args:
        checkpoint: The best checkpoint, already written. Its bytes are
            digested, so it identifies the model rather than the run's name.
        corpus_path: The corpus file the run read.
        best_val_loss: Lowest validation loss reached.
        vocab_size: Distinct characters the model learned over.
        epochs_run: Epochs this process completed.
        gpu_model: Card trained on, or :data:`NO_VALUE` on CPU.
        driver_version: CUDA version behind it, or :data:`NO_VALUE`.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If ``checkpoint`` is not there. A completed run
            has one: the best validation loss starts at infinity, so the
            first epoch to finish improves on it and saves. A run that
            finished with no checkpoint did not train, and reporting that as
            a quiet "no record written" would leave a model-less run looking
            like a successful one.
    """
    return write_run_record(
        checkpoint,
        experiment=TRAINING_EXPERIMENT,
        label=corpus_label(corpus_path),
        observations=(
            Observation(name="best_val_loss", value=best_val_loss),
            Observation(name="epochs_run", value=float(epochs_run)),
            Observation(name="vocab_size", value=float(vocab_size)),
        ),
        fingerprint=training_fingerprint(gpu_model, driver_version),
    )


__all__ = [
    "NO_VALUE",
    "TRAINING_EXPERIMENT",
    "corpus_label",
    "record_training_run",
    "training_fingerprint",
]
