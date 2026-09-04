"""A checkpoint can say what produced it.

Until 2026-09-03 nothing beside a ``*_best.pt`` said which corpus trained
it. That mattered the moment a corpus was found to be wrong: the v3 Kazakh
and Kyrgyz files carried a segment their cited description excludes, the
checkpoints on the cluster were trained under it, and no artifact on disk
distinguished those from ones trained after the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from platform_core.comparability import NO_VALUE
from platform_core.determinism_record import UNPINNED_STACK
from platform_core.environment_record import HostProbe
from platform_core.run_record import decode_run_record

from char_lstm.provenance import RUN_RECORD_SUFFIX, digest_of
from char_lstm.training_record import (
    TRAINING_EXPERIMENT,
    corpus_label,
    record_training_run,
    training_fingerprint,
)


class _FakeProbe:
    """A machine stated rather than owned."""

    def platform(self) -> str:
        """Return a fixed platform string."""
        return "Linux-5.15.0-x86_64"

    def machine(self) -> str:
        """Return a fixed architecture."""
        return "x86_64"

    def logical_cores(self) -> int:
        """Return a fixed core count."""
        return 40


def _versions(distribution: str) -> str:
    """Report a fixed version for any distribution.

    Args:
        distribution: The distribution asked about.

    Returns:
        A version string naming it, so a swap would be visible.
    """
    return f"9.9.9+{distribution}"


def _no_env(name: str) -> str | None:
    """Report an environment that exports nothing.

    Args:
        name: The variable asked about.

    Returns:
        None, always.
    """
    return None


class TestTrainingFingerprint:
    """The one axis where training and scoring must disagree."""

    def test_it_carries_the_card_it_trained_on(self) -> None:
        probe: HostProbe = _FakeProbe()
        fingerprint = training_fingerprint(
            "NVIDIA A100-SXM4-40GB", "12.4", _no_env, probe, _versions
        )

        assert fingerprint["gpu_model"] == "NVIDIA A100-SXM4-40GB"
        assert fingerprint["driver_version"] == "12.4"

    def test_a_cpu_run_states_the_card_absent_rather_than_guessing(self) -> None:
        probe: HostProbe = _FakeProbe()
        fingerprint = training_fingerprint(NO_VALUE, NO_VALUE, _no_env, probe, _versions)

        assert fingerprint["gpu_model"] == NO_VALUE
        assert fingerprint["driver_version"] == NO_VALUE

    def test_the_determinism_stack_is_reported_unpinned(self) -> None:
        """The trainer seeds three sources but pins no algorithm flags, so the
        stack is reported unpinned because that is what was configured.

        Deliberately not because a re-run was expected to disagree. It does
        not disagree -- see the measurement in ``seed_everything`` -- and the
        field would still be wrong if it said pinned, because reproducing
        once is not the same as having asked for reproducibility.
        """
        probe: HostProbe = _FakeProbe()
        fingerprint = training_fingerprint(NO_VALUE, NO_VALUE, _no_env, probe, _versions)

        assert fingerprint["determinism"]["stack"] == UNPINNED_STACK


class TestCorpusLabel:
    """The arm of a training run is the corpus it read."""

    def test_it_names_the_generation_and_the_content(self, tmp_path: Path) -> None:
        generation = tmp_path / "corpora_clean_v4"
        generation.mkdir()
        corpus = generation / "oscar_kk_ipa.txt"
        corpus.write_bytes(b"borsss")

        label = corpus_label(corpus)

        assert label == "corpora_clean_v4:" + digest_of(corpus)[:12]

    def test_two_generations_of_one_name_are_told_apart(self, tmp_path: Path) -> None:
        """The reason the digest is there at all. Five of the seven v4 files
        are byte-identical to v3 and two are not, so the directory name alone
        would report a corpus change that did not happen and miss one that did.
        """
        first = tmp_path / "a" / "corpora_clean_v4"
        second = tmp_path / "b" / "corpora_clean_v4"
        for directory, text in ((first, b"one"), (second, b"two")):
            directory.mkdir(parents=True)
            (directory / "oscar_kk_ipa.txt").write_bytes(text)

        assert corpus_label(first / "oscar_kk_ipa.txt") != corpus_label(second / "oscar_kk_ipa.txt")


class TestRecordTrainingRun:
    def test_it_writes_a_sidecar_naming_the_corpus_and_the_model(self, tmp_path: Path) -> None:
        generation = tmp_path / "corpora_clean_v4"
        generation.mkdir()
        corpus = generation / "oscar_kk_ipa.txt"
        corpus.write_bytes(b"kk corpus")
        checkpoint = tmp_path / "kk_best.pt"
        checkpoint.write_bytes(b"weights")

        written = record_training_run(
            checkpoint=checkpoint,
            corpus_path=corpus,
            best_val_loss=1.75,
            vocab_size=38,
            epochs_run=4,
            gpu_model=NO_VALUE,
            driver_version=NO_VALUE,
        )

        assert written == checkpoint.with_name(checkpoint.name + RUN_RECORD_SUFFIX)
        record = decode_run_record(json.loads(written.read_text(encoding="utf-8")))
        assert record["experiment"] == TRAINING_EXPERIMENT
        assert record["label"] == corpus_label(corpus)
        assert record["payload_digest"] == digest_of(checkpoint)

    def test_the_observations_are_the_numbers_a_reader_would_compare(self, tmp_path: Path) -> None:
        generation = tmp_path / "corpora_clean_v4"
        generation.mkdir()
        corpus = generation / "oscar_kk_ipa.txt"
        corpus.write_bytes(b"kk corpus")
        checkpoint = tmp_path / "kk_best.pt"
        checkpoint.write_bytes(b"weights")

        written = record_training_run(
            checkpoint=checkpoint,
            corpus_path=corpus,
            best_val_loss=1.75,
            vocab_size=38,
            epochs_run=4,
            gpu_model=NO_VALUE,
            driver_version=NO_VALUE,
        )

        record = decode_run_record(json.loads(written.read_text(encoding="utf-8")))
        assert {o["name"]: o["value"] for o in record["observations"]} == {
            "best_val_loss": 1.75,
            "epochs_run": 4.0,
            "vocab_size": 38.0,
        }

    def test_a_missing_checkpoint_raises_rather_than_reporting_nothing(
        self, tmp_path: Path
    ) -> None:
        """A completed run has a checkpoint: best validation loss starts at
        infinity, so the first epoch to finish improves on it and saves. A run
        that finished without one did not train, and returning a quiet "no
        record written" would leave that looking like success.
        """
        generation = tmp_path / "corpora_clean_v4"
        generation.mkdir()
        corpus = generation / "oscar_az_ipa.txt"
        corpus.write_bytes(b"az corpus")
        checkpoint = tmp_path / "az_best.pt"

        with pytest.raises(FileNotFoundError):
            record_training_run(
                checkpoint=checkpoint,
                corpus_path=corpus,
                best_val_loss=float("inf"),
                vocab_size=31,
                epochs_run=1,
                gpu_model=NO_VALUE,
                driver_version=NO_VALUE,
            )

        assert not checkpoint.with_name(checkpoint.name + RUN_RECORD_SUFFIX).exists()
