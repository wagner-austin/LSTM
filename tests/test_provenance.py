"""A results file can say what produced it.

The headline number here is ``excess_cross_entropy`` -- one model's
cross-entropy minus another's on the same positions -- and until 2026-08-28
the only record of what produced a given CSV was its filename. These tests
cover the sidecar that replaced that.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from platform_core.comparability import NO_VALUE, RunFingerprint
from platform_core.determinism_record import UNPINNED_STACK
from platform_core.environment_record import HostProbe
from platform_core.run_record import Observation, decode_run_record

from char_lstm.provenance import (
    RUN_RECORD_SUFFIX,
    SCORING_DISTRIBUTIONS,
    digest_of,
    scoring_fingerprint,
    sidecar_path,
    write_run_record,
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


def _stamped_env(name: str) -> str | None:
    """Report an environment whose launcher exported an image digest.

    Args:
        name: The variable asked about.

    Returns:
        A digest for any variable.
    """
    return "sha256:deadbeef"


def _no_env(name: str) -> str | None:
    """Report an environment that exports nothing.

    Args:
        name: The variable asked about.

    Returns:
        None, always.
    """
    return None


def _fingerprint() -> RunFingerprint:
    """Build a fingerprint from stated facts.

    Returns:
        The fingerprint.
    """
    probe: HostProbe = _FakeProbe()
    return scoring_fingerprint(_no_env, probe, _versions)


class TestTheFingerprintStatesWhatIsTrue:
    """Including the axes that are absent, which is not the same as unknown."""

    def test_it_records_the_machine_and_the_libraries_that_do_the_arithmetic(self) -> None:
        fingerprint = scoring_fingerprint(_no_env, _FakeProbe(), _versions)

        assert fingerprint["host"]["machine"] == "x86_64"
        assert fingerprint["host"]["logical_cores"] == 40
        assert [p["name"] for p in fingerprint["packages"]] == sorted(SCORING_DISTRIBUTIONS)

    def test_the_card_and_driver_are_absent_rather_than_unknown(self) -> None:
        """This evaluation names no device and moves nothing to a GPU, so it
        genuinely has no card. Saying so is what stops it comparing equal to
        a cuda run of the same code."""
        fingerprint = scoring_fingerprint(_no_env, _FakeProbe(), _versions)

        assert fingerprint["gpu_model"] == NO_VALUE
        assert fingerprint["driver_version"] == NO_VALUE

    def test_it_admits_that_nothing_was_pinned(self) -> None:
        """The scoring path sets no BLAS thread count. A record claiming a
        posture the run does not have is worse than one admitting it has
        none, because only the first is believed."""
        fingerprint = scoring_fingerprint(_no_env, _FakeProbe(), _versions)

        assert fingerprint["determinism"]["stack"] == UNPINNED_STACK
        assert fingerprint["determinism"]["settings"] == ()

    def test_it_carries_the_image_digest_a_launcher_exported(self) -> None:
        fingerprint = scoring_fingerprint(_stamped_env, _FakeProbe(), _versions)

        assert fingerprint["image_digest"] == "sha256:deadbeef"


class TestTheSidecarDescribesTheFileBesideIt:
    """A record that cannot be tied to its payload is a record of nothing."""

    def test_it_is_named_for_the_results_file(self, tmp_path: Path) -> None:
        assert sidecar_path(tmp_path / "zero_shot_excess_ce_skip.csv") == (
            tmp_path / ("zero_shot_excess_ce_skip.csv" + RUN_RECORD_SUFFIX)
        )

    def test_it_digests_the_bytes_that_were_written(self, tmp_path: Path) -> None:
        csv = tmp_path / "r.csv"
        csv.write_text("src,tgt\naz,fi\n", encoding="utf-8")

        assert digest_of(csv) == hashlib.sha256(csv.read_bytes()).hexdigest()

    def test_it_round_trips_through_the_shared_contract(self, tmp_path: Path) -> None:
        """Written by this repository, read by anything that knows
        RunRecord -- which is the entire reason for speaking that shape
        rather than inventing a local one."""
        csv = tmp_path / "r.csv"
        csv.write_text("src,tgt,excess_ce\naz,fi,2.05\n", encoding="utf-8")
        observations = (Observation(name="excess_ce.az.fi", value=2.05),)

        path = write_run_record(
            csv, "turkic-zero-shot-excess-ce", "skip", observations, _fingerprint()
        )
        decoded = decode_run_record(json.loads(path.read_text(encoding="utf-8")))

        assert decoded["experiment"] == "turkic-zero-shot-excess-ce"
        assert decoded["label"] == "skip"
        assert decoded["observations"] == observations
        assert decoded["payload_digest"] == digest_of(csv)

    def test_a_changed_results_file_changes_the_digest(self, tmp_path: Path) -> None:
        """The property the digest exists for: two runs can be checked for
        bit-identity without this layer reading a single row."""
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        first.write_text("src,tgt,excess_ce\naz,fi,2.05\n", encoding="utf-8")
        second.write_text("src,tgt,excess_ce\naz,fi,2.06\n", encoding="utf-8")

        assert digest_of(first) != digest_of(second)

    def test_writing_a_record_for_an_absent_results_file_fails(self, tmp_path: Path) -> None:
        """Rather than recording a digest of nothing."""
        with pytest.raises(FileNotFoundError):
            write_run_record(tmp_path / "missing.csv", "e", "l", (), _fingerprint())
