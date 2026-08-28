"""What a scored number here was produced under, in the shared shape.

WHY THIS EXISTS. The headline result of this project is a SUBTRACTION.
``zero_shot_excess_ce_*.csv`` reports ``excess_cross_entropy`` -- one model's
cross-entropy on a passage minus the target language's own model on the same
positions -- for every ordered language pair, with a paired bootstrap
confidence interval, and it is bound for publication. Until 2026-08-28 the
only record of what produced those numbers was the FILENAME: ``_2026-08-13``,
``_fixedfi``, ``_forMoldir``.

TWO AXES VARY BY CONSTRUCTION HERE AND WERE UNRECORDED.

* The card. ``slurm/train_base.sub`` is an array job on the PREEMPTIBLE
  ``free-gpu`` partition with ``--requeue``, and its own comment says the
  scheduler places tasks "across whatever is free". So two arms can train on
  two different GPUs, and a preempted arm can resume on a different card than
  it started on.
* The library. ``torch = "^2.5"`` is a caret range. Arms trained months apart
  can differ in torch minor version, and the ablation that motivated
  :mod:`platform_core.run_record` subtracted arm B from arm A across a torch
  major version for weeks before anyone noticed.

Neither of those makes a result wrong. Both make it unknowable whether two
results may be subtracted, which is the question
:mod:`platform_core.comparability` answers and the reason this project now
speaks that vocabulary instead of its own.

WHAT IS RECORDED AS "NONE", HONESTLY. The evaluation in
``scripts/zero_shot_eval`` runs on CPU -- it names no device and moves nothing
to a GPU -- so the card and driver axes are :const:`NO_VALUE`, which is a
statement that this run genuinely had no card rather than that nobody looked.
And nothing here pins the BLAS thread count, so the determinism axis says
``none``. That is true, and a true record of an unpinned run is worth strictly
more than no record: it says the numbers are attributable, and does not
pretend they are reproducible against themselves.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

from platform_core.comparability import RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.environment_record import (
    HostProbe,
    VersionReader,
    capture_host_record,
    capture_package_versions,
    installed_version,
    stdlib_host_probe,
)
from platform_core.json_utils import dump_json_str
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record

SCORING_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "torch")
"""The libraries whose arithmetic decides a cross-entropy on this stack.

Not every installed distribution: a fingerprint over all of them differs
whenever a linter is bumped, and a difference that cannot reach a log
probability makes the differences that can harder to see. ``wandb`` and
``rich`` report; they do not compute.
"""

RUN_RECORD_SUFFIX = ".runrecord.json"
"""Appended to a results CSV's stem to name its provenance sidecar.

A sidecar rather than columns in the CSV: the fingerprint is one fact about
the whole run, and repeating it on all forty-nine pair rows would invite the
reading that it varies per row.
"""


def scoring_fingerprint(
    get_env: Callable[[str], str | None] = os.environ.get,
    probe: HostProbe | None = None,
    read_version: VersionReader = installed_version,
) -> RunFingerprint:
    """Describe the configuration this process would score under.

    Args:
        get_env: Reader for a process environment variable, for the image
            digest a launcher exports.
        probe: Reader for the host facts. Defaults to the stdlib probe.
        read_version: Reader for one distribution's installed version.

    Returns:
        The fingerprint, with the card and driver axes stated as absent
        because this evaluation genuinely uses neither.
    """
    return cpu_run_fingerprint(
        determinism_record(UNPINNED_STACK, {}),
        get_env,
        capture_host_record(stdlib_host_probe(os.cpu_count) if probe is None else probe),
        capture_package_versions(SCORING_DISTRIBUTIONS, read_version),
    )


def digest_of(path: Path) -> str:
    """Hash a results file so two runs can be checked for bit-identity.

    Args:
        path: The file to digest.

    Returns:
        Its SHA-256, hex-encoded.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar_path(output_csv: Path) -> Path:
    """Name the provenance file that belongs beside a results CSV.

    Args:
        output_csv: The results file.

    Returns:
        The sidecar path, next to it.
    """
    return output_csv.with_name(output_csv.name + RUN_RECORD_SUFFIX)


def write_run_record(
    output_csv: Path,
    experiment: str,
    label: str,
    observations: tuple[Observation, ...],
    fingerprint: RunFingerprint,
) -> Path:
    """Write the provenance sidecar for a results CSV.

    Args:
        output_csv: The results file this describes, already written. Its
            bytes are digested, so it must exist.
        experiment: What was run, stable across its arms.
        label: Which arm within the experiment.
        observations: The named numbers, which
            :func:`~platform_core.run_record.run_record` sorts by name.
        fingerprint: What produced them.

    Returns:
        The path written.
    """
    record: RunRecord = run_record(
        experiment=experiment,
        label=label,
        fingerprint=fingerprint,
        observations=observations,
        payload_digest=digest_of(output_csv),
    )
    path = sidecar_path(output_csv)
    path.write_text(dump_json_str(encode_run_record(record), compact=False, indent=2), "utf-8")
    return path


__all__ = [
    "RUN_RECORD_SUFFIX",
    "SCORING_DISTRIBUTIONS",
    "digest_of",
    "scoring_fingerprint",
    "sidecar_path",
    "write_run_record",
]
