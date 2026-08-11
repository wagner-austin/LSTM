"""Seeding makes a run repeatable, and the tests say so by repeating one.

Every assertion here runs the thing twice and compares. A test that only
checked ``seed_everything`` returns a generator would pass while the
weights still differed between runs, which is the failure this exists to
catch.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from char_lstm.model import CharLSTM
from char_lstm.train import (
    DEFAULT_SEED,
    CorpusSplit,
    TrainConfig,
    _extract_args,
    build_train_config,
    create_dataloaders,
    parse_args,
    seed_everything,
)

CORPUS: CorpusSplit = {
    "train_text": "abcabcabc" * 60,
    "val_text": "abc" * 30,
    "test_text": "abc" * 30,
}
STOI = {"a": 0, "b": 1, "c": 2, "<unk>": 3}


def config_with_seed(seed: int) -> TrainConfig:
    """Build a small training config carrying one seed.

    Args:
        seed: Seed to place in the config.

    Returns:
        A config sized for a test rather than for training.
    """
    return {
        "seq_len": 10,
        "batch_size": 8,
        "num_epochs": 1,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "num_workers": 0,
        "pin_memory": False,
        "seed": seed,
    }


def first_batches(seed: int, how_many: int = 3) -> list[list[int]]:
    """Collect the first training batches produced under one seed.

    Args:
        seed: Seed to apply before building the loader.
        how_many: Number of batches to take.

    Returns:
        The input tensors of those batches, as nested lists.
    """
    config = config_with_seed(seed)
    generator = seed_everything(seed)
    loaders = create_dataloaders(CORPUS, STOI, config, generator)
    batches: list[list[int]] = []
    for index, (inputs, _targets) in enumerate(loaders["train_loader"]):
        if index >= how_many:
            break
        batches.append(inputs.flatten().tolist())
    return batches


def initial_weights(seed: int) -> list[float]:
    """Build a model under one seed and return its starting weights.

    Args:
        seed: Seed to apply before constructing the model.

    Returns:
        Every parameter of a freshly built model, flattened.
    """
    seed_everything(seed)
    model = CharLSTM(vocab_size=8, embed_dim=4, hidden_dim=8, num_layers=1, dropout=0.0)
    return [value for tensor in model.parameters() for value in tensor.flatten().tolist()]


def test_the_same_seed_gives_the_same_initial_weights() -> None:
    """Weight initialisation is the first thing a seed has to pin."""
    assert initial_weights(DEFAULT_SEED) == initial_weights(DEFAULT_SEED)


def test_a_different_seed_gives_different_initial_weights() -> None:
    """The seed is doing the work, rather than the init being constant.

    Without this the test above would pass just as well against a model
    that initialised every weight to zero.
    """
    assert initial_weights(DEFAULT_SEED) != initial_weights(DEFAULT_SEED + 1)


def test_the_same_seed_gives_the_same_batch_order() -> None:
    """Shuffling is the randomness that does not read torch's global RNG.

    It reads the generator handed to the DataLoader, so it would stay
    unpinned if seeding torch alone were treated as enough.
    """
    assert first_batches(DEFAULT_SEED) == first_batches(DEFAULT_SEED)


def test_a_different_seed_gives_a_different_batch_order() -> None:
    """The generator is reaching the loader rather than being ignored."""
    assert first_batches(DEFAULT_SEED) != first_batches(DEFAULT_SEED + 1)


def test_dropout_draws_repeat_under_one_seed() -> None:
    """The third randomness source: the dropout masks drawn while training.

    Called functionally with ``training=True`` rather than by putting a
    module into training mode, because the mask is what is under test and
    nothing here trains.
    """
    ones = torch.ones(64)

    seed_everything(DEFAULT_SEED)
    first = torch.nn.functional.dropout(ones, p=0.5, training=True).tolist()
    seed_everything(DEFAULT_SEED)
    second = torch.nn.functional.dropout(ones, p=0.5, training=True).tolist()

    assert first == second
    assert any(value == 0.0 for value in first), "dropout should have masked something"


def test_the_validation_loader_is_ordered_without_a_seed() -> None:
    """No seed is needed where nothing is shuffled.

    Stated so the docstring's claim that only the training loader takes a
    generator is checked rather than trusted.
    """
    config = config_with_seed(DEFAULT_SEED)
    loaders = create_dataloaders(CORPUS, STOI, config, seed_everything(DEFAULT_SEED))
    other = create_dataloaders(CORPUS, STOI, config, seed_everything(DEFAULT_SEED + 99))

    first = [inputs.flatten().tolist() for inputs, _ in loaders["val_loader"]]
    second = [inputs.flatten().tolist() for inputs, _ in other["val_loader"]]

    assert first == second


def test_the_seed_reaches_the_training_config() -> None:
    """A seed given on the command line is the seed the run uses."""
    args = _extract_args(
        argparse.Namespace(
            lang="tr",
            from_checkpoint=None,
            freeze_embed=False,
            epochs=1,
            lr=1e-4,
            device="cpu",
            seed=99,
        )
    )

    assert args["seed"] == 99
    assert build_train_config(args, use_cuda=False)["seed"] == 99


def test_the_seed_defaults_rather_than_being_left_to_the_clock() -> None:
    """Omitting the flag still produces a reproducible run."""
    assert _extract_args(parse_args(["--lang", "tr"]))["seed"] == DEFAULT_SEED


def test_the_seed_flag_is_read() -> None:
    """A seed given on the command line reaches the parsed arguments."""
    assert _extract_args(parse_args(["--lang", "tr", "--seed", "7"]))["seed"] == 7


def test_a_non_integer_seed_is_rejected() -> None:
    """The seed is typed, and the type is checked at the boundary."""
    with pytest.raises(TypeError, match="Expected int for seed"):
        _extract_args(
            argparse.Namespace(
                lang="tr",
                from_checkpoint=None,
                freeze_embed=False,
                epochs=1,
                lr=1e-4,
                device="cpu",
                seed="1234",
            )
        )
