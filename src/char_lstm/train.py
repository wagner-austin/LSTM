#!/usr/bin/env python3
"""Training script for character-level LSTM on Turkic languages."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Protocol, TypedDict

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader

from char_lstm._console import (
    log_config,
    log_early_stopping,
    log_epoch_result,
    log_epoch_start,
    log_epoch_val,
    log_final_results,
    log_header,
    log_info,
    log_no_improvement,
    log_progress,
    log_saved,
    log_subheader,
)
from char_lstm._types import _get_torch_load
from char_lstm.data import CharDataset, build_vocab_with_unk, load_vocab_json, save_vocab_json
from char_lstm.model import CharLSTM


class OptimizerProtocol(Protocol):
    """Protocol for PyTorch optimizer interface."""

    def zero_grad(self) -> None:
        """Zero out gradients."""
        ...

    def step(self) -> None:
        """Perform optimization step."""
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return optimizer state dict."""
        ...

    @property
    def param_groups(self) -> list[dict[str, float]]:
        """Return parameter groups."""
        ...


def _create_optimizer(model: CharLSTM, lr: float) -> OptimizerProtocol:
    """Create Adam optimizer for model.

    Args:
        model: Model to optimize.
        lr: Learning rate.

    Returns:
        Configured Adam optimizer.
    """
    _optim_module = __import__("torch.optim", fromlist=["Adam"])
    adam_cls = _optim_module.Adam
    result: OptimizerProtocol = adam_cls(model.parameters(), lr=lr)
    return result


# Language configs: code -> (name, corpus_path)
LANGUAGES: dict[str, tuple[str, str]] = {
    "tr": ("Turkish", "09_Downloaded_Corpora/oscar_tr_ipa.txt"),
    "az": ("Azerbaijani", "09_Downloaded_Corpora/oscar_az_ipa.txt"),
    "kk": ("Kazakh", "09_Downloaded_Corpora/oscar_kk_ipa.txt"),
    "ky": ("Kyrgyz", "09_Downloaded_Corpora/oscar_ky_ipa.txt"),
    "uz": ("Uzbek", "09_Downloaded_Corpora/oscar_uz_ipa.txt"),
    "ug": ("Uyghur", "09_Downloaded_Corpora/oscar_ug_ipa.txt"),
    "fi": ("Finnish", "09_Downloaded_Corpora/oscar_fi_ipa.txt"),
}


class CheckpointData(TypedDict):
    """Schema for model checkpoint files."""

    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, Tensor]
    val_loss: float
    epoch: int
    global_step: int
    vocab_size: int


class LoadedCheckpoint(TypedDict, total=False):
    """Schema for loaded checkpoint files (may be partial)."""

    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, Tensor]
    val_loss: float
    epoch: int
    global_step: int
    vocab_size: int


class TrainConfig(TypedDict):
    """Training configuration parameters."""

    seq_len: int
    batch_size: int
    num_epochs: int
    log_every: int
    patience: int
    lr: float
    train_ratio: float
    val_ratio: float
    num_workers: int
    pin_memory: bool


class WandbConfig(TypedDict):
    """Configuration logged to Weights & Biases."""

    # Model architecture
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float

    # Training settings
    seq_len: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    patience: int

    # Data splits
    train_ratio: float
    val_ratio: float
    test_ratio: float

    # Experiment metadata
    language: str
    language_code: str
    is_finetune: bool
    source_checkpoint: str
    freeze_embed: bool
    device: str


class EpochMetrics(TypedDict):
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    train_ppl: float
    val_loss: float
    val_ppl: float
    best_val_loss: float
    learning_rate: float
    epochs_no_improve: int


class RunPaths(TypedDict):
    """Paths and names for a training run."""

    run_name: str
    checkpoint_dir: Path
    vocab_json_path: Path
    checkpoint_best: Path
    source_checkpoint_path: Path


class CorpusSplit(TypedDict):
    """Split corpus text data."""

    train_text: str
    val_text: str
    test_text: str


class VocabData(TypedDict):
    """Vocabulary data from loading or building."""

    stoi: dict[str, int]
    itos: dict[int, str]
    vocab_size: int


class DataLoaders(TypedDict):
    """Container for train/val/test data loaders."""

    train_loader: DataLoader[tuple[Tensor, Tensor]]
    val_loader: DataLoader[tuple[Tensor, Tensor]]
    test_loader: DataLoader[tuple[Tensor, Tensor]]


class ModelSetup(TypedDict):
    """Model, optimizer, and criterion setup."""

    model: CharLSTM
    optimizer: OptimizerProtocol
    criterion: nn.CrossEntropyLoss
    device: torch.device
    checkpoint_save: Path


class TrainState(TypedDict):
    """Mutable training state."""

    global_step: int
    window_sum: float
    window_n: int
    best_val_loss: float
    epochs_no_improve: int


class ParsedArgs(TypedDict):
    """Parsed command-line arguments."""

    lang: str
    from_checkpoint: str | None
    freeze_embed: bool
    epochs: int
    lr: float


def _extract_args(args: argparse.Namespace) -> ParsedArgs:
    """Extract and validate arguments from Namespace.

    Args:
        args: Parsed argparse.Namespace.

    Returns:
        TypedDict with validated arguments.

    Raises:
        TypeError: If argument types are incorrect.
    """
    lang = args.lang
    if not isinstance(lang, str):
        msg = f"Expected str for lang, got {type(lang).__name__}"
        raise TypeError(msg)

    from_checkpoint = args.from_checkpoint
    if from_checkpoint is not None and not isinstance(from_checkpoint, str):
        msg = f"Expected str or None for from_checkpoint, got {type(from_checkpoint).__name__}"
        raise TypeError(msg)
    from_checkpoint_typed: str | None = from_checkpoint

    freeze_embed = args.freeze_embed
    if not isinstance(freeze_embed, bool):
        msg = f"Expected bool for freeze_embed, got {type(freeze_embed).__name__}"
        raise TypeError(msg)

    epochs = args.epochs
    if not isinstance(epochs, int):
        msg = f"Expected int for epochs, got {type(epochs).__name__}"
        raise TypeError(msg)

    lr_val = args.lr
    if not isinstance(lr_val, float):
        msg = f"Expected float for lr, got {type(lr_val).__name__}"
        raise TypeError(msg)

    return {
        "lang": lang,
        "from_checkpoint": from_checkpoint_typed,
        "freeze_embed": freeze_embed,
        "epochs": epochs,
        "lr": lr_val,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/fine-tune char-level LSTM on Turkic languages"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=list(LANGUAGES.keys()),
        help="Language code to train on (tr, az, kk, ky, uz, ug, fi)",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for fine-tuning",
    )
    parser.add_argument(
        "--freeze-embed",
        action="store_true",
        help="Freeze embedding layer during fine-tuning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: CharLSTM,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    vocab_size: int,
) -> tuple[float, float]:
    """Evaluate model on a data loader.

    Args:
        model: The model to evaluate.
        loader: DataLoader to evaluate on.
        criterion: Loss function.
        device: Device to run evaluation on.
        vocab_size: Vocabulary size for reshaping logits.

    Returns:
        Tuple of (average_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss_tensor = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss_value: float = loss_tensor.item()
        total_loss += loss_value
        n += 1
    avg_loss = total_loss / max(1, n)
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


def wb_log(data: dict[str, float | int]) -> None:
    """Log metrics to Weights & Biases if active."""
    if wandb.run is not None:
        wandb.log(data)


def wb_config(config: WandbConfig) -> None:
    """Log configuration to Weights & Biases if active."""
    if wandb.run is None:
        return
    # Use setattr to avoid untyped wandb.config.update() call
    for key, value in config.items():
        setattr(wandb.config, key, value)


def compute_gradient_norm(model: CharLSTM) -> float:
    """Compute total gradient norm across all parameters.

    Args:
        model: Model with computed gradients.

    Returns:
        L2 norm of all gradients concatenated.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm * param_norm
    return math.sqrt(total_norm_sq)


def get_learning_rate(optimizer: OptimizerProtocol) -> float:
    """Extract current learning rate from optimizer.

    Args:
        optimizer: Optimizer with param_groups.

    Returns:
        Learning rate from first param group.
    """
    param_groups = optimizer.param_groups
    first_group: dict[str, float] = param_groups[0]
    lr: float = first_group["lr"]
    return lr


def wb_log_epoch_table(epoch_history: list[EpochMetrics]) -> None:
    """Log epoch summary table to Weights & Biases.

    Args:
        epoch_history: List of metrics for each completed epoch.
    """
    if wandb.run is None or len(epoch_history) == 0:
        return

    columns = [
        "epoch",
        "train_loss",
        "train_ppl",
        "val_loss",
        "val_ppl",
        "best_val_loss",
        "learning_rate",
        "epochs_no_improve",
    ]

    data: list[list[float | int]] = []
    for metrics in epoch_history:
        row: list[float | int] = [
            metrics["epoch"],
            metrics["train_loss"],
            metrics["train_ppl"],
            metrics["val_loss"],
            metrics["val_ppl"],
            metrics["best_val_loss"],
            metrics["learning_rate"],
            metrics["epochs_no_improve"],
        ]
        data.append(row)

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"epoch_summary": table})


def next_checkpoint_path(base: Path) -> Path:
    """Find next available checkpoint path with .ft{n} suffix.

    Args:
        base: Base checkpoint path.

    Returns:
        Path that doesn't exist yet.
    """
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 1
    while True:
        cand = base.with_name(f"{stem}.ft{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def build_train_config(args: ParsedArgs, use_cuda: bool) -> TrainConfig:
    """Build training configuration with GPU optimization.

    Args:
        args: Parsed command-line arguments.
        use_cuda: Whether CUDA is available.

    Returns:
        Training configuration dictionary.
    """
    return {
        "seq_len": 100,
        "batch_size": 256 if use_cuda else 64,
        "num_epochs": args["epochs"],
        "log_every": 100,
        "patience": 1,
        "lr": args["lr"],
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "num_workers": 4 if use_cuda else 0,
        "pin_memory": use_cuda,
    }


def _load_checkpoint_state_dict(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Tensor]:
    """Load model state dict from checkpoint file.

    Internal _load* function for checkpoint loading.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load tensors to.

    Returns:
        Model state dictionary.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    # Use typed torch.load via Protocol
    load_fn = _get_torch_load()
    device_str = str(device)

    # Load as raw state dict (weights_only=True for security)
    state_dict: dict[str, Tensor] = load_fn(
        str(checkpoint_path), map_location=device_str, weights_only=True
    )
    return state_dict


def save_checkpoint(
    path: Path,
    model: CharLSTM,
    optimizer: OptimizerProtocol,
    val_loss: float,
    epoch: int,
    global_step: int,
    vocab_size: int,
) -> None:
    """Save model checkpoint.

    Saves only the model state dict for security (weights_only=True compatible).
    Metadata (optimizer, val_loss, epoch, etc.) is not persisted.

    Args:
        path: Output path.
        model: Model to save.
        optimizer: Optimizer to save (unused, kept for API compatibility).
        val_loss: Validation loss at this checkpoint (unused, kept for API).
        epoch: Current epoch number (unused, kept for API).
        global_step: Current global step (unused, kept for API).
        vocab_size: Vocabulary size (unused, kept for API).
    """
    # Save only model state dict for security (weights_only=True compatible)
    # Unused parameters kept for API stability
    _ = optimizer, val_loss, epoch, global_step, vocab_size
    state_dict: dict[str, Tensor] = model.state_dict()
    torch.save(state_dict, path)


def print_config(
    lang_name: str,
    lang_code: str,
    is_finetune: bool,
    checkpoint_path: Path,
    device: str,
    config: TrainConfig,
    freeze_embed: bool,
    output_path: Path,
) -> None:
    """Print training configuration."""
    log_header("Training config")
    log_config("Language", f"{lang_name} ({lang_code})")
    mode = f"Fine-tune from {checkpoint_path}" if is_finetune else "Train from scratch"
    log_config("Mode", mode)
    log_config("Device", device)
    log_config("Batch size", config["batch_size"])
    log_config("Workers", config["num_workers"])
    log_config("Pin memory", config["pin_memory"])
    log_config("Epochs", config["num_epochs"])
    log_config("LR", config["lr"])
    log_config("Freeze embed", freeze_embed)
    log_config("Output", str(output_path))


def build_run_paths(
    lang_code: str,
    from_checkpoint: str | None,
) -> RunPaths:
    """Build run name and checkpoint paths.

    Args:
        lang_code: Language code (e.g., 'tr', 'az').
        from_checkpoint: Optional path to source checkpoint for fine-tuning.

    Returns:
        RunPaths with all path information.
    """
    checkpoint_dir = Path("checkpoints")
    vocab_json_path = checkpoint_dir / "vocab.json"

    if from_checkpoint is not None:
        checkpoint_path_str: str = from_checkpoint
        base_name = Path(checkpoint_path_str).stem.replace("_best", "")
        run_name = f"{base_name}->{lang_code}"
        checkpoint_name = f"{base_name}_to_{lang_code}.pt"
        source_checkpoint_path = Path(checkpoint_path_str)
    else:
        run_name = f"{lang_code}-train"
        checkpoint_name = f"{lang_code}_best.pt"
        source_checkpoint_path = checkpoint_dir / checkpoint_name

    checkpoint_best = checkpoint_dir / checkpoint_name

    return {
        "run_name": run_name,
        "checkpoint_dir": checkpoint_dir,
        "vocab_json_path": vocab_json_path,
        "checkpoint_best": checkpoint_best,
        "source_checkpoint_path": source_checkpoint_path,
    }


def load_and_split_corpus(data_path: str, config: TrainConfig) -> CorpusSplit:
    """Load corpus and split into train/val/test.

    Args:
        data_path: Path to corpus file.
        config: Training configuration with split ratios.

    Returns:
        CorpusSplit with train, val, test text.
    """
    log_info(f"Loading data from {data_path}...")
    text = Path(data_path).read_text(encoding="utf-8")[:10_000_000]
    total_chars = len(text)
    train_idx = int(total_chars * config["train_ratio"])
    val_idx = int(total_chars * (config["train_ratio"] + config["val_ratio"]))

    train_text = text[:train_idx]
    val_text = text[train_idx:val_idx]
    test_text = text[val_idx:]

    log_config("Loaded", f"{total_chars:,} chars total")
    log_config("Train", f"{len(train_text):,} chars")
    log_config("Val", f"{len(val_text):,} chars")
    log_config("Test", f"{len(test_text):,} chars")

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}


def setup_vocab(
    corpus: CorpusSplit,
    is_finetune: bool,
    vocab_json_path: Path,
) -> VocabData:
    """Load existing vocab or build new one.

    Args:
        corpus: Split corpus data.
        is_finetune: Whether this is a fine-tuning run.
        vocab_json_path: Path to vocab JSON file.

    Returns:
        VocabData with stoi, itos, and vocab_size.
    """
    if is_finetune and vocab_json_path.exists():
        log_info(f"Loading vocab from {vocab_json_path}...")
        stoi, itos, vocab_size, _ = load_vocab_json(vocab_json_path)
    else:
        log_info("Building vocab from training text...")
        full_text = corpus["train_text"] + corpus["val_text"] + corpus["test_text"]
        stoi, itos, vocab_size = build_vocab_with_unk(full_text)
        save_vocab_json(itos, vocab_json_path)

    log_config("Vocab size", vocab_size)
    return {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}


def create_dataloaders(
    corpus: CorpusSplit,
    stoi: dict[str, int],
    config: TrainConfig,
) -> DataLoaders:
    """Create train/val/test data loaders.

    Args:
        corpus: Split corpus data.
        stoi: String-to-index vocabulary mapping.
        config: Training configuration.

    Returns:
        DataLoaders with train, val, test loaders.
    """
    log_info("Creating data loaders...")

    train_dataset = CharDataset(corpus["train_text"], stoi, config["seq_len"])
    val_dataset = CharDataset(corpus["val_text"], stoi, config["seq_len"])
    test_dataset = CharDataset(corpus["test_text"], stoi, config["seq_len"])

    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    log_config("Train batches", f"{len(train_loader):,}")
    log_config("Val batches", f"{len(val_loader):,}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def setup_model_and_optimizer(
    vocab_size: int,
    config: TrainConfig,
    is_finetune: bool,
    checkpoint_best: Path,
    source_checkpoint_path: Path,
    freeze_embed: bool,
) -> ModelSetup:
    """Create model, optimizer, criterion, and load checkpoint if fine-tuning.

    Args:
        vocab_size: Vocabulary size.
        config: Training configuration.
        is_finetune: Whether this is a fine-tuning run.
        checkpoint_best: Path to save best checkpoint.
        source_checkpoint_path: Path to source checkpoint for fine-tuning.
        freeze_embed: Whether to freeze embedding layer.

    Returns:
        ModelSetup with model, optimizer, criterion, device, and save path.

    Raises:
        FileNotFoundError: If fine-tuning but checkpoint not found.
    """
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    log_subheader("Model config")
    log_config("Device", str(device))
    log_config("Embed dim", embed_dim)
    log_config("Hidden dim", hidden_dim)
    log_config("Layers", num_layers)

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log_config("Parameters", f"{num_params:,}")

    lr = config["lr"]
    if is_finetune:
        lr = min(lr, 5e-5)

    optimizer = _create_optimizer(model, lr)
    criterion = nn.CrossEntropyLoss()
    log_config("Optimizer", f"Adam (lr={lr})")

    checkpoint_save = checkpoint_best
    if is_finetune:
        checkpoint_save = next_checkpoint_path(checkpoint_best)
        log_info(f"[FT] Will save fine-tuned checkpoint to: {checkpoint_save}")

    if is_finetune and source_checkpoint_path.exists():
        log_info(f"Loading checkpoint from {source_checkpoint_path}")
        state_dict = _load_checkpoint_state_dict(source_checkpoint_path, device)
        model.load_state_dict(state_dict, strict=True)
        log_info("Loaded model weights for fine-tuning.")
    elif is_finetune:
        msg = f"Checkpoint not found at {source_checkpoint_path}"
        raise FileNotFoundError(msg)

    if freeze_embed:
        for p in model.embedding.parameters():
            p.requires_grad = False
        log_info("Embedding layer frozen.")

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device,
        "checkpoint_save": checkpoint_save,
    }


def train_epoch(
    epoch: int,
    num_epochs: int,
    model: CharLSTM,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: OptimizerProtocol,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    vocab_size: int,
    config: TrainConfig,
    state: TrainState,
    checkpoint_save: Path,
) -> tuple[bool, EpochMetrics]:
    """Run a single training epoch with validation.

    Args:
        epoch: Current epoch number (0-indexed).
        num_epochs: Total number of epochs.
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        vocab_size: Vocabulary size.
        config: Training configuration.
        state: Mutable training state (updated in place).
        checkpoint_save: Path to save checkpoint.

    Returns:
        Tuple of (should_continue, epoch_metrics).
        should_continue is False if early stopping triggered.
        epoch_metrics contains summary for W&B table.
    """
    log_epoch_start(epoch, num_epochs)
    model.train()

    lr = get_learning_rate(optimizer)
    epoch_train_loss = 0.0
    epoch_train_steps = 0

    num_batches = len(train_loader)
    for step, batch in enumerate(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss_tensor = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss_tensor.backward()

        # Compute gradient norm before clipping
        grad_norm = compute_gradient_norm(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value: float = loss_tensor.item()
        epoch_train_loss += loss_value
        epoch_train_steps += 1
        state["global_step"] += 1
        state["window_sum"] += loss_value
        state["window_n"] += 1

        log_every = config["log_every"]
        is_last_step = step == num_batches - 1
        is_log_step = (state["global_step"] % log_every == 0) or is_last_step
        if is_log_step:
            avg_loss = state["window_sum"] / state["window_n"]
            avg_ppl = math.exp(avg_loss)
            wb_log(
                {
                    "train_loss": avg_loss,
                    "train_ppl": avg_ppl,
                    "grad_norm": grad_norm,
                    "learning_rate": lr,
                    "global_step": state["global_step"],
                    "epoch": epoch + 1,
                }
            )
            pct = 100 * (step + 1) / num_batches
            log_progress(epoch, num_epochs, state["global_step"], pct, avg_loss, avg_ppl)
            state["window_sum"], state["window_n"] = 0.0, 0

    # Compute epoch-level train metrics
    epoch_avg_train_loss = epoch_train_loss / max(1, epoch_train_steps)
    epoch_avg_train_ppl = math.exp(epoch_avg_train_loss)

    # Validation
    log_epoch_val(epoch, num_epochs)
    val_loss, val_ppl = evaluate(model, val_loader, criterion, device, vocab_size)
    wb_log(
        {
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "best_val_loss": min(state["best_val_loss"], val_loss),
            "epochs_no_improve": state["epochs_no_improve"],
            "epoch": epoch + 1,
            "global_step": state["global_step"],
        }
    )
    log_epoch_result(epoch, num_epochs, val_loss, val_ppl)

    # Save best
    if val_loss < state["best_val_loss"]:
        state["best_val_loss"] = val_loss
        state["epochs_no_improve"] = 0
        save_checkpoint(
            path=checkpoint_save,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            epoch=epoch,
            global_step=state["global_step"],
            vocab_size=vocab_size,
        )
        log_saved(str(checkpoint_save))
    else:
        state["epochs_no_improve"] += 1
        patience = config["patience"]
        log_no_improvement(state["epochs_no_improve"], patience)
        if state["epochs_no_improve"] >= patience:
            log_early_stopping()
            # Return metrics even on early stop
            epoch_metrics: EpochMetrics = {
                "epoch": epoch + 1,
                "train_loss": epoch_avg_train_loss,
                "train_ppl": epoch_avg_train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "best_val_loss": state["best_val_loss"],
                "learning_rate": lr,
                "epochs_no_improve": state["epochs_no_improve"],
            }
            return False, epoch_metrics

    # Build epoch metrics for summary table
    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": epoch_avg_train_loss,
        "train_ppl": epoch_avg_train_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "best_val_loss": state["best_val_loss"],
        "learning_rate": lr,
        "epochs_no_improve": state["epochs_no_improve"],
    }

    return True, epoch_metrics


def run_final_evaluation(
    model: CharLSTM,
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    vocab_size: int,
    checkpoint_save: Path,
    checkpoint_best: Path,
) -> None:
    """Run final evaluation on test set.

    Args:
        model: Model to evaluate.
        test_loader: Test data loader.
        criterion: Loss function.
        device: Device to run on.
        vocab_size: Vocabulary size.
        checkpoint_save: Path to saved checkpoint.
        checkpoint_best: Fallback checkpoint path.
    """
    log_header("Training complete. Running final evaluation...")
    best_path = checkpoint_save if checkpoint_save.exists() else checkpoint_best
    state_dict = _load_checkpoint_state_dict(best_path, device)
    model.load_state_dict(state_dict)

    test_loss, test_ppl = evaluate(model, test_loader, criterion, device, vocab_size)
    wb_log({"test_loss": test_loss, "test_ppl": test_ppl})
    log_final_results(test_loss, test_ppl)


def main() -> None:
    """Main training entry point."""
    raw_args = parse_args()
    args = _extract_args(raw_args)

    # Get language config
    lang_name, data_path = LANGUAGES[args["lang"]]
    is_finetune = args["from_checkpoint"] is not None

    # Build paths
    paths = build_run_paths(args["lang"], args["from_checkpoint"])
    wandb.init(project="char-level-lstm", name=paths["run_name"])

    # Build configuration
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    config = build_train_config(args, use_cuda)

    # Load and prepare data (needed for vocab_size in wandb config)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    corpus = load_and_split_corpus(data_path, config)
    vocab = setup_vocab(corpus, is_finetune, paths["vocab_json_path"])

    # Log comprehensive config to W&B
    source_ckpt = args["from_checkpoint"] if args["from_checkpoint"] is not None else ""
    wandb_cfg: WandbConfig = {
        # Model architecture
        "vocab_size": vocab["vocab_size"],
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        # Training settings
        "seq_len": config["seq_len"],
        "batch_size": config["batch_size"],
        "num_epochs": config["num_epochs"],
        "learning_rate": config["lr"],
        "patience": config["patience"],
        # Data splits
        "train_ratio": config["train_ratio"],
        "val_ratio": config["val_ratio"],
        "test_ratio": 1.0 - config["train_ratio"] - config["val_ratio"],
        # Experiment metadata
        "language": lang_name,
        "language_code": args["lang"],
        "is_finetune": is_finetune,
        "source_checkpoint": source_ckpt,
        "freeze_embed": args["freeze_embed"],
        "device": device_str,
    }
    wb_config(wandb_cfg)

    print_config(
        lang_name=lang_name,
        lang_code=args["lang"],
        is_finetune=is_finetune,
        checkpoint_path=paths["source_checkpoint_path"],
        device=device_str,
        config=config,
        freeze_embed=args["freeze_embed"],
        output_path=paths["checkpoint_best"],
    )

    loaders = create_dataloaders(corpus, vocab["stoi"], config)

    # Setup model
    model_setup = setup_model_and_optimizer(
        vocab_size=vocab["vocab_size"],
        config=config,
        is_finetune=is_finetune,
        checkpoint_best=paths["checkpoint_best"],
        source_checkpoint_path=paths["source_checkpoint_path"],
        freeze_embed=args["freeze_embed"],
    )

    # Training loop
    state: TrainState = {
        "global_step": 0,
        "window_sum": 0.0,
        "window_n": 0,
        "best_val_loss": float("inf"),
        "epochs_no_improve": 0,
    }

    epoch_history: list[EpochMetrics] = []
    num_epochs = config["num_epochs"]
    batches = len(loaders["train_loader"])
    log_header(f"Starting training: {num_epochs} epochs, {batches} batches/epoch")

    for epoch in range(num_epochs):
        should_continue, epoch_metrics = train_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            model=model_setup["model"],
            train_loader=loaders["train_loader"],
            val_loader=loaders["val_loader"],
            optimizer=model_setup["optimizer"],
            criterion=model_setup["criterion"],
            device=model_setup["device"],
            vocab_size=vocab["vocab_size"],
            config=config,
            state=state,
            checkpoint_save=model_setup["checkpoint_save"],
        )
        epoch_history.append(epoch_metrics)
        if not should_continue:
            break

    # Log epoch summary table to W&B
    wb_log_epoch_table(epoch_history)

    # Final evaluation
    run_final_evaluation(
        model=model_setup["model"],
        test_loader=loaders["test_loader"],
        criterion=model_setup["criterion"],
        device=model_setup["device"],
        vocab_size=vocab["vocab_size"],
        checkpoint_save=model_setup["checkpoint_save"],
        checkpoint_best=paths["checkpoint_best"],
    )


if __name__ == "__main__":
    main()
