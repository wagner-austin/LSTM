#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from model import charLSTM
from data import CharDataset, build_vocab_with_unk, save_vocab_json, load_vocab_json

# Language configs: code -> (name, corpus_path)
LANGUAGES = {
    "tr": ("Turkish", "09_Downloaded_Corpora/oscar_tr_ipa.txt"),
    "az": ("Azerbaijani", "09_Downloaded_Corpora/oscar_az_ipa.txt"),
    "kk": ("Kazakh", "09_Downloaded_Corpora/oscar_kk_ipa.txt"),
    "ky": ("Kyrgyz", "09_Downloaded_Corpora/oscar_ky_ipa.txt"),
    "uz": ("Uzbek", "09_Downloaded_Corpora/oscar_uz_ipa.txt"),
    "ug": ("Uyghur", "09_Downloaded_Corpora/oscar_ug_ipa.txt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/fine-tune char-level LSTM on Turkic languages")
    parser.add_argument("--lang", type=str, required=True, choices=LANGUAGES.keys(),
                        help="Language code to train on (tr, az, kk, ky, uz, ug)")
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Path to checkpoint for fine-tuning (e.g., checkpoints/turkish_best.pt)")
    parser.add_argument("--freeze-embed", action="store_true",
                        help="Freeze embedding layer during fine-tuning")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / max(1, n)
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


def wb_log(d: dict):
    if wandb.run is not None:
        wandb.log(d)


def next_checkpoint_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 1
    while True:
        cand = base.with_name(f"{stem}.ft{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def main():
    args = parse_args()

    # Get language config
    lang_name, data_path = LANGUAGES[args.lang]
    is_finetune = args.from_checkpoint is not None

    # Build run name and checkpoint paths
    if is_finetune:
        base_name = Path(args.from_checkpoint).stem.replace("_best", "")
        run_name = f"{base_name}->{args.lang}"
        checkpoint_name = f"{base_name}_to_{args.lang}.pt"
    else:
        run_name = f"{args.lang}-train"
        checkpoint_name = f"{args.lang}_best.pt"

    wandb.init(project="char-level-lstm", name=run_name)

    CHECKPOINT_DIR = Path("checkpoints")
    VOCAB_JSON = CHECKPOINT_DIR / "vocab.json"
    CHECKPOINT_BEST = CHECKPOINT_DIR / checkpoint_name

    # Hyperparams
    SEQ_LEN = 100
    NUM_EPOCHS = args.epochs
    LOG_EVERY = 100
    PATIENCE = 1
    lr = args.lr

    # Auto-optimize for GPU vs CPU
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    BATCH_SIZE = 256 if use_cuda else 64
    NUM_WORKERS = 4 if use_cuda else 0
    PIN_MEMORY = use_cuda

    train_ratio = 0.75
    val_ratio = 0.25

    # Training / finetuning toggles
    USE_CHECKPOINT = is_finetune
    RESUME_OPTIMIZER = False        # Always fresh optimizer for fine-tuning
    FREEZE_EMBED = args.freeze_embed
    CHECKPOINT_PATH = Path(args.from_checkpoint) if args.from_checkpoint else CHECKPOINT_BEST

    # Print config
    print(f"\n{'='*60}", flush=True)
    print(f"Training config:", flush=True)
    print(f"  Language:    {lang_name} ({args.lang})", flush=True)
    print(f"  Mode:        {'Fine-tune from ' + str(CHECKPOINT_PATH) if is_finetune else 'Train from scratch'}", flush=True)
    print(f"  Device:      {device}", flush=True)
    print(f"  Batch size:  {BATCH_SIZE}", flush=True)
    print(f"  Workers:     {NUM_WORKERS}", flush=True)
    print(f"  Pin memory:  {PIN_MEMORY}", flush=True)
    print(f"  Epochs:      {NUM_EPOCHS}", flush=True)
    print(f"  LR:          {lr}", flush=True)
    print(f"  Freeze embed: {FREEZE_EMBED}", flush=True)
    print(f"  Output:      {CHECKPOINT_BEST}", flush=True)
    print(f"{'='*60}\n", flush=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = data_path

    # Load corpus and split
    print(f"Loading data from {DATA_PATH}...", flush=True)
    text = Path(DATA_PATH).read_text(encoding="utf-8")[:10000000]
    N = len(text)
    train_idx = int(N * train_ratio)
    val_idx = int(N * (train_ratio + val_ratio))
    train_text = text[:train_idx]
    val_text = text[train_idx:val_idx]
    test_text = text[val_idx:]
    print(f"  Loaded {N:,} chars total", flush=True)
    print(f"  Train: {len(train_text):,} chars ({train_ratio*100:.0f}%)", flush=True)
    print(f"  Val:   {len(val_text):,} chars ({val_ratio*100:.0f}%)", flush=True)
    print(f"  Test:  {len(test_text):,} chars", flush=True)

    # Vocab: load existing when resuming/ft, else build & save
    if USE_CHECKPOINT and VOCAB_JSON.exists():
        print(f"Loading vocab from {VOCAB_JSON}...", flush=True)
        stoi, itos, vocab_size, _ = load_vocab_json(VOCAB_JSON)
    else:
        print("Building vocab from training text...", flush=True)
        stoi, itos, vocab_size = build_vocab_with_unk(text)
        save_vocab_json(itos, VOCAB_JSON)
    print(f"  Vocab size: {vocab_size}", flush=True)

    # DataLoaders
    print("Creating data loaders...", flush=True)
    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "pin_memory": PIN_MEMORY}
    train_loader = DataLoader(CharDataset(train_text, stoi, SEQ_LEN), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(CharDataset(val_text, stoi, SEQ_LEN), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(CharDataset(test_text, stoi, SEQ_LEN), shuffle=False, **loader_kwargs)
    print(f"  Train batches: {len(train_loader):,}", flush=True)
    print(f"  Val batches:   {len(val_loader):,}", flush=True)

    # Model / optimizer / loss
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    device = torch.device(device)

    print(f"\nModel config:", flush=True)
    print(f"  Device:     {device}", flush=True)
    print(f"  Embed dim:  {embed_dim}", flush=True)
    print(f"  Hidden dim: {hidden_dim}", flush=True)
    print(f"  Layers:     {num_layers}", flush=True)

    model = charLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print(f"  Optimizer:  Adam (lr={lr})", flush=True)

    # Decide where to save: versioned path if fine-tuning
    CHECKPOINT_SAVE = CHECKPOINT_BEST
    if USE_CHECKPOINT and not RESUME_OPTIMIZER:
        CHECKPOINT_SAVE = next_checkpoint_path(CHECKPOINT_BEST)
        print(f"[FT] Will save fine-tuned checkpoint to: {CHECKPOINT_SAVE}")

    # Optional: load checkpoint
    start_epoch = 0
    best_val_loss = float("inf")

    if USE_CHECKPOINT and Path(CHECKPOINT_PATH).exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

            if RESUME_OPTIMIZER and checkpoint.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("Loaded optimizer state (resume).")
                    best_val_loss = checkpoint.get("val_loss", checkpoint.get("eval_loss", float("inf")))
                    start_epoch = checkpoint.get("epoch", -1) + 1
                except Exception as e:
                    print(f"Could not load optimizer state; starting fresh: {e}")
            else:
                print("Fine-tuning with a fresh optimizer.")
                for g in optimizer.param_groups:
                    g["lr"] = min(g["lr"], 5e-5)  # optional lower LR for FT
                best_val_loss = float("inf")
                start_epoch = 0
        else:
            # raw state_dict
            model.load_state_dict(checkpoint, strict=True)
            print("Loaded raw state_dict. Fine-tuning with fresh optimizer.")
    else:
        if USE_CHECKPOINT:
            print(f"Checkpoint not found at {CHECKPOINT_PATH}; training from scratch.")

    # Optional: freeze embeddings
    if FREEZE_EMBED:
        for p in model.embedding.parameters():
            p.requires_grad = False
        print("Embedding layer frozen.")

    # Train
    epochs_no_improve = 0
    global_step = 0
    window_sum = 0.0
    window_n = 0

    print(f"\n{'='*60}", flush=True)
    print(f"Starting training: {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch", flush=True)
    print(f"{'='*60}\n", flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Starting...", flush=True)
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            window_sum += loss.item()
            window_n += 1

            if (global_step % LOG_EVERY == 0) or (step == len(train_loader) - 1):
                avg_loss = window_sum / window_n
                avg_ppl = math.exp(avg_loss)
                wb_log({"train_loss": avg_loss, "train_ppl": avg_ppl,
                        "global_step": global_step, "epoch": epoch + 1})
                pct = 100 * (step + 1) / len(train_loader)
                print(f"  [{pct:5.1f}%] Step {global_step:,} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}", flush=True)
                window_sum, window_n = 0.0, 0

        # Validation
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Validating...", flush=True)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device, vocab_size)
        wb_log({"val_loss": val_loss, "val_ppl": val_ppl,
                "epoch": epoch + 1, "global_step": global_step})
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}", flush=True)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "global_step": global_step,
                "vocab_size": vocab_size,
            }, CHECKPOINT_SAVE)
            print(f"  -> Saved best model to {CHECKPOINT_SAVE}", flush=True)
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{PATIENCE})", flush=True)
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.", flush=True)
                break

    # Final test on best
    print(f"\n{'='*60}", flush=True)
    print("Training complete. Running final evaluation...", flush=True)
    print(f"{'='*60}", flush=True)
    best_path = CHECKPOINT_SAVE if CHECKPOINT_SAVE.exists() else CHECKPOINT_BEST
    best_ckpt = torch.load(best_path, map_location=device)
    if isinstance(best_ckpt, dict) and "model_state_dict" in best_ckpt:
        model.load_state_dict(best_ckpt["model_state_dict"])
    else:
        model.load_state_dict(best_ckpt)

    test_loss, test_ppl = evaluate(model, test_loader, criterion, device, vocab_size)
    wb_log({"test_loss": test_loss, "test_ppl": test_ppl})
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
