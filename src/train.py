"""
Training script for matched Transformer and RNN models on TinyStories.
Trains all architectures with identical data, optimizer, and schedule.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from models import build_models
from data import load_tinystories, prepare_data


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, model_name, device,
                num_epochs=10, lr=3e-4, grad_clip=1.0, save_dir="results/checkpoints"):
    """Train a single model with early stopping."""
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\nTraining {model_name} ({model.count_params():,} params) on {device}")
    print(f"  Epochs: {num_epochs}, LR: {lr}, Batch size: {train_loader.batch_size}")

    for epoch in range(num_epochs):
        t0 = time.time()

        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 2000 == 0 and batch_idx > 0:
                avg = total_loss / num_batches
                print(f"  Epoch {epoch+1}, batch {batch_idx}: loss={avg:.4f}")

        train_loss = total_loss / num_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss_total += loss.item()
                val_batches += 1

        val_loss = val_loss_total / val_batches
        val_losses.append(val_loss)

        dt = time.time() - t0
        print(f"  Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, perplexity={np.exp(val_loss):.1f}, "
              f"time={dt:.1f}s")

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_best.pt"),
                                      weights_only=True))

    return {
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_perplexity": np.exp(best_val_loss),
        "num_params": model.count_params(),
        "epochs_trained": len(train_losses),
    }


def main():
    set_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data - use 15k stories for feasible training time
    texts = load_tinystories(max_stories=15000)

    # Prepare data with shared tokenizer
    tokenizer, train_dataset, val_dataset = prepare_data(
        texts, vocab_size=8192, seq_len=128
    )

    # Save tokenizer
    os.makedirs("results", exist_ok=True)
    tokenizer.save("results/tokenizer.json")

    # Create data loaders - use large batch for speed
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Build models at small scale
    models = build_models(tokenizer.vocab_size, scale="small")

    # Train each model
    all_results = {}
    for name, model in models.items():
        results = train_model(
            model, train_loader, val_loader, name, device,
            num_epochs=30, lr=3e-4
        )
        all_results[name] = results

        # Save model
        torch.save(model.state_dict(), f"results/checkpoints/{name}_final.pt")

    # Save training results
    with open("results/training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name}: val_loss={res['best_val_loss']:.4f}, "
              f"perplexity={res['best_perplexity']:.1f}, "
              f"params={res['num_params']:,}")

    # Also build and train medium-scale models
    print("\n" + "=" * 60)
    print("TRAINING MEDIUM-SCALE MODELS")
    print("=" * 60)

    models_med = build_models(tokenizer.vocab_size, scale="medium")
    med_results = {}
    for name, model in models_med.items():
        med_name = f"{name}_medium"
        results = train_model(
            model, train_loader, val_loader, med_name, device,
            num_epochs=20, lr=2e-4
        )
        med_results[med_name] = results
        torch.save(model.state_dict(), f"results/checkpoints/{med_name}_final.pt")

    all_results.update(med_results)

    # Save combined results
    with open("results/training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("FULL TRAINING SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name}: val_loss={res['best_val_loss']:.4f}, "
              f"perplexity={res['best_perplexity']:.1f}, "
              f"params={res['num_params']:,}")


if __name__ == "__main__":
    main()
