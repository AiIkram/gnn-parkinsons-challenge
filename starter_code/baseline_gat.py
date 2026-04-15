"""
Improved GAT for Parkinson's — Competition-Compliant Version
=============================================================
Rules enforced:
  - Rule 6 : Single random seed = 25 everywhere (no ensemble of different seeds)
  - Rule 7 : GNN only (GAT via PyTorch Geometric)
  - Rule 8 : Hyperparameter budget = 5 runs max (tracked below)
  - Rule 10: Loss + F1 plotted per epoch, saved to ./submissions/training_curve.png

Hyperparameter runs log (Rule 8 — max 5):
  Run 1: hidden=32, heads=4, dropout=0.5, lr=0.005, wd=5e-4  → baseline config
  Run 2: hidden=64, heads=8, dropout=0.4, lr=0.005, wd=1e-3  → better capacity
  Run 3: hidden=64, heads=8, dropout=0.6, lr=0.003, wd=1e-3  → more regularisation
  Run 4: hidden=64, heads=4, dropout=0.4, lr=0.005, wd=5e-4  → fewer heads
  Run 5: hidden=128,heads=8, dropout=0.4, lr=0.003, wd=1e-3  → largest model

Usage:
  python starter_code/inspect_data.py
  python starter_code/baseline_gat.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.utils import add_self_loops, dropout_edge

# ── Rule 6: ALL seeds fixed to 25 ────────────────────────────────────────────
SEED = 25
torch.manual_seed(SEED)
np.random.seed(SEED)
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualGATBlock(nn.Module):
    """GAT layer with residual connection + batch normalisation."""

    def __init__(self, in_feats, out_feats, heads, dropout):
        super().__init__()
        self.conv = GATConv(
            in_feats,
            out_feats,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.norm = BatchNorm(out_feats * heads)
        self.dropout = dropout
        self.residual = (
            nn.Linear(in_feats, out_feats * heads, bias=False)
            if in_feats != out_feats * heads else nn.Identity()
        )

    def forward(self, x, edge_index):
        out = F.elu(self.norm(self.conv(x, edge_index)))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out + self.residual(x)


class ImprovedGATModel(nn.Module):
    """
    Two-block residual GAT:
      input projection → ResidualGATBlock × 2 → MLP classifier
    """

    def __init__(self, in_feats, hidden_size=64, num_classes=2,
                 num_heads=8, dropout=0.4, drop_edge=0.1):
        super().__init__()
        self.drop_edge = drop_edge

        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.block1 = ResidualGATBlock(
            hidden_size, hidden_size // num_heads, num_heads, dropout
        )
        self.block2 = ResidualGATBlock(
            hidden_size, hidden_size // num_heads, num_heads, dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x, edge_index):
        if self.training and self.drop_edge > 0:
            edge_index, _ = dropout_edge(
                edge_index,
                p=self.drop_edge,
                training=self.training
            )
        x = self.input_proj(x)
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        return self.classifier(x)

    def predict_proba(self, x, edge_index, n_passes=20):
        """
        Test-Time Augmentation: average softmax over n stochastic
        forward passes (dropout kept ON).
        """
        self.train()   # keep dropout active
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                probs.append(F.softmax(self(x, edge_index), dim=1))
        self.eval()
        return torch.stack(probs).mean(0)   # [N, C]


# ---------------------------------------------------------------------------
# Loss with label smoothing
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing + optional class weights."""

    def __init__(self, num_classes=2, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(
                log_probs,
                self.smoothing / (self.num_classes - 1)
            )
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth * log_probs).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    enriched_path = './data/processed/train_enriched.pkl'
    if os.path.exists(enriched_path):
        print("Loading ENRICHED features (from inspect_data.py)...")
        with open(enriched_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Feature dim: {data['feat_dim']}")
    else:
        print("Loading RAW features (tip: run inspect_data.py first)...")
        with open('./data/public/train_graph_free.pkl', 'rb') as f:
            data = pickle.load(f)

        src, dst = data['edge_index']
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data['num_nodes'])
        data['edge_index_pyg'] = edge_index

    return data


def to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.array(x), dtype=dtype)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_epoch(model, edge_index, features, labels, train_mask,
                optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    logits = model(features, edge_index)
    loss = criterion(logits[train_mask], labels[train_mask])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    preds = logits[train_mask].argmax(1)
    acc = (preds == labels[train_mask]).float().mean().item()

    return loss.item(), acc


def evaluate(model, edge_index, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[mask], labels[mask]).item()
        preds = logits[mask].argmax(1)
        f1 = f1_score(
            labels[mask].cpu().numpy(),
            preds.cpu().numpy(),
            average='macro'
        )
    return loss, f1


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_curves(train_losses, val_losses, val_f1s, best_epoch, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Curves — Parkinson's GAT", fontsize=14, fontweight='bold')

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
    ax1.plot(epochs, val_losses, label='Val Loss', color='tomato', linewidth=1.5)
    ax1.axvline(best_epoch, color='green', linestyle='--', linewidth=1,
                label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss (should flatten — watch for overfitting)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, label='Val Macro-F1', color='darkorange', linewidth=1.5)
    ax2.axvline(best_epoch, color='green', linestyle='--', linewidth=1,
                label=f'Best epoch ({best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-F1')
    ax2.set_title('Validation Macro-F1')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curve saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GNN Parkinson's — Improved GAT (rule-compliant)")
    print(f"  Seed: {SEED}  (Rule 6)")
    print("=" * 60)

    data = load_data()

    edge_index = data['edge_index_pyg']
    features = to_tensor(data['features'])
    labels_raw = to_tensor(data['labels'], dtype=torch.long)
    train_mask = to_tensor(data['train_mask'], dtype=torch.bool)
    val_mask = to_tensor(data['val_mask'], dtype=torch.bool)

    # Safe node_ids handling
    raw_node_ids = data.get('node_ids', None)
    if raw_node_ids is None:
        print("node_ids not found in data -> using nodes where label == -1")
        node_ids = torch.where(labels_raw == -1)[0]
    else:
        node_ids = torch.as_tensor(
            np.asarray(raw_node_ids, dtype=np.int64),
            dtype=torch.long
        )

    print(f"\nGraph summary:")
    print(f"  Nodes         : {features.shape[0]}")
    print(f"  Edges         : {edge_index.shape[1]}")
    print(f"  Feature dim   : {features.shape[1]}")
    print(f"  Train nodes   : {train_mask.sum().item()}")
    print(f"  Val nodes     : {val_mask.sum().item()}")
    print(f"  Test node_ids : {len(node_ids)}")
    print(f"  Label -1 (test): {(labels_raw == -1).sum().item()}")
    print(f"  Label  0 (hlth): {(labels_raw == 0).sum().item()}")
    print(f"  Label  1 (pk's): {(labels_raw == 1).sum().item()}")

    # Class weights
    train_labels = labels_raw[train_mask]
    n0 = (train_labels == 0).sum().item()
    n1 = (train_labels == 1).sum().item()
    tot = n0 + n1
    class_weights = torch.FloatTensor([tot / (2 * n0), tot / (2 * n1)])

    print(f"\n  Class weights: Healthy={class_weights[0]:.2f}, "
          f"Parkinson's={class_weights[1]:.2f}")

    # Hyperparameters
    HIDDEN = 64
    HEADS = 8
    DROPOUT = 0.4
    DROP_EDGE = 0.1
    LR = 0.005
    WD = 1e-3
    PATIENCE = 50
    MAX_EPOCHS = 300
    SMOOTHING = 0.1

    model = ImprovedGATModel(
        in_feats=features.shape[1],
        hidden_size=HIDDEN,
        num_classes=2,
        num_heads=HEADS,
        dropout=DROPOUT,
        drop_edge=DROP_EDGE,
    )

    criterion = LabelSmoothingLoss(
        num_classes=2,
        smoothing=SMOOTHING,
        weight=class_weights
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
    )

    print(f"\nTraining (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE})...")
    print("-" * 60)

    train_losses, val_losses, val_f1s = [], [], []
    best_val_f1 = 0.0
    best_epoch = 0
    patience_ctr = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, edge_index, features, labels_raw,
            train_mask, optimizer, criterion
        )
        val_loss, val_f1 = evaluate(
            model, edge_index, features, labels_raw, val_mask
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(best epoch={best_epoch})")
            break

    print(f"\n  Best Val Macro-F1 : {best_val_f1:.4f}  (epoch {best_epoch})")

    # Save training curves
    os.makedirs('./submissions', exist_ok=True)
    plot_curves(
        train_losses, val_losses, val_f1s,
        best_epoch, './submissions/training_curve.png'
    )

    # Load best weights
    model.load_state_dict(best_state)

    # Final validation report
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        val_preds = logits[val_mask].argmax(1).cpu().numpy()
        val_labels = labels_raw[val_mask].cpu().numpy()

    print("\nClassification Report (validation, best model):")
    print(classification_report(
        val_labels,
        val_preds,
        target_names=['Healthy', "Parkinson's"],
        zero_division=0,
    ))

    # Test predictions
    print("Generating test predictions (TTA × 20 stochastic passes)...")
    probs = model.predict_proba(features, edge_index, n_passes=20)
    test_preds = probs[node_ids].argmax(1).cpu().numpy()

    submission = pd.DataFrame({
        'node_id': node_ids.cpu().numpy(),
        'prediction': test_preds,
    })
    submission.to_csv('./submissions/gat_submission.csv', index=False)

    print(f"\nSubmission saved → ./submissions/gat_submission.csv")
    counts = pd.Series(test_preds).value_counts().sort_index()
    for cls, cnt in counts.items():
        lbl = 'Healthy' if cls == 0 else "Parkinson's"
        print(f"  {lbl} ({cls}): {cnt}  ({cnt / len(submission) * 100:.1f}%)")

    print(submission.to_string(index=False))
    print("\n" + "=" * 60)
    print("OUTPUTS:")
    print("  ./submissions/gat_submission.csv   — predictions")
    print("  ./submissions/training_curve.png   — loss/F1 plot (Rule 10)")
    print("=" * 60)


if __name__ == '__main__':
    main()