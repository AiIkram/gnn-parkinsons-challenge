import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from dgl.nn import GraphConv
import os

# Reproducibility
torch.manual_seed(25)
np.random.seed(25)
dgl.seed(25)
# =========================
# 🔥 Improved GCN Model
# =========================
class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.4):
        super(GCNModel, self).__init__()

        self.conv1 = GraphConv(in_feats, hidden_size)
        self.bn1   = nn.BatchNorm1d(hidden_size)

        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.bn2   = nn.BatchNorm1d(hidden_size)

        self.conv3 = GraphConv(hidden_size, hidden_size)
        self.bn3   = nn.BatchNorm1d(hidden_size)

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, g, features):
        h = features

        h1 = F.relu(self.bn1(self.conv1(g, h)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.relu(self.bn2(self.conv2(g, h1)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Residual connection
        h3 = self.conv3(g, h2) + h1
        h3 = F.relu(self.bn3(h3))
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        return self.classifier(h3)


# =========================
# 📦 Load Data
# =========================
def load_data():
    print("Loading data...")

    DATA_FORMAT = "free"  # or "dgl"

    if DATA_FORMAT == "free":
        with open('../data/public/train_graph_free.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/public/test_graph_free.pkl', 'rb') as f:
            test_data = pickle.load(f)

        def rebuild_dgl_graph(d):
            src, dst = d["edge_index"]
            g = dgl.graph((src, dst), num_nodes=d["num_nodes"])
            d["graph"] = g
            return d

        train_data = rebuild_dgl_graph(train_data)
        test_data  = rebuild_dgl_graph(test_data)

    elif DATA_FORMAT == "dgl":
        with open('../data/public/train_graph.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('../data/public/test_graph.pkl', 'rb') as f:
            test_data = pickle.load(f)

    else:
        raise ValueError("Invalid DATA_FORMAT")

    return train_data, test_data


# =========================
# 🏋️ Training
# =========================
def train_epoch(model, g, features, labels, train_mask, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()

    logits = model(g, features)

    loss = F.cross_entropy(
        logits[train_mask],
        labels[train_mask],
        weight=class_weights,
        label_smoothing=0.1   # 🔥 improvement
    )

    loss.backward()
    optimizer.step()

    _, predicted = torch.max(logits[train_mask], 1)
    acc = (predicted == labels[train_mask]).float().mean()

    return loss.item(), acc.item()


# =========================
# 📊 Evaluation
# =========================
def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, predicted = torch.max(logits[mask], 1)

        acc = (predicted == labels[mask]).float().mean()

        f1 = f1_score(
            labels[mask].cpu().numpy(),
            predicted.cpu().numpy(),
            average='macro'
        )

    return acc.item(), f1


# =========================
# 🚀 MAIN
# =========================
def main():
    print("=" * 60)
    print("🔥 Improved GCN Model")
    print("=" * 60)

    train_data, test_data = load_data()

    g          = train_data['graph']
    features   = train_data['features']
    labels     = train_data['labels']
    train_mask = train_data['train_mask']
    val_mask   = train_data['val_mask']

    print(f"\nNodes: {g.num_nodes()}, Edges: {g.num_edges()}")

    # =========================
    # 🔥 Feature Normalization
    # =========================
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)

    # =========================
    # ⚖️ Class Weights
    # =========================
    train_labels = labels[train_mask]

    num_class0 = (train_labels == 0).sum().item()
    num_class1 = (train_labels == 1).sum().item()
    total = num_class0 + num_class1

    w0 = total / (2 * num_class0)
    w1 = total / (2 * num_class1)

    class_weights = torch.FloatTensor([w0, w1])

    print(f"Class weights: {w0:.2f}, {w1:.2f}")

    # =========================
    # 🔧 Hyperparameters
    # =========================
    in_feats = features.shape[1]
    hidden_size = 128
    num_classes = 2

    lr = 0.005
    weight_decay = 5e-4
    num_epochs = 200

    model = GCNModel(in_feats, hidden_size, num_classes)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 🔥 LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )

    print("\nTraining...")
    best_val_f1 = 0
    patience = 50
    patience_counter = 0

    for epoch in range(num_epochs):
        loss, train_acc = train_epoch(
            model, g, features, labels,
            train_mask, optimizer, class_weights
        )

        val_acc, val_f1 = evaluate(
            model, g, features, labels, val_mask
        )

        scheduler.step(val_f1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss {loss:.4f} | Val F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest Val F1: {best_val_f1:.4f}")

    # =========================
    # 🔮 Inference
    # =========================
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    test_g        = test_data['graph']
    test_features = test_data['features']
    test_node_ids = test_data['node_ids']

    # Normalize test features
    test_features = (test_features - test_features.mean(0)) / (test_features.std(0) + 1e-6)

    with torch.no_grad():
        logits = model(test_g, test_features)
        preds = torch.max(logits[test_node_ids], 1)[1]

    submission = pd.DataFrame({
        'node_id': test_node_ids,
        'prediction': preds.cpu().numpy()
    })

    os.makedirs('../submissions', exist_ok=True)
    submission.to_csv('../submissions/gcn_submission.csv', index=False)

    print("\n✅ Submission saved!")
    print(submission.head())
    print("=" * 60)


if __name__ == "__main__":
    main()