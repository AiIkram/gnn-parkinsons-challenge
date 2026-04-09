import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from dgl.nn import GraphConv

torch.manual_seed(25)
np.random.seed(25)
dgl.seed(25)


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, activation=F.relu)
        self.conv2 = GraphConv(hidden_size, hidden_size, activation=F.relu)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, g, features):
        h = F.dropout(features, p=self.dropout, training=self.training)
        h = self.conv1(g, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        return self.classifier(h)


def load_data():
    print("Loading data...")

    DATA_FORMAT = "free"

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
        print("  Loaded DGL-free format")

    else:
        raise ValueError("Use free format")

    return train_data, test_data


def train_epoch(model, g, features, labels, train_mask, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        preds = logits[mask].argmax(dim=1)
        f1 = f1_score(labels[mask].cpu().numpy(), preds.cpu().numpy(), average='macro')
    return f1


def main():
    print("=== GCN Tuned Version ===")

    train_data, test_data = load_data()

    g          = train_data['graph']
    features   = train_data['features']
    labels     = train_data['labels']
    train_mask = train_data['train_mask']
    val_mask   = train_data['val_mask']

    # class weights
    y_train = labels[train_mask]
    w0 = len(y_train) / (2 * (y_train == 0).sum().item())
    w1 = len(y_train) / (2 * (y_train == 1).sum().item())
    class_weights = torch.FloatTensor([w0, w1])

    model = GCNModel(features.shape[1], 64, 2, dropout=0.3)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,              # ✅ stabilized
        weight_decay=5e-4
    )

    best_f1 = 0
    patience = 0

    for epoch in range(200):
        loss = train_epoch(model, g, features, labels, train_mask, optimizer, class_weights)
        val_f1 = evaluate(model, g, features, labels, val_mask)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss {loss:.4f} | F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), 'best_gcn_model.pt')
        else:
            patience += 1

        if patience >= 50:
            break

    print("Best F1:", best_f1)

    # inference
    model.load_state_dict(torch.load('best_gcn_model.pt'))
    model.eval()

    test_g = test_data['graph']
    test_features = test_data['features']
    test_ids = test_data['node_ids']

    with torch.no_grad():
        logits = model(test_g, test_features)
        preds = logits[test_ids].argmax(dim=1)

    df = pd.DataFrame({
        "node_id": test_ids,
        "prediction": preds.numpy()
    })

    df.to_csv("../submissions/gcn_submission.csv", index=False)
    print("Submission saved.")


if __name__ == "__main__":
    main()
