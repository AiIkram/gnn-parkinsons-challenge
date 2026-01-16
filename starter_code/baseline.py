import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from dgl.nn import GraphConv

torch.manual_seed(42)
np.random.seed(42)
dgl.seed(42)


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.dropout = dropout
        
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        return h


def find_data_path():
    """Find data directory regardless of where script is run from"""
    possible_paths = [
        Path('data'),           # Run from project root
        Path('../data'),        # Run from starter_code
        Path('../../data'),     # Run from subdirectory
    ]
    
    for path in possible_paths:
        if path.exists() and (path / 'train_graph.pkl').exists():
            return path
    
    raise FileNotFoundError(
        "Could not find data directory. Make sure you run from project root "
        "or starter_code directory, and data/train_graph.pkl exists."
    )


def load_data():
    """Load data with flexible path handling"""
    print("Loading data...")
    
    data_path = find_data_path()
    train_path = data_path / 'train_graph.pkl'
    test_path = data_path / 'test_graph.pkl'
    
    print(f"✓ Found data directory: {data_path.absolute()}")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, test_data


def find_submissions_path():
    """Find or create submissions directory"""
    possible_paths = [
        Path('submissions'),      # Run from project root
        Path('../submissions'),   # Run from starter_code
    ]
    
    for path in possible_paths:
        if path.parent.exists():
            path.mkdir(exist_ok=True)
            return path
    
    # Fallback: create in current directory
    path = Path('submissions')
    path.mkdir(exist_ok=True)
    return path


def train_epoch(model, g, features, labels, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    
    # Use class weights to handle imbalance
    class_counts = torch.bincount(labels[train_mask])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    
    loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(logits[train_mask], 1)
    train_acc = (predicted == labels[train_mask]).float().mean()
    return loss.item(), train_acc.item()


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, predicted = torch.max(logits[mask], 1)
        accuracy = (predicted == labels[mask]).float().mean()
        f1_macro = f1_score(
            labels[mask].cpu().numpy(),
            predicted.cpu().numpy(),
            average='macro'
        )
    return accuracy.item(), f1_macro


def main():
    print("=" * 60)
    print("GNN Parkinson's Challenge - Baseline GCN Model")
    print("=" * 60)
    
    try:
        train_data, test_data = load_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    g = train_data['graph']
    features = train_data['features']
    labels = train_data['labels']
    train_mask = train_data['train_mask']
    val_mask = train_data['val_mask']
    
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {g.num_nodes()}")
    print(f"  Edges: {g.num_edges()}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Training nodes: {train_mask.sum().item()}")
    print(f"  Validation nodes: {val_mask.sum().item()}")
    
    # Class distribution
    train_labels = labels[train_mask]
    print(f"\nClass distribution in training set:")
    print(f"  Healthy (0): {(train_labels == 0).sum().item()}")
    print(f"  Parkinson's (1): {(train_labels == 1).sum().item()}")
    
    in_feats = features.shape[1]
    hidden_size = 128
    num_classes = 2
    dropout = 0.6
    lr = 0.005
    weight_decay = 5e-4
    num_epochs = 300
    
    print(f"\nModel Hyperparameters:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    
    model = GCNModel(in_feats, hidden_size, num_classes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print("\nTraining...")
    print("-" * 60)
    
    best_val_f1 = 0
    best_epoch = 0
    patience = 80
    patience_counter = 0
    
    for epoch in range(num_epochs):
        loss, train_acc = train_epoch(model, g, features, labels, train_mask, optimizer)
        val_acc, val_f1 = evaluate(model, g, features, labels, val_mask)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-" * 60)
    print(f"\nBest Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, predicted = torch.max(logits[val_mask], 1)
        print("\nValidation Set Performance:")
        print(classification_report(
            labels[val_mask].cpu().numpy(),
            predicted.cpu().numpy(),
            target_names=['Healthy', 'Parkinson\'s'],
            digits=4,
            zero_division=0
        ))
    
    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_node_ids = test_data['node_ids']
    
    model.eval()
    with torch.no_grad():
        test_logits = model(g, features)
        _, all_predictions = torch.max(test_logits, 1)
        test_predictions = all_predictions[test_node_ids].cpu().numpy()
    
    # Create submission with renumbered node_ids (0-38)
    submission = pd.DataFrame({
        'node_id': range(39),  # Always 0-38 for consistency
        'prediction': test_predictions
    })
    
    # Save submission
    submissions_path = find_submissions_path()
    output_file = submissions_path / 'baseline_gcn.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"\n✓ Submission saved to: {output_file.absolute()}")
    print(f"\nSubmission preview:")
    print(submission.head(10))
    
    print("\n" + "=" * 60)
    print("Baseline model training complete!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print(f"  1. Score your submission: python scoring_script.py {output_file}")
    print(f"  2. Try GAT model: python starter_code/baseline_gat.py")


if __name__ == '__main__':
    main()