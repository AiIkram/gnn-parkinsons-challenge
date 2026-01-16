# 🧠 GNN Mini-Challenge: Parkinson's Disease Detection using Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Challenge Overview

Welcome to the **GNN Parkinson's Disease Detection Challenge**! This competition focuses on using Graph Neural Networks (GNNs) to detect Parkinson's Disease from acoustic voice measurements.

**Why GNNs?** Parkinson's Disease affects multiple interconnected biomarkers simultaneously. By modeling these relationships as a graph where:
- **Nodes** represent individual voice recordings/patients
- **Edges** connect similar patients or related acoustic features
- **Node features** contain voice measurements (jitter, shimmer, pitch, etc.)

You can capture complex patterns that traditional ML methods might miss!

### 🏆 Competition Details

- **Task Type**: Node Classification (Binary)
- **Difficulty**: ⭐⭐⭐⭐ (Challenging)
- **Metric**: **Macro F1-Score** (handles class imbalance)
- **Dataset**: UCI Parkinson's Dataset with graph structure
- **Deadline**: Open-ended (rolling leaderboard)

### 🎓 Learning Objectives

This challenge covers concepts from **DGL Lectures 1.1-4.6**:
- Graph construction from tabular data
- Message passing neural networks (MPNN)
- Graph attention mechanisms (GAT)
- Sampling methods for large graphs
- Node classification with GNNs

---

## 📊 Dataset Description

### Source
- **Original Dataset**: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Citation**: Little et al. (2008), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease'

### Features (22 acoustic measurements)
- **Vocal fundamental frequency measures**: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- **Jitter variations**: MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP
- **Shimmer variations**: MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA
- **Harmonics & noise ratios**: NHR, HNR
- **Nonlinear measures**: RPDE, DFA, spread1, spread2, D2, PPE

### Graph Structure
- **Nodes**: 195 voice recordings from 31 subjects (23 PD, 8 healthy)
- **Edges**: K-nearest neighbors (k=5) + subject connections
- **Training**: 156 nodes (80%) - labels provided
- **Test**: 39 nodes (20%) - labels hidden

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/AiIkram/gnn-parkinsons-challenge.git
cd gnn-parkinsons-challenge
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r starter_code/requirements.txt
```

### 3. Generate Data
```bash
cd scripts
python generate_graph_data.py
cd ..
```

### 4. Run Baseline Model
```bash
cd starter_code
python baseline.py
```

Expected baseline F1-score: **~0.72-0.78**

---

## 📁 Repository Structure
```
gnn-parkinsons-challenge/
├── data/
│   ├── train_graph.pkl          # Training graph with labels
│   ├── test_graph.pkl           # Test graph without labels
│   └── feature_names.txt        # Feature descriptions
├── submissions/
│   └── sample_submission.csv    # Example submission
├── starter_code/
│   ├── baseline.py              # GCN baseline
│   ├── baseline_gat.py          # GAT baseline
│   └── requirements.txt         # Dependencies
├── scripts/
│   ├── generate_graph_data.py   # Data preprocessing
│   └── scoring_script.py        # Evaluation
├── .github/workflows/
│   └── score_submission.yml     # Auto-scoring
├── LEADERBOARD.md
├── RULES.md
└── README.md
```

---

## 📤 Making a Submission

### Submission Format

CSV with exactly 39 rows:
```csv
node_id,prediction
0,1
1,0
2,1
...
```

### How to Submit

1. Fork this repository
2. Add your CSV to `submissions/`
3. Create a Pull Request
4. GitHub Actions scores automatically
5. Results posted as comment

---

## 📈 Evaluation Metric

**Macro F1-Score** = (F1_Healthy + F1_Parkinson's) / 2

Why?
- Handles class imbalance
- Equal importance to both classes
- More challenging than accuracy

---

# Add This Section to Your README.md

---

## 🏆 Live Leaderboard

**[👉 View Live Leaderboard](https://AiIkram.github.io/gnn-parkinsons-challenge/leaderboard.html)**

The leaderboard is automatically updated when submissions are merged via Pull Request.

### Current Top 3

| Rank | Team | F1-Score | Model | Date |
|------|------|----------|-------|------|
| 🥇 1 | baseline_gcn | 0.7654 | GCN | 2025-01-15 |
| 🥈 2 | baseline_gat | 0.7521 | GAT | 2025-01-15 |
| 🥉 3 | *Your team here* | - | - | - |

---

## 📤 How to Submit

### 1. Prepare Your Submission

Create two files in the `submissions/` folder:

**`submissions/your_team_name.csv`** (Required):
```csv
node_id,prediction
0,1
1,0
2,1
...
38,0
```

**`submissions/your_team_name_metadata.json`** (Optional but recommended):
```json
{
  "score": 0.8500,
  "model": "GAT",
  "date": "2025-01-16",
  "description": "Graph Attention Network with 3 layers"
}
```

### 2. Submission Requirements

- ✅ **39 rows** (one per test node, node_id 0-38)
- ✅ **2 columns**: `node_id`, `prediction`
- ✅ **Binary predictions**: 0 (Healthy) or 1 (Parkinson's)
- ✅ **No duplicates** in node_id
- ✅ **CSV format** with comma delimiter

### 3. Submit via Pull Request

**Option A: Via GitHub Web Interface**
1. Fork this repository
2. Upload your CSV and metadata files to `submissions/`
3. Create a Pull Request
4. Wait for automatic validation and scoring
5. Check the PR comments for your score

**Option B: Via Git**
```bash
# Fork and clone
git clone https://github.com/AiIkram/gnn-parkinsons-challenge.git
cd gnn-parkinsons-challenge

# Add your files
cp your_submission.csv submissions/your_team_name.csv
cp your_metadata.json submissions/your_team_name_metadata.json

# Commit and push
git add submissions/
git commit -m "Add submission for team: your_team_name"
git push origin main

# Create Pull Request on GitHub
```

### 4. Automated Scoring

When you submit a Pull Request:
- 🤖 GitHub Actions automatically validates your CSV
- 📊 Calculates your F1-Score (macro-averaged)
- 💬 Comments on your PR with results
- 🏆 After merge, updates the live leaderboard

### 5. View Your Ranking

Once merged, your score appears on the **[Live Leaderboard](https://AiIkram.github.io/gnn-parkinsons-challenge/leaderboard.html)** within minutes!
---

## 💡 Tips for Success

1. **Start with baselines**: Test with GCN/GAT before complex models
2. **Validate locally**: Run `python scoring_script.py your_file.csv`
3. **Check format**: Ensure exact CSV format (node_id, prediction)
4. **Add metadata**: Helps others learn from your approach
5. **Iterate**: Submit multiple times to improve your score

---

## 🔗 Quick Links

- 🏆 **[Live Leaderboard](https://AiIkram.github.io/gnn-parkinsons-challenge/leaderboard.html)**
- 📊 **[Competition Homepage](https://AiIkram.github.io/gnn-parkinsons-challenge/)**
- 📂 **[GitHub Repository](https://github.com/AiIkram/gnn-parkinsons-challenge)**
- 📖 **[Setup Guide](SETUP_GUIDE.md)**
- 📋 **[Submission Rules](RULES.md)**

---

## 📞 Support

Having issues? 
1. Check [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)
2. Review [example submissions](submissions/)
3. Open an issue on GitHub

---

**Ready to compete? Submit your first entry today! 🚀**