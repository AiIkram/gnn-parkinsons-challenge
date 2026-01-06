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
git clone https://github.com/YOUR-USERNAME/gnn-parkinsons-challenge.git
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

## 🏅 Current Leaderboard

| Rank | Participant | F1-Score | Model | Date |
|------|-------------|----------|-------|------|
| 🥇 1 | Baseline-GCN | 0.7654 | GCN | 2025-01-05 |
| 🥈 2 | Baseline-GAT | 0.7521 | GAT | 2025-01-05 |
| 🥉 3 | *-* | ??? | ??? | TBD |

*View full leaderboard in [LEADERBOARD.md](LEADERBOARD.md)*

---

## 💡 Tips & Tricks

### For Beginners
1. Start with baseline GCN
2. Try different hidden sizes
3. Vary number of layers (2-4)
4. Add dropout for regularization
5. Use cross-validation

### Advanced
- Experiment with k in KNN graphs
- Add edge weights
- Try GAT attention
- Use skip connections
- Handle class imbalance
- Ensemble models

### Common Pitfalls
⚠️ Overfitting (small dataset)  
⚠️ Over-smoothing (too many layers)  
⚠️ Ignoring class imbalance  
⚠️ Data leakage  

---

## 🔬 Research Context

Top participants may be invited to:
- Co-author NeurIPS 2026 paper (deadline: May 2026)
- Collaborate on research projects
- Contribute to open-source medical AI

---

## 📚 Resources

### Learning GNNs
- [DGL Tutorial](https://docs.dgl.ai/tutorials/blitz/index.html)
- [Distill.pub GNN Intro](https://distill.pub/2021/gnn-intro/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

### Papers
- Kipf & Welling (2017): Semi-Supervised GCNs
- Veličković et al. (2018): Graph Attention Networks
- Hamilton et al. (2017): GraphSAGE

---

## 🎯 Challenge Rules

### Must Do
✅ Use at least one GNN layer  
✅ Only use provided dataset  
✅ Complete inference within 5 minutes  
✅ Set random seeds  
✅ Provide code  

### Cannot Do
❌ Use test labels  
❌ External Parkinson's datasets  
❌ Pure non-GNN models  

See [RULES.md](RULES.md) for complete details.

---

## 🤝 Contributing

- **Bug?** Open an issue
- **Suggestion?** Start a discussion
- **Improvement?** Submit a PR

---

## 📝 Citation
```bibtex
@misc{gnn_parkinsons_challenge2025,
  title={GNN Mini-Challenge: Parkinson's Disease Detection},
  author={Aissiou Ikram},
  year={2025},
  url={https://github.com/AiIkram/gnn-parkinsons-challenge}
}
```

---

## 📧 Contact

- **Issues**: Use GitHub Issues
- **Discussions**: Use GitHub Discussions
- **Email**: aissiouikram47@gmail.com

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

**Dataset License**: UCI Parkinson's Dataset - CC BY 4.0

---

**Ready to start? Fork this repo and submit your solution!** 🚀

Good luck! 🎉