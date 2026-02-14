# 🧠 PARK-GNN Challenge: Parkinson's Disease Detection using Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Challenge Status](https://img.shields.io/badge/status-active-success.svg)](https://aiikram.github.io/gnn-parkinsons-challenge/)

**[🏆 View Live Leaderboard](https://aiikram.github.io/gnn-parkinsons-challenge/leaderboard.html)** | **[📖 View Challenge Website](https://aiikram.github.io/gnn-parkinsons-challenge/)**

---

## 🎯 Challenge Overview

Welcome to the **PARK-GNN Challenge** (**P**arkinson’s **A**coustic **R**epresentation & **K**nowledge with **G**raph **N**eural **N**etworks).

This mini-competition focuses on detecting **Parkinson’s Disease (PD)** from **acoustic voice measurements** using **Graph Neural Networks (GNNs)**.

### Why Graph Neural Networks?

Parkinson’s Disease affects multiple vocal biomarkers **simultaneously and interdependently**. Traditional machine learning models treat samples as independent, ignoring these relationships.

In this challenge, the problem is framed as a **graph learning task**, where:

- **Nodes** represent individual voice recordings (or patients)
- **Edges** encode similarity between patients or shared subject-level information
- **Node features** consist of acoustic voice measurements  
  (e.g., jitter, shimmer, pitch, harmonics, nonlinear features)

By leveraging GNNs, participants can model **relational structure** in the data and capture patterns that classical tabular approaches may miss.

---

### 🏆 Competition Details

| **Aspect** | **Details** |
|------------|-------------|
| **Task Type** | Node Classification (Binary) |
| **Difficulty** | ⭐⭐⭐⭐ (Challenging) |
| **Metric** | **Macro F1-Score** (handles class imbalance) |
| **Dataset** | UCI Parkinson's Dataset with graph structure |
| **Deadline** | Open-ended (rolling leaderboard) |

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
│   ├── scoring_script.py        # Evaluation
│   └── update_leaderboard.py    # Leaderboard management
├── .github/workflows/
│   └── score_submission.yml     # Auto-scoring
├── leaderboard.html             # Live leaderboard page
├── leaderboard.json             # Leaderboard data
├── index.html                   # Challenge homepage
├── _config.yml                  # GitHub Pages config
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

1. **Fork this repository**
2. **Train your model** and generate predictions
3. **Add your CSV** to `submissions/your_name.csv`
4. **Create a Pull Request**
5. **GitHub Actions scores automatically**
6. **Results posted** as comment and added to leaderboard

---

## 📈 Evaluation Metric

**Macro F1-Score** = (F1_Healthy + F1_Parkinson's) / 2

**Why?**
- ✅ Handles class imbalance
- ✅ Equal importance to both classes  
- ✅ More challenging than accuracy
- ✅ Better reflects real-world performance

---

## 💡 Tips & Tricks

### For Beginners
1. ✅ Start with baseline GCN
2. ✅ Try different hidden sizes (32, 64, 128)
3. ✅ Vary number of layers (2-4)
4. ✅ Add dropout for regularization (0.3-0.5)
5. ✅ Use cross-validation

### Advanced
- 🔥 Experiment with k in KNN graphs (3, 5, 7, 10)
- 🔥 Add edge weights based on similarity
- 🔥 Try GAT attention mechanisms
- 🔥 Use skip connections / residual connections
- 🔥 Handle class imbalance (weighted loss, oversampling)
- 🔥 Ensemble multiple models
- 🔥 Try GraphSAGE, GIN, or other architectures

### Common Pitfalls
⚠️ **Overfitting** (small dataset - use regularization!)  
⚠️ **Over-smoothing** (too many layers collapse node representations)  
⚠️ **Ignoring class imbalance** (use weighted metrics)  
⚠️ **Data leakage** (don't use test labels!)

---

## 🎯 Challenge Rules

### ✅ Must Do
- Use at least one GNN layer
- Only use provided dataset
- Complete inference within 5 minutes
- Set random seeds for reproducibility
- Provide code with submission

### ❌ Cannot Do
- Use test labels (obviously!)
- Use external Parkinson's datasets
- Use pure non-GNN models (e.g., just MLP)

**See [RULES.md](RULES.md) for complete details.**

---

## 🤝 Contributing

- **Bug?** [Open an issue](https://github.com/AiIkram/gnn-parkinsons-challenge/issues)
- **Question?** [Start a discussion](https://github.com/AiIkram/gnn-parkinsons-challenge/discussions)
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

- **Issues**: [GitHub Issues](https://github.com/AiIkram/gnn-parkinsons-challenge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AiIkram/gnn-parkinsons-challenge/discussions)
- **Email**: [aissiouikram47@gmail.com](mailto:aissiouikram47@gmail.com)

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

**Dataset License**: UCI Parkinson's Dataset - CC BY 4.0

---

<div align="center">

### 🚀 Ready to start? 

**[View Leaderboard](https://aiikram.github.io/gnn-parkinsons-challenge/leaderboard.html)** | **[Fork Repo](https://github.com/AiIkram/gnn-parkinsons-challenge/fork)** | **[Submit Solution](https://github.com/AiIkram/gnn-parkinsons-challenge/pulls)**

**Good luck! 🎉**

</div>
