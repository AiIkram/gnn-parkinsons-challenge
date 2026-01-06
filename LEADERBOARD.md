# 🏆 Leaderboard

*Last Updated: January 5, 2025*

---

## 🥇 Top Rankings

| Rank | Participant | F1-Score | Model Type | Date | Link |
|------|-------------|----------|------------|------|------|
| 🥇 1 | Baseline-GCN | 0.7654 | 2-layer GCN | 2025-01-05 | [code](starter_code/baseline.py) |
| 🥈 2 | Baseline-GAT | 0.7521 | 2-layer GAT | 2025-01-05 | [code](starter_code/baseline_gat.py) |
| 🥉 3 | *Open* | ??? | ??? | TBD | - |

---

## 📊 Scores by Architecture

### Graph Convolutional Networks (GCN)
1. **Baseline-GCN**: 0.7654

### Graph Attention Networks (GAT)
1. **Baseline-GAT**: 0.7521

### GraphSAGE
*(No submissions yet)*

### Other Architectures
*(No submissions yet)*

---

## 🎯 Performance Milestones

| Threshold | Achievement | Count |
|-----------|-------------|-------|
| ≥ 0.90 | 🏆 Excellent | 0 |
| ≥ 0.85 | 🥇 Great | 0 |
| ≥ 0.80 | 🥈 Good | 0 |
| ≥ 0.75 | 🥉 Decent | 2 |

---

## 📤 How to Submit

1. **Fork** this repository
2. **Train** your GNN model
3. **Generate** predictions on test set (39 rows)
4. **Add** CSV to `submissions/` folder
5. **Create** Pull Request
6. **Wait** for automated scoring
7. **Check** results in PR comment

---

## 💡 Tips from Top Submissions

### What Works
- 2-4 GNN layers
- Dropout 0.5-0.6
- Learning rate 0.005-0.01
- Early stopping
- K=5 for KNN graphs

### Avoid
- Too many layers (>5)
- Ignoring class imbalance
- Overfitting
- Skipping edge weights

---

## 🏅 Special Recognition

### Research Track (NeurIPS 2026)
Top 3 participants invited to co-author paper.

Current candidates:
1. *TBD*
2. *TBD*
3. *TBD*

---

## 📊 Statistics

- **Total Submissions**: 2
- **Unique Participants**: 2
- **Average F1-Score**: 0.7588
- **Best F1-Score**: 0.7654

---

**Good luck!** 🚀
```

**Save and close.**

---

## 🔷 STEP 7: Update .gitignore (IMPORTANT!)

**Open the existing `.gitignore` file** (it should already exist from GitHub)

**Add these lines at the END of the file:**
```
# CRITICAL - Hidden test labels - NEVER COMMIT
data/test_labels.pkl
data/test_labels.csv

# Model files
*.pt
*.pth
best_*.pt

# Results
*_results.txt