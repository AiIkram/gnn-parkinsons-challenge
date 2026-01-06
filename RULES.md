# 📜 Competition Rules & Guidelines

## 🎯 Objective

Develop a Graph Neural Network (GNN) model to accurately classify Parkinson's Disease from voice acoustic measurements. Achieve the highest **Macro F1-Score** on the hidden test set.

---

## ✅ Requirements

### 1. Model Requirements

**MUST USE GNN:**
- Your model **must** include at least one GNN layer
- Acceptable GNN layers:
  - Graph Convolutional Networks (GCN)
  - Graph Attention Networks (GAT)
  - GraphSAGE
  - Graph Isomorphism Network (GIN)
  - Any message-passing neural network
- Can combine GNN with other architectures (MLP, attention, etc.)

**MUST NOT:**
- Use purely non-GNN models
- Use test set labels during training
- Use external Parkinson's datasets
- Manually engineer features using test set info

### 2. Data Usage

**ALLOWED:**
✅ Use all provided training data  
✅ Create validation splits from training  
✅ Feature engineering using training data  
✅ Graph augmentation based on training  
✅ Oversampling/undersampling training data  
✅ Cross-validation on training set  

**PROHIBITED:**
❌ Using test set labels  
❌ Using test node features for training  
❌ External Parkinson's datasets  
❌ Test set distribution information  
❌ Accessing `test_labels.pkl` file  

### 3. Technical Constraints

- **Inference Time**: Max 5 minutes on standard CPU
- **Dependencies**: Only pip/conda installable packages
- **Reproducibility**: Set random seeds, document hyperparameters
- **Model Size**: No explicit limit (be reasonable)

---

## 📤 Submission Requirements

### 1. CSV Format

Must have:
- Exactly **39 rows** (one per test node)
- Two columns: `node_id` and `prediction`
- Binary predictions: 0 (Healthy) or 1 (Parkinson's)
```csv
node_id,prediction
0,1
1,0
...
```

### 2. Code Submission

Include:
- Training script
- Model definition
- README with:
  - Architecture description
  - Hyperparameters
  - Training procedure
  - Training time
  - Hardware used
- requirements.txt

---

## 🏆 Evaluation

### Primary Metric: Macro F1-Score
```
Macro F1 = (F1_Healthy + F1_Parkinson's) / 2
```

**Why Macro F1?**
- Handles class imbalance
- Equal importance to both classes
- Requires good minority class performance

### Scoring Process

1. Submit CSV via Pull Request
2. GitHub Actions scores automatically
3. Results in PR comment
4. Leaderboard updates on merge

### Ranking

- By Macro F1-Score (higher = better)
- Ties broken by submission date
- Only best submission per participant
- Can resubmit anytime

---

## 🎓 Allowed Techniques

### Graph Construction
- Modify k in KNN
- Add/remove edges
- Weight edges by similarity
- Multiple edge types
- Subject connections

### Model Architecture
- Stack GNN layers
- Different aggregations
- Skip connections
- Combine GNN types
- Attention mechanisms
- Batch/layer normalization
- Dropout

### Training
- Cross-validation
- Ensemble methods
- Class weighting
- Focal loss
- Data augmentation
- Semi-supervised learning

### Features
- Different normalization
- Polynomial features
- PCA/dimensionality reduction
- Feature selection
- Domain-specific features

---

## 🚫 Prohibited Actions

### Strict Violations (Disqualification)

1. **Data Leakage** - Using test labels/features
2. **External Data** - Other Parkinson's datasets
3. **Non-GNN Models** - Models without GNN component
4. **Cheating** - Reverse engineering test labels

### Soft Violations (Warning → Disqualification)

1. **Reproducibility Issues** - Missing/broken code
2. **Technical Violations** - Time limit exceeded
3. **Gaming System** - Exploiting loopholes

---

## 👥 Participation

### Individual
- Compete under your GitHub username
- General discussion OK
- No direct code collaboration

### Teams (up to 3)
- Declare members at first submission
- Clear team name in filename
- One submission per team

---

## 🔄 Resubmission Policy

- **Unlimited resubmissions**
- Only best score counts
- Previous submissions visible
- Iterate and improve!

---

## 🎖️ Recognition

### Top 3 Finishers
- Featured on leaderboard
- Invited to NeurIPS 2026 paper
- Research acknowledgment

### All Participants
- Feedback on approaches
- Learning resources
- Community recognition

---

## ⚖️ Fair Play

### Honor Code

By participating, you agree to:
1. Follow all rules honestly
2. Not use test labels/forbidden data
3. Provide reproducible code
4. Not share test label info
5. Compete in learning spirit

### Reporting Violations

If you notice violations:
1. Open confidential issue
2. Contact organizers
3. Provide evidence

All reports investigated. Confirmed violations = disqualification.

---

## 🤝 Collaboration Guidelines

### Encouraged
✅ Discussing GNN architectures  
✅ Sharing learning resources  
✅ Explaining concepts  
✅ Asking questions  
✅ Reviewing documentation  

### Prohibited
❌ Sharing model implementations  
❌ Sharing trained weights  
❌ Sharing hyperparameter searches  
❌ Coordinating submissions  
❌ Reverse engineering together  

---

## 📝 Submission Checklist

Before submitting:

- [ ] CSV has exactly 39 rows
- [ ] Predictions are 0 or 1
- [ ] Node IDs match test set
- [ ] No missing values
- [ ] Model uses GNN layer
- [ ] Code included and documented
- [ ] README explains approach
- [ ] Random seeds set
- [ ] requirements.txt provided
- [ ] No test set info used
- [ ] Inference under 5 minutes

---

## 🎯 Timeline

- **Start**: December 5, 2025
- **End**: Open-ended (rolling)
- **Leaderboard**: Continuous updates
- **NeurIPS 2026**: May deadline
- **Paper Selection**: April 2026

---

## 📧 Questions

### Where to Ask

1. **GitHub Discussions**: General questions
2. **GitHub Issues**: Technical problems
3. **Email**: Research collaboration

### Response Times

- Discussions: 1-2 days
- Issues: 1-3 days
- Email: 3-5 days

---

## 🔄 Rule Updates

- Rules may be clarified
- Changes announced in Discussions
- No retroactive disqualifications
- Check before each submission

---

## 🌟 Focus on Learning!

Remember:
- Understanding GNNs
- Creative solutions
- Learning from failures
- Building fundamentals

The journey matters most! 🚀

---

**Last Updated**: January 3, 2026  
**Version**: 1.0