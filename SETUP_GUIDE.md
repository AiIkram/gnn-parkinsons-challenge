# 🚀 Complete Setup Guide for GNN Parkinson's Challenge

This guide will walk you through setting up the entire competition on GitHub, from creating the repository to launching the challenge.

---

## 📋 Prerequisites

Before starting, ensure you have:
- GitHub account
- Git installed on your computer
- Python 3.8+ installed
- Basic command line knowledge

---

## Step 1: Create GitHub Repository

### 1.1 Create New Repository

1. Go to [GitHub](https://github.com)
2. Click the **+** icon → **New repository**
3. Fill in details:
   - **Repository name**: `gnn-parkinsons-challenge`
   - **Description**: "GNN Mini-Challenge: Parkinson's Disease Detection using Graph Neural Networks"
   - **Visibility**: Public
   - **Initialize with**: README (unchecked, we'll add our own)
   - **License**: MIT License

4. Click **Create repository**

### 1.2 Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/gnn-parkinsons-challenge.git
cd gnn-parkinsons-challenge
```

---

## Step 2: Create Directory Structure

```bash
# Create all directories
mkdir -p data
mkdir -p submissions
mkdir -p starter_code
mkdir -p scripts
mkdir -p .github/workflows

# Create placeholder files
touch data/.gitkeep
touch submissions/.gitkeep
touch starter_code/.gitkeep
```

---

## Step 3: Add Core Files

### 3.1 Main README

Create `README.md` with the complete challenge description (see artifact).

### 3.2 Competition Rules

Create `RULES.md` with detailed rules (see artifact).

### 3.3 License

Create `LICENSE`:

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3.4 Leaderboard Template

Create `LEADERBOARD.md`:

```markdown
# 🏆 Leaderboard

*Updated automatically when submissions are merged*

## Current Rankings

| Rank | Participant | F1-Score | Model Type | Date | Submission |
|------|-------------|----------|------------|------|------------|
| 🥇 1 | Baseline-GCN | 0.7654 | 2-layer GCN | 2025-12-04 | [link](submissions/sample_submission.csv) |
| 🥈 2 | Baseline-GAT | 0.7521 | 2-layer GAT | 2025-12-24 | [link](submissions/gat_submission.csv) |

## Top Scores by Approach

### Graph Convolutional Networks (GCN)
1. Baseline-GCN: 0.7654

### Graph Attention Networks (GAT)
1. Baseline-GAT: 0.7521

### Other Architectures
*(No submissions yet)*

---

## How to Submit

1. Fork this repository
2. Add your submission CSV to `submissions/`
3. Create a Pull Request
4. Your score will be automatically calculated
5. Upon merge, leaderboard updates automatically

---

## Submission Archive

All submissions are preserved in the [submissions](submissions/) folder.
```

---

## Step 4: Add Code Files

### 4.1 Requirements

Create `starter_code/requirements.txt` (see artifact).

### 4.2 Data Generation Script

Create `scripts/generate_graph_data.py` (see artifact).

### 4.3 Baseline Models

Create both:
- `starter_code/baseline.py` (GCN model - see artifact)
- `starter_code/baseline_gat.py` (GAT model - see artifact)

### 4.4 Scoring Script

Create `scripts/scoring_script.py` (see artifact).

### 4.5 GitHub Actions Workflow

Create `.github/workflows/score_submission.yml` (see artifact).

---

## Step 5: Generate Dataset

### 5.1 Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install torch dgl pandas numpy scikit-learn networkx
```

### 5.2 Generate Graph Data

```bash
cd scripts
python generate_graph_data.py
```

This will:
- Download UCI Parkinson's dataset
- Create graph structures
- Generate train/val/test splits
- Save processed data to `data/` folder

**Important Files Created:**
- `data/train_graph.pkl` - Training data (commit this)
- `data/test_graph.pkl` - Test data without labels (commit this)
- `data/test_labels.pkl` - Hidden labels (DO NOT commit - add to .gitignore)
- `data/feature_names.txt` - Feature descriptions (commit this)

### 5.3 Create .gitignore

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Models
*.pt
*.pth
*.pkl.gz
best_*.pt

# Hidden test labels (CRITICAL - never commit these!)
data/test_labels.pkl
data/test_labels.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary
*.log
*.tmp
*_results.txt
```

---

## Step 6: Test the Setup

### 6.1 Run Baseline Model

```bash
cd starter_code
python baseline.py
```

Expected output:
- Training progress
- Validation F1-score around 0.76-0.78
- Generates `submissions/sample_submission.csv`

### 6.2 Test Scoring Script

```bash
cd ..
python scripts/scoring_script.py submissions/sample_submission.csv
```

Should display:
- Validation results
- Macro F1-Score
- Detailed metrics

---

## Step 7: Commit and Push

### 7.1 Stage Files

```bash
git add .
```

### 7.2 Commit

```bash
git commit -m "Initial setup: Add GNN Parkinson's Challenge

- Complete README with challenge description
- Competition rules and guidelines  
- Baseline GCN and GAT models
- Data generation script
- Automated scoring via GitHub Actions
- Training and test graph data
"
```

### 7.3 Push to GitHub

```bash
git push origin main
```

---

## Step 8: Enable GitHub Actions

### 8.1 Check Actions Tab

1. Go to your repository on GitHub
2. Click **Actions** tab
3. If prompted, click **I understand my workflows, go ahead and enable them**

### 8.2 Test Workflow

1. Create a test submission locally
2. Create a new branch: `git checkout -b test-submission`
3. Modify and commit a submission file
4. Push and create a Pull Request
5. Check if GitHub Actions runs automatically

---

## Step 9: Create Announcement

### 9.1 Add Topics

On GitHub repository page:
1. Click ⚙️ next to "About"
2. Add topics:
   - `graph-neural-networks`
   - `parkinsons-disease`
   - `deep-learning`
   - `machine-learning`
   - `competition`
   - `challenge`
   - `healthcare`
   - `gnn`

### 9.2 Update Repository Description

Set description:
"🧠 GNN Challenge: Detect Parkinson's Disease using Graph Neural Networks. Apply message passing, attention mechanisms & graph learning to medical diagnosis."

---

## Step 10: Launch Challenge

### 10.1 Create Discussion Categories

1. Go to **Settings** → **Features**
2. Enable **Discussions**
3. Create categories:
   - **General**: General discussion
   - **Q&A**: Ask questions
   - **Show and Tell**: Share results
   - **Ideas**: Suggest improvements

### 10.2 Post Welcome Announcement

Create first discussion post:

```markdown
# 🎉 Welcome to the GNN Parkinson's Challenge!

We're excited to launch this mini-competition focused on applying Graph Neural Networks to medical diagnosis!

## 🎯 Challenge Goal
Use GNNs to detect Parkinson's Disease from voice acoustic measurements.

## 📊 What Makes This Unique?
- Real medical dataset (UCI Parkinson's)
- Graph-structured learning approach
- Automated scoring via GitHub Actions
- Open-ended learning opportunity

## 🚀 Getting Started
1. Read the [README](../README.md) for full details
2. Clone the repo and run baseline models
3. Experiment with different GNN architectures
4. Submit your best model!

## 🏆 Top Performers
Will be invited to co-author a NeurIPS 2026 paper!

## ❓ Questions?
Ask here in Discussions or open an issue.

Good luck and happy learning! 🎓
```

### 10.3 Share Your Challenge

Share on:
- Twitter/X with hashtags: #MachineLearning #GNN #Healthcare
- LinkedIn
- Reddit: r/MachineLearning, r/deeplearning
- Your university/organization channels
- Discord servers for ML/AI

---

## 📝 Maintenance Checklist

### Daily
- [ ] Monitor new submissions
- [ ] Respond to questions in Discussions
- [ ] Check GitHub Actions are working

### Weekly  
- [ ] Update leaderboard manually if needed
- [ ] Review and merge valid submissions
- [ ] Add helpful comments to participant code

### Monthly
- [ ] Analyze participation metrics
- [ ] Consider adding hints or tips
- [ ] Update documentation if needed

---

## 🔧 Troubleshooting

### Issue: GitHub Actions Not Running

**Solution:**
1. Check `.github/workflows/score_submission.yml` exists
2. Verify Actions are enabled in Settings
3. Check workflow syntax with GitHub's validator

### Issue: Data Files Too Large

**Solution:**
```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

### Issue: Scoring Script Fails

**Solution:**
1. Verify `test_labels.pkl` exists in `data/`
2. Check Python dependencies are installed
3. Test locally first before pushing

### Issue: Can't Download UCI Dataset

**Solution:**
1. Download manually from: https://archive.ics.uci.edu/ml/datasets/parkinsons
2. Place `parkinsons.data` in `data/` folder
3. Modify `generate_graph_data.py` to load local file

---

## 🎓 Educational Resources

Include these in your repository wiki:

1. **GNN Tutorials**
   - Link to DGL tutorials
   - Link to PyG tutorials
   - Your own explanations

2. **Medical Context**
   - Parkinson's Disease overview
   - Why voice measurements?
   - Clinical significance

3. **Competition Tips**
   - Common pitfalls
   - Best practices
   - Advanced techniques

---

## 📞 Support

If you encounter issues during setup:

1. **Check existing issues**: Someone may have solved it
2. **Create new issue**: Describe your problem clearly
3. **Provide details**: OS, Python version, error messages
4. **Be patient**: Maintainers will help!

---

## ✅ Final Checklist

Before considering your setup complete:

- [ ] Repository created and public
- [ ] All core files committed
- [ ] Data generated and committed (except test_labels.pkl)
- [ ] Baseline models run successfully
- [ ] Scoring script works
- [ ] GitHub Actions enabled and tested
- [ ] README is comprehensive and clear
- [ ] RULES are detailed and fair
- [ ] Discussions enabled
- [ ] Topics and description added
- [ ] First announcement posted
- [ ] Challenge shared with community

---

## 🎉 Congratulations!

Your GNN Parkinson's Challenge is now live! 

Watch as participants:
- Fork your repository
- Submit creative solutions
- Learn about GNNs and medical AI
- Compete for top scores

Enjoy running your mini-competition! 🚀

---

**Questions about this guide?**  
Open an issue with label `setup-question`