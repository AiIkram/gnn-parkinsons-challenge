#!/usr/bin/env python3
"""
GNN Parkinson's Challenge - Scoring Script
Evaluates submissions and updates the leaderboard
"""

import pandas as pd
import pickle
import sys
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def load_ground_truth(ground_truth_path=None):
    """Load ground truth labels"""
    # If ground truth path is provided (from GitHub Actions secret)
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"✓ Loading ground truth from: {ground_truth_path}")
        df = pd.read_csv(ground_truth_path)
        
        # Ensure correct column names
        if 'node_id' in df.columns and 'label' in df.columns:
            return df.sort_values('node_id').reset_index(drop=True)
        elif len(df.columns) == 2:
            df.columns = ['node_id', 'label']
            return df.sort_values('node_id').reset_index(drop=True)
    
    # Try multiple possible paths for local testing
    possible_paths = [
        Path('data/test_labels.pkl'),
        Path('../data/test_labels.pkl'),
        Path('../../data/test_labels.pkl'),
        Path('/tmp/ground_truth.csv'),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found ground truth at: {path}")
            
            # Handle CSV files
            if path.suffix == '.csv':
                df = pd.read_csv(path)
                if 'node_id' not in df.columns or 'label' not in df.columns:
                    if len(df.columns) == 2:
                        df.columns = ['node_id', 'label']
                return df.sort_values('node_id').reset_index(drop=True)
            
            # Handle pickle files
            with open(path, 'rb') as f:
                labels = pickle.load(f)
            
            # Convert to DataFrame if needed
            if isinstance(labels, dict):
                df = pd.DataFrame(list(labels.items()), columns=['node_id', 'label'])
            elif isinstance(labels, pd.DataFrame):
                df = labels
            elif isinstance(labels, pd.Series):
                df = pd.DataFrame({'node_id': labels.index, 'label': labels.values})
            else:
                df = pd.DataFrame({'node_id': range(len(labels)), 'label': labels})
            
            return df.sort_values('node_id').reset_index(drop=True)
    
    # If not found, return None
    print("❌ Error: Ground truth labels not found")
    print(f"\n📝 Note for challenge organizers:")
    print(f"   Ground truth should be in: data/test_labels.pkl or /tmp/ground_truth.csv")
    print(f"   Current working directory: {Path.cwd()}")
    return None

def validate_submission(submission_df):
    """Validate submission format"""
    errors = []
    
    # Check columns
    required_cols = ['node_id', 'prediction']
    if not all(col in submission_df.columns for col in required_cols):
        errors.append(f"Missing required columns. Expected: {required_cols}, Got: {list(submission_df.columns)}")
    
    # Check for 39 nodes (0-38)
    if len(submission_df) != 39:
        errors.append(f"Expected 39 predictions, got {len(submission_df)}")
    
    # Check node IDs
    expected_ids = set(range(39))
    actual_ids = set(submission_df['node_id'].values)
    if actual_ids != expected_ids:
        errors.append(f"node_id values must be 0-38. Missing: {expected_ids - actual_ids}, Extra: {actual_ids - expected_ids}")
    
    # Check predictions are binary
    if not all(submission_df['prediction'].isin([0, 1])):
        errors.append("All predictions must be 0 or 1")
    
    return errors

def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    # Try to calculate AUC if probabilities are available
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['auc_roc'] = None
    
    return metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <submission_file> [submission_name] [ground_truth_file]")
        sys.exit(1)
    
    submission_file = Path(sys.argv[1])
    submission_name = sys.argv[2] if len(sys.argv) > 2 else submission_file.stem
    ground_truth_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load submission
    try:
        submission_df = pd.read_csv(submission_file)
        print(f"\n📄 Loaded submission: {submission_file}")
        print(f"   Shape: {submission_df.shape}")
    except Exception as e:
        print(f"❌ Error loading submission: {e}")
        sys.exit(1)
    
    # Validate format
    errors = validate_submission(submission_df)
    if errors:
        print("❌ Submission validation failed:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    print("✓ Submission format is valid")
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_file)
    if ground_truth is None:
        print("\n⚠️  Cannot score submission without ground truth labels.")
        print("   Your submission format is valid and ready to submit!")
        sys.exit(0)
    
    # Merge and calculate metrics
    merged = submission_df.merge(ground_truth, on='node_id', how='inner')
    
    if len(merged) != 39:
        print(f"❌ Error: Could only match {len(merged)}/39 predictions with ground truth")
        sys.exit(1)
    
    metrics = calculate_metrics(merged['label'], merged['prediction'])
    
    # Display results
    print("\n" + "="*60)
    print(f"RESULTS FOR: {submission_name}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if metrics['auc_roc']:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("="*60)
    
    # For GitHub Actions - output in parseable format
    print(f"\nScore: {metrics['f1_score']:.4f}")
    print(f"Rank: 1")  # Placeholder - will be calculated by workflow
    
    print("\n📊 Class distribution in predictions:")
    print(submission_df['prediction'].value_counts())

if __name__ == '__main__':
    main()
