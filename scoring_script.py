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

def load_ground_truth():
    """Load ground truth labels from data directory"""
    # Try multiple possible paths
    possible_paths = [
        Path('data/test_labels.pkl'),
        Path('../data/test_labels.pkl'),
        Path('../../data/test_labels.pkl'),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found ground truth at: {path}")
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
    print(f"   Ground truth should be in: data/test_labels.pkl")
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
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    
    # Try to calculate AUC if probabilities are available
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['auc_roc'] = None
    
    return metrics

def update_leaderboard(submission_name, metrics, submission_file):
    """Update leaderboard JSON file"""
    leaderboard_file = Path('leaderboard.json')
    
    # Load existing leaderboard
    if leaderboard_file.exists():
        with open(leaderboard_file, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {'submissions': []}
    
    # Create new entry with all required fields
    entry = {
        'name': submission_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'file': str(submission_file),
        'accuracy': float(metrics['accuracy']),
        'f1_score': float(metrics['f1_score']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall'])
    }
    
    # Add AUC if available
    if metrics.get('auc_roc') is not None:
        entry['auc_roc'] = float(metrics['auc_roc'])
    
    # Add or update entry
    existing_idx = None
    for i, sub in enumerate(leaderboard['submissions']):
        if sub['name'] == submission_name:
            existing_idx = i
            break
    
    if existing_idx is not None:
        leaderboard['submissions'][existing_idx] = entry
    else:
        leaderboard['submissions'].append(entry)
    
    # Sort by F1 score
    leaderboard['submissions'].sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Save updated leaderboard
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    print(f"\n✓ Leaderboard updated: {leaderboard_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <submission_file> [--verbose] [--name <submission_name>]")
        sys.exit(1)
    
    submission_file = Path(sys.argv[1])
    verbose = '--verbose' in sys.argv
    
    # Get submission name
    if '--name' in sys.argv:
        name_idx = sys.argv.index('--name') + 1
        submission_name = sys.argv[name_idx] if name_idx < len(sys.argv) else submission_file.stem
    else:
        submission_name = submission_file.stem
    
    # Load submission
    try:
        submission_df = pd.read_csv(submission_file)
        if verbose:
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
    
    if verbose:
        print("✓ Submission format is valid")
    
    # Load ground truth
    ground_truth = load_ground_truth()
    if ground_truth is None:
        print("\n⚠️  Cannot score submission without ground truth labels.")
        print("   Your submission format is valid and ready to submit!")
        sys.exit(0)
    
    # Merge and calculate metrics
    merged = submission_df.merge(ground_truth, on='node_id')
    metrics = calculate_metrics(merged['label'], merged['prediction'])
    
    # Display results
    print("\n" + "="*60)
    print(f"RESULTS FOR: {submission_name}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if metrics['auc_roc']:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("="*60)
    
    # Update leaderboard
    try:
        update_leaderboard(submission_name, metrics, submission_file)
    except Exception as e:
        print(f"⚠️  Warning: Could not update leaderboard: {e}")
    
    if verbose:
        print("\n📊 Class distribution in predictions:")
        print(submission_df['prediction'].value_counts())
        print("\n📊 Class distribution in ground truth:")
        print(ground_truth['label'].value_counts())

if __name__ == '__main__':
    main()