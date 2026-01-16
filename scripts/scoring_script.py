#!/usr/bin/env python3
"""
Scoring script for GNN Parkinson's Challenge
Evaluates submission files against ground truth labels
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import sys
import os
import pickle

def load_ground_truth():
    """Load ground truth labels from pickle or CSV"""
    
    # Try multiple possible locations
    possible_paths = [
        'data/test_labels.csv',
        'data/test_labels.pkl',
        '../data/test_labels.csv',
        '../data/test_labels.pkl',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                    print(f"✓ Loaded ground truth from {path}")
                    return df
                elif path.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Convert to DataFrame
                    if isinstance(data, dict):
                        df = pd.DataFrame(list(data.items()), columns=['node_id', 'label'])
                    elif isinstance(data, pd.DataFrame):
                        df = data
                    else:
                        # Assume array of labels
                        df = pd.DataFrame({
                            'node_id': range(len(data)),
                            'label': data
                        })
                    
                    # Ensure node_ids are 0-38
                    if len(df) != 39:
                        if len(df) > 39:
                            df = df.iloc[:39]
                        df['node_id'] = range(39)
                    
                    print(f"✓ Loaded ground truth from {path}")
                    return df
            except Exception as e:
                continue
    
    return None


def validate_submission(submission_df):
    """Validate submission format"""
    
    errors = []
    
    # Check required columns
    if 'node_id' not in submission_df.columns:
        errors.append("Missing 'node_id' column")
    if 'prediction' not in submission_df.columns:
        errors.append("Missing 'prediction' column")
    
    if errors:
        return False, errors
    
    # Check node_id range
    node_ids = submission_df['node_id'].values
    if node_ids.min() < 0 or node_ids.max() > 38:
        errors.append(f"node_id values must be between 0 and 38 (got {node_ids.min()} to {node_ids.max()})")
    
    # Check for 39 unique nodes
    if len(node_ids) != 39:
        errors.append(f"Expected 39 predictions, got {len(node_ids)}")
    
    if len(set(node_ids)) != 39:
        errors.append(f"node_id values must be unique (got {len(set(node_ids))} unique values)")
    
    # Check predictions are 0 or 1
    predictions = submission_df['prediction'].values
    if not all(p in [0, 1] for p in predictions):
        errors.append("prediction values must be 0 or 1")
    
    return len(errors) == 0, errors


def score_submission(submission_path, verbose=False):
    """Score a submission file"""
    
    # Load submission
    try:
        submission_df = pd.read_csv(submission_path)
        if verbose:
            print(f"\n📄 Loaded submission: {submission_path}")
            print(f"   Shape: {submission_df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {submission_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading submission: {e}")
        return None
    
    # Validate submission format
    is_valid, errors = validate_submission(submission_df)
    if not is_valid:
        print(f"❌ Error: Invalid submission format")
        for error in errors:
            print(f"   - {error}")
        return None
    
    if verbose:
        print("✓ Submission format is valid")
    
    # Load ground truth
    ground_truth_df = load_ground_truth()
    
    if ground_truth_df is None:
        print("❌ Error: Ground truth labels not found")
        print("\n📝 Note for challenge organizers:")
        print("   Ground truth should be in: data/test_labels.csv or data/test_labels.pkl")
        print("   Format: CSV with columns 'node_id' (0-38) and 'label' (0 or 1)")
        return None
    
    # Merge and compute scores
    submission_df = submission_df.sort_values('node_id')
    ground_truth_df = ground_truth_df.sort_values('node_id')
    
    y_true = ground_truth_df['label'].values
    y_pred = submission_df['prediction'].values
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    scores = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f} ⭐ (Primary Metric)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=" * 60)
    
    if verbose:
        from sklearn.metrics import confusion_matrix, classification_report
        print("\n📊 Detailed Metrics:")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Healthy', 'Parkinsons'],
                                   digits=4))
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='Score submissions for GNN Parkinson\'s Challenge'
    )
    parser.add_argument('submission', type=str, help='Path to submission CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed metrics')
    
    args = parser.parse_args()
    
    scores = score_submission(args.submission, verbose=args.verbose)
    
    if scores is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()