#!/usr/bin/env python3
"""
Scoring script for GNN Parkinson's Challenge
Validates and scores submission CSV files against ground truth
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pickle
import os

def load_ground_truth():
    """Load test labels from pickle file"""
    try:
        # Try multiple possible paths
        paths = [
            'data/test_labels.pkl',
            'data/test_graph.pkl',
            '../data/test_labels.pkl'
        ]
        
        for path in paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Handle different data structures
                    if hasattr(data, 'ndata') and 'label' in data.ndata:
                        return data.ndata['label'].numpy()
                    elif isinstance(data, np.ndarray):
                        return data
                    elif isinstance(data, list):
                        return np.array(data)
                        
        raise FileNotFoundError("Ground truth labels not found")
        
    except Exception as e:
        print(f"Error loading ground truth: {e}", file=sys.stderr)
        sys.exit(1)

def validate_submission(df, expected_length=39):
    """Validate submission format and content"""
    
    # Check shape
    if len(df) != expected_length:
        print(f"❌ Error: Expected {expected_length} rows, got {len(df)}", file=sys.stderr)
        return False
    
    # Check required columns
    required_cols = ['node_id', 'prediction']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Error: Missing required column '{col}'", file=sys.stderr)
            print(f"   Found columns: {list(df.columns)}", file=sys.stderr)
            return False
    
    # Check node_id range
    if not all(df['node_id'].between(0, expected_length - 1)):
        print("❌ Error: node_id values must be between 0 and 38", file=sys.stderr)
        return False
    
    # Check for duplicates
    if df['node_id'].duplicated().any():
        print("❌ Error: Duplicate node_id values found", file=sys.stderr)
        return False
    
    # Check prediction values
    unique_preds = df['prediction'].unique()
    if not all(p in [0, 1] for p in unique_preds):
        print(f"❌ Error: Predictions must be 0 or 1, found: {unique_preds}", file=sys.stderr)
        return False
    
    return True

def score_submission(csv_path, verbose=False):
    """Score a submission CSV against ground truth"""
    
    # Load submission
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    if not validate_submission(df):
        sys.exit(1)
    
    # Load ground truth
    y_true = load_ground_truth()
    
    if len(y_true) != len(df):
        print(f"❌ Error: Ground truth has {len(y_true)} samples, submission has {len(df)}", file=sys.stderr)
        sys.exit(1)
    
    # Sort by node_id and extract predictions
    df = df.sort_values('node_id').reset_index(drop=True)
    y_pred = df['prediction'].values
    
    # Calculate metrics
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Print primary score
    print(f"{f1_macro:.4f}")
    
    # Verbose output
    if verbose:
        print("\n" + "="*60, file=sys.stderr)
        print("📊 SUBMISSION SCORING REPORT", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"\n✅ Validation: PASSED", file=sys.stderr)
        print(f"📈 F1-Score (Macro):    {f1_macro:.4f}", file=sys.stderr)
        print(f"📈 F1-Score (Weighted): {f1_weighted:.4f}", file=sys.stderr)
        
        print("\n📋 Classification Report:", file=sys.stderr)
        print(classification_report(y_true, y_pred, 
                                   target_names=['Healthy', 'Parkinson\'s'],
                                   digits=4), file=sys.stderr)
        
        print("🔢 Confusion Matrix:", file=sys.stderr)
        cm = confusion_matrix(y_true, y_pred)
        print(f"   Predicted:  Healthy  PD", file=sys.stderr)
        print(f"   Healthy  :  {cm[0,0]:>7}  {cm[0,1]:>3}", file=sys.stderr)
        print(f"   PD       :  {cm[1,0]:>7}  {cm[1,1]:>3}", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
    
    return f1_macro

def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <csv_file> [--verbose]", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python scoring_script.py submissions/my_team.csv", file=sys.stderr)
        print("  python scoring_script.py submissions/my_team.csv --verbose", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    score_submission(csv_path, verbose=verbose)

if __name__ == "__main__":
    main()