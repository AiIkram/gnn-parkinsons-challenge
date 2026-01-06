import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def load_ground_truth():
    try:
        with open('data/test_labels.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
        return ground_truth
    except FileNotFoundError:
        print("Error: test_labels.pkl not found.")
        sys.exit(1)


def validate_submission(submission_df, ground_truth):
    errors = []
    
    required_cols = ['node_id', 'prediction']
    for col in required_cols:
        if col not in submission_df.columns:
            errors.append(f"Missing column: '{col}'")
    
    if errors:
        return False, errors
    
    expected_size = len(ground_truth['node_ids'])
    if len(submission_df) != expected_size:
        errors.append(f"Expected {expected_size} predictions, got {len(submission_df)}")
    
    expected_ids = set(ground_truth['node_ids'])
    actual_ids = set(submission_df['node_id'].values)
    
    if expected_ids != actual_ids:
        errors.append(f"Node IDs don't match")
    
    unique_preds = submission_df['prediction'].unique()
    if not all(pred in [0, 1] for pred in unique_preds):
        errors.append(f"Predictions must be 0 or 1")
    
    if submission_df.isnull().any().any():
        errors.append("Contains missing values")
    
    return len(errors) == 0, errors


def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metrics


def print_results(metrics, submission_file):
    print("\n" + "=" * 70)
    print("SUBMISSION EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nSubmission: {submission_file}")
    print("\n" + "-" * 70)
    print("PRIMARY METRIC")
    print("-" * 70)
    print(f"  Macro F1-Score: {metrics['f1_macro']:.4f}")
    
    print("\n" + "-" * 70)
    print("OVERALL METRICS")
    print("-" * 70)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    cm = metrics['confusion_matrix']
    print("\n           Predicted")
    print("         Healthy  PD")
    print(f"Healthy    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"PD         {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    print("\n" + "=" * 70)
    
    if metrics['f1_macro'] >= 0.90:
        print("\n  🏆 EXCELLENT!")
    elif metrics['f1_macro'] >= 0.85:
        print("\n  🥇 GREAT!")
    elif metrics['f1_macro'] >= 0.80:
        print("\n  🥈 GOOD!")
    elif metrics['f1_macro'] >= 0.75:
        print("\n  🥉 DECENT!")
    else:
        print("\n  ⚠️  Needs improvement.")
    
    print("\n" + "=" * 70 + "\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scoring_script.py <submission_file.csv>")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    
    print("=" * 70)
    print("GNN Parkinson's Challenge - Scoring Script")
    print("=" * 70)
    
    print("\nLoading ground truth...")
    ground_truth = load_ground_truth()
    
    print(f"\nLoading submission: {submission_file}")
    try:
        submission = pd.read_csv(submission_file)
    except FileNotFoundError:
        print(f"Error: File '{submission_file}' not found.")
        sys.exit(1)
    
    print("\nValidating submission...")
    is_valid, errors = validate_submission(submission, ground_truth)
    
    if not is_valid:
        print("\n❌ SUBMISSION INVALID!")
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("✓ Valid")
    
    submission_sorted = submission.sort_values('node_id')
    y_true = ground_truth['labels']
    y_pred = submission_sorted['prediction'].values
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    print_results(metrics, submission_file)
    
    return metrics['f1_macro']


if __name__ == '__main__':
    main()