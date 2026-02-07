# competition/metrics.py
from sklearn.metrics import f1_score
import pandas as pd

def evaluate_predictions(y_true, y_pred):
    """
    Calculate Macro F1-Score for Parkinson's detection
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        float: Macro F1-Score
    """
    return f1_score(y_true, y_pred, average='macro')

def validate_submission(submission_df, test_nodes_df):
    """
    Validate submission format
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if 'node_id' not in submission_df.columns:
        return False, "Missing 'node_id' column"
    
    if 'prediction' not in submission_df.columns:
        return False, "Missing 'prediction' column"
    
    if len(submission_df) != len(test_nodes_df):
        return False, f"Expected {len(test_nodes_df)} rows, got {len(submission_df)}"
    
    if not set(submission_df['node_id']).issubset(set(test_nodes_df['id'])):
        return False, "Invalid node IDs in submission"
    
    return True, None