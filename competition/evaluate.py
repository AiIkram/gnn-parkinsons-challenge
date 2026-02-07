# competition/evaluate.py
import pandas as pd
import argparse
from metrics import evaluate_predictions, validate_submission

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--test_nodes', required=True)
    args = parser.parse_args()
    
    # Load files
    submission = pd.read_csv(args.submission)
    ground_truth = pd.read_csv(args.ground_truth)
    test_nodes = pd.read_csv(args.test_nodes)
    
    # Validate
    is_valid, error = validate_submission(submission, test_nodes)
    if not is_valid:
        print(f"INVALID: {error}")
        exit(1)
    
    # Merge and evaluate
    merged = submission.merge(ground_truth, on='node_id')
    score = evaluate_predictions(merged['label'], merged['prediction'])
    
    print(f"SCORE: {score:.4f}")

if __name__ == '__main__':
    main()
    