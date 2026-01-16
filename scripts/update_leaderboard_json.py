#!/usr/bin/env python3
"""
Generate leaderboard JSON from submissions folder.
"""

import os
import json
from datetime import datetime
import pandas as pd

def scan_submissions():
    """Scan submissions folder and generate leaderboard data."""
    
    submissions_dir = 'submissions'
    submissions = []
    
    if not os.path.exists(submissions_dir):
        print("Submissions directory not found")
        return submissions
    
    # Scan each submission folder
    for username in os.listdir(submissions_dir):
        user_path = os.path.join(submissions_dir, username)
        
        if not os.path.isdir(user_path):
            continue
        
        # Check for metadata file
        metadata_file = os.path.join(user_path, 'metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            submissions.append({
                'username': username,
                'score': metadata.get('score', 0.0),
                'model': metadata.get('model', 'Unknown'),
                'date': metadata.get('date', datetime.now().isoformat()[:10]),
                'code': f'submissions/{username}/',
                'description': metadata.get('description', ''),
                'isNew': is_recent(metadata.get('date', '')),
                'isTop': False  # Will be set later
            })
    
    # Sort by score (descending)
    submissions.sort(key=lambda x: x['score'], reverse=True)
    
    # Assign ranks and mark top 3
    for i, sub in enumerate(submissions):
        sub['rank'] = i + 1
        sub['isTop'] = (i < 3)
    
    return submissions


def is_recent(date_str, days=7):
    """Check if submission is within last N days."""
    try:
        submit_date = datetime.fromisoformat(date_str)
        days_ago = (datetime.now() - submit_date).days
        return days_ago <= days
    except:
        return False


def generate_leaderboard_json(submissions):
    """Generate final JSON file."""
    
    data = {
        'last_updated': datetime.now().isoformat(),
        'baseline_score': 0.7654,
        'total_submissions': len(submissions),
        'top_score': submissions[0]['score'] if submissions else 0.0,
        'submissions': submissions
    }
    
    # Write to file
    os.makedirs('docs', exist_ok=True)
    with open('docs/leaderboard_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Generated leaderboard with {len(submissions)} submissions")
    print(f"   Top score: {data['top_score']:.4f}")


if __name__ == '__main__':
    submissions = scan_submissions()
    generate_leaderboard_json(submissions)