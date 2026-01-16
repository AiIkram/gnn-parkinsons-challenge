#!/usr/bin/env python3
"""
Update leaderboard.json from submissions folder
Scans all CSV files and their metadata to create ranked leaderboard
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

def scan_submissions(submissions_dir='submissions'):
    """Scan submissions folder for CSV files and metadata"""
    
    if not os.path.exists(submissions_dir):
        print(f"⚠️  Warning: {submissions_dir} folder not found", file=sys.stderr)
        return []
    
    submissions = []
    
    # Scan all CSV files
    for filename in os.listdir(submissions_dir):
        if not filename.endswith('.csv'):
            continue
        
        team_name = filename.replace('.csv', '')
        filepath = os.path.join(submissions_dir, filename)
        
        # Look for metadata file
        metadata_file = filepath.replace('.csv', '_metadata.json')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                
                submissions.append({
                    'team_name': team_name,
                    'score': float(meta.get('score', 0.0)),
                    'model': meta.get('model', 'N/A'),
                    'date': meta.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'description': meta.get('description', '')
                })
                
                print(f"✅ Loaded: {team_name} - Score: {meta.get('score', 0.0):.4f}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load metadata for {team_name}: {e}", file=sys.stderr)
                continue
        else:
            # No metadata - skip or use defaults
            print(f"⚠️  Skipping {team_name}: No metadata file found", file=sys.stderr)
            print(f"   Expected: {metadata_file}", file=sys.stderr)
    
    return submissions

def update_leaderboard(submissions_dir='submissions', output_file='leaderboard.json'):
    """Update leaderboard JSON file with current submissions"""
    
    print("\n" + "="*60)
    print("🔄 UPDATING LEADERBOARD")
    print("="*60)
    
    # Scan submissions
    submissions = scan_submissions(submissions_dir)
    
    if not submissions:
        print("\n⚠️  No valid submissions found!")
        print("   Make sure each submission has a metadata JSON file:")
        print("   - submissions/team_name.csv")
        print("   - submissions/team_name_metadata.json")
        
        # Create empty leaderboard
        data = {
            'last_updated': datetime.now().isoformat(),
            'submissions': []
        }
    else:
        # Sort by score (descending)
        submissions.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for i, sub in enumerate(submissions):
            sub['rank'] = i + 1
        
        # Create leaderboard data
        data = {
            'last_updated': datetime.now().isoformat(),
            'submissions': submissions
        }
        
        print(f"\n📊 Leaderboard Summary:")
        print(f"   Total submissions: {len(submissions)}")
        print(f"   Best score: {submissions[0]['score']:.4f} ({submissions[0]['team_name']})")
        print(f"   Worst score: {submissions[-1]['score']:.4f} ({submissions[-1]['team_name']})")
    
    # Write to JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Leaderboard updated successfully!")
        print(f"   Output: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error writing leaderboard: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("="*60 + "\n")

def create_example_metadata():
    """Create example metadata file for reference"""
    
    example = {
        "score": 0.7654,
        "model": "GCN",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "description": "Graph Convolutional Network with 2 layers"
    }
    
    output_path = "submissions/EXAMPLE_metadata.json"
    os.makedirs("submissions", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"📝 Created example metadata: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update leaderboard from submissions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_leaderboard.py
  python update_leaderboard.py --submissions-dir my_submissions
  python update_leaderboard.py --create-example

Metadata Format (team_name_metadata.json):
  {
    "score": 0.8500,
    "model": "GAT",
    "date": "2025-01-16",
    "description": "Graph Attention Network"
  }
        """
    )
    
    parser.add_argument(
        '--submissions-dir',
        default='submissions',
        help='Directory containing submission CSV files (default: submissions)'
    )
    
    parser.add_argument(
        '--output',
        default='leaderboard.json',
        help='Output JSON file (default: leaderboard.json)'
    )
    
    parser.add_argument(
        '--create-example',
        action='store_true',
        help='Create example metadata file'
    )
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_metadata()
        return
    
    update_leaderboard(args.submissions_dir, args.output)

if __name__ == '__main__':
    main()