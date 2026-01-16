#!/usr/bin/env python3
"""
Update leaderboard HTML from JSON data
"""

import json
from pathlib import Path
from datetime import datetime

def generate_leaderboard_html(leaderboard_data):
    """Generate HTML leaderboard from JSON data"""
    
    submissions = leaderboard_data.get('submissions', [])
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Parkinson's Challenge - Leaderboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 3rem;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .stat-card h3 {{
            color: #667eea;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .stat-card p {{
            color: #666;
            font-size: 0.9rem;
        }}
        
        .leaderboard {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th, td {{
            padding: 1rem;
            text-align: left;
        }}
        
        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #e0e0e0;
            transition: background 0.2s;
        }}
        
        tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        tbody tr:last-child {{
            border-bottom: none;
        }}
        
        .rank {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.2rem;
        }}
        
        .medal {{
            font-size: 1.5rem;
        }}
        
        .score {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .metric {{
            color: #666;
            font-size: 0.9rem;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 2rem;
            opacity: 0.8;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: #999;
        }}
        
        .empty-state h3 {{
            margin-bottom: 1rem;
            color: #666;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            
            th, td {{
                padding: 0.75rem 0.5rem;
                font-size: 0.85rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 GNN Parkinson's Challenge</h1>
            <p>Live Leaderboard - Ranked by F1 Score</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>{len(submissions)}</h3>
                <p>Total Submissions</p>
            </div>
            <div class="stat-card">
                <h3>{f"{submissions[0]['f1_score']:.4f}" if submissions and submissions[0].get('f1_score') is not None else 'N/A'}</h3>
                <p>Best F1 Score</p>
            </div>
            <div class="stat-card">
                <h3>{f"{submissions[0]['accuracy']:.4f}" if submissions and submissions[0].get('accuracy') is not None else 'N/A'}</h3>
                <p>Best Accuracy</p>
            </div>
        </div>
        
        <div class="leaderboard">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Team</th>
                        <th>F1 Score</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Submission Date</th>
                    </tr>
                </thead>
                <tbody>
'''
    
    if not submissions:
        html += '''
                    <tr>
                        <td colspan="7" class="empty-state">
                            <h3>No submissions yet</h3>
                            <p>Be the first to submit!</p>
                        </td>
                    </tr>
'''
    else:
        medals = ['🥇', '🥈', '🥉']
        for i, sub in enumerate(submissions, 1):
            medal = medals[i-1] if i <= 3 else ''
            # Safely get values with defaults
            name = sub.get('name', 'Unknown')
            f1 = sub.get('f1_score', 0.0)
            acc = sub.get('accuracy', 0.0)
            prec = sub.get('precision', 0.0)
            rec = sub.get('recall', 0.0)
            date = sub.get('date', 'N/A')
            
            html += f'''
                    <tr>
                        <td class="rank"><span class="medal">{medal}</span> #{i}</td>
                        <td><strong>{name}</strong></td>
                        <td class="score">{f1:.4f}</td>
                        <td class="metric">{acc:.4f}</td>
                        <td class="metric">{prec:.4f}</td>
                        <td class="metric">{rec:.4f}</td>
                        <td class="metric">{date}</td>
                    </tr>
'''
    
    html += f'''
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Last Updated: {last_updated}</p>
            <p><a href="https://github.com/AiIkram/gnn-parkinsons-challenge" style="color: white;">View on GitHub</a></p>
        </div>
    </div>
</body>
</html>
'''
    
    return html

def main():
    # Load leaderboard JSON
    json_path = Path('leaderboard.json')
    
    if not json_path.exists():
        print("Creating new leaderboard.json...")
        leaderboard_data = {'submissions': []}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=2)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            leaderboard_data = json.load(f)
    
    # Generate HTML
    html = generate_leaderboard_html(leaderboard_data)
    
    # Save to docs directory
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    
    html_path = docs_dir / 'leaderboard.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Also save JSON to docs
    json_docs_path = docs_dir / 'leaderboard.json'
    with open(json_docs_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard_data, f, indent=2)
    
    print(f"✓ Leaderboard HTML generated: {html_path}")
    print(f"✓ Leaderboard JSON copied: {json_docs_path}")

if __name__ == '__main__':
    main()