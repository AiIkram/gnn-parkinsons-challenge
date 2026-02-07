# competition/render_leaderboard.py
import pandas as pd
from datetime import datetime

def render_leaderboard():
    # Read CSV
    df = pd.read_csv('docs/leaderboard.csv')
    
    # Sort by score descending
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Generate markdown
    with open('leaderboard/leaderboard.md', 'w') as f:
        f.write('# 🏆 Leaderboard\n\n')
        f.write(f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('| Rank | Team | Score | Model | Date |\n')
        f.write('|------|------|-------|-------|------|\n')
        
        for _, row in df.iterrows():
            medal = '🥇' if row['rank'] == 1 else '🥈' if row['rank'] == 2 else '🥉' if row['rank'] == 3 else ''
            f.write(f"| {medal} {row['rank']} | {row['team']} | {row['score']:.4f} | {row['model']} | {row['date']} |\n")

if __name__ == '__main__':
    render_leaderboard()
