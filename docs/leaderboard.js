let leaderboardData = [];
let sortColumn = 'score';
let sortAsc = false;

async function loadLeaderboard() {
    try {
        const response = await fetch('../leaderboard/leaderboard.csv');
        const csvText = await response.text();
        leaderboardData = parseCSV(csvText);
        renderTable();
    } catch (error) {
        console.error('Failed to load leaderboard:', error);
        document.getElementById('leaderboardTable').innerHTML = 
            '<p style="padding: 40px; text-align: center;">Error loading leaderboard data</p>';
    }
}

function parseCSV(csv) {
    const lines = csv.trim().split('\n');
    const headers = lines[0].split(',');
    
    return lines.slice(1).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, i) => {
            obj[header.trim()] = values[i]?.trim() || '';
        });
        return obj;
    });
}

function renderTable() {
    const searchTerm = document.getElementById('searchBox')?.value.toLowerCase() || '';
    const modelFilter = document.getElementById('modelFilter')?.value || '';
    
    let filtered = leaderboardData.filter(row => {
        const matchesSearch = !searchTerm || 
            Object.values(row).some(val => val.toString().toLowerCase().includes(searchTerm));
        const matchesModel = !modelFilter || row.model === modelFilter;
        return matchesSearch && matchesModel;
    });
    
    filtered.sort((a, b) => {
        let aVal = a[sortColumn];
        let bVal = b[sortColumn];
        
        if (sortColumn === 'score') {
            aVal = parseFloat(aVal);
            bVal = parseFloat(bVal);
        }
        
        if (aVal < bVal) return sortAsc ? -1 : 1;
        if (aVal > bVal) return sortAsc ? 1 : -1;
        return 0;
    });
    
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th onclick="sortBy('rank')">Rank</th>
                <th onclick="sortBy('team')">Team</th>
                <th onclick="sortBy('score')">Score ▼</th>
                <th onclick="sortBy('model')">Model Type</th>
                <th onclick="sortBy('date')">Date</th>
            </tr>
        </thead>
        <tbody>
            ${filtered.map((row, idx) => `
                <tr>
                    <td class="rank-${idx + 1}">${getRankIcon(idx + 1)} ${idx + 1}</td>
                    <td>${row.team || 'Unknown'}</td>
                    <td>${parseFloat(row.score).toFixed(4)}</td>
                    <td>${row.model || 'N/A'}</td>
                    <td>${row.date || 'N/A'}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    document.getElementById('leaderboardTable').innerHTML = '';
    document.getElementById('leaderboardTable').appendChild(table);
}

function getRankIcon(rank) {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return '';
}

function sortBy(column) {
    if (sortColumn === column) {
        sortAsc = !sortAsc;
    } else {
        sortColumn = column;
        sortAsc = false;
    }
    renderTable();
}

document.addEventListener('DOMContentLoaded', () => {
    loadLeaderboard();
    
    document.getElementById('searchBox')?.addEventListener('input', renderTable);
    document.getElementById('modelFilter')?.addEventListener('change', renderTable);
});
