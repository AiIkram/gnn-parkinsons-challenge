let leaderboardData = [];
let sortColumn = 'score';
let sortAsc = false;

async function loadLeaderboard() {
    const tableContainer = document.getElementById('leaderboardTable');
    const emptyState = document.getElementById('emptyState');
    
    try {
        tableContainer.innerHTML = '<div class="loading">⏳ Loading...</div>';
        if (emptyState) emptyState.style.display = 'none';
        
        // FIX: Use correct path for GitHub Pages
        // Try multiple paths to ensure it works both locally and on GitHub Pages
        let csvUrl;
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            // Local development
            csvUrl = './leaderboard.csv';
        } else {
            // GitHub Pages - use raw GitHub URL
            csvUrl = './leaderboard.csv';
        }
        
        console.log('Fetching leaderboard from:', csvUrl);
        
        const response = await fetch(csvUrl);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const csvText = await response.text();
        console.log('CSV loaded, length:', csvText.length);
        
        leaderboardData = parseCSV(csvText);
        console.log('Parsed entries:', leaderboardData.length);
        
        if (leaderboardData.length === 0) {
            tableContainer.innerHTML = '<div class="loading" style="color: orange;">⚠️ No submissions yet</div>';
            if (emptyState) emptyState.style.display = 'block';
            updateStats(0, null, null);
        } else {
            renderTable();
            updateStatsBar();
        }
    } catch (error) {
        console.error('Error loading leaderboard:', error);
        tableContainer.innerHTML = `
            <div class="loading" style="color: red;">
                ❌ Error loading data: ${error.message}<br>
                <small>Check browser console for details</small>
            </div>
        `;
    }
}

function parseCSV(csv) {
    csv = csv.replace(/^\uFEFF/, '').trim();
    if (!csv) {
        console.warn('CSV is empty');
        return [];
    }
    
    const lines = csv.split(/\r?\n/).filter(line => line.trim());
    if (lines.length <= 1) {
        console.warn('CSV has no data rows');
        return [];
    }
    
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    console.log('CSV headers:', headers);
    
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        if (values.length === headers.length && values[0]) {
            const obj = {};
            headers.forEach((header, idx) => {
                obj[header] = values[idx] || '';
            });
            // Must have at least team and score
            if (obj.team && obj.score) {
                data.push(obj);
            }
        }
    }
    
    console.log('Parsed data:', data);
    return data;
}

function updateStatsBar() {
    const scores = leaderboardData.map(r => parseFloat(r.score)).filter(s => !isNaN(s));
    const best = scores.length > 0 ? Math.max(...scores) : null;
    const dates = leaderboardData.map(r => r.date).filter(d => d);
    const lastDate = dates.length > 0 ? dates[dates.length - 1] : 'N/A';
    
    updateStats(leaderboardData.length, best, lastDate);
}

function updateStats(total, best, updated) {
    const totalEl = document.getElementById('totalSubmissions');
    const bestEl = document.getElementById('bestScore');
    const updatedEl = document.getElementById('lastUpdated');
    
    if (totalEl) totalEl.textContent = total;
    if (bestEl) bestEl.textContent = best !== null ? best.toFixed(4) : '-';
    if (updatedEl) updatedEl.textContent = updated || '-';
}

function renderTable() {
    const searchTerm = document.getElementById('searchBox')?.value.toLowerCase() || '';
    const modelFilter = document.getElementById('modelFilter')?.value || '';
    
    let filtered = leaderboardData.filter(row => {
        const matchesSearch = !searchTerm || 
            Object.values(row).some(val => val.toString().toLowerCase().includes(searchTerm));
        const matchesModel = !modelFilter || row.model?.toLowerCase().includes(modelFilter.toLowerCase());
        return matchesSearch && matchesModel;
    });
    
    // Sort by score descending by default
    filtered.sort((a, b) => {
        let aVal = sortColumn === 'score' ? parseFloat(a[sortColumn]) || 0 : (a[sortColumn] || '');
        let bVal = sortColumn === 'score' ? parseFloat(b[sortColumn]) || 0 : (b[sortColumn] || '');
        
        if (aVal < bVal) return sortAsc ? -1 : 1;
        if (aVal > bVal) return sortAsc ? 1 : -1;
        return 0;
    });
    
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th onclick="sortBy('rank')" style="cursor: pointer;">Rank ↕️</th>
                <th onclick="sortBy('team')" style="cursor: pointer;">Team ↕️</th>
                <th onclick="sortBy('score')" style="cursor: pointer;">Score (F1) ↕️</th>
                <th onclick="sortBy('model')" style="cursor: pointer;">Model ↕️</th>
                <th onclick="sortBy('date')" style="cursor: pointer;">Date ↕️</th>
                <th>Run ID</th>
            </tr>
        </thead>
        <tbody>
            ${filtered.map((row, idx) => {
                const rank = idx + 1;
                const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '';
                const score = parseFloat(row.score);
                const scoreDisplay = !isNaN(score) ? score.toFixed(4) : 'N/A';
                
                return `
                    <tr>
                        <td class="rank-cell">${medal} ${rank}</td>
                        <td><strong>${escapeHtml(row.team || 'Unknown')}</strong></td>
                        <td class="score-cell">${scoreDisplay}</td>
                        <td>${escapeHtml(row.model || 'N/A')}</td>
                        <td>${escapeHtml(row.date || 'N/A')}</td>
                        <td>${escapeHtml(row.run_id || '-')}</td>
                    </tr>
                `;
            }).join('')}
        </tbody>
    `;
    
    document.getElementById('leaderboardTable').innerHTML = '';
    document.getElementById('leaderboardTable').appendChild(table);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function sortBy(column) {
    if (sortColumn === column) {
        sortAsc = !sortAsc;
    } else {
        sortColumn = column;
        sortAsc = column === 'score' ? false : true; // Score defaults to descending
    }
    renderTable();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded, initializing leaderboard...');
    loadLeaderboard();
    
    document.getElementById('searchBox')?.addEventListener('input', renderTable);
    document.getElementById('modelFilter')?.addEventListener('change', renderTable);
    
    // Auto-refresh every 5 minutes
    setInterval(loadLeaderboard, 5 * 60 * 1000);
});
