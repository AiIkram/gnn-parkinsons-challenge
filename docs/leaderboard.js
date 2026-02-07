// Leaderboard data and state
let leaderboardData = [];
let sortColumn = 'score';
let sortAsc = false;

// Load leaderboard data from CSV
async function loadLeaderboard() {
    const tableContainer = document.getElementById('leaderboardTable');
    const emptyState = document.getElementById('emptyState');
    
    try {
        // Show loading state
        tableContainer.innerHTML = '<div class="loading">⏳ Loading leaderboard data...</div>';
        emptyState.style.display = 'none';
        
        // Fetch the CSV file from the parent directory
        const response = await fetch('../leaderboard/leaderboard.csv');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const csvText = await response.text();
        
        // Parse CSV
        leaderboardData = parseCSV(csvText);
        
        if (leaderboardData.length === 0) {
            tableContainer.innerHTML = '';
            emptyState.style.display = 'block';
            updateStats(0, null, null);
        } else {
            renderTable();
            updateStatsBar();
        }
        
    } catch (error) {
        console.error('Failed to load leaderboard:', error);
        tableContainer.innerHTML = `
            <div class="loading" style="color: #dc3545;">
                ❌ Error loading leaderboard data<br>
                <small>Make sure leaderboard/leaderboard.csv exists in your repository</small>
            </div>
        `;
        emptyState.style.display = 'none';
        updateStats(0, null, null);
    }
}

// Parse CSV text into array of objects
function parseCSV(csv) {
    // Remove any BOM and trim
    csv = csv.replace(/^\uFEFF/, '').trim();
    
    if (!csv) {
        return [];
    }
    
    // Split by newlines, handling different line endings
    const lines = csv.split(/\r?\n/);
    
    if (lines.length <= 1) {
        return [];
    }
    
    // Parse header
    const headers = parseCSVLine(lines[0]);
    
    // Parse data rows
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // Skip empty lines
        if (!line) {
            continue;
        }
        
        const values = parseCSVLine(line);
        
        // Only add if we have values
        if (values.length > 0 && values[0]) {
            const obj = {};
            headers.forEach((header, idx) => {
                obj[header] = values[idx] || '';
            });
            
            // Only add rows that have at least a team or score
            if (obj.team || obj.score) {
                data.push(obj);
            }
        }
    }
    
    return data;
}

// Parse a single CSV line, handling quoted fields
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add the last field
    result.push(current.trim());
    
    return result;
}

// Update statistics bar
function updateStatsBar() {
    if (leaderboardData.length === 0) {
        updateStats(0, null, null);
        return;
    }
    
    const scores = leaderboardData.map(row => parseFloat(row.score)).filter(s => !isNaN(s));
    const bestScore = scores.length > 0 ? Math.max(...scores) : null;
    
    // Get most recent date
    const dates = leaderboardData.map(row => row.date).filter(d => d);
    const lastUpdated = dates.length > 0 ? dates[dates.length - 1] : 'N/A';
    
    updateStats(leaderboardData.length, bestScore, lastUpdated);
}

// Update stats display
function updateStats(total, best, updated) {
    document.getElementById('totalSubmissions').textContent = total;
    document.getElementById('bestScore').textContent = best !== null ? best.toFixed(4) : '-';
    document.getElementById('lastUpdated').textContent = updated || '-';
}

// Render the leaderboard table
function renderTable() {
    const searchTerm = document.getElementById('searchBox')?.value.toLowerCase() || '';
    const modelFilter = document.getElementById('modelFilter')?.value || '';
    
    // Filter data
    let filtered = leaderboardData.filter(row => {
        const matchesSearch = !searchTerm || 
            Object.values(row).some(val => 
                val.toString().toLowerCase().includes(searchTerm)
            );
        
        const matchesModel = !modelFilter || 
            row.model?.toLowerCase().includes(modelFilter.toLowerCase());
        
        return matchesSearch && matchesModel;
    });
    
    // Sort data
    filtered.sort((a, b) => {
        let aVal = a[sortColumn];
        let bVal = b[sortColumn];
        
        // Parse numeric values
        if (sortColumn === 'score') {
            aVal = parseFloat(aVal) || 0;
            bVal = parseFloat(bVal) || 0;
        }
        
        if (aVal < bVal) return sortAsc ? -1 : 1;
        if (aVal > bVal) return sortAsc ? 1 : -1;
        return 0;
    });
    
    // Build table HTML
    const table = document.createElement('table');
    
    // Table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th onclick="sortBy('rank')" class="${sortColumn === 'rank' ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : ''}">
                Rank
            </th>
            <th onclick="sortBy('team')" class="${sortColumn === 'team' ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : ''}">
                Team
            </th>
            <th onclick="sortBy('score')" class="${sortColumn === 'score' ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : ''}">
                Score (F1)
            </th>
            <th onclick="sortBy('model')" class="${sortColumn === 'model' ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : ''}">
                Model
            </th>
            <th onclick="sortBy('date')" class="${sortColumn === 'date' ? (sortAsc ? 'sorted-asc' : 'sorted-desc') : ''}">
                Date
            </th>
            <th>Notes</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Table body
    const tbody = document.createElement('tbody');
    
    filtered.forEach((row, idx) => {
        const rank = idx + 1;
        const medal = getMedal(rank);
        const rankClass = `rank-${rank}`;
        
        const score = parseFloat(row.score);
        const scoreDisplay = !isNaN(score) ? score.toFixed(4) : 'N/A';
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="rank-cell ${rank <= 3 ? rankClass : ''}">
                ${medal} ${rank}
            </td>
            <td><strong>${escapeHtml(row.team || 'Unknown')}</strong></td>
            <td class="score-cell">${scoreDisplay}</td>
            <td>${escapeHtml(row.model || 'N/A')}</td>
            <td>${escapeHtml(row.date || 'N/A')}</td>
            <td>${escapeHtml(row.notes || '-')}</td>
        `;
        tbody.appendChild(tr);
    });
    
    table.appendChild(tbody);
    
    // Update DOM
    const container = document.getElementById('leaderboardTable');
    container.innerHTML = '';
    container.appendChild(table);
}

// Get medal emoji for rank
function getMedal(rank) {
    if (rank === 1) return '<span class="medal">🥇</span>';
    if (rank === 2) return '<span class="medal">🥈</span>';
    if (rank === 3) return '<span class="medal">🥉</span>';
    return '';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Sort table by column
function sortBy(column) {
    if (sortColumn === column) {
        sortAsc = !sortAsc;
    } else {
        sortColumn = column;
        sortAsc = column === 'score' ? false : true; // Default: score descending, others ascending
    }
    renderTable();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadLeaderboard();
    
    // Set up event listeners
    document.getElementById('searchBox')?.addEventListener('input', renderTable);
    document.getElementById('modelFilter')?.addEventListener('change', renderTable);
    
    // Auto-refresh every 5 minutes
    setInterval(loadLeaderboard, 5 * 60 * 1000);
});
