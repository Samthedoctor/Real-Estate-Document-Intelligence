// ==================== DOM ELEMENTS ====================
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const answerBox = document.getElementById('answerBox');
const answerText = document.getElementById('answerText');
const metricsContainer = document.getElementById('metricsContainer');
const resultsContainer = document.getElementById('resultsContainer');
const resultsList = document.getElementById('resultsList');

// Metrics
const searchLatency = document.getElementById('searchLatency');
const llmLatency = document.getElementById('llmLatency');
const serverLatency = document.getElementById('serverLatency');
const statusIndicator = document.getElementById('statusIndicator');

// Stats
const totalChunks = document.getElementById('totalChunks');
const totalDocs = document.getElementById('totalDocs');

// Example buttons
const exampleBtns = document.querySelectorAll('.example-btn');

// ==================== STATE ====================
let isSearching = false;

// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', () => {
    // Load system stats
    loadSystemStats();
    
    // Setup event listeners
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
    
    // Example button clicks
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            searchInput.value = btn.dataset.query;
            performSearch();
        });
    });
});

// ==================== LOAD SYSTEM STATS ====================
async function loadSystemStats() {
    try {
        const response = await fetch('/stats');
        if (response.ok) {
            const data = await response.json();
            totalChunks.textContent = data.total_chunks || '-';
            totalDocs.textContent = data.unique_documents || '-';
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ==================== PERFORM SEARCH ====================
async function performSearch() {
    const query = searchInput.value.trim();
    
    if (!query || isSearching) return;
    
    isSearching = true;
    
    // Reset UI
    hideAll();
    loadingIndicator.classList.remove('hidden');
    searchBtn.disabled = true;
    searchBtn.textContent = '⏳ Searching...';
    
    try {
        const startTime = performance.now();
        
        // Call streaming endpoint
        const response = await fetch('/query/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: query,
                k: 5,
                include_metadata: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Process SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let buffer = '';
        let resultsData = [];
        let searchTime = 0;
        let llmTime = 0;
        let serverTime = 0;
        let llmAnswer = '';
        
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, {stream: true});
            
            // Process complete SSE messages
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'results') {
                            // Got search results
                            resultsData = data.results;
                            searchTime = data.search_latency;
                            llmTime = data.llm_latency || 0;
                            serverTime = data.total_server_latency || (searchTime + llmTime);
                            
                            // Hide loading, show metrics
                            loadingIndicator.classList.add('hidden');
                            showMetrics(searchTime, llmTime, serverTime);
                            
                            // Show answer box (empty, will fill with streaming)
                            answerBox.classList.remove('hidden');
                            answerText.textContent = '';
                            answerText.classList.add('streaming');
                            
                        } else if (data.type === 'token') {
                            // Stream LLM answer token
                            llmAnswer += data.content;
                            answerText.textContent = llmAnswer;
                            
                        } else if (data.type === 'done') {
                            // Final answer
                            llmAnswer = data.answer || llmAnswer;
                            answerText.textContent = llmAnswer;
                            answerText.classList.remove('streaming');
                            
                            // Show results
                            showResults(resultsData);
                            
                            // Update final metrics
                            const clientTime = performance.now() - startTime;
                            console.log(`Client total: ${clientTime.toFixed(0)}ms`);
                            
                        } else if (data.type === 'error') {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Search error:', error);
        alert(`Error: ${error.message}`);
        hideAll();
    } finally {
        isSearching = false;
        searchBtn.disabled = false;
        searchBtn.textContent = '🚀 Search';
    }
}

// ==================== SHOW METRICS ====================
function showMetrics(search, llm, server) {
    metricsContainer.classList.remove('hidden');
    
    searchLatency.textContent = `${Math.round(search)}ms`;
    llmLatency.textContent = llm > 0 ? `${Math.round(llm)}ms` : 'N/A';
    serverLatency.textContent = `${Math.round(server)}ms`;
    
    // Status indicator
    if (server < 2000) {
        statusIndicator.textContent = '✅ <2s';
        statusIndicator.style.color = '#2e7d32';
    } else {
        statusIndicator.textContent = '⚠️ >2s';
        statusIndicator.style.color = '#d32f2f';
    }
}

// ==================== SHOW RESULTS ====================
function showResults(results) {
    resultsContainer.classList.remove('hidden');
    resultsList.innerHTML = '';
    
    // Show top 3 sources
    const topResults = results.slice(0, 3);
    
    topResults.forEach((result, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-item';
        resultDiv.style.animationDelay = `${index * 0.1}s`;
        
        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-title">📄 Source ${result.rank} - Page ${result.page}</div>
                <div class="result-score">Score: ${result.score.toFixed(4)}</div>
            </div>
            <div class="result-content">${escapeHtml(result.content)}</div>
            <div class="result-meta">
                <span class="meta-badge">📄 ${escapeHtml(result.source)}</span>
                <span class="meta-badge">📖 Page ${result.page}</span>
            </div>
        `;
        
        resultsList.appendChild(resultDiv);
    });
}

// ==================== UTILITY ====================
function hideAll() {
    loadingIndicator.classList.add('hidden');
    answerBox.classList.add('hidden');
    metricsContainer.classList.add('hidden');
    resultsContainer.classList.add('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}