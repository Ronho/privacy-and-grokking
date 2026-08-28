import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import math

# Load the data
print("Loading parquet data...")
df = pd.read_parquet('grokking_metrics.parquet')
print(f"Loaded {len(df)} rows.")

# Extract unique runs and parameters
param_cols = [c for c in df.columns if c.startswith('params.')]
runs_df = df[['run_id'] + param_cols].drop_duplicates('run_id')
# Replace NaNs with None for JSON serialization
runs_df = runs_df.where(pd.notnull(runs_df), None)
runs_list = runs_df.to_dict(orient='records')

app = FastAPI()

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grokking Run Viewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-hover: #60a5fa;
            --gap-color: #ef4444;
            --train-color: #10b981;
            --test-color: #3b82f6;
        }

        body, html {
            margin: 0;
            padding: 0;
            height: 100%;
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .navbar {
            padding: 1rem 2rem;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }

        .navbar h1 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .controls {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .run-counter {
            font-size: 0.875rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
        }

        .sidebar {
            width: 320px;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border-right: 1px solid var(--border-color);
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            z-index: 5;
        }
        
        .sidebar::-webkit-scrollbar {
            width: 6px;
        }
        .sidebar::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }

        .run-info-card {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .run-id-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .run-id {
            font-size: 1.125rem;
            font-weight: 600;
            word-break: break-all;
            margin-bottom: 1rem;
            line-height: 1.2;
        }

        .param-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }

        .param-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .param-label {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .param-value {
            font-size: 0.875rem;
            font-weight: 500;
            background: rgba(255,255,255,0.05);
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .chart-container {
            flex: 1;
            padding: 2rem;
            position: relative;
            display: flex;
            flex-direction: column;
        }
        
        .chart-wrapper {
            flex: 1;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            position: relative;
        }

        canvas {
            width: 100% !important;
            height: 100% !important;
        }

        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(15, 23, 42, 0.5);
            backdrop-filter: blur(4px);
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 1rem;
            z-index: 20;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top-color: var(--accent);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .keyboard-hint {
            position: absolute;
            bottom: 2rem;
            right: 2rem;
            background: rgba(0,0,0,0.5);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(8px);
            pointer-events: none;
        }
        
        .key {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 0.1rem 0.4rem;
            border-radius: 0.25rem;
            color: white;
            font-family: monospace;
        }
    </style>
</head>
<body>

    <div class="navbar">
        <h1>Grokking Metrics Explorer</h1>
        <div class="controls">
            <span class="run-counter" id="runCounter">Loading...</span>
            <button class="btn" id="prevBtn" onclick="prevRun()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
                Prev
            </button>
            <button class="btn" id="nextBtn" onclick="nextRun()">
                Next
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
        </div>
    </div>

    <div class="main-container">
        <div class="sidebar">
            <div class="run-info-card">
                <div class="run-id-title">Current Run ID</div>
                <div class="run-id" id="runIdDisplay">-----</div>
            </div>
            
            <div class="run-info-card">
                <div class="run-id-title">Hyperparameters</div>
                <div class="param-grid" id="paramGrid">
                    <!-- Params will be populated here -->
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-wrapper">
                <div class="loading-overlay" id="loadingOverlay">
                    <div class="spinner"></div>
                </div>
                <canvas id="metricsChart"></canvas>
            </div>
            <div class="keyboard-hint">
                Use <span class="key">&larr;</span> and <span class="key">&rarr;</span> to navigate runs
            </div>
        </div>
    </div>

    <script>
        let runs = [];
        let currentIndex = 0;
        let chart = null;

        // Initialize Chart.js
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";
        
        const ctx = document.getElementById('metricsChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Train Accuracy',
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Test Accuracy',
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Generalization Gap',
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.1,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            boxWidth: 8
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#f8fafc',
                        bodyColor: '#f8fafc',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: true,
                        boxPadding: 4
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Step'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 0,
                        max: 1.05,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                }
            }
        });

        async function fetchRuns() {
            try {
                const response = await fetch('/api/runs');
                runs = await response.json();
                if (runs.length > 0) {
                    currentIndex = 0;
                    loadRun(currentIndex);
                }
            } catch (error) {
                console.error("Error fetching runs:", error);
                document.getElementById('runCounter').innerText = "Error loading runs";
            }
        }

        async function loadRun(index) {
            if (index < 0 || index >= runs.length) return;
            
            const run = runs[index];
            
            // Update UI
            document.getElementById('runCounter').innerText = `Run ${index + 1} of ${runs.length}`;
            document.getElementById('runIdDisplay').innerText = run.run_id;
            
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === runs.length - 1;

            // Populate Params
            const paramGrid = document.getElementById('paramGrid');
            paramGrid.innerHTML = '';
            
            // Sort keys to look nice
            const keys = Object.keys(run).filter(k => k.startsWith('params.')).sort();
            keys.forEach(k => {
                const val = run[k];
                if (val !== null && val !== undefined) {
                    const labelName = k.replace('params.', '').replace(/_/g, ' ');
                    const div = document.createElement('div');
                    div.className = 'param-item';
                    div.innerHTML = `
                        <span class="param-label">${labelName}</span>
                        <span class="param-value">${val}</span>
                    `;
                    paramGrid.appendChild(div);
                }
            });

            // Fetch data
            document.getElementById('loadingOverlay').classList.add('active');
            try {
                const response = await fetch(`/api/runs/${run.run_id}`);
                const data = await response.json();
                
                // Assuming data is { step: [...], 'eval/train/accuracy': [...], etc }
                
                chart.data.labels = data.step;
                chart.data.datasets[0].data = data['eval/train/accuracy'] || [];
                chart.data.datasets[1].data = data['eval/test/accuracy'] || [];
                chart.data.datasets[2].data = data['eval/generalization_gap'] || [];
                
                // For log-like steps, sometimes a log scale on x axis is preferred, but linear is fine
                chart.update();
            } catch (error) {
                console.error("Error fetching run data:", error);
            } finally {
                document.getElementById('loadingOverlay').classList.remove('active');
            }
        }

        function prevRun() {
            if (currentIndex > 0) {
                currentIndex--;
                loadRun(currentIndex);
            }
        }

        function nextRun() {
            if (currentIndex < runs.length - 1) {
                currentIndex++;
                loadRun(currentIndex);
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                prevRun();
            } else if (e.key === 'ArrowRight') {
                nextRun();
            }
        });

        // Init
        fetchRuns();
    </script>
</body>
</html>
"""

@app.get("/")
def read_root():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api/runs")
def get_runs():
    return runs_list

@app.get("/api/runs/{run_id}")
def get_run_data(run_id: str):
    # Filter for the run
    run_data = df[df['run_id'] == run_id]
    
    # Drop duplicates in case a step was logged multiple times for the same metric
    run_data = run_data.drop_duplicates(subset=['step', 'metric_name'], keep='last')
    
    # Pivot so we have one row per step, and columns are metrics
    pivot_df = run_data.pivot(index='step', columns='metric_name', values='value').reset_index()
    
    # Fill missing values with None for JSON serialization
    pivot_df = pivot_df.where(pd.notnull(pivot_df), None)
    
    return pivot_df.to_dict(orient='list')

if __name__ == "__main__":
    print("Starting Grokking Run Viewer on http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080)
