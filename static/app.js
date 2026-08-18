let currentTransactionId = null;

// Search triggers
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        investigateTransaction();
    }
}

function setAndSearch(id) {
    document.getElementById('transactionId').value = id;
    investigateTransaction();
}

async function investigateTransaction() {
    const txIdInput = document.getElementById('transactionId').value.trim();
    if (!txIdInput) return;

    const txId = parseInt(txIdInput, 10);
    if (isNaN(txId)) {
        alert("Please enter a valid numeric Transaction ID");
        return;
    }

    currentTransactionId = txId;
    
    // UI State
    document.getElementById('searchBtn').querySelector('.btn-text').classList.add('hidden');
    document.getElementById('searchBtn').querySelector('.loader').classList.remove('hidden');
    document.getElementById('dashboardContent').classList.add('hidden');
    document.getElementById('welcomeState').classList.add('hidden');
    
    // Reset feedback
    document.getElementById('feedbackForm').reset();
    document.getElementById('feedbackStatus').textContent = "";
    document.getElementById('feedbackStatus').className = "feedback-status";

    try {
        const response = await fetch('/api/v1/investigate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ transaction_id: txId })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Investigation failed");
        }

        renderDashboard(data);
        document.getElementById('dashboardContent').classList.remove('hidden');
    } catch (error) {
        alert(`Error: ${error.message}`);
        document.getElementById('welcomeState').classList.remove('hidden');
    } finally {
        document.getElementById('searchBtn').querySelector('.btn-text').classList.remove('hidden');
        document.getElementById('searchBtn').querySelector('.loader').classList.add('hidden');
    }
}

function renderDashboard(data) {
    // 1. Final Decision
    const decisionCard = document.getElementById('decisionCard');
    const decisionValue = document.getElementById('finalDecisionValue');
    
    // Clear previous classes
    decisionCard.className = 'glass-panel card decision-card';
    decisionValue.className = 'decision-value';
    
    decisionValue.textContent = data.final_decision;
    const decisionBadge = document.getElementById('decisionBadge');

    if (data.final_decision === 'APPROVE') {
        decisionCard.classList.add('bg-approve');
        decisionValue.classList.add('status-approve');
        decisionBadge.textContent = "Low Risk";
    } else if (data.final_decision === 'REVIEW') {
        decisionCard.classList.add('bg-review');
        decisionValue.classList.add('status-review');
        decisionBadge.textContent = "Medium Risk";
    } else {
        decisionCard.classList.add('bg-decline');
        decisionValue.classList.add('status-decline');
        decisionBadge.textContent = "High Risk";
    }

    // 2. Risk Scores
    renderGauge('model', data.risk.model_score, data.risk.model_level);
    renderGauge('behavior', data.behavioral_risk.score, data.behavioral_risk.level);

    // 3. Cold Start
    renderColdStart(data.cold_start);

    // 4. Triggered Rules
    renderRules(data.behavioral_risk.rules_triggered);

    // 5. AI Report
    const reportContent = document.getElementById('reportContent');
    // Using marked.js to render markdown
    reportContent.innerHTML = marked.parse(data.report || "*No report generated.*");
}

function renderGauge(type, score, level) {
    const gaugeFill = document.getElementById(`${type}GaugeFill`);
    const scoreValue = document.getElementById(`${type}ScoreValue`);
    const levelIndicator = document.getElementById(`${type}LevelIndicator`);

    // Score is 0 to 1, turn is 0.5 (half circle)
    const rotation = score * 0.5;
    gaugeFill.style.transform = `rotate(${rotation}turn)`;
    
    scoreValue.textContent = score.toFixed(4);
    levelIndicator.textContent = level;

    // Colors
    let colorClass = 'status-approve';
    let bgColor = 'var(--risk-low)';
    
    if (level === 'HIGH' || score >= 0.9) {
        colorClass = 'status-decline';
        bgColor = 'var(--risk-high)';
    } else if (level === 'MEDIUM' || score >= 0.7) {
        colorClass = 'status-review';
        bgColor = 'var(--risk-medium)';
    }

    levelIndicator.className = `level-indicator ${colorClass} bg-${colorClass.split('-')[1]}`;
    gaugeFill.style.backgroundColor = bgColor;
}

function renderColdStart(coldStart) {
    const list = document.getElementById('coldStartList');
    list.innerHTML = '';
    
    const signals = [
        { label: 'New Card', value: coldStart.is_new_card },
        { label: 'New Device', value: coldStart.is_new_device },
        { label: 'Card History Available', value: coldStart.card_history_available },
        { label: 'Device History Available', value: coldStart.device_history_available }
    ];

    signals.forEach(s => {
        const li = document.createElement('li');
        li.className = 'signal-item';
        
        const icon = document.createElement('span');
        icon.className = 'signal-icon';
        // Check mark or X depending on value (boolean logic depends on the specific field context)
        if (s.label.includes('New')) {
            icon.innerHTML = s.value ? '⚠️' : '✅';
        } else {
            icon.innerHTML = s.value ? '✅' : '❌';
        }

        const text = document.createElement('span');
        text.textContent = s.label;

        li.appendChild(icon);
        li.appendChild(text);
        list.appendChild(li);
    });
}

function renderRules(rules) {
    const container = document.getElementById('rulesContainer');
    container.innerHTML = '';

    if (!rules || rules.length === 0) {
        container.innerHTML = '<div class="rule-reason">No rules triggered.</div>';
        return;
    }

    rules.forEach(rule => {
        const tag = document.createElement('div');
        tag.className = `rule-tag ${rule.severity.toLowerCase()}`;
        
        const title = document.createElement('div');
        title.className = 'rule-title';
        title.textContent = `${rule.rule_id} [${rule.severity}]`;
        
        const reason = document.createElement('div');
        reason.className = 'rule-reason';
        reason.textContent = rule.reason;

        tag.appendChild(title);
        tag.appendChild(reason);
        container.appendChild(tag);
    });
}

async function submitFeedback(event) {
    event.preventDefault();
    if (!currentTransactionId) return;

    const label = document.getElementById('feedbackLabel').value;
    const comment = document.getElementById('feedbackComment').value;
    const statusDiv = document.getElementById('feedbackStatus');

    const modelScore = parseFloat(document.getElementById('modelScoreValue').textContent);
    const behaviorScore = parseFloat(document.getElementById('behaviorScoreValue').textContent);
    const finalDecision = document.getElementById('finalDecisionValue').textContent;

    try {
        const response = await fetch(`/api/v1/investigations/${currentTransactionId}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                label: label,
                analyst_comment: comment,
                model_score: modelScore,
                behavioral_score: behaviorScore,
                final_decision: finalDecision,
                model_version: "1.0.0"
            })
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Failed to submit feedback");
        }

        statusDiv.textContent = "Feedback recorded successfully!";
        statusDiv.className = "feedback-status success-text";
        document.getElementById('feedbackForm').reset();
    } catch (error) {
        statusDiv.textContent = error.message;
        statusDiv.className = "feedback-status error-text";
    }
}
