/**
 * Medical RAG — Chat UI Application
 * Connects to FastAPI backend at /api/chat
 */

(() => {
    'use strict';

    // ── Config ──────────────────────────────────────────
    const API_URL = '/api/chat';
    const HEALTH_URL = '/health';

    // ── DOM References ──────────────────────────────────
    const chatArea = document.getElementById('chatArea');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const messagesContainer = document.getElementById('messagesContainer');
    const queryInput = document.getElementById('queryInput');
    const sendBtn = document.getElementById('sendBtn');
    const topNSelect = document.getElementById('topNSelect');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const newChatBtn = document.getElementById('newChatBtn');

    // ── State ───────────────────────────────────────────
    let isLoading = false;
    let lastQuery = '';

    // ── Initialize ──────────────────────────────────────
    function init() {
        bindEvents();
        checkHealth();
        queryInput.focus();
    }

    // ── Events ──────────────────────────────────────────
    function bindEvents() {
        // Send message
        sendBtn.addEventListener('click', handleSend);
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });

        // Auto-resize textarea
        queryInput.addEventListener('input', () => {
            autoResize(queryInput);
            sendBtn.disabled = !queryInput.value.trim();
        });

        // Sidebar toggle
        sidebarToggle.addEventListener('click', toggleSidebar);
        mobileMenuBtn.addEventListener('click', openMobileSidebar);

        // New chat
        newChatBtn.addEventListener('click', clearChat);

        // Suggestion clicks (sidebar)
        document.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', () => {
                const query = item.getAttribute('data-query');
                if (query) sendQuery(query);
            });
        });

        // Welcome card clicks
        document.querySelectorAll('.welcome-card').forEach(card => {
            card.addEventListener('click', () => {
                const query = card.getAttribute('data-query');
                if (query) sendQuery(query);
            });
        });

        // Click outside sidebar (mobile)
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 &&
                sidebar.classList.contains('mobile-open') &&
                !sidebar.contains(e.target) &&
                !mobileMenuBtn.contains(e.target)) {
                closeMobileSidebar();
            }
        });
    }

    // ── Chat Logic ──────────────────────────────────────
    function handleSend() {
        const query = queryInput.value.trim();
        if (!query || isLoading) return;
        sendQuery(query);
    }

    function sendQuery(query) {
        if (isLoading) return;

        // Hide welcome screen
        welcomeScreen.classList.add('hidden');

        // Set input
        queryInput.value = '';
        autoResize(queryInput);
        sendBtn.disabled = true;

        // Display user message
        appendUserMessage(query);

        // Show loading
        const thinkingEl = appendThinking();

        // Update status
        setStatus('loading', 'Thinking...');
        isLoading = true;
        lastQuery = query;

        // Close mobile sidebar
        if (window.innerWidth <= 768) {
            closeMobileSidebar();
        }

        // Call API
        const topN = parseInt(topNSelect.value, 10);
        fetchAnswer(query, topN)
            .then(data => {
                thinkingEl.remove();
                appendAIMessage(data.answer, data.sources);
                setStatus('ready', 'Ready');
            })
            .catch(err => {
                thinkingEl.remove();
                appendErrorMessage(err.message, query);
                setStatus('error', 'Error');
            })
            .finally(() => {
                isLoading = false;
                queryInput.focus();
            });
    }

    async function fetchAnswer(query, topN) {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_n: topN })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Server error (${response.status})`);
        }

        return response.json();
    }

    // ── Message Rendering ───────────────────────────────
    function appendUserMessage(text) {
        const html = `
            <div class="message user">
                <div class="message-avatar">You</div>
                <div class="message-content">
                    <div class="message-bubble">${escapeHTML(text)}</div>
                    <div class="message-time">${formatTime()}</div>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', html);
        scrollToBottom();
    }

    function appendAIMessage(answer, sources) {
        const sourcesHTML = renderSources(sources);
        const formattedAnswer = formatAnswer(answer);

        const html = `
            <div class="message ai">
                <div class="message-avatar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="currentColor" opacity="0.3"/>
                        <path d="M12 6v6l4 2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="message-content">
                    <div class="message-bubble">${formattedAnswer}</div>
                    ${sourcesHTML}
                    <div class="message-time">${formatTime()}</div>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', html);

        // Bind source toggle
        const lastMessage = messagesContainer.lastElementChild;
        const toggle = lastMessage.querySelector('.sources-toggle');
        const list = lastMessage.querySelector('.sources-list');
        if (toggle && list) {
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('expanded');
                list.classList.toggle('visible');
            });
        }

        scrollToBottom();
    }

    function appendThinking() {
        const html = `
            <div class="message ai thinking-message">
                <div class="message-avatar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="currentColor" opacity="0.3"/>
                        <path d="M12 6v6l4 2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="thinking-indicator">
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', html);
        scrollToBottom();
        return messagesContainer.lastElementChild;
    }

    function appendErrorMessage(errorText, query) {
        const html = `
            <div class="message ai">
                <div class="message-avatar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="currentColor" opacity="0.3"/>
                        <path d="M12 6v6l4 2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="message-content">
                    <div class="error-message">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
                            <path d="M8 5v4M8 11v.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                        </svg>
                        <div>
                            <div>${escapeHTML(errorText)}</div>
                            <button class="error-retry-btn" onclick="window.__retryQuery('${escapeHTML(query).replace(/'/g, "\\'")}')">
                                ↻ Retry
                            </button>
                        </div>
                    </div>
                    <div class="message-time">${formatTime()}</div>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', html);
        scrollToBottom();
    }

    // Expose retry globally
    window.__retryQuery = (query) => {
        if (!isLoading) sendQuery(query);
    };

    // ── Sources Rendering ───────────────────────────────
    function renderSources(sources) {
        if (!sources || sources.length === 0) return '';

        const sourceCards = sources.map((src, i) => {
            const condition = src.metadata?.condition || 'Unknown';
            const section = src.metadata?.section || '';
            const title = section ? `${condition} — ${section}` : condition;
            const distance = src.dense_distance !== null && src.dense_distance !== undefined
                ? `Score: ${(1 - src.dense_distance).toFixed(4)}`
                : '';
            const preview = src.content
                ? src.content.substring(0, 180) + (src.content.length > 180 ? '...' : '')
                : '';

            return `
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-title">[${i + 1}] ${escapeHTML(title)}</span>
                        ${distance ? `<span class="source-score">${distance}</span>` : ''}
                    </div>
                    <div class="source-preview">${escapeHTML(preview)}</div>
                </div>
            `;
        }).join('');

        return `
            <div class="sources-panel">
                <button class="sources-toggle">
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                        <path d="M3 4.5l3 3 3-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    ${sources.length} source${sources.length > 1 ? 's' : ''} retrieved
                </button>
                <div class="sources-list">
                    ${sourceCards}
                </div>
            </div>
        `;
    }

    // ── Format Answer ───────────────────────────────────
    function formatAnswer(text) {
        if (!text) return '';

        // Convert markdown-like formatting
        let formatted = escapeHTML(text);

        // Bold: **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic: *text*
        formatted = formatted.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        // Bullet points: - item or * item
        formatted = formatted.replace(
            /(?:^|<br>)\s*[-•]\s+(.*?)(?=<br>|$)/g,
            '<br>• $1'
        );

        // Numbered lists: 1. item
        formatted = formatted.replace(
            /(?:^|<br>)\s*(\d+)\.\s+(.*?)(?=<br>|$)/g,
            '<br>$1. $2'
        );

        return formatted;
    }

    // ── Sidebar ─────────────────────────────────────────
    function toggleSidebar() {
        sidebar.classList.toggle('collapsed');
    }

    function openMobileSidebar() {
        sidebar.classList.add('mobile-open');
        // Create overlay
        let overlay = document.querySelector('.sidebar-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'sidebar-overlay';
            document.body.appendChild(overlay);
            overlay.addEventListener('click', closeMobileSidebar);
        }
        overlay.classList.add('visible');
    }

    function closeMobileSidebar() {
        sidebar.classList.remove('mobile-open');
        const overlay = document.querySelector('.sidebar-overlay');
        if (overlay) overlay.classList.remove('visible');
    }

    // ── Clear Chat ──────────────────────────────────────
    function clearChat() {
        messagesContainer.innerHTML = '';
        welcomeScreen.classList.remove('hidden');
        queryInput.value = '';
        autoResize(queryInput);
        sendBtn.disabled = true;
        setStatus('ready', 'Ready');
        queryInput.focus();

        if (window.innerWidth <= 768) {
            closeMobileSidebar();
        }
    }

    // ── Health Check ────────────────────────────────────
    async function checkHealth() {
        try {
            const res = await fetch(HEALTH_URL);
            if (res.ok) {
                setStatus('ready', 'Ready');
            } else {
                setStatus('error', 'API unavailable');
            }
        } catch {
            setStatus('error', 'Cannot connect');
        }
    }

    // ── Utilities ───────────────────────────────────────
    function setStatus(state, text) {
        statusDot.className = 'status-dot';
        if (state === 'loading') statusDot.classList.add('loading');
        if (state === 'error') statusDot.classList.add('error');
        statusText.textContent = text;
    }

    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatArea.scrollTo({
                top: chatArea.scrollHeight,
                behavior: 'smooth'
            });
        });
    }

    function escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatTime() {
        return new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    }

    // ── Boot ────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();
