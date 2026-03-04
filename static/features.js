// ─── Advanced Features: Quiz Mode, Outline, Glossary ──────────────────────────
// This file is loaded after index.html's main script block.
// It wires up the Quiz, Outline, and Glossary buttons added to the header.

(function () {
    'use strict';

    // ── Quiz Mode ─────────────────────────────────────────────────────────────
    let quizAnswered = 0, quizCorrect = 0;

    const quizBtn = document.getElementById('quizBtn');
    const quizClose = document.getElementById('quizClose');
    const quizNewBtn = document.getElementById('quizNewBtn');
    const quizOverlay = document.getElementById('quizOverlay');

    if (quizBtn) quizBtn.addEventListener('click', openQuiz);
    if (quizClose) quizClose.addEventListener('click', () => quizOverlay.classList.remove('active'));
    if (quizNewBtn) quizNewBtn.addEventListener('click', openQuiz);

    async function openQuiz() {
        const sid = window.currentSessionId;
        if (!sid) return;
        const body = document.getElementById('quizBody');
        const scoreEl = document.getElementById('quizScore');
        quizOverlay.classList.add('active');
        quizAnswered = 0; quizCorrect = 0;
        if (scoreEl) scoreEl.textContent = '';
        body.innerHTML = '<div class="feature-loading"><span class="typing-dots"><span></span><span></span><span></span></span> Generating questions...</div>';
        try {
            const res = await fetch(`/quiz/${sid}?n=6`);
            const data = await res.json();
            const qs = data.questions || [];
            if (!qs.length) {
                body.innerHTML = '<div class="feature-empty">&#9888; Not enough definition-style text to auto-generate a quiz. Try a document with more formal definitions.</div>';
                return;
            }
            body.innerHTML = '';
            qs.forEach((q, qi) => {
                const card = document.createElement('div');
                card.className = 'quiz-card';
                card.innerHTML =
                    `<div class="quiz-question"><span class="quiz-num">Q${qi + 1}.</span> ${q.question.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</div>
                    <div class="quiz-options" id="qopts-${qi}">${q.options.map((o, oi) =>
                        `<button class="quiz-option" data-qi="${qi}" data-oi="${oi}" data-correct="${oi === q.answer_index}">${String.fromCharCode(65 + oi)}. ${o}</button>`
                    ).join('')
                    }</div>
                    <div class="quiz-feedback" id="qfb-${qi}" style="display:none"></div>
                    <div class="quiz-source">&#128196; Source: Page ${q.source_page}</div>`;
                body.appendChild(card);
            });
            body.querySelectorAll('.quiz-option').forEach(btn => {
                btn.addEventListener('click', function () {
                    const qi = parseInt(this.dataset.qi);
                    const correct = this.dataset.correct === 'true';
                    const optsEl = document.getElementById(`qopts-${qi}`);
                    if (optsEl.classList.contains('answered')) return;
                    optsEl.classList.add('answered');
                    optsEl.querySelectorAll('.quiz-option').forEach(b => {
                        b.disabled = true;
                        if (b.dataset.correct === 'true') b.classList.add('correct');
                        else if (b === this && !correct) b.classList.add('wrong');
                    });
                    const fb = document.getElementById(`qfb-${qi}`);
                    fb.style.display = 'block';
                    fb.className = 'quiz-feedback ' + (correct ? 'fb-correct' : 'fb-wrong');
                    fb.textContent = correct ? '\u2713 Correct!' : '\u2717 Incorrect \u2014 the correct answer is highlighted.';
                    quizAnswered++;
                    if (correct) quizCorrect++;
                    if (scoreEl) scoreEl.textContent = `Score: ${quizCorrect}/${quizAnswered}`;
                });
            });
        } catch (e) {
            body.innerHTML = '<div class="feature-empty">Failed to load quiz. Please try again.</div>';
        }
    }

    // ── Document Outline ─────────────────────────────────────────────────────
    const outlineBtn = document.getElementById('outlineBtn');
    const outlineClose = document.getElementById('outlineClose');
    const outlineOverlay = document.getElementById('outlineOverlay');

    if (outlineBtn) outlineBtn.addEventListener('click', openOutline);
    if (outlineClose) outlineClose.addEventListener('click', () => outlineOverlay.classList.remove('active'));

    async function openOutline() {
        const sid = window.currentSessionId;
        if (!sid) return;
        const body = document.getElementById('outlineBody');
        outlineOverlay.classList.add('active');
        body.innerHTML = '<div class="feature-loading"><span class="typing-dots"><span></span><span></span><span></span></span> Building outline...</div>';
        try {
            const res = await fetch(`/outline/${sid}`);
            const data = await res.json();
            const items = data.outline || [];
            if (!items.length) {
                body.innerHTML = '<div class="feature-empty">&#9888; No clear section headers detected in this document.</div>';
                return;
            }
            body.innerHTML = '<div class="outline-list">' +
                items.map(item =>
                    `<div class="outline-item level-${item.level}" data-title="${encodeURIComponent(item.title)}">
                        <span class="outline-title">${item.title}</span>
                        <span class="outline-page">p.${item.page}</span>
                    </div>`
                ).join('') + '</div>';
            body.querySelectorAll('.outline-item').forEach(el => {
                el.addEventListener('click', () => {
                    const title = decodeURIComponent(el.dataset.title);
                    outlineOverlay.classList.remove('active');
                    const userInput = document.getElementById('userInput');
                    if (userInput) {
                        userInput.value = `Explain: ${title}`;
                        userInput.dispatchEvent(new Event('input'));
                        if (typeof window.smartSend === 'function') window.smartSend();
                    }
                });
            });
        } catch (e) {
            body.innerHTML = '<div class="feature-empty">Failed to build outline.</div>';
        }
    }

    // ── Glossary ─────────────────────────────────────────────────────────────
    const glossaryBtn = document.getElementById('glossaryBtn');
    const glossaryClose = document.getElementById('glossaryClose');
    const glossaryOverlay = document.getElementById('glossaryOverlay');

    if (glossaryBtn) glossaryBtn.addEventListener('click', openGlossary);
    if (glossaryClose) glossaryClose.addEventListener('click', () => glossaryOverlay.classList.remove('active'));

    async function openGlossary() {
        const sid = window.currentSessionId;
        if (!sid) return;
        const body = document.getElementById('glossaryBody');
        glossaryOverlay.classList.add('active');
        body.innerHTML = '<div class="feature-loading"><span class="typing-dots"><span></span><span></span><span></span></span> Extracting definitions...</div>';
        try {
            const res = await fetch(`/glossary/${sid}`);
            const data = await res.json();
            const terms = data.glossary || [];
            if (!terms.length) {
                body.innerHTML = '<div class="feature-empty">&#9888; No formal definitions found in this document.</div>';
                return;
            }
            // Group alphabetically
            const groups = {};
            terms.forEach(t => {
                const l = (t.term[0] || '#').toUpperCase();
                if (!groups[l]) groups[l] = [];
                groups[l].push(t);
            });
            let html = '<div class="glossary-search-wrap"><input class="glossary-search" id="glossarySearch" placeholder="&#128269; Search terms..."></div><div class="glossary-list" id="glossaryList">';
            Object.keys(groups).sort().forEach(l => {
                html += `<div class="glossary-letter">${l}</div>`;
                groups[l].forEach(t => {
                    html += `<div class="glossary-item" data-term="${(t.term || '').toLowerCase()}">
                        <div class="glossary-term">${t.term} <span class="glossary-page">p.${t.page}</span></div>
                        <div class="glossary-def">${t.definition}</div>
                    </div>`;
                });
            });
            body.innerHTML = html + '</div>';

            const searchInput = document.getElementById('glossarySearch');
            if (searchInput) {
                searchInput.addEventListener('input', function () {
                    const q = this.value.toLowerCase();
                    document.querySelectorAll('#glossaryList .glossary-item').forEach(el => {
                        el.style.display = (!q || (el.dataset.term || '').includes(q)) ? '' : 'none';
                    });
                });
            }
        } catch (e) {
            body.innerHTML = '<div class="feature-empty">Failed to extract glossary.</div>';
        }
    }

    // Expose currentSessionId to window so this script can access it
    // (set by the main script block inside index.html)
    // The main script sets `let currentSessionId = null;`
    // We patch loadSession to also update window.currentSessionId
    const _origLoad = window.loadSession;
    if (typeof _origLoad === 'function') {
        window.loadSession = async function (id, title) {
            window.currentSessionId = id;
            return _origLoad(id, title);
        };
    }
})();
