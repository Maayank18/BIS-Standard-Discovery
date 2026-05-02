import React, { useState, useEffect, useRef, useCallback } from 'react';
import { queryStandards, getHealth, getExamples } from './api';
import './App.css';

/* ─── Tiny icon components ─────────────────────────────────────────────── */
const SearchIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
);
const BoltIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
  </svg>
);
const ClockIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
  </svg>
);
const ChevronDown = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m6 9 6 6 6-6"/>
  </svg>
);
const ShieldIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const LayersIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>
    <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>
  </svg>
);
const SparkleIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17 5.8 21.3l2.4-7.4L2 9.4h7.6z"/>
  </svg>
);

/* ─── Category badge colours ───────────────────────────────────────────── */
const CATEGORY_COLORS = {
  'Cement and Concrete': '#f5a814',
  'Structural Steels': '#3b82f6',
  'Concrete Reinforcement': '#8b5cf6',
  'Stones': '#6b7280',
  'Timber': '#10b981',
  'Glass': '#06b6d4',
  'Bitumen and Tar Products': '#374151',
  'Thermal Insulation Materials': '#f97316',
  'Plastics': '#ec4899',
  'General': '#6b7280',
};

const getCategoryColor = cat => CATEGORY_COLORS[cat] || '#6b7280';

/* ─── Standard Result Card ─────────────────────────────────────────────── */
function StandardCard({ std, index, visible }) {
  const [expanded, setExpanded] = useState(false);
  const catColor = getCategoryColor(std.category);

  return (
    <div
      className={`standard-card ${visible ? 'visible' : ''}`}
      style={{ '--delay': `${index * 80}ms`, '--cat-color': catColor }}
    >
      <div className="card-rank">
        <span className="rank-num">#{index + 1}</span>
        <div className="rank-bar" style={{ height: `${Math.max(20, 100 - index * 18)}%` }} />
      </div>

      <div className="card-body">
        <div className="card-header">
          <div className="is-number-row">
            <span className="is-badge">
              <ShieldIcon />
              {std.is_number_full}
            </span>
            <span className="category-chip" style={{ background: `${catColor}18`, color: catColor, borderColor: `${catColor}40` }}>
              {std.category}
            </span>
          </div>
          <h3 className="std-title">{std.title || 'BIS Standard'}</h3>
        </div>

        <div className={`rationale-block ${expanded ? 'expanded' : ''}`}>
          <p className="rationale-text">{std.rationale}</p>
        </div>

        <div className="card-footer">
          <div className="score-pill">
            <SparkleIcon />
            <span>Score: {(std.score * 100).toFixed(1)}</span>
          </div>
          <span className="std-year">Year: {std.year}</span>
          {std.rationale && std.rationale.length > 120 && (
            <button className="expand-btn" onClick={() => setExpanded(e => !e)}>
              {expanded ? 'Less' : 'More'}
              <span style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0)', display: 'inline-block', transition: '0.2s' }}>
                <ChevronDown />
              </span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── Example pill ─────────────────────────────────────────────────────── */
function ExamplePill({ text, onSelect }) {
  const short = text.length > 72 ? text.slice(0, 70) + '…' : text;
  return (
    <button className="example-pill" onClick={() => onSelect(text)} title={text}>
      {short}
    </button>
  );
}

/* ─── Metrics Bar ──────────────────────────────────────────────────────── */
function MetricsBar({ result }) {
  return (
    <div className="metrics-bar">
      <div className="metric">
        <ClockIcon />
        <span>{result.latency_seconds.toFixed(3)}s</span>
        <label>Latency</label>
      </div>
      <div className="metric-divider" />
      <div className="metric">
        <LayersIcon />
        <span>{result.total_found}</span>
        <label>Standards found</label>
      </div>
      <div className="metric-divider" />
      <div className="metric">
        <BoltIcon />
        <span>Hybrid RAG</span>
        <label>FAISS + BM25 + CE</label>
      </div>
    </div>
  );
}

/* ─── Architecture Diagram (static) ───────────────────────────────────── */
function ArchDiagram() {
  const steps = [
    { icon: '🔍', label: 'Query', sub: 'Product description' },
    { icon: '⚡', label: 'BGE Embed', sub: 'bge-base-en-v1.5' },
    { icon: '🧮', label: 'FAISS + BM25', sub: 'RRF Fusion' },
    { icon: '🎯', label: 'Cross-Encoder', sub: 'ms-marco rerank' },
    { icon: '🤖', label: 'LLM Rationale', sub: 'Claude / GPT' },
    { icon: '📋', label: 'Results', sub: 'Top 5 standards' },
  ];
  return (
    <div className="arch-diagram">
      {steps.map((s, i) => (
        <React.Fragment key={s.label}>
          <div className="arch-step">
            <div className="arch-icon">{s.icon}</div>
            <div className="arch-label">{s.label}</div>
            <div className="arch-sub">{s.sub}</div>
          </div>
          {i < steps.length - 1 && <div className="arch-arrow">→</div>}
        </React.Fragment>
      ))}
    </div>
  );
}

/* ─── Main App ─────────────────────────────────────────────────────────── */
export default function App() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [examples, setExamples] = useState([]);
  const [health, setHealth] = useState(null);
  const [cardsVisible, setCardsVisible] = useState(false);
  const textareaRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getExamples().then(d => setExamples(d.examples || [])).catch(() => {});
  }, []);

  const handleQuery = useCallback(async (queryText) => {
    const q = (queryText || query).trim();
    if (!q || q.length < 10) {
      setError('Please provide a more detailed product description (at least 10 characters).');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setCardsVisible(false);

    try {
      const data = await queryStandards(q, 5, true);
      setResult(data);
      setTimeout(() => setCardsVisible(true), 50);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to connect to the API.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleQuery();
  };

  const selectExample = (text) => {
    setQuery(text);
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  return (
    <div className="app">
      {/* ── Background grid ── */}
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-glow" aria-hidden="true" />

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-group">
            <div className="logo-mark">BIS</div>
            <div className="logo-text">
              <span className="logo-title">Standards Engine</span>
              <span className="logo-sub">AI-Powered BIS Compliance</span>
            </div>
          </div>
          <div className="header-right">
            {health && (
              <div className={`status-pill ${health.status === 'healthy' ? 'healthy' : 'degraded'}`}>
                <span className="status-dot" />
                {health.status === 'healthy'
                  ? `${health.total_standards?.toLocaleString()} vectors indexed`
                  : 'Index not loaded'}
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Hero ── */}
        <section className="hero">
          <div className="hero-eyebrow">
            <BoltIcon />
            <span>BIS SP 21 · RAG · Hackathon 2026</span>
          </div>
          <h1 className="hero-title">
            Find the right BIS standard
            <span className="hero-accent"> in seconds.</span>
          </h1>
          <p className="hero-sub">
            Describe your product and our hybrid retrieval engine surfaces the exact
            Bureau of Indian Standards regulations that apply — with AI-generated rationale.
          </p>
        </section>

        {/* ── Architecture ── */}
        <section className="arch-section">
          <ArchDiagram />
        </section>

        {/* ── Search box ── */}
        <section className="search-section">
          <div className="search-card">
            <label className="search-label">Product Description</label>
            <div className="textarea-wrap">
              <textarea
                ref={textareaRef}
                className="search-textarea"
                placeholder="e.g. We manufacture 33 Grade Ordinary Portland Cement and need to know which BIS standards apply to our chemical and physical requirements…"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={4}
              />
              <div className="textarea-hint">
                <kbd>Ctrl</kbd><kbd>↵</kbd> to search
              </div>
            </div>

            <button
              className={`search-btn ${loading ? 'loading' : ''}`}
              onClick={() => handleQuery()}
              disabled={loading || !query.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Retrieving standards…
                </>
              ) : (
                <>
                  <SearchIcon />
                  Find Applicable Standards
                </>
              )}
            </button>
          </div>

          {/* Examples */}
          {examples.length > 0 && (
            <div className="examples-section">
              <div className="examples-label">Try an example</div>
              <div className="examples-grid">
                {examples.slice(0, 6).map((ex, i) => (
                  <ExamplePill key={i} text={ex} onSelect={selectExample} />
                ))}
              </div>
            </div>
          )}
        </section>

        {/* ── Error ── */}
        {error && (
          <div className="error-banner">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <section className="results-section" ref={resultsRef}>
            <div className="results-header">
              <div>
                <h2 className="results-title">Recommended Standards</h2>
                <p className="results-query">"{result.query.slice(0, 80)}{result.query.length > 80 ? '…' : ''}"</p>
              </div>
              <MetricsBar result={result} />
            </div>

            <div className="standards-list">
              {result.recommendations.map((std, i) => (
                <StandardCard
                  key={std.is_number_full}
                  std={std}
                  index={i}
                  visible={cardsVisible}
                />
              ))}
            </div>

            {result.recommendations.length === 0 && (
              <div className="empty-state">
                <p>No standards found for this query. Try refining your product description.</p>
              </div>
            )}
          </section>
        )}

        {/* ── Footer stats ── */}
        <footer className="footer">
          <div className="footer-inner">
            <div className="footer-tag">BIS × Sigma Squad Hackathon 2026</div>
            <div className="footer-stack">
              {['FAISS', 'BGE-base', 'BM25', 'Cross-Encoder', 'RRF Fusion', 'Claude AI'].map(t => (
                <span key={t} className="stack-tag">{t}</span>
              ))}
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
