import React, { useState, useEffect, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { ControlSection, SliderField, StatusBanner } from '../components/UIControls';

// ─── Types ────────────────────────────────────────────────────────────────────

interface TopologyResponse {
  labels: string[];
  nodesConfig: number[][];
  distanceMatrix: number[][];
  adjacencyMask: number[][];
  N: number;
  F: number;
  undirected?: boolean;
  intentPresets?: Record<string, number[]>;
  accent?: string;
}

interface SimState {
  node_counts: number[];
  edge_counts: number[][];
  active_transit: number;
  active_stationary: number;
}

interface DemoGroup {
  name: string;
  val: number[];
  color: string;
  pct: number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Normalize an array of percentages to sum to exactly 1.0 */
function normalizePercentages(arr: number[]): number[] {
  const sum = arr.reduce((a, b) => a + b, 0);
  if (sum === 0) return arr.map(() => 1 / arr.length);
  return arr.map(v => v / sum);
}

/**
 * Derive a safe sponsor target index.
 * BUG-6 FIX: instead of blindly using N-1, pick the node whose label
 * contains "outlet", "gate", "retail", or "distribution" (case-insensitive),
 * falling back to N-1 only if nothing matches.
 */
function deriveSponsorTarget(labels: string[]): number {
  const keywords = ['outlet', 'gate', 'retail', 'distribution'];
  for (let i = labels.length - 1; i >= 0; i--) {
    const lc = labels[i].toLowerCase();
    if (keywords.some(k => lc.includes(k))) return i;
  }
  return labels.length - 1;
}

// ─── App ─────────────────────────────────────────────────────────────────────

const DEMO_COLORS = ['#f5a623', '#34d399', '#4f8ef5', '#9b6cf7'];

const MallPort: React.FC = () => {
  const fgRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // BUG-2 FIX: track whether the WS is open and ready to accept messages
  const wsReadyRef = useRef(false);

  const [topology, setTopology] = useState<TopologyResponse | null>(null);
  const [simState, setSimState] = useState<SimState | null>(null);
  const [wsError, setWsError] = useState(false);

  // Controls
  const [totalAgents, setTotalAgents] = useState(1000);
  const [temperature, setTemperature] = useState(1.0);
  const [sponsor, setSponsor] = useState<'none' | 'beta' | 'friction'>('none');
  const [sponsorDecay, setSponsorDecay] = useState(0.0);
  const [demoDist, setDemoDist] = useState<DemoGroup[]>([]);

  // ── WebSocket: connect ONLY after topology is fully fetched ──────────────
  // BUG-2 FIX: WS is created inside the .then() after topology is set,
  //            so the initial send always has a valid demoDist.
  useEffect(() => {
    let cancelled = false;

    fetch('/api/topology')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((topo: TopologyResponse) => {
        if (cancelled) return;
        setTopology(topo);
        setWsError(false);

        // Build initial demographic distribution from topology intent presets
        let initialDist: DemoGroup[] = [];
        if (topo.intentPresets) {
          const keys = Object.keys(topo.intentPresets);
          initialDist = keys.map((k, i) => ({
            name: k,
            val: topo.intentPresets![k],
            color: DEMO_COLORS[i % DEMO_COLORS.length],
            pct: 1.0 / keys.length,
          }));
          setDemoDist(initialDist);
        }

        // Open WebSocket only now — topology & groups are both ready
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/mall/stream`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          wsReadyRef.current = true;
          // Send the full initial state in one frame
          ws.send(JSON.stringify({
            K: totalAgents,
            temperature,
            groups: initialDist.map(d => [d.pct, d.val]),
          }));
        };

        ws.onmessage = (e) => {
          try {
            setSimState(JSON.parse(e.data));
          } catch { /* ignore malformed frames */ }
        };

        ws.onerror = () => setWsError(true);
        ws.onclose = () => { wsReadyRef.current = false; };

        wsRef.current = ws;
      })
      .catch(err => {
        console.error('Topology fetch failed:', err);
        setWsError(true);
      });

    return () => {
      cancelled = true;
      wsReadyRef.current = false;
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // run once on mount — totalAgents/temperature captured via closure for the first send only

  // ── Sync controls → WebSocket on any change ───────────────────────────────
  // BUG-6 FIX: use deriveSponsorTarget instead of hard-coded N-1
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || !wsReadyRef.current || !topology) return;

    const N = topology.labels.length;
    const beta = Array.from({ length: N }, () => Array(N).fill(5.0));
    const sponsor_friction = Array.from({ length: N }, () => Array(N).fill(0.0));

    const sponsorDest = deriveSponsorTarget(topology.labels);
    const sponsorSrc  = Math.min(1, N - 1); // second node as source, clamped

    if (sponsor === 'beta')     beta[sponsorSrc][sponsorDest] = 15.0;
    if (sponsor === 'friction') sponsor_friction[sponsorSrc][sponsorDest] = 8.0;

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        K: totalAgents,
        temperature,
        groups: demoDist.map(d => [d.pct, d.val]),
        beta,
        sponsor_friction,
        sponsor_decay: sponsorDecay,
      }));
    }
  }, [totalAgents, temperature, sponsor, sponsorDecay, demoDist, topology]);

  // ── Demographic slider handler: adjust one group and renormalize ──────────
  const handleDemoSlider = useCallback((idx: number, rawPct: number) => {
    setDemoDist(prev => {
      const next = prev.map((g, i) => i === idx ? { ...g, pct: rawPct } : g);
      const normalized = normalizePercentages(next.map(g => g.pct));
      return next.map((g, i) => ({ ...g, pct: normalized[i] }));
    });
  }, []);

  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(prev => (prev === node.id ? null : node.id));
  }, []);

  // ── Graph structure (static topology shape only) ──────────────────────────
  const graphData = React.useMemo(() => {
    if (!topology) return { nodes: [], links: [] };

    const accent = topology.accent ?? '#4f8ef5';
    const sponsorDest = deriveSponsorTarget(topology.labels);

    const nodes = topology.labels.map((label, idx) => ({
      id: idx,
      name: label,
      val: idx === 0 ? 3 : 2,
      // Highlight entry (0), sponsor target, and everything else
      color: idx === 0
        ? accent
        : idx === sponsorDest
          ? '#9b6cf7'
          : '#4f8ef5',
    }));

    const links: any[] = [];
    const N = topology.labels.length;

    if (topology.undirected !== false) {
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          if (topology.adjacencyMask[i][j] || topology.adjacencyMask[j][i]) {
            links.push({ source: i, target: j });
          }
        }
      }
    } else {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (i !== j && topology.adjacencyMask[i][j]) {
            links.push({ source: i, target: j });
          }
        }
      }
    }
    return { nodes, links };
  }, [topology]);

  // ── Canvas: draw agent counts on nodes ────────────────────────────────────
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (!simState) return;
    const count = simState.node_counts[node.id] || 0;
    const fontSize = 12 / globalScale;
    ctx.font = `600 ${fontSize}px Inter`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'white';
    ctx.fillText(node.name, node.x, node.y - 12 / globalScale);
    ctx.fillStyle = count > 0 ? '#34d399' : '#7a82a0';
    ctx.fillText(`${count}`, node.x, node.y + 6 / globalScale);
  }, [simState]);

  // ── Canvas: draw transit counts on edges ─────────────────────────────────
  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (!simState) return;
    const s = link.source;
    const t = link.target;
    if (typeof s !== 'object' || typeof t !== 'object') return;

    const totalTransit = (simState.edge_counts[s.id]?.[t.id] || 0)
                       + (simState.edge_counts[t.id]?.[s.id] || 0);
    if (totalTransit === 0) return;

    const midX = s.x + (t.x - s.x) / 2;
    const midY = s.y + (t.y - s.y) / 2;
    ctx.font = `600 ${10 / globalScale}px JetBrains Mono`;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(midX - 16 / globalScale, midY - 7 / globalScale, 32 / globalScale, 14 / globalScale);
    ctx.fillStyle = '#f5a623';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${totalTransit}`, midX, midY);
  }, [simState]);

  // ── Loading / Error states ────────────────────────────────────────────────
  if (!topology) {
    return (
      <div className="loading-screen">
        <div className="loading-ring"/>
        {wsError
          ? <span style={{ color: '#f87171' }}>Backend unreachable. Is uvicorn running?</span>
          : <span>Connecting to agent flow simulator...</span>
        }
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* ── SIDEBAR ─────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-brand" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1>Agent Flow Simulator</h1>
            <p>Discrete Agent Mechanics</p>
          </div>
          <button
            onClick={() => window.open('/api/export?format=csv', '_blank')}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
              color: '#f0f2f8', padding: '6px 10px', borderRadius: 6, fontSize: 10,
              fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
              textTransform: 'uppercase', letterSpacing: '0.04em'
            }}
          >
            Export State
          </button>
        </div>

        {/* WS health indicator */}
        {wsError && (
          <StatusBanner tone="red">
            Stream disconnected. Reload to reconnect.
          </StatusBanner>
        )}

        {/* Global Stats */}
        <div className="stats-bar">
          <div className="stat-pill">
            <div className="label">Stationary</div>
            <div className="value green">{simState?.active_stationary ?? '--'}</div>
          </div>
          <div className="stat-pill">
            <div className="label">In Transit</div>
            <div className="value amber">{simState?.active_transit ?? '--'}</div>
          </div>
        </div>

        <div className="sidebar-content">

          {/* ── Agent Count ── */}
          <ControlSection title="Total Agents">
            <SliderField
              label="Active population"
              min={10}
              max={5000}
              step={10}
              value={totalAgents}
              valueLabel={`${totalAgents.toLocaleString()} agents`}
              minLabel="Min"
              maxLabel="Max"
              tone="green"
              onChange={value => setTotalAgents(Math.round(value))}
            />
          </ControlSection>

          {/* ── Demographics ── */}
          {demoDist.length > 0 && (
            <ControlSection
              title="Agent Mix"
              description="Drag to adjust the intent profile. Values auto-normalize to 100%."
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {demoDist.map((group, idx) => {
                  const pctStyle = { '--pct': `${group.pct * 100}%` } as React.CSSProperties;
                  return (
                    <div key={group.name}>
                      <div style={{
                        display: 'flex', justifyContent: 'space-between',
                        alignItems: 'center', marginBottom: 6
                      }}>
                        <span style={{ fontSize: 12, fontWeight: 600, color: group.color }}>
                          {group.name}
                        </span>
                        <span style={{
                          fontFamily: 'JetBrains Mono', fontSize: 11,
                          background: `${group.color}22`,
                          color: group.color,
                          padding: '2px 8px', borderRadius: 4,
                          border: `1px solid ${group.color}44`,
                        }}>
                          {(group.pct * 100).toFixed(0)}%
                        </span>
                      </div>
                      <input
                        type="range" min="0" max="1" step="0.01"
                        value={group.pct}
                        style={{
                          ...pctStyle,
                          background: `linear-gradient(to right, ${group.color} 0%, ${group.color} var(--pct, 0%), var(--border-mid) var(--pct, 0%))`,
                        }}
                        onChange={e => handleDemoSlider(idx, parseFloat(e.target.value))}
                      />
                    </div>
                  );
                })}
              </div>

              {/* Visual distribution bar */}
              <div style={{
                display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden',
                marginTop: 14, gap: 1
              }}>
                {demoDist.map(g => (
                  <div key={g.name} style={{
                    flex: g.pct, background: g.color,
                    transition: 'flex 0.3s ease',
                    minWidth: g.pct > 0.01 ? 2 : 0,
                  }}/>
                ))}
              </div>
            </ControlSection>
          )}

          {/* ── Temperature ── */}
          <ControlSection title="Exploration Tau">
            <SliderField
              label="Route softness"
              min={0.01}
              max={5}
              step={0.01}
              value={temperature}
              valueLabel={`tau ${temperature.toFixed(2)}`}
              minLabel="Greedy"
              maxLabel="Explore"
              onChange={setTemperature}
            />
          </ControlSection>

          {/* Intervention channel */}
          <ControlSection title="Intervention Mode">
            <div className="sponsor-cards">
              <button
                type="button"
                className={`sponsor-card ${sponsor === 'none' ? 'active-blue' : ''}`}
                onClick={() => setSponsor('none')}
                aria-pressed={sponsor === 'none'}
              >
                <div className="sponsor-dot"/>
                <div className="sponsor-text">
                  <strong>Baseline</strong>
                  <span>Standard topology routing.</span>
                </div>
              </button>
              <button
                type="button"
                className={`sponsor-card ${sponsor === 'beta' ? 'active-purple' : ''}`}
                onClick={() => setSponsor('beta')}
                aria-pressed={sponsor === 'beta'}
              >
                <div className="sponsor-dot"/>
                <div className="sponsor-text">
                  <strong>Preference Bias (beta)</strong>
                  <span>Alignment-coupled amplification.</span>
                </div>
              </button>
              <button
                type="button"
                className={`sponsor-card ${sponsor === 'friction' ? 'active-amber' : ''}`}
                onClick={() => setSponsor('friction')}
                aria-pressed={sponsor === 'friction'}
              >
                <div className="sponsor-dot"/>
                <div className="sponsor-text">
                  <strong>Friction Reduction</strong>
                  <span>Alignment-independent cost reduction.</span>
                </div>
              </button>
            </div>

            {/* Decay slider */}
            {sponsor !== 'none' && (
              <div style={{ marginTop: 14 }}>
                <SliderField
                  label="Intervention decay"
                  min={0}
                  max={0.05}
                  step={0.001}
                  value={sponsorDecay}
                  valueLabel={sponsorDecay === 0 ? 'Off' : `${(sponsorDecay * 100).toFixed(1)}% / step`}
                  minLabel="Persistent"
                  maxLabel="Fast decay"
                  tone="amber"
                  onChange={setSponsorDecay}
                />
                {sponsorDecay > 0 && (
                  <div style={{
                    fontSize: 10, color: 'var(--text-muted)', marginTop: 6,
                    fontStyle: 'italic', lineHeight: 1.5,
                  }}>
                    Half-life about {Math.round(Math.log(2) / sponsorDecay)} steps
                  </div>
                )}
              </div>
            )}
          </ControlSection>

          {/* ── Live Node Breakdown ── */}
          {simState && (
            <section>
              <div className="section-heading">Node Occupancy</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {topology.labels.map((label, i) => {
                  const count = simState.node_counts[i] || 0;
                  const maxCount = Math.max(...simState.node_counts, 1);
                  return (
                    <div key={i}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 2 }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                        <span style={{ fontFamily: 'JetBrains Mono', color: 'var(--green)' }}>{count}</span>
                      </div>
                      <div style={{ height: 3, background: 'var(--border-mid)', borderRadius: 2 }}>
                        <div style={{
                          height: '100%',
                          width: `${(count / maxCount) * 100}%`,
                          background: 'var(--green)',
                          borderRadius: 2,
                          transition: 'width 0.15s ease',
                        }}/>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

        </div>
      </aside>

      {/* ── GRAPH CANVAS ────────────────────────────────── */}
      <main className="graph-area">
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          nodeRelSize={5}
          nodeColor="color"
          linkColor={() => 'rgba(255,255,255,0.08)'}
          linkWidth={2}
          linkDirectionalArrowLength={topology?.undirected === false ? 5 : 0}
          linkDirectionalArrowRelPos={0.85}
          linkDirectionalArrowColor={() => 'rgba(255,255,255,0.35)'}
          nodeCanvasObjectMode={() => 'after'}
          nodeCanvasObject={paintNode}
          linkCanvasObjectMode={() => 'after'}
          linkCanvasObject={paintLink}
          d3AlphaDecay={0.03}
          d3VelocityDecay={0.3}
          onEngineStop={() => fgRef.current?.zoomToFit(500, 80)}
          backgroundColor="transparent"
          onNodeClick={handleNodeClick}
          onBackgroundClick={() => setSelectedNode(null)}
        />

        {/* ── Live Node Inspector ─────────────────────────── */}
        {selectedNode !== null && simState && topology && selectedNode < topology.labels.length && (() => {
          const id = selectedNode;
          const total = simState.node_counts.reduce((a, b) => a + b, 0) || 1;
          const occ = simState.node_counts[id] || 0;
          const outflow = (simState.edge_counts[id] || []).reduce((a, b) => a + b, 0);
          const inflow = simState.edge_counts.reduce((a, row) => a + (row[id] || 0), 0);
          const outbound = topology.labels
            .map((label, j) => ({ label, transit: simState.edge_counts[id]?.[j] || 0 }))
            .filter(x => x.transit > 0)
            .sort((a, b) => b.transit - a.transit)
            .slice(0, 6);
          return (
            <div style={{
              position: 'absolute', top: 20, left: 20, zIndex: 100,
              background: 'var(--bg-panel)', border: '1px solid var(--border-bright)',
              borderRadius: 'var(--radius-md)', padding: '16px 18px', minWidth: 220,
              boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--amber)' }}>{topology.labels[id]}</h3>
                <button type="button" aria-label="Close node inspector" onClick={() => setSelectedNode(null)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: 0 }}>×</button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14 }}>
                {([
                  ['Agents here', `${occ}`, 'var(--green)'],
                  ['Share of population', `${(occ / total * 100).toFixed(1)}%`, 'var(--blue)'],
                  ['Inflow (transits)', `${inflow}`, 'var(--purple)'],
                  ['Outflow (transits)', `${outflow}`, 'var(--amber)'],
                ] as [string, string, string][]).map(([label, val, col]) => (
                  <div key={label} style={{
                    display: 'flex', justifyContent: 'space-between', fontSize: 11,
                    padding: '3px 0', borderBottom: '1px solid var(--border-dim)',
                  }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                    <span style={{ fontFamily: 'JetBrains Mono', color: col }}>{val}</span>
                  </div>
                ))}
              </div>
              {outbound.length > 0 && (
                <>
                  <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8 }}>
                    Transits by edge
                  </div>
                  {outbound.map(({ label, transit }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 4 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>→ {label}</span>
                      <span style={{ fontFamily: 'JetBrains Mono', color: 'var(--text-primary)' }}>{transit}</span>
                    </div>
                  ))}
                </>
              )}
            </div>
          );
        })()}
      </main>
    </div>
  );
};

export default MallPort;
