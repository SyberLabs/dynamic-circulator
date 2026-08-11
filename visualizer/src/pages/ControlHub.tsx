import { useCallback, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity,
  BrainCircuit,
  Building2,
  GitCompareArrows,
  Landmark,
  Network,
  Plane,
  Route,
  ShoppingBag,
  Waypoints,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface PresetMeta {
  name: string;
  description: string;
  undirected: boolean;
  nodeCount: number;
}

const PRESET_CONFIGS: Record<string, { accent: string; Icon: LucideIcon }> = {
  mall:         { accent: '#4f8ef5', Icon: ShoppingBag },
  airport:      { accent: '#34d399', Icon: Plane },
  museum:       { accent: '#9b6cf7', Icon: Landmark },
  supply_chain: { accent: '#f5a623', Icon: Route },
  wheel_city:   { accent: '#f5a623', Icon: Building2 },
  rhizome_city: { accent: '#34d399', Icon: Network },
  neural_dense: { accent: '#9b6cf7', Icon: BrainCircuit },
  social_media: { accent: '#ef4444', Icon: Waypoints },
};

interface PortCard {
  route: string;
  kicker: string;
  title: string;
  accent: string;
  Icon: LucideIcon;
  body: ReactNode;
}

const TOPOLOGY_VIEWS: PortCard[] = [
  {
    route: '/base',
    kicker: 'Inspector',
    title: 'Kernel Inspector',
    accent: '#4f8ef5',
    Icon: Waypoints,
    body: (
      <>
        Change agent intent and exploration temperature to see how the transition matrix
        reshapes local route probabilities.
      </>
    ),
  },
  {
    route: '/mall',
    kicker: 'Simulator',
    title: 'Agent Flow Simulator',
    accent: '#34d399',
    Icon: Activity,
    body: (
      <>
        Adjust population size, agent mix, and intervention channel to watch live
        occupancy and edge traffic respond.
      </>
    ),
  },
];

const FIXED_LABS: PortCard[] = [
  {
    route: '/compare',
    kicker: 'Lab',
    title: 'Topology Comparison',
    accent: '#f5a623',
    Icon: GitCompareArrows,
    body: (
      <>
        Run the same agent mix through two graph structures to compare congestion,
        mixing, entropy, and intervention response.
      </>
    ),
  },
];

function PortCardView({ card }: { card: PortCard }) {
  const navigate = useNavigate();
  const Icon = card.Icon;

  return (
    <button
      type="button"
      className="hub-port-card"
      style={{ '--accent': card.accent } as React.CSSProperties}
      onClick={() => navigate(card.route)}
    >
      <span className="hub-card-kicker">{card.kicker}</span>
      <span className="hub-card-title">
        <Icon size={18} />
        {card.title}
      </span>
      <span className="hub-card-copy">{card.body}</span>
      <span className="hub-card-action">Open</span>
    </button>
  );
}

function GroupLabel({ children, detail }: { children: ReactNode; detail: ReactNode }) {
  return (
    <div className="hub-group-label">
      <strong>{children}</strong>
      <span>{detail}</span>
    </div>
  );
}

export default function ControlHub() {
  const [presets, setPresets] = useState<Record<string, PresetMeta>>({});
  const [activePreset, setActivePreset] = useState<string>('mall');
  const [loading, setLoading] = useState<string | null>(null);
  const [activePresetName, setActivePresetName] = useState<string>('Enterprise Mall');
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [presetsError, setPresetsError] = useState(false);

  const fetchPresets = useCallback((attempt = 0) => {
    setPresetsLoading(true);
    setPresetsError(false);

    fetch('/api/topology/presets')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(data => {
        setPresets(data);
        setPresetsLoading(false);
      })
      .catch(() => {
        if (attempt < 4) {
          window.setTimeout(() => fetchPresets(attempt + 1), 500 * Math.pow(2, attempt));
        } else {
          setPresetsLoading(false);
          setPresetsError(true);
        }
      });
  }, []);

  useEffect(() => {
    fetchPresets();

    fetch('/api/topology')
      .then(r => r.json())
      .then(d => {
        if (d.presetName) setActivePresetName(d.presetName);
        if (d.preset) setActivePreset(d.preset);
      })
      .catch(() => {});
  }, [fetchPresets]);

  const handleSelectPreset = async (key: string) => {
    if (key === activePreset && loading === null) return;
    setLoading(key);
    try {
      const res = await fetch('/api/topology/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset: key }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.presetName) setActivePresetName(data.presetName);
      setActivePreset(key);
    } catch (e) {
      console.error('Topology load failed:', e);
    } finally {
      setLoading(null);
    }
  };

  const renderPresetArea = () => {
    if (presetsLoading) {
      return (
        <div className="hub-preset-grid">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="hub-preset-card skeleton" />
          ))}
        </div>
      );
    }

    if (presetsError) {
      return (
        <div className="hub-error">
          <strong>Backend unreachable</strong>
          <span>Could not load topology presets. Start the FastAPI server and retry.</span>
          <button type="button" onClick={() => fetchPresets()}>Retry Connection</button>
        </div>
      );
    }

    if (Object.keys(presets).length === 0) {
      return <div className="hub-empty">No presets returned from server.</div>;
    }

    return (
      <div className="hub-preset-grid">
        {Object.entries(presets).map(([key, meta]) => {
          const cfg = PRESET_CONFIGS[key] ?? { accent: '#7a82a0', Icon: Waypoints };
          const Icon = cfg.Icon;
          const isActive = activePreset === key;
          const isLoading = loading === key;

          return (
            <button
              key={key}
              type="button"
              className={`hub-preset-card ${isActive ? 'active' : ''}`}
              style={{ '--accent': cfg.accent } as React.CSSProperties}
              onClick={() => handleSelectPreset(key)}
              aria-pressed={isActive}
              disabled={loading !== null}
            >
              <Icon size={22} />
              <strong>{meta.name}</strong>
              <span>{meta.description}</span>
              <em>
                {meta.nodeCount} nodes
                <i>{meta.undirected ? 'undirected' : 'directed'}</i>
              </em>
              {isLoading && <b className="hub-card-spinner" aria-label="Loading preset" />}
            </button>
          );
        })}
      </div>
    );
  };

  return (
    <main className="hub-shell">
      <header className="hub-hero">
        <span>SyberLabs · Towards a Dynamic Topology</span>
        <h1>Dynamic Circulator</h1>
        <p>
          A circulation kernel: agents route over a weighted graph by telemetry-conditioned
          transition probabilities that co-evolve with where the agents have been. Choose a
          graph, then inspect its transition field or run live agent flow.
        </p>
      </header>

      <section className="hub-section">
        <GroupLabel detail="Feeds the Kernel Inspector and Agent Flow Simulator.">
          Choose Topology
        </GroupLabel>
        {renderPresetArea()}
        {!presetsError && (
          <div className="hub-active">
            Active environment: <strong>{activePresetName}</strong>
          </div>
        )}
      </section>

      <section className="hub-section">
        <GroupLabel detail="These views consume the topology selected above.">
          Topology-Driven Views
        </GroupLabel>
        <div className="hub-port-grid">
          {TOPOLOGY_VIEWS.map((card) => (
            <PortCardView key={card.route} card={card} />
          ))}
        </div>
      </section>

      <section className="hub-section">
        <GroupLabel detail="Self-contained labs with fixed comparison graphs.">
          Fixed Scenario Labs
        </GroupLabel>
        <div className="hub-port-grid">
          {FIXED_LABS.map((card) => (
            <PortCardView key={card.route} card={card} />
          ))}
        </div>
      </section>
    </main>
  );
}
