# Dynamic Circulator

*A telemetry-conditioned circulation kernel for non-stationary routing over weighted graphs.*

Agents move over a directed, feature-decorated graph. Where each agent goes next is a
softmax over edge weights that depend on three factors: the physical friction of an
edge, how well the agent's current **telemetry** (its intent/state) aligns with the
destination, and any intervention applied to that edge. After a move, the visited node
writes back into the agent's telemetry. Routing shapes state; state reshapes routing. The
position process alone is non-stationary; the joint (position, telemetry) process is Markov.

This repository is the **software** artifact of the project: the kernel, a population
simulator, a small set of demo topologies, a FastAPI service, and a browser visualizer.
The conceptual motivation, and the boundary of what "dynamic" means here — is in
[`PAPER.md`](PAPER.md).

## Quickstart

```bash
# backend
pip install -r requirements.txt
uvicorn api:app --port 8000

# visualizer (separate terminal)
cd visualizer
npm install
npm run dev            # opens the Control Hub; proxies /api to :8000
```

## What you can do

The **Control Hub** is the front door. Pick a graph (mall, airport, museum, city
topologies, …), then open one of three surfaces:

- **Kernel Inspector** — change agent intent and exploration temperature and watch the
  transition field reshape route probabilities in real time.
- **Agent Flow Simulator** — release a population and watch live node occupancy and edge
  traffic respond to crowd mix and interventions.
- **Topology Comparison** — run two graphs side by side under matched conditions.

## The kernel in one object

```python
from kernel import DynamicCirculator   # alias of DynamicTopologyKernel
```

Beyond routing, the kernel carries the research instruments the project was built to ask
questions with: a stationary-**leverage** field (`edge_leverage`, `stationary_leverage`) —
the first-order sensitivity of long-run circulation to an edge intervention — and an
effective-topology primitive (`set_edge_active`) that activates or prunes edges over a fixed
substrate. These are the seeds of the direction named in the paper: *towards a dynamic
topology*.

## Scope and Limits

The **process** is dynamic and non-stationary; the **graph** is, today, fixed within a run.
True combinatorial self-rewriting, a topology that reorganizes itself through use, is a
research direction the kernel is built toward, not a delivered guarantee. Claims here are
kept to what the code demonstrably does.

## Tests

```bash
pytest tests/     # kernel, simulator, edge-learning, memory law, API surface
```
