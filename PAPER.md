# Dynamic Circulator: Towards a Dynamic Topology

*A conceptual note.*

## The question

Most models of movement over a network treat the network as scenery: fixed edges, fixed
costs, and a process that flows across them. Intervention analysis inherits this: change a
cost, expect a proportional change in flow. But circulation systems, whether of goods,
attention, or intent, are cybernetic: the act of circulating changes the circulator, and
over longer horizons it changes the very structure through which circulation happens. A
worn path becomes a road. A habit becomes an institution. A frequently-taken route becomes
infrastructure that then constrains what routes are taken next.

The Dynamic Circulator is a small kernel built to hold that loop in one inspectable object,
and to ask a sequence of questions with it.

## The kernel

An agent at a node carries a **telemetry** vector: its intent or state. The probability
of each next move is a temperature-controlled softmax over edge weights

```
W(i→j) = α·distance(i,j) − β·⟨telemetry, features(j)⟩ − intervention(i,j)
```

so a move is jointly a function of physical friction, alignment between who the agent is and
where it might go, and any lever placed on the edge. After the move, the destination writes
back into the telemetry. Two feedback timescales layer on top: **preference memory** on
edges (traffic and reward leave a trace that decays), and, at the population scale, a shared
field that couples agents through the environment they collectively shape. The position
process alone is non-stationary; the joint process is Markov. This is the cybernetic
minimum: state and structure-of-choice each conditioning the other.

## The questions it lets us ask

**1. Where can an intervention even act?** A lever on an edge changes nothing if the agent
at that edge has no alternative: a single admissible next step is a foregone conclusion, and
no finite change to its cost alters a softmax over one option. Effectiveness is a property
of the *intervention–topology pair*, not of the lever's size. The kernel makes this
first-order and computable: the **leverage field** `∂π/∂S` gives the sensitivity of the
long-run distribution to each edge, and it is exactly zero where there is no genuine choice.
Counting choices, though, is not enough, as an edge can carry many options and still be unable to change the outcome if those options lead to the same downstream future.

**2. What does circulation remember, and when does memory mislead?** Preference memory lets
useful routes consolidate and stale ones fade. But the same mechanism that records a good
habit records it too well: a route reinforced under one regime keeps drawing flow after the
regime that justified it is gone. The kernel exposes this memory–ecology mismatch directly,
as a phase question in the ratio of reinforcement to evaporation.

**3. When should circulation become structure, and how is structure undone?** This is the
edge of the project and the meaning of *towards a dynamic topology*. The kernel includes the
primitive for it with edges that can be activated or pruned over a fixed physical substrate
and the beginnings of an answer: consolidation that is too eager builds infrastructure that
then traps the system, and only a release mechanism (not the same force that built it)
undoes it. A topology that reshapes itself through use, and can also let go of what it built,
is the object this kernel is aimed at.

## Scope and Limits

The **process** here is genuinely dynamic and non-stationary. The **graph** is, within a
run, fixed; self-rewriting is scaffolding and a direction, not a result. The demo topologies
are abstractions for exploring mechanism, not calibrated models of any real
system. The value of the object is that it puts attractiveness, feasibility, adaptation, and
the *reachability of an intervention* into one place where they can be distinguished and
where the difference between a route that is attractive, a route that is usable, a route that
is selected, and a route that remains chosen after the environment changes is observable and measurable.

As the title alludes to, our next investigation will lie in expressing self-modifying topology rigorously via typed rules under stochastic events. 

— SyberLabs, August 2026

