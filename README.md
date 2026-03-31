# ORE — Oxford Research Engine

> *"The greatest minds in one room, fleshing out one idea."*

ORE is a CLI tool that orchestrates multiple LLM agents in structured research loops. Instead of a single chatbot generating monologues, ORE puts **asymmetric agents** — a Builder, a Skeptic, a Historian, and a Referee — into intellectual tension with each other. Every output is a typed **research object** (conjecture, objection, toy model, synthesis), not free-form prose. The result is a compressed, high-quality research artifact.

## How It Works

```
You ask a question
  → Builder proposes a conjecture
    → Skeptic attacks it
      → Builder refines or pivots
        → Historian synthesizes the round
          → Referee scores and decides: continue, stop, or pivot
            → Loop until convergence or budget exhaustion
```

Each agent has a fixed role, a system prompt, and constrained output types:

| Agent | Role | Outputs |
|-------|------|---------|
| **Builder** | Proposes ideas, models, novel framings | Conjecture, ToyModel, Reframe |
| **Skeptic** | Attacks claims, finds gaps, proposes tests | Objection, FalsificationTest, CounterExample |
| **Historian** | Tracks surviving/killed ideas, finds patterns | Synthesis, PatternNote, OpenQuestion |
| **Referee** | Scores each round, decides when to stop | RoundScore |

## Installation

```bash
pip install ore-research
```

Or install from source:

```bash
git clone https://github.com/christianalteri/oxford-research-engine.git
cd oxford-research-engine
pip install -e .
```

## Quick Start

Set your API key(s):

```bash
export OPENAI_API_KEY="sk-..."
# and/or
export ANTHROPIC_API_KEY="sk-ant-..."
```

Run a research session:

```bash
ore run "What resolves the tension between quantum mechanics and general relativity?"
```

## Usage

### Run with a config file

```bash
ore run --config examples/quantum_gravity.yaml
```

### Override settings

```bash
ore run "Is consciousness emergent?" --rounds 5 --budget 2.00
```

### Resume a stopped session

```bash
ore resume abc12345
```

### List past sessions

```bash
ore list
```

### Export a session

```bash
ore export abc12345 --format markdown
ore export abc12345 --format json
ore export abc12345 --format both
```

## Configuration

Create a YAML config file to control models, rounds, and budget:

```yaml
question: "Your research question here"
max_rounds: 10
budget_usd: 5.00

agents:
  builder:
    model: "anthropic/claude-sonnet-4-20250514"
    temperature: 0.8
  skeptic:
    model: "openai/gpt-4o"
    temperature: 0.3
  historian:
    model: "openai/gpt-4o-mini"
    temperature: 0.5
  referee:
    model: "openai/gpt-4o-mini"
    temperature: 0.1
```

ORE uses [LiteLLM](https://docs.litellm.ai/) under the hood, so any model it supports works here — OpenAI, Anthropic, Ollama, Azure, Bedrock, Vertex, and [100+ more](https://docs.litellm.ai/docs/providers).

### Mix models by role

This is a key feature. Use frontier models for the hard thinking and cheap models for utility work:

```yaml
agents:
  builder:
    model: "anthropic/claude-sonnet-4-20250514"  # needs creativity
  skeptic:
    model: "openai/gpt-4o"          # needs rigor
  historian:
    model: "openai/gpt-4o-mini"     # summarization is cheaper
  referee:
    model: "openai/gpt-4o-mini"     # scoring is cheaper
```

### Run with local models

```yaml
agents:
  builder:
    model: "ollama/llama3.1:70b"
  skeptic:
    model: "ollama/llama3.1:70b"
  historian:
    model: "ollama/llama3.1:8b"
  referee:
    model: "ollama/llama3.1:8b"
```

## Stopping Rules

The engine stops when any of these conditions are met:

- **Referee stops it** — intellectual convergence detected
- **Max rounds** — configurable (default 10)
- **Budget exceeded** — token cost limit hit
- **Ctrl+C** — graceful stop, still produces a report

## Output

Sessions are saved to `~/.ore/sessions/<session-id>/`:

```
~/.ore/sessions/abc12345/
  config.yaml          # session configuration
  memory.json          # all research objects
  report.md            # markdown research report
  round_001.json       # per-round snapshots
  round_002.json
  ...
```

The markdown report includes all research objects organized by round, with score tables and a final synthesis.

## What It's Good For

ORE is strongest on **ambiguous, critique-heavy problems** where ideas need pressure, comparison, and repeated refinement:

- **Scientific questions** — quantum gravity, consciousness, origin of life
- **Startup strategy** — go-to-market, pricing, competitive positioning
- **Technical architecture** — system design trade-offs, migration planning
- **Philosophy** — ethics, epistemology, metaphysics
- **Literature synthesis** — connecting findings across papers
- **Worldbuilding** — fiction, game design, alternate histories

## Architecture

```
ore run "question"
  │
  ├── EngineConfig (from YAML or CLI args)
  ├── JsonMemoryStore (session persistence)
  ├── LLMProvider (LiteLLM wrapper + cost tracking)
  │
  └── ResearchEngine (loop orchestrator)
       ├── BuilderAgent  → Conjecture | ToyModel | Reframe
       ├── SkepticAgent  → Objection | FalsificationTest | CounterExample
       ├── HistorianAgent → Synthesis | PatternNote | OpenQuestion
       └── RefereeAgent  → RoundScore (verdict: continue/stop/pivot)
```

## Development

```bash
git clone https://github.com/christianalteri/oxford-research-engine.git
cd oxford-research-engine
pip install -e ".[dev]"
pytest
```

## License

MIT
