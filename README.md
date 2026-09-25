# Route Optimization via Reinforcement Learning (RORL)

Decentralized Reinforcement Learning vs. Dynamic Dijkstra and Static Shortest Path on Dynamic Network Topologies.
A decentralized MaskablePPO reinforcement-learning agent trained to route packets on a simulated NSFNET topology, benchmarked against Dynamic Dijkstra and Static SPF. Diagnosed and fixed a two-hop routing-loop failure mode via split-horizon ingress masking, then showed the RL agent consistently outperforms Dijkstra once Dijkstra's global network state is realistically delayed (stale), across a 5-seed, multi-staleness-level sweep.
---
## Live Demo

---
## Canonical Python Environment

> **IMPORTANT**: The single canonical Python environment for this project is:
> ```bash
> /opt/anaconda3/bin/python  (Python 3.12)
> ```
> All training, benchmark evaluations, automated tests, and Streamlit execution must target `/opt/anaconda3/bin`.
>
> *(Note: Any other Python environment, such as system Python 3.13, was patched once for emergency compatibility and is **not** maintained as a supported dual environment. Always use the canonical Anaconda environment below to avoid dependency divergence).*

---

## Running the Streamlit Dashboard

To launch the dashboard with the guaranteed canonical environment:

```bash
./run_dashboard.sh
```

Or directly via the canonical Streamlit executable:

```bash
/opt/anaconda3/bin/streamlit run app.py
```

The dashboard runs locally at `http://localhost:8501`.

---

## Running Tests

Always execute tests using the canonical environment's `pytest`:

```bash
PYTHONPATH=. /opt/anaconda3/bin/pytest tests/
```

To run the live inference regression suite specifically:

```bash
PYTHONPATH=. /opt/anaconda3/bin/pytest tests/test_live_inference_regression.py -v
```

---
## Project structure

    src/              — network simulator, routing algorithms, RL environment, training pipeline
    scripts/          — demo and evaluation scripts
    tests/            — automated test suite (53+ tests)
    checkpoints/      — trained model weights (mostly gitignored; two canonical
                        checkpoints below are committed for the live demo)
    assets/           — benchmark results, plots, telemetry CSVs (tracked in git)
    app.py            — Streamlit dashboard (3 tabs: benchmark, path tracer, custom network)

---
## Repository Structure

- `app.py`: Streamlit dashboard with permanent dark mode, 3 tabs (Policy Benchmark, Path Tracer, Custom Network Builder).
- `run_dashboard.sh`: Canonical launch script binding to `/opt/anaconda3/bin/streamlit`.
- `src/`:
  - `topology.py`: NSFNET topology and custom Erdős–Rényi / Barabási–Albert / Ring-Mesh network generators.
  - `network_sim.py`: NSFNET simulation environment with M/M/1 queuing and live link loads.
  - `dijkstra.py`: Dynamic Dijkstra, Static SPF, and Periodic/Stale Dijkstra routers.
  - `packet_traffic.py`: Packet traffic generator and closed-loop routing simulation runner.
  - `routing_env.py`: Gymnasium-compatible decentralized packet routing environment with action masking.
  - `train_agent.py`: MaskablePPO training pipeline.
  - `metrics.py`: Aggregate telemetry computation (latency, loss, throughput, queuing delay).
- `checkpoints/`: Pre-trained MaskablePPO checkpoints (`candidate_a.zip`, `final_model_200k.zip`).
- `assets/`: Audited benchmark comparison telemetry and evaluation curves.
- `tests/`: Unit, integration, and regression test suites.

## Canonical model checkpoints

The audited, reported results in this project use:

    checkpoints/reward_tuning/candidate_a.zip   — tuned reward config, w1=2.0, w3=4.0,
                                                    run with Split-Horizon enabled (default)
    checkpoints/final_model_200k.zip            — baseline reward config, 200k training steps

Do not confuse either of these with `checkpoints/final_model_50k.zip`
(an earlier, superseded checkpoint) if present locally.

## Reproducing results

    pytest tests/                                          # run full test suite
    python -m src.train_agent --seed 42                    # retrain from scratch
    python scripts/evaluate_agent.py --model checkpoints/reward_tuning/candidate_a.zip

## Key results (Seed 100 benchmark, 2,000 packets, traffic shock at t=20)

| Policy                          | Delivered | Loss Rate | Mean Latency |
|----------------------------------|-----------|-----------|--------------|
| Dynamic Dijkstra (oracle)         | 1,984     | 0.80%     | 15.06 ms     |
| Candidate A + Split-Horizon (RL)  | 1,978     | 1.10%     | 19.56 ms     |
| Static SPF                        | 1,957     | 2.15%     | 15.75 ms     |

Under realistic stale global-state conditions (τ ≥ 5 timesteps of link-state
delay, mirroring real OSPF flooding intervals), the RL agent outperforms
Dijkstra on 5 of 5 evaluation seeds — at the cost of a ~5ms latency penalty
from taking longer, congestion-avoiding paths.

## Dashboard tabs

1. **Policy Benchmark & Evaluation** — audited comparison tables and plots
   across Dijkstra, Static SPF, Stale Dijkstra, and the RL agent.
2. **Interactive Path Tracer** — pick any source/destination on real NSFNET
   cities, inject link congestion, and watch all three policies route live.
3. **Custom Network Builder & Analyzer** — generate synthetic topologies
   (Erdős–Rényi / Barabási–Albert / ring-mesh) and inspect graph metrics.
   Note: the trained RL policy is NSFNET-specific (fixed 14-node input
   shape) and does not run unmodified on custom graphs — this tab uses a
   separate, untrained routing method for comparison.

## Known limitations

- Single centralized policy (not full multi-agent RL — one router per node).
- Trained and evaluated on NSFNET only; cross-topology generalization is
  future work (would require a Graph Neural Network state encoding).
- PPO training hyperparameters were not exhaustively tuned; only the reward
  function's weighting was tuned across a small, hand-selected sweep.
