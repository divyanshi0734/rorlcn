# Route Optimization via Reinforcement Learning (RORL)

Decentralized Reinforcement Learning vs. Dynamic Dijkstra and Static Shortest Path on Dynamic Network Topologies.

---

## 🐍 Canonical Python Environment

> **IMPORTANT**: The single canonical Python environment for this project is:
> ```bash
> /opt/anaconda3/bin/python  (Python 3.12)
> ```
> All training, benchmark evaluations, automated tests, and Streamlit execution must target `/opt/anaconda3/bin`.
>
> *(Note: Any other Python environment, such as system Python 3.13, was patched once for emergency compatibility and is **not** maintained as a supported dual environment. Always use the canonical Anaconda environment below to avoid dependency divergence).*

---

## 🚀 Running the Streamlit Dashboard

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

## 🧪 Running Tests

Always execute tests using the canonical environment's `pytest`:

```bash
PYTHONPATH=. /opt/anaconda3/bin/pytest tests/
```

To run the live inference regression suite specifically:

```bash
PYTHONPATH=. /opt/anaconda3/bin/pytest tests/test_live_inference_regression.py -v
```

---

## 📁 Repository Structure

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
