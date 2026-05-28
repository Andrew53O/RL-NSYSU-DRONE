# Mane et al., 2024 — EROAS: Efficient Reactive Obstacle Avoidance System

**Citation / Goal**
- Mane et al., 2024 (arXiv preprint / accepted for IEEE Journal of Oceanic Engineering). EROAS: reactive obstacle avoidance for AUVs using 2.5D forward-looking sonar.
- Goal: handle partial observability, occlusions, and hardware constraints of forward-looking sonar with a lightweight, reactive, provably safer system.

**Sensors / Context**
- Forward‑looking 2D/2.5D sonar profiles with limited field-of-view; readings are noisy and can be occluded.
- Evaluated in simulation and HIL (hardware-in-the-loop) experiments.

**Core ideas**
- Short-term sonar memory: keep a small temporal feature set (previous readings, sliding-window minimums, simple trends) rather than trying large recurrent nets.
- Sonar-profile-guided directional control: interpret sectorized sonar profiles into preferred avoidance directions.
- Control‑barrier-function (CBF) safety layer / safety filter: deterministic runtime filter that constrains or overrides nominal commands when safety thresholds are breached.

**Short-term memory features**
- `previous_front_*` (one-step history) for each sector.
- `trend = prev_norm - curr_norm` (positive trend → obstacle getting closer).
- `min_recent = min(range[t-w..t])` for a short window (e.g., w = 5–20 steps).

**Safety filter / CBF**
- Runtime module that examines proposed control (from policy or planner) and enforces safety actions when sectors violate emergency thresholds.
- Suggested thresholds: caution ≈ 1.5 m, emergency ≈ 0.25–0.5 m (tune for vehicle/sensor).
- Filter behavior examples: cap forward velocity when front-center risk > threshold; push vertical command upward if down-sector emergency; when invalid sensor, stop or throttle down.

**Design implications**
- Partial observability: a single sonar frame is unreliable; short-term memory and trend features are necessary for early detection of closing obstacles.
- Safety: keep a small, explainable filter rather than relying on the policy to always be safe.

**Implementation notes**
- Log when the safety filter acts and include a small training penalty so the learned policy avoids depending on the filter.
- Keep memory features compact to avoid increasing policy input dimensionality substantially.
- Use the same normalized units as the rest of the observation vector.

**Report language snippets**
- "EROAS motivates including short-term sonar memory and a lightweight control‑barrier-function safety layer. We include previous-sector ranges, a recent minimum, and a trend feature in the observation, and apply a small safety filter at publish time to prevent clearly dangerous commands." 

---

*Notes for implementers:* Implement trend and min_recent in the environment wrapper; implement a safety_filter function that can alter `vx, vy, vz` given sector risks and log its interventions so training can penalize them lightly.
