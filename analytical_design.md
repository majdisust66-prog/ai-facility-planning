# Phase 3 - Analytical Design

> All data in this project is SYNTHETIC. Results validate the method, not real contract performance.

## 1. KPI definitions

| KPI | Definition |
|---|---|
| Total work orders | Unique `wo_id` after de-duplication |
| Completed | `status = Completed` |
| SLA breach | Completion hours (created to completed) > completion SLA of the priority |
| SLA compliance % | 1 - breaches / work orders with a known outcome |
| Backlog (at time t) | Work orders created before t and not completed at t (cancelled and invalid rows excluded) |
| Repeat work order | Same asset and scope reported again within 30 days |

Excluded from SLA KPIs: cancelled, data-quality failures, and open work orders still within SLA (outcome unknown).

## 2. Prediction task
- **Question:** at the moment a corrective work order is created, what is the probability it will breach its completion SLA?
- **Target:** `sla_breach` (1 = breach).
- **Prediction point:** creation time. The contractor is known then, because it follows from contract and scope.
- **Use:** rank new work orders so the planner follows up the riskiest first. A human decides the action.

## 3. Features (only information available at creation)

| Feature | Meaning |
|---|---|
| priority, scope, region, facility_id, contract_id, contractor_id | Work order attributes |
| facility_type, area_m2, building_age_years | Facility master data |
| hour, day_of_week, month, after_hours | Creation time (weekend = Friday, Saturday) |
| contractor_backlog, facility_backlog | Open work orders at creation |
| contractor_new_7d | Work orders assigned to the contractor in the previous 7 days |
| asset_repeat_30d, asset_missing | Earlier reports on the same asset and scope |
| contractor_breach_90d, facility_breach_180d | Breach rate of outcomes already known at creation, smoothed toward the overall rate |

**Excluded as leakage (known only later):** assigned_at, response_hours, actual_start, completed_at, resolution_hours, status, reopened, age_hours_at_extract.
**Excluded as uncertain:** fault_code (in real CMMS data it is often entered at closure).

## 4. Validation design (time-based)

| Split | Created | Use |
|---|---|---|
| Warm-up | Jul-Sep 2024 | Builds history only |
| Train | Oct 2024 - Dec 2025 (outcome known before 1 Jan 2026) | Fit models |
| Validation | Jan - Mar 2026 | Compare and choose models and threshold |
| Test | Apr 2026 - 23 Jun 2026 | Used once, for the final model |

Work orders created in the last 7 days before extraction are left out, so every outcome had time to mature.

## 5. Evaluation
- **Primary:** PR-AUC (the breach class is the minority). A random ranking scores about the breach rate.
- **Secondary:** ROC-AUC, Brier score (probability quality).
- **Operational:** precision and recall when the planner can follow up the top 20% of work orders. False alarms = 1 - precision; missed breaches = 1 - recall.
- **Baselines:** (a) same probability for all; (b) breach rate by contractor and priority. A model must beat (b) to be worth using.
- **Lead time:** at creation the warning arrives a full SLA window before the breach (4 to 168 hours by priority).

## 6. Risk score design (to be built next, separate from the ML model)
- **Level:** facility and contract, over a rolling 90 days.
- **Indicators:** breach rate, open backlog, backlog age, repeat rate, share of P1/P2 breaches.
- **Normalisation:** each indicator scaled 0-100; rates smoothed for small samples.
- **Weights:** start with equal weights, then compare with a documented priority-based weighting, and run a sensitivity analysis (does the top-10 list change?). Equal weights are a transparent starting assumption, not a proven optimum.
- **Classes:** proposed High / Medium / Low by percentile; thresholds are a management choice.
- **Validation:** does the score at time t rank facilities by their breach rate in the following period?

## 7. Limitations
- Synthetic data with planted relationships; real data will be noisier.
- Cancellation time is not recorded, so cancelled work orders are left out of backlog, which a live system could not know in advance.
- Rows failing quality checks are left out of history features.
- Associations found by the model are not causal evidence.
