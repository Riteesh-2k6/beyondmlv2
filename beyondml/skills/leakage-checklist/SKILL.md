---
name: leakage-checklist
description: "How to recognise data leakage: features unavailable at prediction time or proxies for the target."
metadata:
  beyondml:
    kind: knowledge
---
Leakage is a feature that would not exist when the prediction is made, or that is a proxy for the target.

- Recorded after the outcome: a churn model's Cancellation_Date, a fraud model's chargeback flag.
- Direct functions of the target: Tax_Amount for Price, total = sum of parts that include the target.
- Treatment implies diagnosis: a disease model's Prescription_A.
- IDs and timestamps that encode the split or the label order.
- Too good to be true: a single column that scores 0.98+ on its own almost always leaks.

Action: drop high-risk columns; for medium risk, keep them only with a written justification.
