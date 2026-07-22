# Gate E2 vs C — results

items: 60/60 | conflict: 20/20

## Conflict (n=20)
- C=1.30  E2=1.40  **E2-C=+0.10**
- outcome: **PARITY** (WIN>=+0.30, LOSS<=-0.10)

## AGG guard (n=60)
- C=1.03 E2=0.98  E2-C=-0.05  guard HOLDS

## Style-premium control
- E2 answer attribution: {'retrieval': 41, 'both': 13, 'neither': 6}
  - retrieval: n=41 E2=0.95 C=1.00
  - both: n=13 E2=1.08 C=1.15
  - neither: n=6 E2=1.00 C=1.00
- mean answer length: E2=128 C=139

## Conflict by block content (explicit edges vs trajectories-only)
  - with-conflict-section: n=16 C=1.38 E2=1.50 E2-C=+0.12
  - trajectories-only: n=4 C=1.00 E2=1.00 E2-C=+0.00

## Conflict per-item (C -> E2, style, section)
  p17|2|2      C=1 E2=0 d=-1 style=neither CS
  p13|19|2     C=2 E2=1 d=-1 style=retrieval CS
  p14|12|2     C=0 E2=0 d=+0 style=retrieval traj
  p13|14|3     C=2 E2=2 d=+0 style=retrieval CS
  p12|20|4     C=2 E2=2 d=+0 style=neither CS
  p17|12|1     C=2 E2=2 d=+0 style=retrieval CS
  p5|6|2       C=0 E2=0 d=+0 style=retrieval traj
  p6|15|2      C=0 E2=0 d=+0 style=retrieval CS
  p15|16|0     C=1 E2=1 d=+0 style=retrieval CS
  p17|4|1      C=2 E2=2 d=+0 style=retrieval CS
  p8|6|1       C=2 E2=2 d=+0 style=retrieval traj
  p8|16|2      C=2 E2=2 d=+0 style=both traj
  p13|6|2      C=0 E2=0 d=+0 style=retrieval CS
  p7|5|1       C=2 E2=2 d=+0 style=retrieval CS
  p18|13|3     C=2 E2=2 d=+0 style=neither CS
  p13|13|2     C=2 E2=2 d=+0 style=retrieval CS
  p6|8|3       C=2 E2=2 d=+0 style=retrieval CS
  p7|1|2       C=2 E2=2 d=+0 style=retrieval CS
  p13|11|2     C=0 E2=2 d=+2 style=neither CS
  p18|7|2      C=0 E2=2 d=+2 style=retrieval CS