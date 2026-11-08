# WP3 exact L1 winding solver: pre-registered gates

Written and committed **before** any of the measurements below were taken. A result that misses its gate is
reported as a miss; gates are not adjusted after the fact.

Problem: `min_{x in Z^n} sum_e w_e |x_j - x_i - d_e| + sum_i u_i |x_i - pi_i|`, anchors fixed (eliminated).
"Certified" means the solver stopped because no +-1 min-cut move improves the (integer-quantised) energy, which for
this L-natural-convex objective is a proof of optimality of the quantised problem.

| id | gate | measurement | pass if |
|---|---|---|---|
| G1 | exactness on lattices | `make()` lattices 12x12x12 (and one 20^3), 5% / 20% wrong edges and the burst wall, 20 seeds each | certified in 100% of runs **and** E(solution) <= E(truth) in 100% of runs |
| G2 | no silent overflow | a graph whose natural integer capacities reach 3e9 | the solver rescales (per-node capacity < 2^31-1 asserted) or raises; it never returns a move from a silently failed max-flow (cut value from S must equal flow - N) |
| G3 | fixed-node elimination | 200 random small graphs (<= 12 nodes) with anchors, priors | optimal energy equals the big-M formulation and brute force (where enumerable), exactly in integer units |
| G4 | warm start | 20 lattices: cold (tree init) vs warm (IRLS-L1 rounding and a supplied perturbed optimum) | identical certified energy |
| G5 | stopping flag | `max_iter=1` from a far start | `stopped_by == 'max_iter'`, `certified False`; otherwise `'certificate'` |
| G6 | confidence proxy | 1000 sampled nodes of a 20k-node sheetlet graph: proxy `min(c1, cblk)` vs exact `c*` (one forced-node max-flow per node and sign) | Spearman >= 0.7 **and** proxy > 2 c* on <= 5% of nodes |
| G7 | chirality | sheetlet graph built with chi = +1 (true) and its mirror chi = -1 | picks the true sign; ratio E(wrong)/E(right) >= 1.5 |
| G8 | BK backend | if PyMaxflow is installable: certified energies on every G1/G4 graph | identical to scipy, else the backend is rejected |
| G9 | scale | global polish of a sheetlet-like graph with 1e6 supernodes / 1e7 edges from a stitched (slab -> block) start, on 1 CPU | wall <= 1 h and certified; otherwise BK is evaluated |
| G10 | accuracy (report, sanity floor) | sheetlet-like graphs, 3-10% wrong edges + correlated walls | fraction of nodes at the true winding (up to the global constant) reported for cold, stitched and polished solutions; floor: polished >= stitched >= 0.9 |

Notes fixed in advance:
* The certificate is for the quantised weights; the quantisation error `sum|w*scale - round(w*scale)| / scale` and the
  resulting energy bound `sum_e |dw_e| |r_e(x)|` are reported with every solve.
* Truth for the synthetic graphs is known up to one additive constant per connected component without anchors; accuracy
  is measured after the best constant shift per component.

## Clarifications recorded before the formal G6 / G7 / G9 / G10 runs (after development on seeds 0-3 only)

* **c\* reference.** Without anchors a whole component shifts for free (c\* = 0), and with one anchor node the
  cheapest set is "everything but the anchor" (one cut shared by all nodes). c\* is therefore defined as the minimal
  flip cost over sets S containing i, **confined to the nodes within radius R = sqrt(3) * (largest block) of i**
  (all other nodes held at the solved x) - the region the block proxy can reach.
* **G6 graph.** `sheetlets(20000, p_wrong=0.05, scale=10, seed=1000)` (pitch 120 vx, ~1400 x 1400 x 2400 vx), blocks
  (256, 512, 1024) L0 vx (2048 exceeds the 20k-node graph), R = 1774 vx, 1000 sampled nodes, BK backend (the cut
  values are cross-checked against the scipy definition by the G8 equality test).
* **Synthetic sheetlet graphs** (all formal runs): SAME edges = 6 nearest neighbours on the same wrap, cross edges =
  the nearest node on the next wrap outward (d = +1), weights U(0.3, 1.5) (x1.5 for SAME), p_wrong random offsets,
  8 correlated walls (a 0.5 rad x 30% height sector-slab in which 60% of cross edges are off by +1). Formal seeds:
  1000-1009 (never used during development).

## Round 2 (registered after the round-1 results, before any round-2 measurement)

**G6b - confidence with tie flagging (fix for the round-1 G6 failure; gate thresholds unchanged).**
* TIE definition: fix the gauge with one anchor per connected component (the node nearest the component centroid, at
  its solved value) unless anchors or priors are supplied; at the certified x run one max-flow per sign (scipy Dinic,
  whose residual reachability gives the *maximal* min-cut sink side); TIE_i = i lies in the maximal zero-cost +1 or -1
  move set, i.e. some +-1 move containing i and no anchor costs nothing. TIE nodes get conf = 0 (status flag
  TIE/LOWMARGIN).
* Measurement: `sheetlets(20000, p_wrong=0.05, scale=10)` for seeds 2000, 2001, 2002 (new), 1000 nodes sampled per
  graph from the **non-TIE** nodes, blocks (256, 512, 1024), c\* radius sqrt(3) * 1024 as in round 1.
* Gate (each seed): Spearman(proxy, c\*) >= 0.7 **and** proxy > 2 c\* on <= 5 % of the sampled non-TIE nodes.
  Reported alongside, not gated: TIE share of all nodes; recall of TIE for c\* = 0 on a separate 300-node sample drawn
  from all nodes.

**G7b - chirality on informative edges. This is a POST-HOC gate change** (made after the registered total-energy gate
G7 failed; the round-1 G7 FAIL stands in the report). Informative edges are the edges whose offset differs between
the chi = +1 and chi = -1 builds (here: sheets crossing the angular seam); each build is evaluated at its own optimum.
* Measurement: `sheetlets(20000, p_wrong, scale=10, chi=+-1)` for seeds 2100-2104 with p_wrong 0.05 (2100-2102) and
  0.10 (2103-2104); true chirality +1.
* Gate (each seed): the chosen sign is +1 **and** informative ratio >= 1.5 **and** `decided` is True.

**Amendment to G6b (recorded before any G6b measurement on the formal seeds 2000+).** A development check on seed 3
showed that the round-1 diagnosis was wrong: the c\* = 0 nodes were not ties. For all 20 of 200 sampled nodes with
c\* = 0 the radius-1774 vx ball contained the *entire* 20k-node graph (0 nodes outside it; >= 8 outside for every
c\* > 0 node), so the "move" was the free global shift - an artifact of the c\* definition on a graph smaller than the
ball, not a property of the solution. The TIE detector (gauge fixed) flags 0 % of that graph. G6b is therefore measured
on graphs large enough that every ball leaves nodes outside it: `sheetlets(100000, p_wrong=0.05, scale=10)` (about
2000 x 2000 x 4100 vx), seeds 2000, 2001, 2002, 1000 non-TIE nodes each; any sampled node whose ball still covers the
whole graph is excluded and counted. Thresholds unchanged (Spearman >= 0.7, proxy > 2 c\* on <= 5 %). The round-1
G6 result is re-labelled "FAIL caused by the c\* definition (whole-graph balls)", not "ties".
