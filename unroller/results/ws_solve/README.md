# WP3 exact L1 winding solver: results

Code: `unroller/ws/solve.py` (solver, contraction, slab pipeline, chirality, synthetic graphs),
`unroller/ws/confidence.py`, tests `tests/ws/test_solve.py`, benchmark `scripts/ws_bench_solve.py`.
Gates were pre-registered in `gates.md` (commit 4096fe5) before any measurement; the c* definition and the formal
synthetic configs were recorded (5ac7689) after development on seeds 0-3 and before the formal runs on seeds 1000+.
Machine: 4-core container, 1 CPU (`taskset -c 0`, OMP_NUM_THREADS=1) for the scale runs; backend BK (PyMaxflow 1.3.2)
unless stated.

| gate | result | verdict |
|---|---|---|
| G1 exactness | 60/60 lattices (12^3; 5 %, 20 %, wall; 20 seeds) certified, E(sol) <= E(truth) in 60/60 | PASS |
| G2 overflow | weights 1e6 x degree ~20: scale auto-reduced, max node capacity < 2^31-1, optimum equals the unit-weight one x 1e6; an over-limit capacity raises OverflowError; a max-flow that returns 0 (scipy's silent failure) raises via the cut cross-check | PASS |
| G3 elimination | 200 random graphs with anchors (+ priors on half): eliminated == big-M energy in 200/200, == brute force (<= 4 free nodes, range +-6) in all enumerable cases | PASS |
| G4 warm start | 20 lattices: tree, IRLS-L1 and perturbed-optimum starts reach the identical certified integer energy | PASS |
| G5 stop flag | max_iter=1 -> stopped_by 'max_iter', not certified; else 'certificate' | PASS |
| G6 confidence | 20k sheetlet graph (seed 1000), 1000 nodes: Spearman(proxy, c*) **0.802**; proxy > 2 c* on **10.0 %** (gate <= 5 %) | **FAIL**, caused by the c* definition (see below) |
| G6b (round 2) | 1e5 graphs, seeds 2000/2001/2002, 1000 non-TIE nodes each: Spearman 0.993 / 0.996 / 0.996; proxy > 2 c* on 0.0 / 0.1 / 0.1 %; TIE share 0 / 0 / 0; whole-graph balls 0 | PASS |
| G7 chirality | 20k graph (seed 1001): correct sign; total-energy ratio **1.096** (gate >= 1.5); informative-edge ratio 8.40 on 385 seam edges | **FAIL as registered** |
| G7b (round 2, post-hoc gate) | seeds 2100-2104 (p_wrong 5,5,5,10,10 %): sign correct 5/5; informative ratio 11.3 / 3.29 / 6.00 / 6.36 / 4.22 on 334-397 seam edges; total ratio 1.04-1.08 | PASS |
| G8 BK backend | identical certified integer energies to scipy on all lattice tests and a 1e5 sheetlet graph; 4.1x faster (29 s vs 120 s) | PASS, BK accepted |
| G9 scale | 1e6 nodes / 1.06e7 edges (6.9 % wrong incl. walls), stitched start: global polish **141 s, 3 iterations, certified**, 2.8 GB peak; whole slab->block->polish pipeline 24.0 min (8 slabs 21.5 min). Cold (spanning-tree start) for comparison: 3252 s, 48 iterations, the identical certified energy 917424.079 and accuracy 0.999969 | PASS |
| G10 accuracy | 1e5 graphs, p_wrong 3/5/10 % (+ walls): stitched 0.99955 / 0.99948 / 0.99840, polished 0.99995 / 0.99965 / 0.99885, cold = polished energy; 1e6: polished 0.99997; E(sol) < E(truth) everywhere | PASS |

**G6 failure - corrected diagnosis.** The round-1 write-up attributed the 9.9 % of nodes with c* = 0 to ties. That was
wrong: every one of them had a c* ball (radius 1774 vx) that held the *whole* 20k-node graph, so the "move" was the
free global shift - an artifact of the c* definition on a graph smaller than the ball. The gauge-fixed TIE detector
(`confidence.ties`, union of zero-cost +-1 moves from the certificate's final max-flows, scipy residual reachability =
maximal min-cut side) flags 0 % of these graphs. Re-measured as pre-registered (G6b, amended before measuring) on 1e5
graphs where no ball covers the graph, the proxy is tight: Spearman >= 0.993, loose (> 2 c*) on <= 0.1 %. The TIE path
is exercised by a unit test (equal-weight conflicting edges); the synthetic sheetlet graphs contain no genuine ties.

**G7 failure as registered.** Mirroring the handedness changes only the offsets of edges that cross the angular seam;
noise and walls cost the same under both. The total-energy ratio is 1 + (seam evidence)/(noise energy) -> 1 on large
graphs (~1.1 at 20k nodes, smaller at whole-scroll scale) even when the seam evidence is unanimous. `choose_chirality`
reports both and decides on the ratio restricted to the informative edges (each build at its own optimum).

**Iteration counts / cost.** BK move times grow from 0.06 s to ~1.4 s per move on 1e5 nodes as the solution
converges; scipy's Dinic is 4-6x slower. Cold starts from a spanning-tree propagation need 13-57 iterations
(tree-init accuracy 7-13 %); stitched starts 2-3.

**Backend.** BK = PyMaxflow 1.3.2 from PyPI (`pip install PyMaxflow==1.3.2`; wheel cp313 manylinux_2_17_x86_64).

**Quantisation.** scale 1e6 on these graphs; per-solve quantisation error sum|w*scale - round| / scale = 2.65 for
1.06e7 edges; energy bound 0.20 on an optimum of 917 424 (2e-7 relative).
