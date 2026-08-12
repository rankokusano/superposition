# Known confounds / open issues (v4)

Running list of design concerns that are known at the time a decision is
made but deliberately not blocking progress. Not a task list — see the
v4 instruction doc for the gated experiment plan (R2/R3/R4/R5).

## A-1's green-aversion may structurally suppress R5's self-projection signal

**Raised:** 2026-08-12, during A-2 reward redesign discussion (§5.2).

A-1's reward (`train_rl_v3.py`) is red-approach (+1.0) **and green-avoidance
(-1.0)**. A-2's reward, as of the symmetric redesign below, mirrors this:
green-approach (+1.0) and **red-avoidance (-1.0)**.

R5 ("self-projection bias") asks whether VE's Q̂² correlates first with
A-1's own (Red-based) value structure early in training, then shifts to
correlate with A-2's true (Green-based) value structure later — the
self-to-other developmental shift reported in Repacholi & Gopnik-style
false-belief-adjacent literature.

**The concern:** because A-1 experiences Green with a strong *negative*
reward, the Shared Module may learn "Green = bad" as a general feature of
its own (self) value representation. If so, VE — which reads from the
Shared Module's representation of the other's visual scene — may find it
structurally difficult to output a *positive* value for Green, regardless
of how well it has come to represent A-2's true preference. This would
show up as a ceiling/floor effect that looks like "no self-to-other
transition happened" even if the underlying representation did shift,
confounding the R5 measurement.

**Status:** not addressed. Proceeding with the symmetric reward design
(A-1: Red+/Green-, A-2: Green+/Red-) per explicit instruction, because:
- it keeps A-1 and A-2 built from the same reward *principle* (vision
  pixel threshold, not privileged position), which the current design
  work depends on;
- fixing it now (e.g. dropping A-1's Green-aversion) would mean
  retraining A-1, which currently behaves correctly (see the position
  spread probes in this same discussion) — a second simultaneous change,
  which the "one change per gate" rule is specifically meant to avoid.

**Revisit at:** R5. If the self-to-other shift fails to appear, this is
one candidate explanation to check before concluding the hypothesis is
falsified — e.g. by probing whether VE's output is systematically
compressed/clipped on the positive side, or by an ablation that retrains
A-1 without the Green-aversion term for comparison.

## A-1's and A-2's Q-value ranges differ substantially between training runs

**Raised:** 2026-08-12, after the 3-seed A-1 retrain (continuous reward,
§5.2 revision).

Probe-Q spatial map (K=8 directional probes, 10x10 grid,
`analyze/probe_q_spatial_map.py`), same continuous reward principle for
both agents:

| | Q range | notes |
|---|---|---|
| A-1 seed0 (adopted) | [0.075, 4.032] | healthy critic |
| A-1 seed1 | [-2.831, 3.422] | also healthy, different range/sign entirely |
| A-2 (pre-3-seed-retrain) | [3.380, 5.219] | healthy critic |

Seed0 and seed1 are **both healthy A-1 critics** trained with the identical
reward, architecture, and hyperparameters — only the seed differs — yet
their Q ranges don't even share the same sign convention (seed0 is
strictly positive, seed1 spans negative to positive). This means Q's
absolute scale/offset is not a reproducible property of the reward design;
it's set by arbitrary details of each SAC run (critic initialization,
replay buffer composition early in training, etc).

**Why this matters:** §5.3's probe-Q-vector design normalizes Q(s, a_probe)
before feeding it into the Shared Module as `m_t`'s replacement. The
normalization statistics (μ, σ) have to come from *somewhere* — either a
shared statistic across process-1 and process-2, or a separate statistic
per process. Given the ranges above:
- **Shared statistic**: superposition's own premise (the Shared Module
  interprets process-2 using process-1's mechanism) argues for this, but
  if A-1 and A-2's raw Q ranges don't overlap much, whichever side has the
  narrower/offset range gets compressed into a small corner of the
  normalized space (or clipped, if using tanh) — effectively the same
  failure mode that produced v3's h² collapse, just moved from position
  space to Q space.
- **Per-process statistic**: keeps both usable, but then the Shared Module
  is no longer interpreting process-2's Q the same way it interprets its
  own — and A-2's normalization constants would have to be computed from
  data A-1 could never have access to, which weakens the "self mechanism
  interprets other" claim the whole project rests on.

**Status:** tried retraining A-2 across 3 seeds to find one whose range
naturally lands close to A-1 seed0's. Result: only 1/3 A-2 seeds produced a
healthy (non-collapsed) critic (seed0: range [3.118, 4.380], spatial
variance 0.082 — about 1/10th of A-1 seed0's 0.842); the other two
collapsed outright. The one usable A-2 seed still doesn't overlap A-1's
low end and has a much flatter gradient. A root-cause investigation (color
detection code diff, actual rendered RGB at all 4 corners, cross-agent
avatar-color contamination, full red+green pixel-vs-distance scans from
both camera roles, initial-pose/orientation check) found **no environmental
or rendering asymmetry between red and green** — the pixel-count-vs-distance
curves for the two colors are exact mirror images of each other
(green(y) = red(-y) along the shared wall, confirmed identical from both
camera roles). Current read: this is SAC training variance (also visible
within A-1 alone, seed0 vs seed1 above) rather than a fixable structural
cause, and RL-side tuning has been deprioritized (2026-08-12) since Q-value
generation is a means to an end for this project, not the research target.
Canonical models: A-1 seed0, A-2 seed0 (the only healthy A-2 run) — adopted
as-is despite the range mismatch.

**Revisit at:** R3 (probe-Q-vector implementation), when a normalization
scheme has to be picked concretely. The range mismatch documented above is
expected to still be present at that point and will need an actual decision
(shared vs. per-process normalization stats), not just more seed-hunting.

## Convergence asymmetry: A-1 reaches its landmark far more precisely than A-2

**Raised:** 2026-08-12, same investigation as above.

With the healthy canonical models (continuous reward, stochastic rollout,
ε=0.1), A-1 lands within 1.5 units of Red 65.5% of the time; A-2 lands
within 1.5 units of Green only 1.8% of the time (though 80.6% within 3.0 —
A-2 sits in a band around Green rather than converging on the exact
corner). The root-cause check above ruled out environmental asymmetry as
the explanation (color detection, rendering, avatar cross-contamination,
and the distance-vs-pixel-count curves are all provably symmetric between
red and green).

One partial, unresolved observation from that same check: scanning green
pixel count along the shared Red/Green wall (x=-9), the count *peaks at
y=-7 (121px), not at the landmark's actual coordinate y=-9 (96px)* — the
visual/reward peak is offset from the true geometric corner. This alone
would explain an RL agent settling in a band near-but-not-at the corner.
But the same offset pattern exists on the red side by symmetry (already
confirmed: green(y) = red(-y) exactly), and A-1 *does* reach within 1.5 of
its own true corner 65.5% of the time despite it — so the visual-peak
explanation doesn't actually distinguish A-1 from A-2. Why the same
peak-offset "trap" catches A-2 but not A-1 is unresolved.

**Status:** not investigated further per 2026-08-12 decision to stop RL
tuning and proceed to R2. R2 doesn't depend on Q-values or on A-2 reaching
the exact corner (its gate criteria are about h²→position R², not position
precision directly), so this isn't blocking. Left here so it isn't
forgotten.

**Revisit at:** once R2's actual results are in — if h²→A-2-position comes
out fine despite A-2's band-shaped (not point-shaped) distribution, this
confound may turn out not to matter for the regression-based gates at all,
and R3+ (where Q-value precision starts to matter more directly) is a more
useful place to dig further if needed.
