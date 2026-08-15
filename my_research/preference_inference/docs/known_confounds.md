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

**Update 2026-08-14 (normalization scheme decided):** Confirmed by
simulation (`analyze` scripts + `q_normalization_simulation.json`) that
this range mismatch bites hardest under tanh normalization. Using A-1's
own real-data statistics (μ=1.9233, σ=0.9466, from `self_vision` +
8-probe critic evaluation) as the shared normalization source (this is
"self's own experience used as the yardstick," not a pooled A-1+A-2
statistic — keeps the superposition claim intact):

|                     | A-1 normalized std | A-2 normalized std | A-2 variance retained |
|---|---|---|---|
| tanh((Q-μ)/σ)       | 0.629               | 0.0090              | ~5.5% |
| (Q-μ)/(3σ), clipped | 0.333               | 0.0581              | ~35% |

**Decision: tanh, not the linear clip.** R3 only exercises process-1 (A-2
is a zero vector there), so A-1's own normalized variance is what R3's
outcome actually depends on — and tanh preserves nearly 2x more of it
(0.629 vs 0.333) than the linear clip, which maps ±3σ to ±1 and leaves
most real data compressed into a ±0.33 band, far short of how the
original paper's `m_t` used the full (-1,1) range. **A-2's collapse under
tanh (std=0.0090) does not block R3** (process-2 is zero regardless of
normalization scheme there) **but is a real open question for R4**, where
VE has to produce a value for process-2 that lands somewhere usable in
this same tanh-normalized space. The ~5.5%-variance-retained figure is
the number to check against once VE exists: if VE's raw output range
can't be usefully distinguished after passing through this normalization,
R4 will need one of: retraining A-2's critic for a wider natural Q range,
revisiting the reward design so Q separates more across A-2's operating
region, or (least preferred, since it reopens this whole tradeoff)
switching just A-2's downstream normalization path to something gentler
than tanh at the cost of breaking the "same normalization for both
processes" property.

**Revisit at:** R4, when VE's output actually needs to pass through this
normalization for the first time.

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

**Update 2026-08-14 (R2 results in):** h²→A-2-position came out *fine*
(0.8811, exceeding exp3's own 0.7122) — the band-shaped distribution did
not, in fact, hurt this regression. The actual problem R2 surfaced was
different and unrelated to position precision at all: see "A value-driven
policy structurally starves the Motion Generator" below.

## A value-driven policy structurally starves the Motion Generator

**Raised:** 2026-08-14, after diagnosing R2's gate failure.

R2 (A-2's Cycler replaced by the trained RL Green-policy, architecture
otherwise identical to exp3, MG present and directly wired into SM's
process-2 input per the original paper) failed its gate: h²→self = 0.3421
(need <0.15), though h²→other = 0.8811 comfortably passed.

**Root cause, confirmed by direct measurement, not inference:** A-2's RL
policy converged to `other_motion` with std_y = 0.0154, vs. A-1's own
reference std_y = 0.678 and exp3's Cycler-driven A-2 std_y = 0.690 — a
~44x collapse, mean_y = -0.988 (i.e. essentially constant near-maximal
speed toward Green). Two hypotheses were checked and ruled out first: A-1/
A-2 position correlation (measured near-zero, <0.003 R² ceiling, nowhere
near the observed 0.34-0.49 leakage) and a regional/localized MG failure
(error is uniformly moderate across A-2's entire real operating region,
not concentrated in specific cells). The motion-variance collapse is the
only measured factor that actually explains the failure: MG has almost no
y-axis signal to learn from, produces inaccurate `om` fed straight into
SM (exp3's direct wiring, unlike v3's Approach-B design), and h² falls
back on directly-visible A-1 features instead of representing A-2 — this
also explains why *vision reconstruction* losses were simultaneously
**better** than exp3's (self_vision 11.65 vs 16.79) while *self/other
separation* got worse: falling back on A-1's own directly-visible features
is a genuinely easier reconstruction target.

**Why this isn't a bug to fix — it's a structural property worth stating
plainly:** A SAC agent that has converged to a good policy is *supposed*
to move purposefully toward whatever maximizes reward. A-2 heading
straight for Green is the policy working correctly, not failing. Cycler's
omnidirectional motion (which let exp3's MG learn) came from an entirely
different source: a scripted agent with no value function at all, cycling
between arbitrary targets. There is a real tension between "motion serves
value" (what an RL agent is trained to do) and "motion is directionally
diverse enough to be predictable by an unsupervised observer" (what MG
needs) — a value-maximizing policy will generically *reduce* its own
motion entropy as it improves, which is close to the opposite of what a
motion-generation objective needs. This is worth stating as a finding in
its own right when writing this up: extending the original paper's
architecture from motion-observation to value-observation isn't a
drop-in substitution, because motion and value have structurally
different relationships to behavioral diversity.

**Why it doesn't block progress:** R3 removes MG from the architecture
entirely (process-2 input becomes a zero vector — §5.4), and R4 replaces
it with VE, which reads value directly rather than trying to predict
motion. Neither depends on MG converging. R2's job — telling apart "the
v3 h² collapse was caused by Q-value stuff" vs. "caused by A-2's policy"
— is done: swapping only A-2's policy took h²→self from 0.8812 (v3) to
0.3421 (R2), and R2's remaining gap traces to MG specifically, which won't
exist in R3+.

**Status:** treating R2 as a conditional pass on this basis (2026-08-14
decision) and proceeding to R3 rather than iterating further on A-2's RL
policy. Recorded here in case it needs to be cited later, e.g. if a
future stage reintroduces a motion-prediction objective and hits the same
wall.

**Revisit at:** would only matter again if a future design reintroduces
a motion-generation objective fed by a value-driven agent's actions.

## Fig.5 (viewpoint-taking) reproduction: corrected after root-repo audit

**Raised:** 2026-08-14/15, R4-prerequisite check (independent of the R3
gate itself).

**Original (incorrect) conclusion:** an ad-hoc reproduction attempt
(`train_fig5_decoder.py`, since discarded) trained a fresh
`VisionDecoderModule` on top of frozen exp1_l1 vision encoders, with a
plain L1 autoencoder objective on `self_random_other_stay` (~100k steps,
paper's batch=10). This was reported as failing to reproduce the paper's
claim (other_other=0.5506 not < other_self=0.3536).

**Correction (2026-08-15):** the root repo (outside `my_research/`) was
not audited before attempting this from scratch. It turns out Fig.5 is
already fully reproduced by the *original* pipeline:
`config/exp/exp2.yml` (`model.name: Autoencoder`, a distinct class in
root `model/model.py`, not `VisionDecoderModule`), trained via
`run_training.sh` with **weight_decay=3.0** (~1000x a typical value —
evidently a deliberate regularization choice to force the decoder to
generalize from Encoder-1's output to Encoder-2's, rather than overfit
to Encoder-1 alone) and evaluated on the **`grid`** dataset (self and
other exhaustively placed on every cell of a 21x21 grid, not a natural
random trajectory) via `analyze/analyze_vpt.py` (root-level, called from
`run_analysis_exp2.sh`). The precomputed result is sitting at
`data/result/exp2/0/test/grid/save/vpt/1/histogram/result.txt`:

| | mean | std |
|---|---|---|
| self_true_and_self_rec | 0.0513 | 0.0170 |
| other_true_and_other_rec | 0.1125 | 0.0366 |
| self_true_and_other_rec | 0.1393 | 0.0400 |
| other_true_and_self_rec | 0.1465 | 0.0506 |

`other_other (0.1125) < other_self (0.1465)` — **the paper's claim holds**
in the original repo's own pipeline. The ad-hoc reproduction's failure and
~5x larger absolute errors were a methodology gap (wrong model class,
missing the heavy weight_decay, wrong eval dataset), not a real property
of this codebase's encoders.

**R4 implication retracted:** the earlier note that Encoder-2's output
(`ov_enc`) might carry too little A-2-specific information to decode
usefully is withdrawn — for **exp1_l1's** encoders, A-2 viewpoint
information is confirmed present and decodable. Whether the same holds
for **R3-A's own** (retrained-from-scratch superposition_module, still
using exp1_l1's frozen vision encoders) is a separate, still-open
question, since R3-A's SM was trained on Probe-Q not motion input — the
vision encoders themselves are unchanged/frozen through R3-A, so this is
expected to still hold, but hasn't been directly checked against R3-A's
own checkpoint.

**Assets now protected from deletion** (do not remove/regenerate):
`data/result/exp2/` (~10GB, root repo) and the `grid` dataset under
`data/data/` — required to reproduce Fig.5d/5e. Also: `exp1_l1_1000`,
`exp1_mse_1000`, `exp3_1000` configs and `run_analysis_1000.sh` (root,
dated 6/20-21 — an undocumented but likely self-authored variant
pipeline) are no longer delete candidates; provenance unconfirmed but
probably prior work by the same researcher, not paper-original material
to discard as irrelevant.

**Status:** applying the real exp2/vpt methodology to R3-A's own
checkpoint is deferred — not urgent, since R3's pass/fail gate is
h¹→self's R², and Fig.5-equivalent viewpoint-taking is only an R4
prerequisite check, not an R3 criterion. Revisit once R3-A's final
(corrected-weight-transfer) results are confirmed.

**Revisit at:** before R4, if VE's decoded output needs a Fig.5-style
qualitative/quantitative viewpoint-taking check — at that point, prefer
adapting `analyze_vpt.py` / the `Autoencoder`+grid-dataset approach over
another ad-hoc implementation.
