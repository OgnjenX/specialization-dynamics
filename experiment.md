# Module-level compositional generalization

This protocol supersedes the **invalid two-module pilot** below. The original
`composition_*` artifacts and commits `51a2315`, `7cc1986`, `5bdc38d` remain intact.
Their parity-confounded holdouts, neuron-level geometry, absent communicating
specialization, and global max readout prevent testing the present hypothesis.
They are historical diagnostics, not confirmatory evidence. No history is rewritten.

## Hypothesis and scope

An intermediate, sparse, distance-dependent **module graph** may preserve source
specialization while enabling cooperation and improve held-out composition by at
least five percentage points relative to both strongly local and uniform controls.
The tested grid has exponential and power-law shortcut rules, two edge budgets,
and two exact total cable-cost budgets. A cost-matched benefit would support a
better generalization/wiring tradeoff in this grid. Existence outside this grid,
biological wiring laws, and general compositional reasoning remain outside scope.

Béna and Goodman, *Dynamics of specialization in neural modules under resource
constraints*, https://doi.org/10.1038/s41467-024-55188-9, motivate restricted
communication, separate inputs, and post-training digit probes. This is an
**empirically gated reduced symbolic experiment**, not a replication of their
MNIST/EMNIST study or their winner-take-all architecture. Their perceptual inputs
and original specialization metrics/sweeps are not reproduced. Do not describe a
passing gate as replication of the original paper.

## Architecture and task

Six modules occupy coordinates `(0,0)` through `(5,0)` on a line. Euclidean module
distance is `abs(i-j)`: distance 1 is local, 2 medium, and 3–5 long-range. Dense
intra-module recurrence is charged identically across conditions and is excluded
from variable inter-module cost. Modules are atomic areas; there are no neuron
coordinates, geometrically cheap neuron swaps, or distance-dependent transmission
latencies. Geometry concerns cable length, not measured physical tissue.

Each module has 24 tanh RNN units. Only module 0 sees digit A and module 5 sees B.
Other modules have no sensory input. A stimulus is a noisy 10-component one-hot
vector per digit, noise SD .05, sampled anew per example and held constant for six
synchronous time steps. Every outgoing module message has **four hard binary
bits**, from a learned hidden-to-message projection. The same message is sent
along each outgoing graph edge; each receiver has its own learned projection.
A sigmoid surrogate derivative trains the threshold (the forward values remain
0/1). There is no analog cross-module route. Biases and local recurrence are
allowed; initial states are zero and independent of the inputs.

The update is `h_i(t+1) = tanh(W_i x_i + R_i h_i(t) + bias_i +
sum_j A_ji C_ji binary(P_j h_j(t)))`. Information traverses at most one graph edge
per time step. **A single linear readout sees only final module 3.** No other
module produces task logits; no max-confidence selection, pooled hidden state,
shared readout, input skip, normalization across examples, or auxiliary digit loss
exists. The implementation uses independent parameter storage and one recurrent
state per module, avoiding the upstream separately evolved core/comms RNN sum.

Reuse the upstream task: output A when A+B is odd, otherwise B. There is no
hand-coded parity, digit relay, curriculum, or digit supervision in training.
Only the ten-way task loss is optimized.

## Splits and controlled budgets

Hold out `(B-A) mod 10` offsets `(1,2)`, `(3,4)`, `(5,6)` respectively. Each test
partition has 20 ordered pairs and its train complement has 80. The test partitions
are mutually disjoint (training partitions overlap, as repeated-measures designs
do). Each partition has equal counts of each digit in both positions, equal counts
of every target label, and exactly half A-selected/half B-selected examples. Each
branch separately covers all ten targets equally. Tests enumerate all 100 pairs
and prove these statements; balanced digits alone are insufficient.

All connected graph conditions share a bidirectional adjacent-module backbone
(five undirected edges). This guarantees routes from both sources to output and
prevents disconnected draws from masquerading as a distance effect. Conclusions
are conditional on this imposed backbone. Additional edges are bidirectional,
without self edges or duplicate edges, selected among the ten nonadjacent pairs.
Before conditioning on budgets, shortcut admissions are independent with priors:

- Strongly local: probability .5 at distance 2 and zero at greater distances.
- Uniform sparse: probability .35 at every candidate distance.
- Exponential: probability `.6 * exp(-.75*(d-2))`.
- Power law: probability `.6 * (d/2)^(-1.5)`.

Enumerate the 1,024 shortcut subsets and sample their Bernoulli likelihoods
**conditional on the exact requested budget**. Thus the *prior* decays with
distance; realized marginal probabilities change under conditioning. Do not claim
independent final edge admissions or isolate distance from degree effects. The
backbone is deterministic; rare long-range shortcuts are the manipulated factor.

Fixed edge counts: **14 or 18 directed module edges**, including the backbone.
Fixed total cable cost: **22 or 26 module-distance units**, summed over directed
edges. A bidirectional edge of length d contributes 2d. Four parallel binary
channels multiply this cable cost by four; both costs are reported. Exact cost
matching can change edge counts and active parameter counts; it controls cable,
not capacity. Fixed-edge comparisons control active connection capacity but allow
cable cost to vary. All conditions allocate the same dense parameters and execute
the same masked operations; active counts are reported separately. These are
complementary estimands, not a claim that capacity and cable are simultaneously
matched. Two budgets per view provide a small, descriptive Pareto comparison.

Also run the bare local chain and a fully disconnected graph as diagnostics, not
matched-budget controls. The disconnected output has no input and must be a
constant prediction with 10% balanced accuracy. A complete graph is used only for
the baseline gate; it is not an extra final condition.

## Baseline validity gate and development disclosure

Exploration used **only full-pair training/evaluation**, never held-out compositions.
A prototype seed 91 checked 2/4/8-bit communication for 2,000 updates. Two-bit
chain communication learned poorly; four-bit chain communication could reach 100%
with strong source specialization. An eight-seed check (1000–1007) found the
four-bit chain less reliable (mean approximately 91% task accuracy), while complete
four-bit graphs learned approximately 99.9%, with own-minus-other source probe
accuracy approximately .275. These are development observations, not final
baseline or topology evidence. The selected 3,000-update design and gate are
frozen before fresh baseline seeds 2000–2007, distinct from final seeds 0–47.

Run full-pair **complete-graph** reference models first, with eight fresh seeds and
identical architecture/training. Also run eight chain diagnostics. Baseline validity
requires complete-reference mean task accuracy >=95%, each branch mean >=95%,
source specialization >=.10, own-digit probe >=.90, and communication-cut drop
>=.20. At least six of eight seeds must individually have task accuracy >=90%
and specialization >=.10. Chain diagnostics do not determine the gate. This gate
establishes cooperation and specialization coexist in this reduced architecture;
it does not establish that every sparse graph is learnable.

The executable refuses **all final topology runs** unless it recomputes this gate
from the complete raw baseline panel and verifies the same frozen source/protocol
hashes and execution revision. No summary flag override is accepted. A failed gate
means revise/replicate the baseline and preregister a new study, without examining
held-out topology outcomes. Baseline results cannot select topology parameters.

## Training, diagnostics, and uncertainty

CPU float32, one thread per model, AdamW lr .003, weight decay 1e-5, norm clipping
1, 3,000 updates, batch 128; no checkpoint selection or early stopping. Training
pairs are sampled uniformly with replacement. Initialization, sampled minibatches,
and noisy stimuli are paired across conditions within seed/split. A separate RNG
controls graphs, shared across splits within seed (splits repeat the same graph).
Complete baseline and chain also share initialization/stimuli. Up to eight worker
processes (default four) accelerate independent jobs without changing per-model compute.

No final held-out evaluation happens during training. Evaluate 100 fresh draws
per pair, macro-average over pairs, and report both branches for seen and held-out
sets. Communication intervention: recompute dynamics on identical inputs while
cutting all outgoing edges, or only outgoing source-A/source-B edges. Parameters,
initial states, readout, and sensory drives remain fixed. Report all-cut held-out
accuracy/drop, each source-cut drop, and the appropriate branch drop. These are
acute interventions on the trained system and can shift the state distribution;
they establish causal dependence, not an optimal retrained-control comparison.
Cuts are forward arguments and do not mutate or retrain the network.

After task training, detach final states from all 100 pairs, fit fixed ridge .01
linear probes with unpenalized intercept on 30 fresh draws per pair, and test on
30 independent draws. Every module is probed for A and B. Probe labels never enter
task training, model selection, or readout. Using held-pair labels for post-training
diagnostics is disclosed. The primary specialization measure is
`[(acc(0,A)-acc(0,B))+(acc(5,B)-acc(5,A))]/1.8`, normalized by chance .1. Also report
all six module probe pairs and mean absolute module preference. Relays/output may
be generalists; source specialization does not establish specialization of every
module, a cognitive role, or nonlinear information absence.

Report total cable and channel cable, edge count, active/allocated parameters,
local/medium/long edge counts, degree spread, clustering, mean shortest path, and
source-to-output hops. The empirical mean performance/cost frontier is descriptive;
it is not an uncertainty-certified Pareto frontier. Exact-cost contrasts provide
the inferential wiring-cost view.

Use **48 independent paired seeds**, with three splits averaged within each seed
before any interval or contrast. Never treat 144 split measurements as independent
replicates. Prospectively assume paired seed SD .08: Student-t nominal 95% interval
half-width is approximately .0232, below the .025 target. If actual variance is
larger, report failed precision and do not add seeds after inspecting results.
Bonferroni simultaneous 95% coverage for 16 planned candidate/control contrasts
has wider intervals (approximately .036 under this SD). This resolves sizable
advantages; it cannot reliably establish a true effect exactly at the five-point
boundary. Report every seed difference, nominal and simultaneous intervals, and
whether nominal half-width meets the target. This is a prospective precision
argument, not retrospective power or a guarantee about observed uncertainty.

## Support, falsification, and inconclusive outcomes

There are eight distance-dependent candidates (two rules x four budget cells).
Compare each to both local and uniform at its identical budget: 16 contrasts.

- Support in this setup requires the baseline gate; candidate mean seen >=.95,
  held >=.80, specialization >=.10, all-cut drop >=.20; and both comparator mean
  seen >=.95. Against **each** control, the point gain must be >=.05 and the
  simultaneous confidence lower bound >0. Report whether support is in the
  fixed-edge or fixed-cost view; only the latter establishes cost-matched benefit.
- Falsify the stated >=5-point advantage **in this tested grid** only if baseline
  passes and every candidate has learned comparators and an upper simultaneous
  bound <.05 against at least one of its two controls. This rejects the required
  advantage over both controls for every tested candidate, not every possible
  topology or the broad biological hypothesis.
- Otherwise inconclusive. Failed baseline, learning, specialization, absolute
  generalization, cooperation, or precision must be reported explicitly. Ancillary
  failure alone cannot establish a precisely falsified topology advantage.

All raw baseline/final runs and exact graph masks are retained as small CSV files;
configurations record splits, protocol, environment, revision and hashes. Summaries
are regenerated from complete raw panels; missing/duplicate/unplanned runs are
rejected. No model/data blobs, hidden tuning, omitted failed seeds, or result-based
selection. Commit protocol/code/tests **before baseline and final runs**. The
baseline/final configs identify that commit. Later changes to numerical code or
thresholds require a disclosed new study; reporting edits may follow results.

## Reproduce

Use the existing pinned Python 3.11.9 environment (`uv sync --locked`). For the
recorded source, check out the revision in `module_baseline_config.json` or
`module_final_config.json` in a separate checkout. Fresh prefixes preserve results.

```sh
uv run --locked python -m unittest -v test_composition
uv run --locked ruff check --isolated dynspec/composition.py test_composition.py
uv run --locked ruff format --isolated --check dynspec/composition.py test_composition.py
uv run --locked python -m dynspec.composition --stage baseline --output reproduced_baseline --workers 4
uv run --locked python -m dynspec.composition --stage final --baseline reproduced_baseline --output reproduced_final --workers 4
git diff --check
```

The existing historical artifacts retain their original meaning and numbers. The
old command without `--stage` now runs the new baseline gate, not the invalid
pilot. To reproduce the pilot, use its historical commit in a separate checkout.
Numerical identity across library versions or machines is not guaranteed.

## Historical invalid pilot (2026-10-01)

Everything below describes the superseded pilot and its original decision rule.
Its original "inconclusive" statistic is retained for audit, but its design is
invalid for the revised hypothesis; passing its learning gate did not validate it.

# Compositional generalization experiment

Protocol frozen before final runs; see the protocol commit in git history. Upstream:
Gabriel Béna and Dan F. M. Goodman, *Dynamics of specialization in neural modules
under resource constraints*, https://doi.org/10.1038/s41467-024-55188-9;
code baseline 520dd88183a6fcf51ae6995956829ea10c2d906c.

## Hypothesis and scope

At equal communication budget, exponentially distance-biased wiring improves
accuracy on unseen digit combinations by at least 5 percentage points over
uniform sparse wiring, while retaining specialization and reliance on cooperation.
This tests one synthetic operationalization, not all sparse networks or biological
wiring. Geometry is imposed on exchangeable hidden units, not semantic inputs.

Reuse upstream Community (two RNNs summed, analog communication, two local linear
readouts, upstream max-confidence decision) and exact parity-digits target: return
A if A+B is odd, otherwise B. Inputs are ten-dimensional one-hot digit identities
with independent Gaussian noise SD .05, repeated for four timesteps with fresh
noise. No digit supervision, auxiliary loss, hard-coded composition, or joint
readout. Analog communication is an upstream option; the notebook uses binary.

## Fixed design

CPU float32, one thread, hidden size 64/module, AdamW lr .003, weight decay 1e-5,
1,500 updates of 128 examples, no early stopping/checkpoint selection. Seeds 0–4;
three fixed split offsets (1,3), (2,4), (5,7). Hold out all ordered pairs satisfying
(B-A) mod 10 in the split offsets: 20 test pairs, 80 train pairs. Each digit occurs
in both positions in both sets, and all output labels are represented. Sample
training pairs uniformly with replacement. Evaluate 100 independent noisy draws
per pair, reporting pair-macro accuracy. No final test evaluation during training.
Initialization and minibatches are paired between conditions; masks use an
independent RNG. Same optimizer, dense allocated parameters, active intra-module
connections, input/readout masks, updates and examples in primary conditions.

Uniform: 409 unique edges per directed 64x64 inter-module block (upstream uniform
sampling without replacement, floor(.1*64²)). Distance: exactly 409 per block,
sampling without replacement with weights exp(-d/.2). Each module has neurons on
a unit interval, fixed independently shuffled coordinates per seed; d includes a
fixed inter-module gap .1: sqrt((x_i-x_j)^2+.1^2). Geometry/scale are never tuned.
Report realized mean edge length; degree distributions are not controlled, so
inferences concern the whole topology rule, not distance independently of degree.
Disconnected diagnostic uses zero communication edges; it is not budget matched.
No dense or power-law primary condition: the claim is relative to uniform sparse.

First run a reduced upstream sanity check on all 100 pairs, uniform .1 versus
fully connected communication, same training budget and seeds, with no combination
holdout. This is not a replication of the MNIST paper or its complete sweeps.

## Measurements and decision rule

Use the final model. Report seen-pair and held-out pair accuracies. Cooperation:
held-out accuracy loss after zeroing communication masks (restore afterward;
post-training intervention, potentially distribution-shifting). Specialization:
fit fixed ridge linear digit probes (ridge .01 with intercept) on detached final
states using 20 fresh noisy draws/all 100 pairs, evaluate on independent 20 draws.
For each module measure own-digit accuracy minus other-digit accuracy; average the
two. Probes are diagnostic only, never feed task training. Their use of all pairs
is disclosed and does not change the task predictor.

Average three split results within each seed; five seed means are independent
units. Primary uncertainty: paired mean distance-minus-uniform accuracy difference,
two-sided 95% Student-t interval (df=4). Splits are repeated measurements, not 15
independent replicates. Thresholds are set now, before results:

- Valid learning gate: both primary conditions mean seen accuracy >= .95.
- Support requires the gate; lower paired CI > .05; distance mean held-out >= .90;
  distance mean specialization >= .10; distance mean cooperation drop >= .10.
- Falsify the >=5-point advantage in this setup if the gate passes and upper CI
  < .05 (regardless of ancillary metrics). This does not falsify all wiring rules.
- Otherwise inconclusive; explicitly report failed learning/ancillary gates.

All raw run-level measurements, exact splits, environment/configuration, and a
machine-readable decision will be saved in small text files. No hyperparameter
selection using held-out performance. Test repairs may change code correctness,
but any protocol change after results requires a disclosed amendment/new study.

## Results (2026-10-01)

Protocol: commit `51a2315`; execution code: `7cc1986`. All 55 prescribed runs
completed (10 full-pair sanity checks, 45 held-out runs). No protocol amendments,
extra seeds, tuned geometry, checkpoint selection, or omitted runs. Later code
edits only format equivalent expressions; a representative distance run reproduced
all recorded numerical values exactly after formatting.

| Condition | Seen pairs | Held-out pairs | Own probe | Other probe | Specialization | Communication ablation drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform sparse | 99.996% | 39.210% | 100.000% | 100.000% | 0.00000 | 13.653 pp |
| Exponential distance | 99.997% | 41.017% | 100.000% | 99.998% | 0.00002 | 11.947 pp |
| Disconnected diagnostic | 64.376% | 13.653% | 100.000% | 9.703% | 0.90297 | 0.000 pp |

Distance-minus-uniform held-out gain: **+1.807 percentage points**, 95% paired
Student-t interval **[-8.732, +12.345] pp**. Seed-level differences after averaging
splits: +6.317, -5.617, -7.333, +2.517, +13.150 pp. Learning gate passed; joint
ancillary gate failed (held-out performance <90%; specialization <.10).
Machine-readable preregistered decision: **inconclusive**.

The hypothesis is **not supported in this setup**. The connected models fit the
seen combinations nearly perfectly but fail the absolute generalization and
specialization criteria. The interval is too wide to either establish or falsify
the specific >=5-point distance advantage. Calling that advantage falsified would
misstate the evidence. Post-training ablation supports reliance on communication,
and the disconnected control has high input specificity but low task performance;
none of these conditions demonstrates the desired three-way combination.

Full-pair baseline accuracy was 99.998% for both .1 and 1.0 connectivity, with
specialization 0.00000 and -0.00005 respectively; communication ablation drops
were 42.228 and 51.322 pp. These runs establish task learnability using upstream
machinery, **not replication of the paper's specialization findings**. The original
MNIST/EMNIST notebook and parameter sweeps were not run. The reduced symbolic
check removes perception and uses analog instead of notebook binary communication;
its lack of specialization is a substantive limit on external validity.

Mean inter-module edge length was .36785 (uniform) versus .21498 (distance), a
41.6% reduction, so the geometry manipulation worked. Both use 818 edges and
40,468 allocated parameter entries. Upstream core and communication Parameter
objects share underlying weight storage (21,524 unique stored scalar weights,
with separate optimizer entries); this behavior is retained identically across
conditions. Counts of dense allocated entries are not counts of active edges.

Additional limits: five seeds give low precision; the three circular-offset
splits cover one family of compositions; geometry is arbitrary at hidden-unit
level and intra-module recurrence remains dense. Edge counts, not summed cable
length or degree distributions, are matched. Distance and degree-pattern effects
cannot be separated. Probes may saturate and only assess linear decodability;
no causal claim about specific digit-specialist modules follows. The upstream
max-confidence decision sees both local readouts, so communication is not a
structural necessity imposed by the readout. Ablation shifts learned dynamics;
its drop is not a standalone proof of useful compositional cooperation. Noise
draws are repeated observations, not additional independent seeds. These results
do not rule out other spatial scales, budgets, tasks, or learning mechanisms.

## Reproduce and verify

Run from the repository root. Recorded environment: macOS arm64, Python 3.11.9,
PyTorch 2.13.0, NumPy 2.3.4, SciPy 1.16.2, CPU float32, one thread. No CUDA or
external datasets are required. Numerical identity across other platforms or
library builds is not guaranteed.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . torch==2.13.0 numpy==2.3.4 scipy==1.16.2
python -m unittest -v test_composition
python -m dynspec.composition --output reproduced
uvx ruff check --isolated dynspec/composition.py test_composition.py
uvx ruff format --isolated --check dynspec/composition.py test_composition.py
git diff --check
```

The fresh output prefix protects existing results. Tracked `composition_runs.csv`
contains every run; `composition_config.json` records the execution revision,
script hash, environment, seeds and explicit train/test pairs;
`composition_summary.json` contains means, paired differences, uncertainty and
decision. For exact historical source, check out `7cc1986` in a separate checkout
before running. Subsequent formatting does not alter the experiment.

Validation: six integrity/decision tests passed; independent recalculation from
all 55 CSV rows matched the summary interval and budgets; one distance run
(seed 0, split 0) repeated with exact numerical agreement. New experiment files
passed Ruff checks/format verification and the branch passed whitespace checks.
No pretrained model blobs or large generated artifacts are tracked.
