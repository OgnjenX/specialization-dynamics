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
