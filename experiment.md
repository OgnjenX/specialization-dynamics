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
