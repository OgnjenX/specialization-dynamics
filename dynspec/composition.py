"""Fixed-budget, held-out digit composition experiment; protocol in experiment.md."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from torch.nn import functional as F

from dynspec.decision import get_decision
from dynspec.models import Community
from dynspec.tasks import get_single_task

SPLITS = ((1, 3), (2, 4), (5, 7))
CONDITIONS = ("uniform", "distance", "disconnected")
HIDDEN = 64
STEPS = 1500


def pairs_for_split(offsets):
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    held = torch.tensor([int((b - a) % 10) in offsets for a, b in pairs])
    return pairs[~held], pairs[held]


def topology(condition, seed, density=.1):
    rng = np.random.default_rng(seed + 10000)
    coordinates = np.stack([rng.permutation(np.linspace(0, 1, HIDDEN)) for _ in range(2)])
    mask = torch.zeros(2 * HIDDEN, 2 * HIDDEN)
    lengths = []
    for receiver, sender in ((0, 1), (1, 0)):
        distance = np.sqrt((coordinates[receiver, :, None] - coordinates[sender, None, :]) ** 2 + .1 ** 2)
        k = 0 if condition == "disconnected" else int(density * HIDDEN ** 2)
        weights = np.exp(-distance.ravel() / .2) if condition == "distance" else np.ones(HIDDEN ** 2)
        edges = rng.choice(HIDDEN ** 2, k, replace=False, p=weights / weights.sum())
        block = torch.zeros(HIDDEN ** 2)
        block[edges] = 1
        mask[receiver * HIDDEN:(receiver + 1) * HIDDEN, sender * HIDDEN:(sender + 1) * HIDDEN] = block.reshape(HIDDEN, HIDDEN)
        lengths.extend(distance.ravel()[edges].tolist())
    return mask, float(np.mean(lengths)) if lengths else 0.


def make_model(condition, seed, density=.1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask, length = topology(condition, seed, density)
    config = {
        "input": {"input_size": 10, "common_input": False},
        "modules": {"n_modules": 2, "hidden_size": HIDDEN, "n_layers": 1,
                    "dropout": 0., "cell_type": str(torch.nn.RNN)},
        "connections": {"sparsity": density, "binary": False, "mask": mask},
        "readout": {"output_size": 10, "common_readout": False},
    }
    return Community(config), length


def inputs(pairs, rng):
    base = F.one_hot(pairs, 10).float().reshape(-1, 20)
    return base.unsqueeze(0).repeat(4, 1, 1) + .05 * torch.randn(4, len(pairs), 20, generator=rng)


def predict(model, x):
    output, states = model(x)
    return get_decision(output, "last", "max")[0], states[-1]


@torch.no_grad()
def accuracy(model, pairs, seed, repeats=100):
    rng = torch.Generator().manual_seed(seed)
    expanded = pairs.repeat_interleave(repeats, 0)
    target = get_single_task("parity-digits", expanded, 10)
    correct = []
    for p, y in zip(expanded.split(256), target.split(256)):
        out, _ = predict(model, inputs(p, rng))
        correct.extend((out.argmax(-1) == y).tolist())
    return np.asarray(correct).reshape(len(pairs), repeats).mean(1).mean().item()


@torch.no_grad()
def specialization(model, seed):
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10)).repeat_interleave(20, 0)
    states = []
    for offset in (0, 1):
        rng = torch.Generator().manual_seed(seed + offset)
        states.append(torch.cat([predict(model, inputs(p, rng))[1] for p in pairs.split(256)]))
    probe_acc = []
    for module in range(2):
        train = states[0][:, module * HIDDEN:(module + 1) * HIDDEN]
        test = states[1][:, module * HIDDEN:(module + 1) * HIDDEN]
        train = torch.cat((train, torch.ones(len(train), 1)), 1)
        test = torch.cat((test, torch.ones(len(test), 1)), 1)
        penalty = torch.eye(HIDDEN + 1) * .01
        penalty[-1, -1] = 0
        acc = []
        for digit in (module, 1 - module):
            labels = F.one_hot(pairs[:, digit], 10).float()
            weight = torch.linalg.solve(train.T @ train + penalty, train.T @ labels)
            acc.append(((test @ weight).argmax(-1) == pairs[:, digit]).float().mean().item())
        probe_acc.append(acc)
    return float(np.mean([a - b for a, b in probe_acc])), probe_acc


def run(condition, seed, split, baseline=False, density=.1):
    model, length = make_model(condition, seed, density)
    train, test = pairs_for_split(() if baseline else SPLITS[split])
    if baseline:
        test = train
    rng = torch.Generator().manual_seed(20000 + seed * 100 + split)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-5)
    for _ in range(STEPS):
        batch = train[torch.randint(len(train), (128,), generator=rng)]
        out, _ = predict(model, inputs(batch, rng))
        loss = F.cross_entropy(out, get_single_task("parity-digits", batch, 10))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    eval_seed = 30000 + seed * 100 + split
    seen = accuracy(model, train, eval_seed)
    held = accuracy(model, test, eval_seed + 1)
    spec, probes = specialization(model, eval_seed + 2)
    comm_mask = model.comms.parametrizations.weight_hh_l0[0].mask
    saved = comm_mask.clone()
    comm_mask.zero_()
    ablated = accuracy(model, test, eval_seed + 1)
    comm_mask.copy_(saved)
    return dict(stage="baseline" if baseline else "composition", seed=seed, split=split,
                condition=condition, density=density, seen=seen, held=held,
                specialization=spec, cooperation_drop=held - ablated,
                own_probe=float(np.mean([p[0] for p in probes])),
                other_probe=float(np.mean([p[1] for p in probes])),
                edges=int(saved.sum()), mean_length=length,
                parameters=sum(p.numel() for p in model.parameters()), final_loss=loss.item())


def summarize(rows):
    primary = [r for r in rows if r['stage'] == 'composition']
    means = {c: {k: float(np.mean([r[k] for r in primary if r['condition'] == c]))
                 for k in ('seen', 'held', 'specialization', 'cooperation_drop', 'own_probe', 'other_probe', 'mean_length')}
             for c in CONDITIONS}
    differences = []
    for seed in range(5):
        values = {c: np.mean([r['held'] for r in primary if r['condition'] == c and r['seed'] == seed]) for c in CONDITIONS}
        differences.append(values['distance'] - values['uniform'])
    delta = float(np.mean(differences))
    margin = float(t.ppf(.975, 4) * np.std(differences, ddof=1) / np.sqrt(5))
    interval = [delta - margin, delta + margin]
    learning = all(means[c]['seen'] >= .95 for c in ('uniform', 'distance'))
    ancillary = means['distance']['held'] >= .90 and means['distance']['specialization'] >= .10 and means['distance']['cooperation_drop'] >= .10
    decision = 'supported' if learning and interval[0] > .05 and ancillary else (
        'falsified in this setup' if learning and interval[1] < .05 else 'inconclusive')
    baseline = {str(d): {k: float(np.mean([r[k] for r in rows if r['stage'] == 'baseline' and r['density'] == d]))
                         for k in ('seen', 'specialization', 'cooperation_drop')}
                for d in (.1, 1.)}
    return dict(means=means, paired_seed_differences=differences, difference=delta,
                ci95=interval, learning_gate=learning, ancillary_gate=ancillary,
                decision=decision, baseline=baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='composition')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    paths = {name: Path(f'{output}_{name}') for name in ('config.json', 'runs.csv', 'summary.json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Choose a fresh output prefix to preserve earlier results')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    torch=torch.__version__, numpy=np.__version__, threads=1, device='cpu',
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    script_sha256=source_hash, hidden=HIDDEN, updates=STEPS, batch=128,
                    seeds=list(range(5)), offsets=SPLITS, splits=[{'train': pairs_for_split(s)[0].tolist(),
                    'test': pairs_for_split(s)[1].tolist()} for s in SPLITS])
    paths['config.json'].write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    # Baseline precedes all held-out composition runs. No interim test evaluations.
    jobs = [(c, seed, 0, True, d) for seed in range(5) for c, d in [('uniform', .1), ('uniform', 1.)]]
    jobs += [(c, seed, split, False, .1) for seed in range(5) for split in range(3) for c in CONDITIONS]
    for c, seed, split, baseline, density in jobs:
        row = run(c, seed, split, baseline, density)
        rows.append(row)
        with paths['runs.csv'].open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    summary = summarize(rows)
    paths['summary.json'].write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
