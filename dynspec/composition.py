"""Module-level composition study; frozen design and interpretation in experiment.md."""

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from torch import nn
from torch.nn import functional as F

from dynspec.tasks import get_single_task

SPLITS = ((1, 2), (3, 4), (5, 6))
MODULES, HIDDEN, CHANNELS, TIMESTEPS = 6, 24, 4, 6
OUTPUT_MODULE = 3
STEPS, BATCH = 3000, 128
SEEDS = tuple(range(48))
BASELINE_SEEDS = tuple(range(2000, 2008))
RULES = ("local", "uniform", "exponential", "power")
CELLS = (("edges", 14), ("edges", 18), ("cost", 22), ("cost", 26))
PROTOCOL = {
    "version": 2,
    "coordinates": [[i, 0] for i in range(MODULES)],
    "source_modules": [0, 5],
    "output_module": OUTPUT_MODULE,
    "hidden": HIDDEN,
    "channels": CHANNELS,
    "timesteps": TIMESTEPS,
    "updates": STEPS,
    "batch": BATCH,
    "lr": 0.003,
    "weight_decay": 1e-5,
    "clip_norm": 1.0,
    "input_noise_sd": 0.05,
    "seeds": SEEDS,
    "baseline_seeds": BASELINE_SEEDS,
    "offsets": SPLITS,
    "cells": CELLS,
    "shortcut_prior": {
        "uniform": 0.35,
        "exponential": [0.6, 0.75],
        "power": [0.6, 1.5],
    },
    "eval_repeats": 100,
    "probe_repeats": 30,
    "probe_ridge": 0.01,
    "baseline_gate": {
        "seen": 0.95,
        "branch": 0.95,
        "specialization": 0.1,
        "cooperation_drop": 0.2,
        "own_probe": 0.9,
    },
    "final_gate": {
        "seen": 0.95,
        "held": 0.8,
        "specialization": 0.1,
        "cooperation_drop": 0.2,
    },
    "minimum_gain": 0.05,
    "family_contrasts": 16,
    "ci_half_width_target": 0.025,
}


def pairs_for_split(offsets):
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    held = torch.tensor([int((b - a) % 10) in offsets for a, b in pairs])
    return pairs[~held], pairs[held]


def target(pairs):
    return get_single_task("parity-digits", pairs, 10)


@lru_cache(None)
def graph_distribution(rule, view, budget):
    """Exact independent-edge prior conditioned on edge count or cable cost."""
    candidates = [(a, b) for a in range(MODULES) for b in range(a + 2, MODULES)]
    distance = np.array([b - a for a, b in candidates])
    if rule == "local":
        probability = np.where(distance == 2, 0.5, 0.0)
    elif rule == "uniform":
        probability = np.full(len(candidates), 0.35)
    elif rule == "exponential":
        probability = 0.6 * np.exp(-0.75 * (distance - 2))
    elif rule == "power":
        probability = 0.6 * (distance / 2.0) ** -1.5
    else:
        raise ValueError(f"Unknown rule: {rule}")
    subsets, masses = [], []
    for subset in itertools.product((0, 1), repeat=len(candidates)):
        chosen = np.array(subset, dtype=bool)
        value = 2 * (5 + (chosen.sum() if view == "edges" else distance[chosen].sum()))
        if value != budget:
            continue
        mass = np.prod(np.where(chosen, probability, 1 - probability))
        if mass > 0:
            subsets.append(subset)
            masses.append(mass)
    if not masses:
        raise ValueError("No graph satisfies the requested rule and budget")
    masses = np.asarray(masses)
    return candidates, np.asarray(subsets), masses / masses.sum()


def topology(rule, seed, view="edges", budget=14):
    graph = np.zeros((MODULES, MODULES), dtype=np.float32)
    if rule == "disconnected":
        return torch.from_numpy(graph)
    if rule == "dense":
        return torch.ones(MODULES, MODULES) - torch.eye(MODULES)
    for a in range(MODULES - 1):
        graph[a, a + 1] = graph[a + 1, a] = 1
    if rule != "chain":
        candidates, subsets, masses = graph_distribution(rule, view, budget)
        # Common random number couples graph draws; training RNG is independent.
        u = np.random.default_rng(10000 + seed).random()
        index = min(int(np.searchsorted(np.cumsum(masses), u)), len(masses) - 1)
        for (a, b), chosen in zip(candidates, subsets[index]):
            if chosen:
                graph[a, b] = graph[b, a] = 1
    return torch.from_numpy(graph)


def topology_stats(graph):
    adjacency = graph.numpy().astype(bool)
    distance = np.abs(np.arange(MODULES)[:, None] - np.arange(MODULES)[None, :])
    paths = np.where(adjacency, 1.0, np.inf)
    np.fill_diagonal(paths, 0)
    for k in range(MODULES):
        paths = np.minimum(paths, paths[:, k, None] + paths[None, k, :])
    triangle = np.trace(adjacency.astype(float) @ adjacency @ adjacency) / 6
    degree = adjacency.sum(1)
    triples = (degree * (degree - 1) / 2).sum()
    edges = int(adjacency.sum())
    cable = int((distance * adjacency).sum())
    finite = paths[~np.eye(MODULES, dtype=bool)]
    return {
        "edges": edges,
        "cable_cost": cable,
        "channel_cable_cost": cable * CHANNELS,
        "mean_length": cable / edges if edges else 0.0,
        "local_edges": int((adjacency & (distance == 1)).sum()),
        "long_edges": int((adjacency & (distance >= 3)).sum()),
        "medium_edges": int((adjacency & (distance == 2)).sum()),
        "degree_min": int(degree.min()),
        "degree_max": int(degree.max()),
        "degree_sd": float(degree.std()),
        "clustering": float(3 * triangle / triples) if triples else 0.0,
        "mean_path": float(finite.mean()) if np.isfinite(finite).all() else -1.0,
        "a_to_output_hops": int(paths[0, OUTPUT_MODULE])
        if np.isfinite(paths[0, OUTPUT_MODULE])
        else -1,
        "b_to_output_hops": int(paths[5, OUTPUT_MODULE])
        if np.isfinite(paths[5, OUTPUT_MODULE])
        else -1,
        "graph": "".join(map(str, adjacency.astype(int).ravel())),
    }


class ModuleNetwork(nn.Module):
    """Six synchronous RNN states; every cross-module signal is a binary message."""

    def __init__(self, graph, seed):
        super().__init__()
        if graph.shape != (MODULES, MODULES) or torch.any(graph.diag()):
            raise ValueError("Graph must have six modules and no self edges")
        if not torch.all((graph == 0) | (graph == 1)):
            raise ValueError("Graph must be binary")
        torch.manual_seed(seed)
        self.input_weight = nn.Parameter(torch.randn(2, 10, HIDDEN) * 0.3)
        self.recurrent_weight = nn.Parameter(
            torch.randn(MODULES, HIDDEN, HIDDEN) / HIDDEN**0.5
        )
        self.message_weight = nn.Parameter(
            torch.randn(MODULES, HIDDEN, CHANNELS) / HIDDEN**0.5
        )
        self.connection_weight = nn.Parameter(
            torch.randn(MODULES * CHANNELS, MODULES * HIDDEN)
            / (MODULES * CHANNELS) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(MODULES, 1, HIDDEN))
        self.readout = nn.Linear(HIDDEN, 10)
        self.register_buffer("graph", graph.clone())

    @staticmethod
    def binary_messages(value):
        # Hard 0/1 forward values with a sigmoid surrogate derivative.
        soft = torch.sigmoid(value)
        return (value > 0).to(value.dtype).detach() + (soft - soft.detach())

    def connection_mask(self, cut_sources=()):
        graph = self.graph.clone()
        if cut_sources:
            graph[list(cut_sources)] = 0
        return graph.repeat_interleave(CHANNELS, 0).repeat_interleave(HIDDEN, 1)

    def forward(self, x, cut_sources=()):
        projected = torch.bmm(x.transpose(0, 1), self.input_weight)
        drive = x.new_zeros(MODULES, len(x), HIDDEN)
        drive[0], drive[5] = projected[0], projected[1]
        state = torch.zeros_like(drive)
        weights = self.connection_weight * self.connection_mask(cut_sources)
        trace = []
        for _ in range(TIMESTEPS):
            bits = self.binary_messages(torch.bmm(state, self.message_weight))
            incoming = (
                (bits.transpose(0, 1).reshape(len(x), -1) @ weights)
                .reshape(len(x), MODULES, HIDDEN)
                .transpose(0, 1)
            )
            state = torch.tanh(
                drive + torch.bmm(state, self.recurrent_weight) + incoming + self.bias
            )
            trace.append(state.transpose(0, 1))
        return self.readout(state[OUTPUT_MODULE]), torch.stack(trace)


def make_model(rule, seed, view="edges", budget=14):
    return ModuleNetwork(topology(rule, seed, view, budget), seed)


def inputs(pairs, rng):
    # Fresh noisy symbolic stimulus, held constant during the six time steps.
    return F.one_hot(pairs, 10).float() + 0.05 * torch.randn(
        len(pairs), 2, 10, generator=rng
    )


@torch.no_grad()
def accuracy(model, pairs, seed, repeats=100, cut_sources=()):
    rng = torch.Generator().manual_seed(seed)
    expanded = pairs.repeat_interleave(repeats, 0)
    correct = []
    for p in expanded.split(256):
        out, _ = model(inputs(p, rng), cut_sources)
        correct.extend((out.argmax(-1) == target(p)).tolist())
    correct = np.asarray(correct).reshape(len(pairs), repeats).mean(1)
    a_selected = pairs.sum(1).remainder(2).bool().numpy()
    return {
        "accuracy": float(correct.mean()),
        "a_selected": float(correct[a_selected].mean()),
        "b_selected": float(correct[~a_selected].mean()),
    }


@torch.no_grad()
def specialization(model, seed):
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10)).repeat_interleave(
        PROTOCOL["probe_repeats"], 0
    )
    states = []
    for offset in (0, 1):
        rng = torch.Generator().manual_seed(seed + offset)
        states.append(
            torch.cat([model(inputs(p, rng))[1][-1] for p in pairs.split(256)])
        )
    probes = []
    for module in range(MODULES):
        train = torch.cat((states[0][:, module], torch.ones(len(pairs), 1)), 1)
        test = torch.cat((states[1][:, module], torch.ones(len(pairs), 1)), 1)
        penalty = torch.eye(HIDDEN + 1) * PROTOCOL["probe_ridge"]
        penalty[-1, -1] = 0
        acc = []
        for digit in (0, 1):
            labels = F.one_hot(pairs[:, digit], 10).float()
            weight = torch.linalg.solve(train.T @ train + penalty, train.T @ labels)
            acc.append(
                ((test @ weight).argmax(-1) == pairs[:, digit]).float().mean().item()
            )
        probes.append(acc)
    own = (probes[0][0] + probes[5][1]) / 2
    other = (probes[0][1] + probes[5][0]) / 2
    signed = [(a - b) / 0.9 for a, b in probes]
    return {
        "specialization": (own - other) / 0.9,
        "own_probe": own,
        "other_probe": other,
        "mean_module_specificity": float(np.mean(np.abs(signed))),
        "module_probes": json.dumps(probes, separators=(",", ":")),
    }


def run(
    rule, seed, split=0, stage="composition", view="edges", budget=14, updates=STEPS
):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = make_model(rule, seed, view, budget)
    train, held = pairs_for_split(() if stage == "baseline" else SPLITS[split])
    if stage == "baseline":
        held = train
    rng = torch.Generator().manual_seed(20000 + seed * 100 + split)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"]
    )
    for _ in range(updates):
        batch = train[torch.randint(len(train), (BATCH,), generator=rng)]
        out, _ = model(inputs(batch, rng))
        loss = F.cross_entropy(out, target(batch))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["clip_norm"])
        optimizer.step()
    model.eval()
    eval_seed = 30000 + seed * 100 + split
    seen = accuracy(model, train, eval_seed)
    test = accuracy(model, held, eval_seed + 1)
    ablated = accuracy(model, held, eval_seed + 1, cut_sources=tuple(range(MODULES)))
    cuts = {
        name: accuracy(model, held, eval_seed + 1, cut_sources=(source,))
        for name, source in (("a", 0), ("b", 5))
    }
    graph_stats = topology_stats(model.graph)
    fixed = sum(
        p.numel() for name, p in model.named_parameters() if name != "connection_weight"
    )
    return {
        "stage": stage,
        "seed": seed,
        "split": split,
        "condition": rule,
        "view": view,
        "budget": budget,
        "updates": updates,
        "seen": seen["accuracy"],
        "seen_a": seen["a_selected"],
        "seen_b": seen["b_selected"],
        "held": test["accuracy"],
        "held_a": test["a_selected"],
        "held_b": test["b_selected"],
        **specialization(model, eval_seed + 2),
        "cooperation_drop": test["accuracy"] - ablated["accuracy"],
        "ablated": ablated["accuracy"],
        "cut_a_drop": test["accuracy"] - cuts["a"]["accuracy"],
        "cut_b_drop": test["accuracy"] - cuts["b"]["accuracy"],
        "cut_a_a_selected_drop": test["a_selected"] - cuts["a"]["a_selected"],
        "cut_b_b_selected_drop": test["b_selected"] - cuts["b"]["b_selected"],
        **graph_stats,
        "parameters": sum(p.numel() for p in model.parameters()),
        "active_parameters": fixed + graph_stats["edges"] * HIDDEN * CHANNELS,
        "final_loss": loss.item(),
    }


def baseline_gate(rows):
    selected = [
        r for r in rows if r["stage"] == "baseline" and r["condition"] == "dense"
    ]
    if len(selected) != len(BASELINE_SEEDS) or {r["seed"] for r in selected} != set(
        BASELINE_SEEDS
    ):
        return {"passed": False, "reason": "Incomplete baseline seed panel"}
    means = {
        key: float(np.mean([r[key] for r in selected]))
        for key in (
            "seen",
            "seen_a",
            "seen_b",
            "specialization",
            "cooperation_drop",
            "own_probe",
        )
    }
    thresholds = PROTOCOL["baseline_gate"]
    checks = {
        "learning": means["seen"] >= thresholds["seen"],
        "both_branches": min(means["seen_a"], means["seen_b"]) >= thresholds["branch"],
        "specialization": means["specialization"] >= thresholds["specialization"],
        "own_decoding": means["own_probe"] >= thresholds["own_probe"],
        "cooperation": means["cooperation_drop"] >= thresholds["cooperation_drop"],
        "seed_consistency": sum(
            r["seen"] >= 0.9 and r["specialization"] >= 0.1 for r in selected
        )
        >= 6,
    }
    return {"passed": all(checks.values()), "checks": checks, "means": means}


def interval(values, confidence=0.95):
    values = np.asarray(values)
    mean = float(values.mean())
    margin = float(
        t.ppf((1 + confidence) / 2, len(values) - 1)
        * values.std(ddof=1)
        / len(values) ** 0.5
    )
    return [mean - margin, mean + margin]


def summarize(rows, gate):
    expected = {
        (seed, split, rule, view, budget)
        for seed in SEEDS
        for split in range(len(SPLITS))
        for view, budget in CELLS
        for rule in RULES
    }
    expected |= {
        (seed, split, rule, "diagnostic", 0)
        for seed in SEEDS
        for split in range(len(SPLITS))
        for rule in ("chain", "disconnected")
    }
    keys = [
        (r["seed"], r["split"], r["condition"], r["view"], r["budget"]) for r in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "Final panel is incomplete or contains duplicate/unplanned runs"
        )
    metrics = (
        "seen",
        "seen_a",
        "seen_b",
        "held",
        "held_a",
        "held_b",
        "specialization",
        "cooperation_drop",
        "own_probe",
        "other_probe",
        "cable_cost",
        "channel_cable_cost",
        "edges",
        "active_parameters",
        "local_edges",
        "long_edges",
        "medium_edges",
        "mean_length",
        "mean_path",
        "clustering",
        "degree_sd",
        "a_to_output_hops",
        "b_to_output_hops",
        "mean_module_specificity",
        "cut_a_drop",
        "cut_b_drop",
    )
    groups = {}
    seed_metrics = {}
    for view, budget, rule in sorted(
        {(r["view"], r["budget"], r["condition"]) for r in rows}
    ):
        label = f"{view}:{budget}:{rule}"
        group = [
            r
            for r in rows
            if (r["view"], r["budget"], r["condition"]) == (view, budget, rule)
        ]
        seed_metrics[label] = {
            k: [float(np.mean([r[k] for r in group if r["seed"] == s])) for s in SEEDS]
            for k in metrics
        }
        groups[label] = {k: float(np.mean(v)) for k, v in seed_metrics[label].items()}
        groups[label]["held_ci95"] = interval(seed_metrics[label]["held"])
        groups[label]["specialization_ci95"] = interval(
            seed_metrics[label]["specialization"]
        )
    comparisons, candidates = [], []
    for view, budget in CELLS:
        for rule in ("exponential", "power"):
            label = f"{view}:{budget}:{rule}"
            candidate = groups[label]
            eligible = gate["passed"] and all(
                candidate[k] >= v for k, v in PROTOCOL["final_gate"].items()
            )
            gains, precise, refuted = [], [], []
            for control in ("local", "uniform"):
                reference = f"{view}:{budget}:{control}"
                differences = (
                    np.asarray(seed_metrics[label]["held"])
                    - seed_metrics[reference]["held"]
                )
                nominal = interval(differences)
                simultaneous = interval(
                    differences, 1 - 0.05 / PROTOCOL["family_contrasts"]
                )
                valid = (
                    gate["passed"]
                    and min(candidate["seen"], groups[reference]["seen"])
                    >= PROTOCOL["final_gate"]["seen"]
                )
                delta = float(differences.mean())
                comparisons.append(
                    {
                        "candidate": label,
                        "control": reference,
                        "difference": delta,
                        "ci95": nominal,
                        "family_ci95": simultaneous,
                        "seed_differences": differences.tolist(),
                        "valid_learning": valid,
                        "target_precision_met": (nominal[1] - nominal[0]) / 2
                        <= PROTOCOL["ci_half_width_target"],
                    }
                )
                gains.append(
                    valid and delta >= PROTOCOL["minimum_gain"] and simultaneous[0] > 0
                )
                precise.append(valid)
                refuted.append(valid and simultaneous[1] < PROTOCOL["minimum_gain"])
            candidates.append(
                {
                    "candidate": label,
                    "joint_metrics_pass": eligible,
                    "supported": eligible and all(gains),
                    "advantage_refuted": all(precise) and any(refuted),
                }
            )
    # A failed baseline or unlearned comparator cannot falsify the topology claim.
    supported = any(c["supported"] for c in candidates)
    falsified = gate["passed"] and all(c["advantage_refuted"] for c in candidates)
    decision = (
        "supported in this setup"
        if supported
        else "falsified in this tested grid"
        if falsified
        else "inconclusive"
    )
    pareto = [
        label
        for label, point in groups.items()
        if not label.endswith(":disconnected")
        and not any(
            other["cable_cost"] <= point["cable_cost"]
            and other["held"] >= point["held"]
            and (
                other["cable_cost"] < point["cable_cost"]
                or other["held"] > point["held"]
            )
            for name, other in groups.items()
            if name != label
        )
    ]
    return {
        "baseline_gate": gate,
        "means": groups,
        "comparisons": comparisons,
        "candidates": candidates,
        "descriptive_pareto_frontier": pareto,
        "decision": decision,
        "n_seeds": len(SEEDS),
        "splits_are_repeated_measures": True,
    }


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def metadata():
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for name in ("dynspec/composition.py", "test_composition.py", "experiment.md"):
        data = (root / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
        if data != committed:
            raise RuntimeError(
                f"Commit the protocol and implementation before running: {name}"
            )
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "hashes": hashes,
        "protocol_hash": protocol_hash(),
        "protocol": PROTOCOL,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": 1,
        "splits": [
            {
                "train": pairs_for_split(s)[0].tolist(),
                "test": pairs_for_split(s)[1].tolist(),
            }
            for s in SPLITS
        ],
    }


def read_rows(path):
    with Path(path).open(newline="") as handle:
        result = list(csv.DictReader(handle))
    text_keys = {"stage", "condition", "view", "graph", "module_probes"}
    int_keys = {
        "seed",
        "split",
        "budget",
        "updates",
        "edges",
        "cable_cost",
        "channel_cable_cost",
        "local_edges",
        "long_edges",
        "medium_edges",
        "degree_min",
        "degree_max",
        "a_to_output_hops",
        "b_to_output_hops",
        "parameters",
        "active_parameters",
    }
    return [
        {
            k: v if k in text_keys else int(v) if k in int_keys else float(v)
            for k, v in row.items()
        }
        for row in result
    ]


def verified_baseline(prefix, meta):
    config = json.loads(Path(f"{prefix}_config.json").read_text())
    if (
        config["protocol_hash"] != meta["protocol_hash"]
        or config["hashes"] != meta["hashes"]
        or config["revision"] != meta["revision"]
    ):
        raise RuntimeError(
            "Baseline must use the same frozen protocol and source revision"
        )
    rows = read_rows(f"{prefix}_runs.csv")
    expected = {(s, c) for s in BASELINE_SEEDS for c in ("dense", "chain")}
    if (
        {(r["seed"], r["condition"]) for r in rows} != expected
        or len(rows) != len(expected)
        or any(r["stage"] != "baseline" or r["updates"] != STEPS for r in rows)
    ):
        raise RuntimeError("Baseline panel is incomplete or inconsistent")
    gate = baseline_gate(rows)
    if not gate["passed"]:
        raise RuntimeError(
            f"Baseline failed; final topology runs are forbidden: {gate}"
        )
    return gate


def run_job(job):
    return run(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "final"), default="baseline")
    parser.add_argument("--output", default="module_baseline")
    parser.add_argument("--baseline", default="module_baseline")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use one to eight workers; each run uses one CPU thread")
    meta = metadata()
    # Gate before scheduling, writing final artifacts, or evaluating any holdout.
    gate = verified_baseline(args.baseline, meta) if args.stage == "final" else None
    paths = {
        name: Path(f"{args.output}_{name}")
        for name in ("config.json", "runs.csv", "summary.json")
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(
            "Choose a fresh output prefix to preserve earlier results"
        )
    meta.update(
        stage=args.stage,
        workers=args.workers,
        baseline_prefix=args.baseline if gate else None,
    )
    paths["config.json"].write_text(json.dumps(meta, indent=2) + "\n")
    if args.stage == "baseline":
        jobs = [
            (c, s, 0, "baseline", "diagnostic", 0)
            for s in BASELINE_SEEDS
            for c in ("dense", "chain")
        ]
    else:
        jobs = [
            (c, s, split, "composition", v, b)
            for s in SEEDS
            for split in range(len(SPLITS))
            for v, b in CELLS
            for c in RULES
        ]
        jobs += [
            (c, s, split, "composition", "diagnostic", 0)
            for s in SEEDS
            for split in range(len(SPLITS))
            for c in ("chain", "disconnected")
        ]
    rows = []
    with (
        paths["runs.csv"].open("w", newline="") as handle,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        writer = None
        for row in pool.map(run_job, jobs, chunksize=1):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "total": len(jobs),
                        "seed": row["seed"],
                        "split": row["split"],
                        "condition": row["condition"],
                        "view": row["view"],
                        "budget": row["budget"],
                    }
                ),
                flush=True,
            )
    summary = (
        {"baseline_gate": baseline_gate(rows)}
        if args.stage == "baseline"
        else summarize(rows, gate)
    )
    paths["summary.json"].write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
