"""Scientific and implementation invariants for the module-level protocol."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dynspec.composition import (
    BASELINE_SEEDS,
    CELLS,
    CHANNELS,
    HIDDEN,
    MODULES,
    OUTPUT_MODULE,
    RULES,
    SEEDS,
    SPLITS,
    STEPS,
    ModuleNetwork,
    accuracy,
    baseline_gate,
    graph_distribution,
    inputs,
    make_model,
    pairs_for_split,
    protocol_hash,
    read_rows,
    summarize,
    target,
    topology,
    topology_stats,
    verified_baseline,
)


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formal_split_balance(self):
        held_sets = []
        for offsets in SPLITS:
            train, held = pairs_for_split(offsets)
            train_set, held_set = (set(map(tuple, p.tolist())) for p in (train, held))
            self.assertFalse(train_set & held_set)
            self.assertEqual(len(train_set), 80)
            self.assertEqual(len(held_set), 20)
            self.assertEqual(len(train_set | held_set), 100)
            for pairs in (train, held):
                for axis in (0, 1):
                    self.assertTrue(
                        torch.all(
                            torch.bincount(pairs[:, axis], minlength=10)
                            == len(pairs) // 10
                        )
                    )
                self.assertTrue(
                    torch.all(
                        torch.bincount(target(pairs), minlength=10) == len(pairs) // 10
                    )
                )
                branch = pairs.sum(1).remainder(2)
                self.assertTrue(
                    torch.all(torch.bincount(branch, minlength=2) == len(pairs) // 2)
                )
                for source in (0, 1):
                    labels = target(pairs)[branch == source]
                    self.assertTrue(
                        torch.all(
                            torch.bincount(labels, minlength=10) == len(pairs) // 20
                        )
                    )
            held_sets.append(held_set)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(held_sets[a] & held_sets[b])
        # The invalid pilot offsets must fail the branch condition.
        for offsets in ((1, 3), (2, 4), (5, 7)):
            held = pairs_for_split(offsets)[1]
            self.assertEqual(held.sum(1).remainder(2).unique().numel(), 1)

    def test_module_geometry_and_budgets(self):
        for seed in SEEDS:
            for view, budget in CELLS:
                params, active = [], []
                for rule in RULES:
                    graph = topology(rule, seed, view, budget)
                    stats = topology_stats(graph)
                    self.assertEqual(
                        stats["edges" if view == "edges" else "cable_cost"], budget
                    )
                    self.assertEqual(
                        stats["channel_cable_cost"], CHANNELS * stats["cable_cost"]
                    )
                    self.assertTrue(torch.equal(graph, graph.T))
                    self.assertEqual(graph.diag().sum(), 0)
                    self.assertTrue(
                        torch.equal(graph, topology(rule, seed, view, budget))
                    )
                    self.assertGreaterEqual(stats["a_to_output_hops"], 1)
                    self.assertGreaterEqual(stats["b_to_output_hops"], 1)
                    if rule == "local":
                        self.assertEqual(stats["long_edges"], 0)
                    model = make_model(rule, seed, view, budget)
                    params.append(sum(p.numel() for p in model.parameters()))
                    active.append(stats["edges"] * HIDDEN * CHANNELS)
                self.assertEqual(len(set(params)), 1)
                if view == "edges":
                    self.assertEqual(len(set(active)), 1)
        chain = topology_stats(topology("chain", 0))
        self.assertEqual((chain["edges"], chain["cable_cost"]), (10, 10))
        self.assertEqual((chain["a_to_output_hops"], chain["b_to_output_hops"]), (3, 2))

    def test_distance_dependent_graph_law(self):
        for rule in ("exponential", "power"):
            candidates, subsets, mass = graph_distribution(rule, "edges", 14)
            inclusion = mass @ subsets
            by_distance = {
                d: np.mean(
                    [p for (a, b), p in zip(candidates, inclusion) if b - a == d]
                )
                for d in range(2, 6)
            }
            self.assertTrue(
                all(by_distance[d] > by_distance[d + 1] for d in range(2, 5))
            )
            self.assertGreater(by_distance[5], 0)
        _, subsets, mass = graph_distribution("uniform", "edges", 14)
        self.assertTrue(np.allclose(mass @ subsets, 0.2))
        with self.assertRaises(ValueError):
            topology("local", 0, "cost", 23)
        with self.assertRaises(ValueError):
            topology("unrecognized", 0)

    def test_paired_weights_batches_and_masked_gradients(self):
        a, b = make_model("uniform", 9), make_model("exponential", 9)
        for left, right in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(left, right))
        pairs = torch.tensor([[2, 3], [4, 4]])
        x = inputs(pairs, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(x, inputs(pairs, torch.Generator().manual_seed(3))))
        a(x)[0].sum().backward()
        gradient = a.connection_weight.grad
        mask = a.connection_mask()
        self.assertTrue(torch.all(gradient[mask == 0] == 0))
        self.assertGreater(gradient[mask != 0].abs().sum(), 0)
        pointers = [p.untyped_storage().data_ptr() for p in a.parameters()]
        self.assertEqual(len(pointers), len(set(pointers)))

    def test_binary_messages_and_single_output(self):
        value = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0], requires_grad=True)
        bits = ModuleNetwork.binary_messages(value)
        self.assertTrue(torch.equal(bits, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])))
        bits.sum().backward()
        self.assertTrue(torch.all(value.grad > 0))
        model = make_model("dense", 0)
        x = inputs(torch.tensor([[2, 3], [7, 8]]), torch.Generator().manual_seed(1))
        logits, trace = model(x)
        self.assertEqual(model.readout.in_features, HIDDEN)
        self.assertTrue(torch.equal(logits, model.readout(trace[-1, :, OUTPUT_MODULE])))
        # Diagnostic states never enter a shared output/readout.
        altered = trace[-1].clone()
        altered[:, :OUTPUT_MODULE] += 100
        altered[:, OUTPUT_MODULE + 1 :] -= 100
        self.assertTrue(torch.equal(logits, model.readout(altered[:, OUTPUT_MODULE])))

    def test_no_global_input_or_aggregation_bypass(self):
        model = make_model("disconnected", 0)
        pairs = torch.tensor([[2, 3], [2, 9], [7, 3]])
        x = torch.nn.functional.one_hot(pairs, 10).float()
        out, trace = model(x)
        self.assertTrue(torch.equal(out[0], out[1]))
        self.assertTrue(torch.equal(out[0], out[2]))
        self.assertTrue(torch.equal(trace[:, 0, 0], trace[:, 1, 0]))
        self.assertTrue(torch.equal(trace[:, 0, 5], trace[:, 2, 5]))
        self.assertTrue(
            torch.equal(trace[:, 0, OUTPUT_MODULE], trace[:, 2, OUTPUT_MODULE])
        )
        out.sum().backward()
        self.assertEqual(model.input_weight.grad.abs().sum(), 0)
        self.assertEqual(model.message_weight.grad.abs().sum(), 0)
        score = accuracy(model, pairs_for_split(SPLITS[0])[1], 5, repeats=10)
        self.assertAlmostEqual(score["accuracy"], 0.1)

    def test_causal_cuts_and_hop_delays(self):
        x = torch.nn.functional.one_hot(torch.tensor([[2, 3], [7, 3]]), 10).float()
        model = make_model("chain", 3)
        # Install a deterministic one-bit route; random hard thresholds can
        # legitimately suppress an untrained input difference.
        with torch.no_grad():
            model.input_weight.zero_()
            model.input_weight[0, 2, 0] = 1
            model.recurrent_weight.zero_()
            model.message_weight.zero_()
            model.message_weight[:, 0, 0] = 1
            model.connection_weight.zero_()
            for sender in range(MODULES):
                for receiver in range(MODULES):
                    model.connection_weight[sender * CHANNELS, receiver * HIDDEN] = 1
            model.readout.weight.zero_()
            model.readout.bias.zero_()
            model.readout.weight[0, 0] = 1
        _, trace = model(x)
        self.assertTrue(
            torch.equal(trace[:3, 0, OUTPUT_MODULE], trace[:3, 1, OUTPUT_MODULE])
        )
        self.assertFalse(
            torch.equal(trace[-1, 0, OUTPUT_MODULE], trace[-1, 1, OUTPUT_MODULE])
        )
        saved = copy.deepcopy(model.state_dict())
        out, _ = model(x, cut_sources=(0,))
        self.assertTrue(torch.equal(out[0], out[1]))
        y = torch.nn.functional.one_hot(torch.tensor([[2, 3], [2, 9]]), 10).float()
        out, _ = model(y, cut_sources=(5,))
        self.assertTrue(torch.equal(out[0], out[1]))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(saved[key], value))
        out, _ = model(y, cut_sources=tuple(range(MODULES)))
        self.assertTrue(torch.equal(out[0], out[1]))

    def test_state_roundtrip_and_validation(self):
        model, restored = make_model("power", 3), make_model("uniform", 2)
        restored.load_state_dict(model.state_dict())
        x = inputs(torch.tensor([[0, 1]]), torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(model(x)[0], restored(x)[0]))
        for graph in (
            torch.eye(MODULES),
            torch.zeros(2, 2),
            torch.ones(MODULES, MODULES) * 0.5,
        ):
            with self.assertRaises(ValueError):
                ModuleNetwork(graph, 0)

    @staticmethod
    def baseline_rows(seen=1.0, specialization=0.3):
        return [
            {
                "stage": "baseline",
                "condition": c,
                "seed": s,
                "updates": STEPS,
                "seen": seen,
                "seen_a": seen,
                "seen_b": seen,
                "specialization": specialization,
                "cooperation_drop": 0.8,
                "own_probe": 0.95,
            }
            for s in BASELINE_SEEDS
            for c in ("dense", "chain")
        ]

    def test_baseline_gate_blocks_before_holdout(self):
        good = self.baseline_rows()
        self.assertTrue(baseline_gate(good)["passed"])
        self.assertFalse(baseline_gate(good[2:])["passed"])
        self.assertFalse(baseline_gate(self.baseline_rows(specialization=0))["passed"])
        self.assertFalse(baseline_gate(self.baseline_rows(seen=0.9))["passed"])
        for row in good:
            row["seen_a"] = 0.8
        self.assertFalse(baseline_gate(good)["passed"])

    @staticmethod
    def panel(gain=0.08, seen=1.0, specialization=0.3):
        rows = []
        conditions = [(r, v, b) for v, b in CELLS for r in RULES] + [
            (r, "diagnostic", 0) for r in ("chain", "disconnected")
        ]
        for seed in SEEDS:
            for split in range(len(SPLITS)):
                for rule, view, budget in conditions:
                    row = {
                        "seed": seed,
                        "split": split,
                        "condition": rule,
                        "view": view,
                        "budget": budget,
                    }
                    row.update(topology_stats(topology(rule, seed, view, budget)))
                    value = 0.82 + (gain if rule in ("exponential", "power") else 0)
                    row.update(
                        {
                            "seen": seen,
                            "seen_a": seen,
                            "seen_b": seen,
                            "held": value,
                            "held_a": value,
                            "held_b": value,
                            "specialization": specialization,
                            "cooperation_drop": 0.7,
                            "own_probe": 0.95,
                            "other_probe": 0.5,
                            "active_parameters": 1000,
                            "mean_module_specificity": 0.3,
                            "cut_a_drop": 0.3,
                            "cut_b_drop": 0.3,
                        }
                    )
                    rows.append(row)
        return rows

    def test_decision_support_falsification_and_invalidity(self):
        gate = baseline_gate(self.baseline_rows())
        summary = summarize(self.panel(), gate)
        self.assertEqual(summary["decision"], "supported in this setup")
        self.assertEqual(len(summary["comparisons"]), 16)
        self.assertEqual(
            summarize(self.panel(gain=0), gate)["decision"],
            "falsified in this tested grid",
        )
        self.assertEqual(
            summarize(self.panel(seen=0.8), gate)["decision"], "inconclusive"
        )
        self.assertEqual(
            summarize(self.panel(specialization=0), gate)["decision"], "inconclusive"
        )
        self.assertEqual(
            summarize(self.panel(), {"passed": False})["decision"], "inconclusive"
        )
        with self.assertRaises(ValueError):
            summarize(self.panel()[:-1], gate)
        duplicate = self.panel()
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            summarize(duplicate, gate)

    def test_repeated_splits_are_not_independent_seeds(self):
        rows = self.panel(gain=0.03)
        for row in rows:
            if row["condition"] in ("exponential", "power"):
                row["held"] += (
                    0.06 * (-1 if row["seed"] % 2 else 1)
                    + [-0.01, 0, 0.01][row["split"]]
                )
        summary = summarize(rows, baseline_gate(self.baseline_rows()))
        self.assertEqual(summary["decision"], "inconclusive")
        values = summary["comparisons"][0]["seed_differences"]
        self.assertEqual(len(values), len(SEEDS))
        self.assertTrue(
            np.allclose(values, [0.03 + 0.06 * (-1 if s % 2 else 1) for s in SEEDS])
        )
        self.assertGreater(
            summary["comparisons"][0]["family_ci95"][1],
            summary["comparisons"][0]["ci95"][1],
        )

    def test_baseline_file_verification(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "baseline")
            meta = {
                "protocol_hash": protocol_hash(),
                "hashes": {"code": "frozen"},
                "revision": "protocol-commit",
            }
            Path(f"{prefix}_config.json").write_text(json.dumps(meta))
            rows = self.baseline_rows()
            with Path(f"{prefix}_runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            self.assertTrue(verified_baseline(prefix, meta)["passed"])
            self.assertEqual(
                read_rows(f"{prefix}_runs.csv")[0]["seed"], BASELINE_SEEDS[0]
            )
            with self.assertRaises(RuntimeError):
                verified_baseline(prefix, dict(meta, revision="different"))
            rows[0]["specialization"] = -10
            with Path(f"{prefix}_runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(RuntimeError):
                verified_baseline(prefix, meta)


if __name__ == "__main__":
    unittest.main()
