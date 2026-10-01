"""Integrity checks for the composition protocol and explicit masks."""
import unittest

import numpy as np
import torch

from dynspec.composition import HIDDEN, SPLITS, inputs, make_model, pairs_for_split, topology
from dynspec.models import Community
from dynspec.tasks import get_single_task


class CompositionTests(unittest.TestCase):
    def test_splits_are_disjoint_and_cover_marginals_and_labels(self):
        for split in SPLITS:
            train, test = pairs_for_split(split)
            train_set = set(map(tuple, train.tolist()))
            test_set = set(map(tuple, test.tolist()))
            self.assertFalse(train_set & test_set)
            self.assertEqual(len(train_set), 80)
            self.assertEqual(len(test_set), 20)
            self.assertEqual(len(train_set | test_set), 100)
            for pairs in (train, test):
                for axis in (0, 1):
                    counts = torch.bincount(pairs[:, axis], minlength=10)
                    self.assertTrue(torch.all(counts == len(pairs) // 10))
                self.assertEqual(len(get_single_task('parity-digits', pairs, 10).unique()), 10)

    def test_matched_budget_and_distance_rule(self):
        lengths = {c: [] for c in ('uniform', 'distance')}
        for seed in range(5):
            for c in lengths:
                mask, length = topology(c, seed)
                self.assertEqual(mask[:HIDDEN, HIDDEN:].sum(), 409)
                self.assertEqual(mask[HIDDEN:, :HIDDEN].sum(), 409)
                self.assertEqual(mask[:HIDDEN, :HIDDEN].sum(), 0)
                self.assertEqual(mask[HIDDEN:, HIDDEN:].sum(), 0)
                lengths[c].append(length)
                self.assertTrue(torch.equal(mask, topology(c, seed)[0]))
        self.assertLess(np.mean(lengths['distance']), np.mean(lengths['uniform']))

    def test_pairing_and_masked_gradients(self):
        uniform, _ = make_model('uniform', 0)
        distance, _ = make_model('distance', 0)
        for a, b in zip(uniform.parameters(), distance.parameters()):
            self.assertTrue(torch.equal(a, b))
        x = inputs(torch.tensor([[2, 3], [4, 4]]), torch.Generator().manual_seed(9))
        uniform(x)[0].sum().backward()
        param = uniform.comms.parametrizations.weight_hh_l0.original
        mask = uniform.comms.parametrizations.weight_hh_l0[0].mask
        self.assertTrue(torch.all(param.grad[mask == 0] == 0))
        self.assertGreater(param.grad[mask != 0].abs().sum(), 0)

    def test_disconnected_module_state_cannot_see_other_input(self):
        model, _ = make_model('disconnected', 0)
        # Identical input A, different input B. Max readout can still see both modules.
        x = torch.nn.functional.one_hot(torch.tensor([[2, 3], [2, 9]]), 10).float().reshape(2, 20)
        _, states = model(x.unsqueeze(0).repeat(4, 1, 1))
        self.assertTrue(torch.equal(states[:, 0, :HIDDEN], states[:, 1, :HIDDEN]))

    def test_custom_mask_roundtrip_and_validation(self):
        model, _ = make_model('distance', 3)
        restored, _ = make_model('uniform', 2)
        restored.load_state_dict(model.state_dict())
        x = inputs(torch.tensor([[0, 1]]), torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(model(x)[0], restored(x)[0]))
        config = dict(input=model.input_config, modules=model.modules_config,
                      connections={'sparsity': .1, 'mask': torch.eye(2 * HIDDEN)},
                      readout=model.readout_config)
        with self.assertRaises(ValueError):
            Community(config)
        config['connections']['mask'] = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            Community(config)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
