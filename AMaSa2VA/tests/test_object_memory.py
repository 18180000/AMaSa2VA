import tempfile
import unittest

import torch

from projects.amasa2va.models.memory.adapter import load_paper_adapter
from projects.amasa2va.models.memory.fusion import FusionConfig, fuse_and_route
from projects.amasa2va.models.memory.object_memory import ObjectCentricMemoryBank


class ObjectMemoryTests(unittest.TestCase):
    def test_mask_pooling_and_object_identity(self):
        bank = ObjectCentricMemoryBank()
        features = torch.tensor([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
        bank.stage("v", 0, features)
        self.assertTrue(bank.commit("v", 0, 9, torch.tensor([[1, 0], [0, 1]])))
        result = bank.retrieve("v", 1, torch.tensor([4.0, 5.0]))
        self.assertIsNotNone(result)
        self.assertEqual(result.entries[0].object_id, 9)
        torch.testing.assert_close(result.evidence, torch.tensor([4.0, 5.0]))

    def test_one_staged_frame_can_commit_multiple_objects(self):
        bank = ObjectCentricMemoryBank()
        features = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).repeat(256, 1, 1)
        bank.stage("v", 0, features)
        mask_a = torch.zeros(4, 4, dtype=torch.bool)
        mask_b = torch.zeros(4, 4, dtype=torch.bool)
        mask_a[:2, :2] = True
        mask_b[2:, 2:] = True
        self.assertTrue(bank.commit("v", 0, 1, mask_a))
        self.assertTrue(bank.commit("v", 0, 2, mask_b))
        self.assertEqual(bank.trajectory_lengths("v"), {1: 1, 2: 1})

    def test_strict_causality_uses_stored_frame_numbers(self):
        bank = ObjectCentricMemoryBank(retrieval_k=None)
        for frame in (0, 4):
            bank.stage("v", frame, torch.ones(2, 1, 1) * (frame + 1))
            bank.commit("v", frame, 0, torch.ones(1, 1))
        result = bank.retrieve("v", 3, torch.ones(2))
        self.assertEqual([entry.frame_end for entry in result.entries], [0])
        offline = bank.retrieve_offline("v", torch.ones(2), retrieval_k=8)
        self.assertEqual(sorted(entry.frame_end for entry in offline.entries), [0, 4])

    def test_similarity_merge_is_adjacent_and_bounded(self):
        bank = ObjectCentricMemoryBank(capacity=2, compression="similarity_merge", retrieval_k=None)
        tokens = [torch.tensor([1.0, 0.0]), torch.tensor([0.9, 0.1]), torch.tensor([-1.0, 0.0])]
        for frame, token in enumerate(tokens):
            bank.stage("v", frame, token[:, None, None])
            bank.commit("v", frame, 3, torch.ones(1, 1))
        entries = list(bank._bank["v"][3])
        self.assertEqual(len(entries), 2)
        self.assertEqual((entries[0].frame_start, entries[0].frame_end), (0, 1))
        torch.testing.assert_close(entries[0].token, torch.tensor([0.95, 0.05]))

    def test_topk_temperature_weighted_retrieval(self):
        bank = ObjectCentricMemoryBank(retrieval_k=2, temperature=0.4, compression="none")
        for frame, token in enumerate((torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0]))):
            bank.stage("v", frame, token[:, None, None])
            bank.commit("v", frame, frame, torch.ones(1, 1))
        result = bank.retrieve("v", 3, torch.tensor([1.0, 0.0]))
        self.assertEqual([entry.frame_end for entry in result.entries], [0, 1])
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)

    def test_route_falls_back_below_threshold(self):
        original = torch.tensor([[[1.0, 0.0]]])
        adapted = torch.tensor([[[-1.0, 0.0]]])
        routed, gate, used = fuse_and_route(
            adapted, original, FusionConfig(cosine_c0=0.0, cosine_c1=1.0, route_threshold=0.5)
        )
        self.assertEqual(gate, 0.0)
        self.assertFalse(used)
        torch.testing.assert_close(routed, original)

    def test_recovered_legacy_checkpoint_fails_contract_check(self):
        legacy = {"net.0.weight": torch.zeros(1024, 4352)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = f"{directory}/legacy.pt"
            torch.save(legacy, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "16 flattened memory slots"):
                load_paper_adapter(checkpoint, torch.device("cpu"))

    def test_clear_removes_training_records(self):
        bank = ObjectCentricMemoryBank()
        bank.training_records.append({"frame_idx": 1})
        bank.clear()
        self.assertEqual(bank.training_records, [])


if __name__ == "__main__":
    unittest.main()
