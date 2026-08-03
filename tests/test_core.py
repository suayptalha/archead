import unittest

import torch

from archead.lm_head_methods import compress_head_baseline, pack_int4, unpack_int4


class QuantizationTests(unittest.TestCase):
    def test_signed_int4_round_trip(self):
        values = torch.tensor([[-8, -7, -1, 0, 1, 7]], dtype=torch.int8)
        restored = unpack_int4(pack_int4(values))
        torch.testing.assert_close(restored, values)

    def test_group_int4_head_shape_and_storage(self):
        torch.manual_seed(0)
        dense = torch.randn(32, 64, dtype=torch.float32)
        head = compress_head_baseline(dense, "group_int4", device="cpu")
        hidden = torch.randn(3, 64, dtype=torch.float16)
        self.assertEqual(tuple(head(hidden).shape), (3, 32))
        self.assertLess(head.stats["byte_ratio_vs_bf16"], 1.0)


if __name__ == "__main__":
    unittest.main()
