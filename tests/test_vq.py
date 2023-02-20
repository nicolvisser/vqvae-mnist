import unittest
import torch
from vqvae import VectorQuantizer


class TestUtils(unittest.TestCase):
    def test_use_ema(self):
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64, commitment_cost=0.25, dim=-1, decay=None,
                             epsilon=1e-5)
        self.assertFalse(vq._use_ema)
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64, commitment_cost=0.25, dim=-1, decay=0.0,
                             epsilon=1e-5)
        self.assertFalse(vq._use_ema)
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64, commitment_cost=0.25, dim=-1, decay=0.99,
                             epsilon=1e-5)
        self.assertTrue(vq._use_ema)

    def test_shapes(self):
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64, commitment_cost=0.25, dim=-4, decay=0.99,
                             epsilon=1e-5)
        x = torch.rand((32, 64, 3, 8, 8))
        x_flat = torch.moveaxis(x, -4, -1).contiguous().view(-1, 64)
        quantized_flat, encodings_indices_flat, encodings_one_hot_flat = vq._encode(x_flat)
        quantized, encodings_indices, encodings_one_hot = vq.encode(x)
        quantized2, _, _ = vq.forward(x)

        self.assertEqual(quantized_flat.shape, torch.Size((32 * 3 * 8 * 8, 64)))
        self.assertEqual(encodings_indices_flat.shape, torch.Size((32 * 3 * 8 * 8,)))
        self.assertEqual(encodings_one_hot_flat.shape, torch.Size((32 * 3 * 8 * 8, 512)))

        self.assertEqual(quantized.shape, torch.Size((32, 64, 3, 8, 8)))
        self.assertEqual(encodings_indices.shape, torch.Size((32, 1, 3, 8, 8)))
        self.assertEqual(encodings_one_hot.shape, torch.Size((32, 512, 3, 8, 8)))

        self.assertEqual(quantized2.shape, torch.Size((32, 64, 3, 8, 8)))

        if __name__ == '__main__':
            unittest.main()
