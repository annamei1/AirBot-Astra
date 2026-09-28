import os
import unittest
from unittest.mock import patch

import torch

from harness import config  # Loads SAM3_HOME before importing the segmenter.
from perception import sam3_cpu
from perception.sam3_segmenter import SAM3Segmenter
import sam3.model_builder as builder
from sam3.model.vitdet import Mlp


class CPUCompatibilityTests(unittest.TestCase):
    def test_device_selection(self):
        for available, requested, expected in [(False, 'auto', 'cpu'), (True, 'auto', 'cuda'),
                                                (True, 'cpu', 'cpu'), (True, 'cuda', 'cuda')]:
            with self.subTest(available=available, requested=requested), \
                 patch.dict(os.environ, {'SAM3_DEVICE': requested}), \
                 patch('torch.cuda.is_available', return_value=available):
                self.assertEqual(sam3_cpu.select_device(), expected)
        with patch.dict(os.environ, {'SAM3_DEVICE': 'cuda'}), \
             patch('torch.cuda.is_available', return_value=False), self.assertRaises(RuntimeError):
            sam3_cpu.select_device()
        with patch.dict(os.environ, {'SAM3_DEVICE': 'bad'}), self.assertRaises(ValueError):
            sam3_cpu.select_device()

    def test_mlp_stays_float32(self):
        mlp = Mlp(4, hidden_features=8).eval()
        x = torch.randn(2, 4)
        expected = mlp.fc2(mlp.act(mlp.fc1(x)))
        actual = sam3_cpu._cpu_mlp_forward(mlp, x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.float32)

    def test_cpu_builder_restores_overrides_on_failure(self):
        pos, coords = builder._create_position_encoding, builder.TransformerDecoder._get_coords
        def fail(**kwargs):
            self.assertEqual(kwargs['device'], 'cpu')
            self.assertIsNot(builder._create_position_encoding, pos)
            self.assertEqual(builder.TransformerDecoder._get_coords(2, 2, 'cuda')[0].device.type, 'cpu')
            raise RuntimeError('bad checkpoint')
        with patch.object(builder, 'build_sam3_image_model', side_effect=fail), \
             patch('torch.set_num_threads'), self.assertRaisesRegex(RuntimeError, 'bad checkpoint'):
            sam3_cpu.build_image_model('missing.pt', 'cpu')
        self.assertIs(builder._create_position_encoding, pos)
        self.assertIs(builder.TransformerDecoder._get_coords, coords)

    def test_cuda_builder_is_unmodified(self):
        pos, coords = builder._create_position_encoding, builder.TransformerDecoder._get_coords
        def build(**kwargs):
            self.assertEqual(kwargs['device'], 'cuda')
            self.assertIs(builder._create_position_encoding, pos)
            self.assertIs(builder.TransformerDecoder._get_coords, coords)
            return 'cuda-model'
        with patch.object(builder, 'build_sam3_image_model', side_effect=build), \
             patch('torch.set_num_threads') as threads:
            self.assertEqual(sam3_cpu.build_image_model('weights.pt', 'cuda'), 'cuda-model')
            threads.assert_not_called()

    def test_cpu_cache_cleanup_never_calls_cuda(self):
        segmenter = object.__new__(SAM3Segmenter)
        segmenter.device = 'cpu'
        with patch('torch.cuda.empty_cache', side_effect=AssertionError('CUDA touched')):
            segmenter._clear_cuda_cache()


if __name__ == '__main__':
    unittest.main()
