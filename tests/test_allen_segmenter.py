"""
Tests for Allen Cell Segmenter Integration
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Mock modules before importing backends
sys.modules['aicssegmentation'] = MagicMock()
sys.modules['aicssegmentation.structure_wrapper'] = MagicMock()
sys.modules['segmenter_model_zoo'] = MagicMock()
sys.modules['segmenter_model_zoo.zoo'] = MagicMock()

from allen_segmenter.backends import ClassicSegmenterBackend, MlSegmenterBackend, get_segmenter_backend
from allen_segmenter.utils import to_segmenter_array

class TestAllenSegmenterUtils(unittest.TestCase):
    def test_to_segmenter_array(self):
        # Test 2D -> 3D
        img2d = np.zeros((10, 10))
        res = to_segmenter_array(img2d)
        self.assertEqual(res.shape, (1, 10, 10))
        
        # Test 3D -> 3D
        img3d = np.zeros((5, 10, 10))
        res = to_segmenter_array(img3d)
        self.assertEqual(res.shape, (5, 10, 10))
        
        # Test 4D -> 4D (C, Z, Y, X)
        img4d = np.zeros((2, 5, 10, 10))
        res = to_segmenter_array(img4d, channel_axis=0)
        self.assertEqual(res.shape, (2, 5, 10, 10))

class TestClassicBackend(unittest.TestCase):
    def setUp(self):
        # Setup mock registry
        self.backend = ClassicSegmenterBackend()
        # Inject a mock wrapper
        self.mock_wrapper_class = MagicMock()
        self.mock_wrapper_instance = MagicMock()
        self.mock_wrapper_class.return_value = self.mock_wrapper_instance
        self.backend._registry = {'TEST_STRUCT': self.mock_wrapper_class}

    def test_segment_call(self):
        img = np.zeros((5, 10, 10))
        self.mock_wrapper_instance.segment.return_value = np.ones((5, 10, 10))
        
        result = self.backend.segment(img, structure_id='TEST_STRUCT')
        
        self.mock_wrapper_class.assert_called_once()
        self.mock_wrapper_instance.segment.assert_called_once()
        self.assertTrue(np.all(result))

    def test_invalid_structure(self):
        img = np.zeros((5, 10, 10))
        with self.assertRaises(ValueError):
            self.backend.segment(img, structure_id='INVALID')

class TestMlBackend(unittest.TestCase):
    @patch('allen_segmenter.backends.supports_segmenter_ml', return_value=True)
    def setUp(self, mock_support):
        with patch('segmenter_model_zoo.zoo.ModelZoo') as MockZoo:
            self.mock_zoo_instance = MockZoo.return_value
            self.mock_zoo_instance.get_model_names.return_value = ['model1', 'model2']
            self.backend = MlSegmenterBackend()

    def test_segment_call(self):
        img = np.zeros((5, 10, 10))
        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones((5, 10, 10))
        self.backend.zoo.get_model.return_value = mock_model
        
        result = self.backend.segment(img, structure_id='model1')
        
        self.backend.zoo.get_model.assert_called_with('model1')
        mock_model.predict.assert_called()
        self.assertTrue(np.all(result))

if __name__ == '__main__':
    unittest.main()
