"""
Unit tests for configuration loading
"""

import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from safe_fraud_detection.utils.config import Config


CONFIG_DIR = Path(__file__).parent.parent.parent / 'configs'


class TestConfig(unittest.TestCase):
    """Test cases for Config."""

    def test_bundled_configs_load(self):
        """Test every YAML file in configs/ loads into Config."""
        config_paths = sorted(CONFIG_DIR.glob('*.yaml'))
        self.assertGreater(len(config_paths), 0)

        for path in config_paths:
            with self.subTest(config=path.name):
                config = Config.from_yaml(str(path))
                self.assertIn(config.device, ('cpu', 'cuda'))

    def test_auto_device_resolves(self):
        """Test 'auto' resolves to a concrete device."""
        self.assertIn(Config(device='auto').device, ('cpu', 'cuda'))
        self.assertIn(Config.from_dict({'device': 'auto'}).device, ('cpu', 'cuda'))

    def test_explicit_device_kept(self):
        """Test an explicit device is not changed."""
        self.assertEqual(Config.from_dict({'device': 'cpu'}).device, 'cpu')

    def test_unknown_section_key_raises(self):
        """Test an unknown key inside a section names the key and section."""
        with self.assertRaisesRegex(ValueError, r"'training'.*num_epochs"):
            Config.from_dict({'training': {'num_epochs': 10}})

    def test_unknown_top_level_key_raises(self):
        """Test an unknown top-level key is reported."""
        with self.assertRaisesRegex(ValueError, "simulation"):
            Config.from_dict({'simulation': {'num_cards': 10}})

    def test_round_trip(self):
        """Test to_dict/from_dict preserves values."""
        config = Config.from_dict({'model': {'input_dim': 10}, 'device': 'cpu'})
        self.assertEqual(Config.from_dict(config.to_dict()).to_dict(), config.to_dict())


if __name__ == '__main__':
    unittest.main()
