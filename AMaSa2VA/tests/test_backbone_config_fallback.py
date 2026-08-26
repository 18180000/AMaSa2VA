import json
import tempfile
import unittest
from pathlib import Path

from projects.amasa2va.models.backbones import QWEN, resolve_backbone


class BackboneConfigFallbackTests(unittest.TestCase):
    def test_generic_config_allows_qwen_path_fallback(self):
        with tempfile.TemporaryDirectory(prefix="Sa2VA-Qwen2_5-VL-") as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "sa2va_chat"}), encoding="utf-8"
            )
            self.assertEqual(resolve_backbone(str(path)), QWEN)


if __name__ == "__main__":
    unittest.main()
