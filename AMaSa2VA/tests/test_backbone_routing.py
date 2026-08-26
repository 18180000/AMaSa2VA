import json
import tempfile
import unittest
from pathlib import Path

from projects.amasa2va.models.backbones import (
    INTERNVL,
    QWEN,
    normalize_backbone,
    resolve_backbone,
)


class BackboneRoutingTests(unittest.TestCase):
    def test_aliases_are_normalized(self):
        self.assertEqual(normalize_backbone("qwen2.5"), QWEN)
        self.assertEqual(normalize_backbone("phi"), INTERNVL)

    def test_local_config_beats_an_ambiguous_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "renamed-checkpoint"
            path.mkdir()
            (path / "config.json").write_text(
                json.dumps({"architectures": ["Sa2VAChatModelQwen"]}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_backbone(str(path)), QWEN)

    def test_internvl_config_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "internvl_chat"}), encoding="utf-8"
            )
            self.assertEqual(resolve_backbone(str(path)), INTERNVL)

    def test_unknown_auto_detection_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "--backbone"):
            resolve_backbone("/models/renamed")

    def test_explicit_choice_does_not_need_local_weights(self):
        self.assertEqual(resolve_backbone("remote/model", "qwen"), QWEN)

    def test_standard_sa2va_name_uses_native_family(self):
        self.assertEqual(resolve_backbone("/models/Sa2VA-4B"), INTERNVL)


if __name__ == "__main__":
    unittest.main()
