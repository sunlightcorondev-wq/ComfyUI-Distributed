import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_utilities_module():
    module_path = Path(__file__).resolve().parents[1] / "nodes" / "utilities.py"
    package_name = "dist_divider_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    nodes_pkg = types.ModuleType(f"{package_name}.nodes")
    nodes_pkg.__path__ = []
    sys.modules[f"{package_name}.nodes"] = nodes_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    plan_spec = importlib.util.spec_from_file_location(
        f"{package_name}.utils.video_plan",
        Path(__file__).resolve().parents[1] / "utils" / "video_plan.py",
    )
    plan_module = importlib.util.module_from_spec(plan_spec)
    sys.modules[f"{package_name}.utils.video_plan"] = plan_module
    plan_spec.loader.exec_module(plan_module)

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.nodes.utilities",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


utils = _load_utilities_module()


class ImageBatchDividerTests(unittest.TestCase):
    def test_divides_images_into_contiguous_chunks(self):
        divider = utils.ImageBatchDivider()
        images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)

        outputs = divider.divide_batch(images, 3)

        self.assertEqual(outputs[0].shape[0], 4)
        self.assertEqual(outputs[1].shape[0], 3)
        self.assertEqual(outputs[2].shape[0], 3)
        self.assertEqual(outputs[0][:, 0, 0, 0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(outputs[1][:, 0, 0, 0].tolist(), [4.0, 5.0, 6.0])
        self.assertEqual(outputs[2][:, 0, 0, 0].tolist(), [7.0, 8.0, 9.0])

    def test_auto_divide_by_keeps_the_batch_whole_without_orchestration(self):
        divider = utils.ImageBatchDivider()
        images = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)

        outputs = divider.divide_batch(images, 0)

        self.assertEqual(len(outputs), utils.MAX_DIVIDER_OUTPUTS)
        self.assertEqual(outputs[0].shape[0], 4)
        self.assertEqual(outputs[1].shape[0], 0)

    def test_divide_by_supports_more_than_ten_parts(self):
        divider = utils.ImageBatchDivider()
        images = torch.arange(16, dtype=torch.float32).reshape(16, 1, 1, 1)

        outputs = divider.divide_batch(images, 16)

        self.assertEqual([out.shape[0] for out in outputs[:16]], [1] * 16)

    def test_unused_image_outputs_are_empty(self):
        divider = utils.ImageBatchDivider()
        images = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)

        outputs = divider.divide_batch(images, 2)

        self.assertEqual(len(outputs), utils.MAX_DIVIDER_OUTPUTS)
        for idx in range(2, utils.MAX_DIVIDER_OUTPUTS):
            self.assertEqual(outputs[idx].shape[0], 0)


class AudioBatchDividerTests(unittest.TestCase):
    def test_divides_audio_samples_into_contiguous_chunks(self):
        divider = utils.AudioBatchDivider()
        audio = {
            "waveform": torch.arange(10, dtype=torch.float32).reshape(1, 1, 10),
            "sample_rate": 24000,
        }

        outputs = divider.divide_audio(audio, 3)

        self.assertEqual(outputs[0]["waveform"][0, 0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(outputs[1]["waveform"][0, 0].tolist(), [4.0, 5.0, 6.0])
        self.assertEqual(outputs[2]["waveform"][0, 0].tolist(), [7.0, 8.0, 9.0])

    def test_unused_audio_outputs_are_empty(self):
        divider = utils.AudioBatchDivider()
        audio = {
            "waveform": torch.arange(8, dtype=torch.float32).reshape(1, 1, 8),
            "sample_rate": 24000,
        }

        outputs = divider.divide_audio(audio, 2)

        self.assertEqual(len(outputs), utils.MAX_DIVIDER_OUTPUTS)
        for idx in range(2, utils.MAX_DIVIDER_OUTPUTS):
            self.assertEqual(outputs[idx]["waveform"].shape[-1], 0)


if __name__ == "__main__":
    unittest.main()
