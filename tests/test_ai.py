import unittest
from unittest.mock import patch

from game2048.acceleration import AccelerationBackend, detect_acceleration_backend
from game2048.ai import HeuristicExpectimaxAI
from game2048.core import Game2048


class TestAI(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game2048()
        self.ai = HeuristicExpectimaxAI(self.game, depth=2)

    def test_ai_returns_valid_move_when_possible(self) -> None:
        board = [
            [2, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2],
            [4, 8, 16, 32],
        ]
        move, values = self.ai.choose_move(board)
        self.assertIn(move, {"left", "right", "up", "down"})
        self.assertEqual(set(values.keys()), {"up", "down", "left", "right"})

    def test_ai_returns_none_when_game_over(self) -> None:
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        move, values = self.ai.choose_move(board)
        self.assertIsNone(move)
        self.assertTrue(all(v == float("-inf") for v in values.values()))

    def test_snake_score_prefers_left_bottom_anchor(self) -> None:
        anchored = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [16, 0, 0, 0],
        ]
        drifting = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 16],
        ]

        self.assertGreater(self.ai._snake_score(anchored), self.ai._snake_score(drifting))

    def test_detect_acceleration_backend_defaults_to_cpu(self) -> None:
        with patch("game2048.acceleration.load_cupy", return_value=None), patch("game2048.acceleration._load_module", return_value=None):
            backend = detect_acceleration_backend()

        self.assertEqual(backend.name, "CPU")
        self.assertTrue(backend.active)
        self.assertIn("GPU", backend.detail)

    def test_detect_acceleration_backend_uses_cupy_when_available(self) -> None:
        class FakeRuntime:
            @staticmethod
            def getDeviceCount() -> int:
                return 1

            @staticmethod
            def getDeviceProperties(_index: int) -> dict:
                return {"name": b"Fake GPU"}

        class FakeCuda:
            runtime = FakeRuntime()

        class FakeCuPy:
            cuda = FakeCuda()

        with patch("game2048.acceleration.probe_cupy", return_value=(FakeCuPy(), 1, "Fake GPU")), patch(
            "game2048.acceleration.load_cupy", return_value=FakeCuPy()
        ):
            backend = detect_acceleration_backend()

        self.assertEqual(backend.name, "CuPy")
        self.assertTrue(backend.active)
        self.assertIn("Fake GPU", backend.detail)

    def test_detect_acceleration_backend_reports_disabled_gpu(self) -> None:
        class FakeRuntime:
            @staticmethod
            def getDeviceCount() -> int:
                return 1

            @staticmethod
            def getDeviceProperties(_index: int) -> dict:
                return {"name": b"Fake GPU"}

        class FakeCuda:
            runtime = FakeRuntime()

        class FakeCuPy:
            cuda = FakeCuda()

        with patch("game2048.acceleration.probe_cupy", return_value=(FakeCuPy(), 1, "Fake GPU")):
            backend = detect_acceleration_backend(prefer_gpu=False)

        self.assertEqual(backend.name, "CPU")
        self.assertTrue(backend.active)
        self.assertIn("已停用", backend.detail)

    def test_ai_exposes_backend_summary(self) -> None:
        backend = AccelerationBackend(
            name="CUDA",
            available=True,
            active=False,
            detail="偵測到測試 GPU；目前演算法仍使用 CPU 搜尋",
        )

        with patch("game2048.ai.detect_acceleration_backend", return_value=backend):
            ai = HeuristicExpectimaxAI(self.game, depth=2)

        self.assertEqual(ai.backend, backend)
        self.assertIn("CUDA", ai.backend_summary)

    def test_evaluate_dispatches_to_gpu_when_backend_is_active(self) -> None:
        self.ai._cupy = object()
        self.ai.backend = AccelerationBackend(
            name="CuPy",
            available=True,
            active=True,
            detail="GPU ready",
        )

        with patch.object(self.ai, "_evaluate_gpu", return_value=123.0) as evaluate_gpu:
            value = self.ai.evaluate(self.game.board)

        self.assertEqual(value, 123.0)
        evaluate_gpu.assert_called_once_with(self.game.board)

    def test_set_gpu_enabled_updates_backend(self) -> None:
        disabled_backend = AccelerationBackend("CPU", True, True, "GPU 可用，但目前已停用")
        enabled_backend = AccelerationBackend("CuPy", True, True, "GPU ready")

        with patch("game2048.ai.load_cupy", side_effect=[None, object()]), patch(
            "game2048.ai.detect_acceleration_backend", side_effect=[disabled_backend, enabled_backend]
        ):
            ai = HeuristicExpectimaxAI(self.game, depth=2, use_gpu=False)
            self.assertFalse(ai.use_gpu)
            self.assertEqual(ai.backend, disabled_backend)

            ai.set_gpu_enabled(True)

        self.assertTrue(ai.use_gpu)
        self.assertEqual(ai.backend, enabled_backend)


if __name__ == "__main__":
    unittest.main()
