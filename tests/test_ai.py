import unittest

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


if __name__ == "__main__":
    unittest.main()
