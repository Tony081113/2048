import unittest

from game2048.core import Game2048


class TestGame2048Core(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game2048()

    def test_simulate_move_left_merges_once(self) -> None:
        board = [
            [2, 2, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        result = self.game.simulate_move(board, "left")
        self.assertTrue(result.moved)
        self.assertEqual(result.score_gained, 4)
        self.assertEqual(result.board[0], [4, 4, 0, 0])

    def test_simulate_move_no_change(self) -> None:
        board = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 64],
        ]
        result = self.game.simulate_move(board, "left")
        self.assertFalse(result.moved)
        self.assertEqual(result.score_gained, 0)

    def test_can_move_false_when_full_and_blocked(self) -> None:
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        self.assertFalse(self.game.can_move(board))


if __name__ == "__main__":
    unittest.main()
