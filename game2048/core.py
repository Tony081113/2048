import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


Board = List[List[int]]
Direction = str


@dataclass
class MoveResult:
    board: Board
    moved: bool
    score_gained: int


class Game2048:
    def __init__(self) -> None:
        self.size = 4
        self.score = 0
        self.best_score = 0
        self.board = self.new_board()
        self.add_random_tile()
        self.add_random_tile()

    def new_board(self) -> Board:
        return [[0 for _ in range(self.size)] for _ in range(self.size)]

    def clone_board(self, board: Optional[Board] = None) -> Board:
        src = self.board if board is None else board
        return [row[:] for row in src]

    def reset(self) -> None:
        self.score = 0
        self.board = self.new_board()
        self.add_random_tile()
        self.add_random_tile()

    def empty_cells(self, board: Optional[Board] = None) -> List[Tuple[int, int]]:
        b = self.board if board is None else board
        return [(r, c) for r in range(self.size) for c in range(self.size) if b[r][c] == 0]

    def add_random_tile(self, board: Optional[Board] = None) -> None:
        b = self.board if board is None else board
        empties = self.empty_cells(b)
        if not empties:
            return
        r, c = random.choice(empties)
        b[r][c] = 4 if random.random() < 0.1 else 2

    def _compress_line(self, line: List[int]) -> Tuple[List[int], int]:
        non_zero = [x for x in line if x != 0]
        result = []
        gained = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged = non_zero[i] * 2
                result.append(merged)
                gained += merged
                i += 2
            else:
                result.append(non_zero[i])
                i += 1
        result.extend([0] * (self.size - len(result)))
        return result, gained

    def simulate_move(self, board: Board, direction: Direction) -> MoveResult:
        b = [row[:] for row in board]
        moved = False
        total_gained = 0

        def set_row(r: int, line: List[int]) -> None:
            nonlocal moved
            if b[r] != line:
                moved = True
            b[r] = line

        def set_col(c: int, line: List[int]) -> None:
            nonlocal moved
            original = [b[r][c] for r in range(self.size)]
            if original != line:
                moved = True
            for r in range(self.size):
                b[r][c] = line[r]

        if direction == "left":
            for r in range(self.size):
                merged, gained = self._compress_line(b[r])
                total_gained += gained
                set_row(r, merged)
        elif direction == "right":
            for r in range(self.size):
                reversed_row = list(reversed(b[r]))
                merged, gained = self._compress_line(reversed_row)
                total_gained += gained
                set_row(r, list(reversed(merged)))
        elif direction == "up":
            for c in range(self.size):
                col = [b[r][c] for r in range(self.size)]
                merged, gained = self._compress_line(col)
                total_gained += gained
                set_col(c, merged)
        elif direction == "down":
            for c in range(self.size):
                col = [b[r][c] for r in range(self.size)][::-1]
                merged, gained = self._compress_line(col)
                total_gained += gained
                set_col(c, list(reversed(merged)))
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return MoveResult(board=b, moved=moved, score_gained=total_gained)

    def move(self, direction: Direction) -> bool:
        result = self.simulate_move(self.board, direction)
        if not result.moved:
            return False
        self.board = result.board
        self.score += result.score_gained
        self.best_score = max(self.best_score, self.score)
        self.add_random_tile()
        return True

    def can_move(self, board: Optional[Board] = None) -> bool:
        b = self.board if board is None else board
        if self.empty_cells(b):
            return True
        for r in range(self.size):
            for c in range(self.size):
                current = b[r][c]
                for dr, dc in ((1, 0), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if nr < self.size and nc < self.size and b[nr][nc] == current:
                        return True
        return False

    def max_tile(self, board: Optional[Board] = None) -> int:
        b = self.board if board is None else board
        return max(max(row) for row in b)
