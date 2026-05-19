import random
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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


class HeuristicExpectimaxAI:
    def __init__(self, game: Game2048, depth: int = 2) -> None:
        self.game = game
        self.depth = depth
        self.weights = {
            "empty_cells": 2.7,
            "monotonicity": 1.0,
            "merge_potential": 1.5,
            "max_tile": 1.0,
        }

    def choose_move(self, board: Board) -> Tuple[Optional[Direction], Dict[Direction, float]]:
        best_move: Optional[Direction] = None
        best_value = float("-inf")
        values: Dict[Direction, float] = {}

        for direction in ("up", "down", "left", "right"):
            result = self.game.simulate_move(board, direction)
            if not result.moved:
                values[direction] = float("-inf")
                continue
            value = self.expectimax(result.board, self.depth - 1, is_chance=True)
            values[direction] = value
            if value > best_value:
                best_value = value
                best_move = direction

        return best_move, values

    def expectimax(self, board: Board, depth: int, is_chance: bool) -> float:
        if depth <= 0 or not self.game.can_move(board):
            return self.evaluate(board)

        if is_chance:
            empties = self.game.empty_cells(board)
            if not empties:
                return self.evaluate(board)
            total = 0.0
            prob_per_cell = 1.0 / len(empties)
            for r, c in empties:
                for val, p in ((2, 0.9), (4, 0.1)):
                    next_board = [row[:] for row in board]
                    next_board[r][c] = val
                    total += prob_per_cell * p * self.expectimax(next_board, depth - 1, False)
            return total

        best = float("-inf")
        for direction in ("up", "down", "left", "right"):
            result = self.game.simulate_move(board, direction)
            if not result.moved:
                continue
            best = max(best, self.expectimax(result.board, depth - 1, True))
        return best if best != float("-inf") else self.evaluate(board)

    def evaluate(self, board: Board) -> float:
        empties = len(self.game.empty_cells(board))
        max_tile = self.game.max_tile(board)
        monotonicity = self._monotonicity(board)
        merge_potential = self._merge_potential(board)

        return (
            self.weights["empty_cells"] * empties
            + self.weights["monotonicity"] * monotonicity
            + self.weights["merge_potential"] * merge_potential
            + self.weights["max_tile"] * (max_tile.bit_length() - 1 if max_tile > 0 else 0)
        )

    def _monotonicity(self, board: Board) -> float:
        score = 0.0
        for row in board:
            for i in range(3):
                score -= abs(row[i] - row[i + 1]) / 1024.0
        for c in range(4):
            for r in range(3):
                score -= abs(board[r][c] - board[r + 1][c]) / 1024.0
        return score

    def _merge_potential(self, board: Board) -> float:
        score = 0.0
        for r in range(4):
            for c in range(4):
                current = board[r][c]
                if current == 0:
                    continue
                if c + 1 < 4 and board[r][c + 1] == current:
                    score += 1
                if r + 1 < 4 and board[r + 1][c] == current:
                    score += 1
        return score


class GameGUI:
    TILE_COLORS = {
        0: "#cdc1b4",
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f2b179",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc850",
        1024: "#edc53f",
        2048: "#edc22e",
    }

    def __init__(self) -> None:
        self.game = Game2048()
        self.ai = HeuristicExpectimaxAI(self.game, depth=2)

        self.root = tk.Tk()
        self.root.title("2048 + AI Training")
        self.root.resizable(False, False)

        self.mode = tk.StringVar(value="MANUAL")
        self.episode = 0
        self.ai_games = 0
        self.ai_wins = 0
        self.best_ai_tile = 0
        self.last_values: Dict[str, float] = {}

        self.delay_ms = tk.IntVar(value=200)
        self.ai_running = False

        self._build_ui()
        self._bind_keys()
        self.draw()

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#faf8ef")
        container.pack(padx=12, pady=12)

        left = tk.Frame(container, bg="#faf8ef")
        left.grid(row=0, column=0, sticky="n")

        right = tk.Frame(container, bg="#f3eee6", bd=1, relief="solid")
        right.grid(row=0, column=1, padx=(12, 0), sticky="ns")

        header = tk.Frame(left, bg="#faf8ef")
        header.pack(fill="x", pady=(0, 8))

        self.score_label = tk.Label(header, text="Score: 0", font=("Arial", 12, "bold"), bg="#faf8ef")
        self.score_label.pack(side="left", padx=(0, 8))

        self.best_label = tk.Label(header, text="Best Score: 0", font=("Arial", 12, "bold"), bg="#faf8ef")
        self.best_label.pack(side="left")

        self.status_label = tk.Label(left, text="Mode: MANUAL", font=("Arial", 11), bg="#faf8ef", fg="#555")
        self.status_label.pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(left, width=420, height=420, bg="#bbada0", highlightthickness=0)
        self.canvas.pack()

        control_row = tk.Frame(left, bg="#faf8ef")
        control_row.pack(fill="x", pady=(8, 0))

        self.toggle_btn = tk.Button(control_row, text="Switch to AI Mode (M)", command=self.toggle_mode)
        self.toggle_btn.pack(side="left", padx=(0, 8))

        tk.Button(control_row, text="Restart (R)", command=self.restart_game).pack(side="left")

        tk.Label(right, text="AI 資訊面板", font=("Arial", 12, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 8))

        self.episode_label = tk.Label(right, text="Episode: 0", bg="#f3eee6", anchor="w")
        self.episode_label.pack(fill="x", padx=10)

        self.winrate_label = tk.Label(right, text="Win Rate (>=2048): 0.00%", bg="#f3eee6", anchor="w")
        self.winrate_label.pack(fill="x", padx=10)

        self.max_tile_label = tk.Label(right, text="Best AI Tile: 0", bg="#f3eee6", anchor="w")
        self.max_tile_label.pack(fill="x", padx=10)

        tk.Label(right, text="Decision Weights", font=("Arial", 10, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 4))
        self.weights_label = tk.Label(right, bg="#f3eee6", justify="left", anchor="w")
        self.weights_label.pack(fill="x", padx=10)

        tk.Label(right, text="Last Move Scores", font=("Arial", 10, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 4))
        self.values_label = tk.Label(right, bg="#f3eee6", justify="left", anchor="w")
        self.values_label.pack(fill="x", padx=10)

        tk.Label(right, text="AI Speed (Delay ms)", bg="#f3eee6").pack(anchor="w", padx=10, pady=(12, 2))
        self.speed_scale = tk.Scale(
            right,
            from_=20,
            to=1000,
            resolution=10,
            orient="horizontal",
            variable=self.delay_ms,
            bg="#f3eee6",
            highlightthickness=0,
            length=230,
        )
        self.speed_scale.pack(padx=10, pady=(0, 10))

        tk.Label(right, text="Hotkeys: Arrow keys=move, M=mode, +/-=speed", bg="#f3eee6", fg="#555").pack(
            anchor="w", padx=10, pady=(0, 10)
        )

    def _bind_keys(self) -> None:
        self.root.bind("<Up>", lambda _e: self.manual_move("up"))
        self.root.bind("<Down>", lambda _e: self.manual_move("down"))
        self.root.bind("<Left>", lambda _e: self.manual_move("left"))
        self.root.bind("<Right>", lambda _e: self.manual_move("right"))
        self.root.bind("m", lambda _e: self.toggle_mode())
        self.root.bind("M", lambda _e: self.toggle_mode())
        self.root.bind("r", lambda _e: self.restart_game())
        self.root.bind("R", lambda _e: self.restart_game())
        self.root.bind("plus", lambda _e: self.adjust_speed(-20))
        self.root.bind("minus", lambda _e: self.adjust_speed(20))
        self.root.bind("equal", lambda _e: self.adjust_speed(-20))
        self.root.bind("underscore", lambda _e: self.adjust_speed(20))

    def adjust_speed(self, delta: int) -> None:
        value = max(20, min(1000, self.delay_ms.get() + delta))
        self.delay_ms.set(value)
        self.draw()

    def restart_game(self) -> None:
        self.game.reset()
        self.last_values = {}
        self.draw()

    def toggle_mode(self) -> None:
        if self.mode.get() == "MANUAL":
            self.mode.set("AI")
            self.toggle_btn.configure(text="Switch to Manual Mode (M)")
            self.ai_running = True
            self.ai_step()
        else:
            self.mode.set("MANUAL")
            self.toggle_btn.configure(text="Switch to AI Mode (M)")
            self.ai_running = False
        self.draw()

    def manual_move(self, direction: Direction) -> None:
        if self.mode.get() != "MANUAL":
            return
        moved = self.game.move(direction)
        if moved and not self.game.can_move():
            self.status_label.configure(text="Mode: MANUAL (Game Over, press R to restart)")
        self.draw()

    def ai_step(self) -> None:
        if not self.ai_running or self.mode.get() != "AI":
            return

        if not self.game.can_move():
            self.finish_ai_episode()
            self.game.reset()

        move, values = self.ai.choose_move(self.game.board)
        self.last_values = values
        if move is not None:
            self.game.move(move)

        if not self.game.can_move():
            self.finish_ai_episode()
            self.game.reset()

        self.draw()
        self.root.after(self.delay_ms.get(), self.ai_step)

    def finish_ai_episode(self) -> None:
        self.episode += 1
        self.ai_games += 1
        tile = self.game.max_tile()
        self.best_ai_tile = max(self.best_ai_tile, tile)
        if tile >= 2048:
            self.ai_wins += 1

    def draw(self) -> None:
        self.score_label.configure(text=f"Score: {self.game.score}")
        self.best_label.configure(text=f"Best Score: {self.game.best_score}")
        self.status_label.configure(text=f"Mode: {self.mode.get()}")
        self.episode_label.configure(text=f"Episode: {self.episode}")

        win_rate = (self.ai_wins / self.ai_games * 100.0) if self.ai_games else 0.0
        self.winrate_label.configure(text=f"Win Rate (>=2048): {win_rate:.2f}%")
        self.max_tile_label.configure(text=f"Best AI Tile: {self.best_ai_tile}")

        self.weights_label.configure(
            text="\n".join([f"{k}: {v:.2f}" for k, v in self.ai.weights.items()])
        )

        lines = []
        for d in ("up", "down", "left", "right"):
            v = self.last_values.get(d)
            text = "-inf" if v == float("-inf") else f"{v:.2f}" if v is not None else "N/A"
            lines.append(f"{d:<5}: {text}")
        self.values_label.configure(text="\n".join(lines))

        self.canvas.delete("all")
        margin = 10
        size = 95
        gap = 10

        for r in range(4):
            for c in range(4):
                x1 = margin + c * (size + gap)
                y1 = margin + r * (size + gap)
                x2 = x1 + size
                y2 = y1 + size
                value = self.game.board[r][c]
                color = self.TILE_COLORS.get(value, "#3c3a32")
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
                if value:
                    text_color = "#f9f6f2" if value >= 8 else "#776e65"
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(value),
                        fill=text_color,
                        font=("Arial", 20, "bold"),
                    )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = GameGUI()
    app.run()


if __name__ == "__main__":
    main()
