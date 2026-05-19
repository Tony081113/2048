import tkinter as tk
from typing import Dict

from .ai import HeuristicExpectimaxAI
from .core import Direction, Game2048


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

        self.weights_label.configure(text="\n".join([f"{k}: {v:.2f}" for k, v in self.ai.weights.items()]))

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
