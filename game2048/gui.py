import tkinter as tk
from tkinter import messagebox
from typing import Dict, Optional

from .ai import HeuristicExpectimaxAI
from .core import Direction, Game2048

# 嘗試匯入圖表模組；若 matplotlib 未安裝則優雅降級
try:
    from .charts import ChartWindow
    _CHARTS_AVAILABLE = True
except ImportError:
    ChartWindow = None  # type: ignore
    _CHARTS_AVAILABLE = False


# 方塊顏色對應表（數值 → 背景色）
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
        # 初始化遊戲核心與 AI
        self.game = Game2048()
        self.ai = HeuristicExpectimaxAI(self.game, depth=4)

        self.root = tk.Tk()
        self.root.title("2048 + AI 訓練")
        self.root.state("zoomed")  # 啟動時最大化視窗

        # 預設為手動模式
        self.mode = tk.StringVar(value="手動")
        self.episode = 0       # AI 累計回合數
        self.ai_games = 0      # AI 累計局數
        self.ai_wins = 0       # AI 達到 2048 的次數
        self.best_ai_tile = 0  # AI 出現過的最高方塊值
        self.last_values: Dict[str, float] = {}  # 上一步各方向評估分數

        self.delay_ms = tk.IntVar(value=200)  # AI 每步延遲（毫秒）
        self.ai_running = False
        self._fullscreen = False  # 真正全螢幕狀態（F11 切換）
        self._chart_win: Optional["ChartWindow"] = None  # 圖表視窗參考

        self._build_ui()
        self._bind_keys()
        self.draw()

    def _build_ui(self) -> None:
        # 根視窗設定可伸縮（全螢幕支援）
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = tk.Frame(self.root, bg="#faf8ef")
        container.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        container.columnconfigure(0, weight=1)   # 左側棋盤區可伸縮
        container.columnconfigure(1, weight=0)   # 右側面板固定寬度
        container.rowconfigure(0, weight=1)

        # 左側：標題、狀態、棋盤、控制按鈕
        left = tk.Frame(container, bg="#faf8ef")
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)  # 第 2 列（Canvas）可垂直伸縮

        # 右側 AI 資訊面板（固定寬度）
        right = tk.Frame(container, bg="#f3eee6", bd=1, relief="solid")
        right.grid(row=0, column=1, padx=(12, 0), sticky="ns")

        # 頂部分數列
        header = tk.Frame(left, bg="#faf8ef")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.score_label = tk.Label(header, text="分數: 0", font=("Arial", 14, "bold"), bg="#faf8ef")
        self.score_label.pack(side="left", padx=(0, 12))

        self.best_label = tk.Label(header, text="最高分: 0", font=("Arial", 14, "bold"), bg="#faf8ef")
        self.best_label.pack(side="left")

        self.status_label = tk.Label(left, text="模式: 手動", font=("Arial", 12), bg="#faf8ef", fg="#555")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(0, 8))

        # 遊戲畫布（隨視窗大小縮放）
        self.canvas = tk.Canvas(left, bg="#bbada0", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self.draw())  # 視窗縮放時重繪

        # 控制按鈕列
        control_row = tk.Frame(left, bg="#faf8ef")
        control_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self.toggle_btn = tk.Button(control_row, text="切換至 AI 模式 (M)", command=self.toggle_mode)
        self.toggle_btn.pack(side="left", padx=(0, 8))

        tk.Button(control_row, text="重新開始 (R)", command=self.restart_game).pack(side="left", padx=(0, 8))

        # 觀察圖表按鈕（開啟獨立圖表視窗）
        chart_btn_color = "#e8f5e9" if _CHARTS_AVAILABLE else "#ffebee"
        tk.Button(control_row, text="觀察圖表 (C)", command=self.open_charts,
                  bg=chart_btn_color, activebackground="#c8e6c9").pack(side="left")

        # 右側 AI 資訊面板
        tk.Label(right, text="AI 資訊面板", font=("Arial", 12, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 8))

        self.episode_label = tk.Label(right, text="回合: 0", bg="#f3eee6", anchor="w")
        self.episode_label.pack(fill="x", padx=10)

        self.winrate_label = tk.Label(right, text="勝率 (>=2048): 0.00%", bg="#f3eee6", anchor="w")
        self.winrate_label.pack(fill="x", padx=10)

        self.max_tile_label = tk.Label(right, text="AI 最高方塊: 0", bg="#f3eee6", anchor="w")
        self.max_tile_label.pack(fill="x", padx=10)

        tk.Label(right, text="決策權重（已學習）", font=("Arial", 10, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 4))
        self.weights_label = tk.Label(right, bg="#f3eee6", justify="left", anchor="w")
        self.weights_label.pack(fill="x", padx=10)

        tk.Label(right, text="學習進度", font=("Arial", 10, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 4))
        self.baseline_label = tk.Label(right, text="基準獎勵: 0", bg="#f3eee6", anchor="w")
        self.baseline_label.pack(fill="x", padx=10)
        self.last_reward_label = tk.Label(right, text="上局獎勵: 0", bg="#f3eee6", anchor="w")
        self.last_reward_label.pack(fill="x", padx=10)
        self.learn_ep_label = tk.Label(right, text="訓練局數: 0", bg="#f3eee6", anchor="w")
        self.learn_ep_label.pack(fill="x", padx=10)

        tk.Label(right, text="上次移動分數", font=("Arial", 10, "bold"), bg="#f3eee6").pack(anchor="w", padx=10, pady=(10, 4))
        self.values_label = tk.Label(right, bg="#f3eee6", justify="left", anchor="w")
        self.values_label.pack(fill="x", padx=10)

        tk.Label(right, text="AI 速度 (延遲 ms)", bg="#f3eee6").pack(anchor="w", padx=10, pady=(12, 2))
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

        tk.Label(right, text="快捷鍵: 方向鍵=移動, M=模式\n+/-=速度, C=圖表, F11=全螢幕", bg="#f3eee6", fg="#555", justify="left").pack(
            anchor="w", padx=10, pady=(0, 10)
        )

    def _bind_keys(self) -> None:
        # 使用 bind_all 確保任何元件有焦點時皆可觸發
        self.root.bind_all("<Up>", lambda _e: self.manual_move("up"))
        self.root.bind_all("<Down>", lambda _e: self.manual_move("down"))
        self.root.bind_all("<Left>", lambda _e: self.manual_move("left"))
        self.root.bind_all("<Right>", lambda _e: self.manual_move("right"))
        self.root.bind_all("m", lambda _e: self.toggle_mode())
        self.root.bind_all("M", lambda _e: self.toggle_mode())
        self.root.bind_all("r", lambda _e: self.restart_game())
        self.root.bind_all("R", lambda _e: self.restart_game())
        self.root.bind_all("c", lambda _e: self.open_charts())
        self.root.bind_all("C", lambda _e: self.open_charts())
        # + 鍵加速（含數字鍵盤）
        self.root.bind_all("+", lambda _e: self.adjust_speed(-20))
        self.root.bind_all("=", lambda _e: self.adjust_speed(-20))
        self.root.bind_all("<KP_Add>", lambda _e: self.adjust_speed(-20))
        # - 鍵減速（含數字鍵盤）
        self.root.bind_all("-", lambda _e: self.adjust_speed(20))
        self.root.bind_all("_", lambda _e: self.adjust_speed(20))
        self.root.bind_all("<KP_Subtract>", lambda _e: self.adjust_speed(20))
        # F11 切換真正全螢幕
        self.root.bind_all("<F11>", lambda _e: self._toggle_fullscreen())
        self.root.bind_all("<Escape>", lambda _e: self._exit_fullscreen())

    def _toggle_fullscreen(self) -> None:
        # 切換真正全螢幕（無標題列）
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self) -> None:
        # Escape 退出全螢幕
        if self._fullscreen:
            self._fullscreen = False
            self.root.attributes("-fullscreen", False)

    def adjust_speed(self, delta: int) -> None:
        # 調整 AI 步驟延遲，限制在 20~1000 ms 之間
        value = max(20, min(1000, self.delay_ms.get() + delta))
        self.delay_ms.set(value)
        self.draw()

    def restart_game(self) -> None:
        # 重置遊戲狀態與上次評估值
        self.game.reset()
        self.last_values = {}
        self.draw()

    def toggle_mode(self) -> None:
        # 在手動模式與 AI 模式之間切換
        if self.mode.get() == "手動":
            self.mode.set("AI")
            self.toggle_btn.configure(text="切換至手動模式 (M)")
            self.ai_running = True
            self.ai_step()
        else:
            self.mode.set("手動")
            self.toggle_btn.configure(text="切換至 AI 模式 (M)")
            self.ai_running = False
        self.draw()

    def open_charts(self) -> None:
        """開啟或重新整理獨立圖表視窗（與遊戲視窗分開）"""
        if not _CHARTS_AVAILABLE:
            messagebox.showerror(
                "缺少 matplotlib",
                "圖表功能需要 matplotlib，請在命令列執行：\n\n"
                "  pip install matplotlib\n\n"
                "安裝後重新啟動遊戲即可使用圖表。"
            )
            return

        if self._chart_win is None or not self._chart_win.win.winfo_exists():
            self._chart_win = ChartWindow(self.root, self.ai)
        else:
            self._chart_win.win.lift()
            self._chart_win.win.focus_force()
            self._chart_win.refresh()

    def manual_move(self, direction: Direction) -> None:
        # 手動模式下處理玩家移動
        if self.mode.get() != "手動":
            return
        moved = self.game.move(direction)
        if moved and not self.game.can_move():
            self.status_label.configure(text="模式: 手動 (遊戲結束，按 R 重新開始)")
        self.draw()

    def ai_step(self) -> None:
        # AI 自動走一步，並排程下一步
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
        # 結算一局 AI 遊戲，並觸發 ES 權重更新
        self.ai_games += 1
        tile = self.game.max_tile()
        self.best_ai_tile = max(self.best_ai_tile, tile)
        if tile >= 2048:
            self.ai_wins += 1

        # 將本局分數與最高方塊回饋給 AI 學習（獎懲機制）
        self.ai.on_episode_end(self.game.score, tile)
        self.episode = self.ai.episode_count

    def draw(self) -> None:
        # 更新所有 UI 標籤與棋盤繪製
        self.score_label.configure(text=f"分數: {self.game.score}")
        self.best_label.configure(text=f"最高分: {self.game.best_score}")
        self.status_label.configure(text=f"模式: {self.mode.get()}")
        self.episode_label.configure(text=f"回合: {self.episode}")

        win_rate = (self.ai_wins / self.ai_games * 100.0) if self.ai_games else 0.0
        self.winrate_label.configure(text=f"勝率 (>=2048): {win_rate:.2f}%")
        self.max_tile_label.configure(text=f"AI 最高方塊: {self.best_ai_tile}")

        # 顯示已學習的基準權重（非擾動值）
        self.weights_label.configure(text="\n".join([f"{k}: {v:.3f}" for k, v in self.ai.base_weights.items()]))

        # 學習進度
        self.baseline_label.configure(text=f"基準獎勵: {self.ai.baseline:.2f}")
        self.last_reward_label.configure(text=f"上局獎勵: {self.ai.last_reward:.2f}")
        self.learn_ep_label.configure(text=f"訓練局數: {self.ai.episode_count}")

        # 顯示上次各方向的評估分數
        dir_labels = {"up": "上", "down": "下", "left": "左", "right": "右"}
        lines = []
        for d in ("up", "down", "left", "right"):
            v = self.last_values.get(d)
            text = "-inf" if v == float("-inf") else f"{v:.2f}" if v is not None else "N/A"
            lines.append(f"{dir_labels[d]}: {text}")
        self.values_label.configure(text="\n".join(lines))

        # 動態計算方塊尺寸（依 Canvas 實際大小縮放）
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        board_px = min(w, h)
        margin = max(8, int(board_px * 0.024))
        gap = max(5, int(board_px * 0.024))
        size = (board_px - 2 * margin - 3 * gap) // 4
        font_size = max(12, size // 4)

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
                        font=("Arial", font_size, "bold"),
                    )

    def _on_close(self) -> None:
        # 關閉前強制儲存最新權重
        self.ai._save_weights()
        self.root.destroy()

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()
