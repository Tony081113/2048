"""
獨立圖表觀察視窗（Toplevel），與遊戲主視窗分開顯示。
分三個分頁：
  分頁 1：每局分數折線 + 累積勝率（右軸）
  分頁 2：每局最高方塊散點 + 移動平均
  分頁 3：6 條權重學習曲線（每 BATCH_SIZE 局一個點）
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ai import HeuristicExpectimaxAI

import matplotlib
import matplotlib.ticker
import matplotlib.patches as mpatches
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 自動刷新間隔（毫秒），AI 跑圖時圖表跟著更新
AUTO_REFRESH_MS = 5000


def _moving_avg(data, window: int = 20):
    """簡單移動平均，不足 window 個點時取現有平均"""
    result = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        result.append(sum(data[lo : i + 1]) / (i - lo + 1))
    return result


class ChartWindow:
    """獨立圖表視窗，與遊戲主視窗分開、可同時開著。"""

    def __init__(self, parent: tk.Tk, ai: "HeuristicExpectimaxAI") -> None:
        self.ai = ai
        self._auto_refresh_id = None

        self.win = tk.Toplevel(parent)
        self.win.title("AI 學習觀察圖表")
        self.win.geometry("960x600")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # 分頁容器
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        tab_score = tk.Frame(notebook, bg="white")
        tab_tile = tk.Frame(notebook, bg="white")
        tab_weights = tk.Frame(notebook, bg="white")
        notebook.add(tab_score,   text="  分數 & 勝率  ")
        notebook.add(tab_tile,    text="  最高方塊  ")
        notebook.add(tab_weights, text="  權重學習  ")

        self._fig_score,   self._cv_score   = self._make_canvas(tab_score)
        self._fig_tile,    self._cv_tile     = self._make_canvas(tab_tile)
        self._fig_weights, self._cv_weights  = self._make_canvas(tab_weights)

        # 底部工具列
        bar = tk.Frame(self.win, bg="#f0f0f0", pady=5)
        bar.pack(side="bottom", fill="x", padx=8)
        tk.Button(bar, text="🔄 重新整理", command=self.refresh).pack(side="left", padx=4)
        self._auto_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="自動刷新（每 5 秒）",
                       variable=self._auto_var, bg="#f0f0f0",
                       command=self._toggle_auto).pack(side="left", padx=8)
        self._status = tk.Label(bar, text="", fg="#666", bg="#f0f0f0")
        self._status.pack(side="left", padx=8)

        self.refresh()
        self._schedule_auto_refresh()

    # ── 工具 ─────────────────────────────────────────────────────

    @staticmethod
    def _make_canvas(parent: tk.Frame):
        fig = Figure(figsize=(8.8, 4.5), dpi=96, facecolor="#fafafa")
        cv = FigureCanvasTkAgg(fig, master=parent)
        cv.get_tk_widget().pack(fill="both", expand=True)
        return fig, cv

    @staticmethod
    def _empty_plot(fig, cv, msg: str) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=13, color="#aaa", transform=ax.transAxes)
        ax.set_axis_off()
        cv.draw()

    def _on_close(self) -> None:
        if self._auto_refresh_id:
            self.win.after_cancel(self._auto_refresh_id)
        self.win.destroy()

    def _toggle_auto(self) -> None:
        if self._auto_var.get():
            self._schedule_auto_refresh()

    def _schedule_auto_refresh(self) -> None:
        if self._auto_var.get() and self.win.winfo_exists():
            self.refresh()
            self._auto_refresh_id = self.win.after(AUTO_REFRESH_MS, self._schedule_auto_refresh)

    # ── 公開 API ─────────────────────────────────────────────────

    def refresh(self) -> None:
        """重新繪製全部三個分頁。"""
        n = len(self.ai.score_history)
        b = len(self.ai.weights_history)
        self._status.configure(text=f"資料：{n} 局 / {b} 批次")
        self._draw_score()
        self._draw_tile()
        self._draw_weights()

    # ── 分頁一：分數 & 勝率 ──────────────────────────────────────

    def _draw_score(self) -> None:
        scores = self.ai.score_history
        if not scores:
            self._empty_plot(self._fig_score, self._cv_score,
                             "尚無數據，請先切換至 AI 模式執行幾局")
            return

        tiles = self.ai.tile_history
        episodes = list(range(1, len(scores) + 1))
        avg_scores = _moving_avg(scores, 20)
        cum_wr = [
            sum(1 for t in tiles[:i + 1] if t >= 2048) / (i + 1) * 100
            for i in range(len(tiles))
        ]

        fig = self._fig_score
        fig.clear()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.fill_between(episodes, scores, alpha=0.12, color="#2196F3")
        ax1.plot(episodes, scores,     color="#90CAF9", linewidth=0.8, alpha=0.6, label="每局分數")
        ax1.plot(episodes, avg_scores, color="#1565C0", linewidth=2.0, label="移動平均（20局）")
        ax2.plot(episodes, cum_wr,     color="#F44336", linewidth=1.8,
                 linestyle="--", label="累積勝率 %")

        ax1.set_xlabel("局數", fontsize=9)
        ax1.set_ylabel("分數", fontsize=9, color="#1565C0")
        ax2.set_ylabel("累積勝率 (%)", fontsize=9, color="#F44336")
        ax1.tick_params(axis="y", labelcolor="#1565C0")
        ax2.tick_params(axis="y", labelcolor="#F44336")
        ax2.set_ylim(0, 105)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

        win_pct = cum_wr[-1] if cum_wr else 0
        fig.suptitle(
            f"分數趨勢 & 勝率（共 {len(scores)} 局，當前勝率 {win_pct:.1f}%）",
            fontsize=10)
        fig.tight_layout()
        self._cv_score.draw()

    # ── 分頁二：最高方塊 ─────────────────────────────────────────

    def _draw_tile(self) -> None:
        tiles = self.ai.tile_history
        if not tiles:
            self._empty_plot(self._fig_tile, self._cv_tile,
                             "尚無數據，請先切換至 AI 模式執行幾局")
            return

        episodes = list(range(1, len(tiles) + 1))
        avg_tiles = _moving_avg(tiles, 20)

        colors = []
        for t in tiles:
            if t >= 2048:   colors.append("#4CAF50")
            elif t >= 1024: colors.append("#FF9800")
            elif t >= 512:  colors.append("#FFC107")
            else:           colors.append("#90A4AE")

        fig = self._fig_tile
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(episodes, tiles, c=colors, s=16, alpha=0.75, zorder=2)
        ax.plot(episodes, avg_tiles, color="#9C27B0", linewidth=2.0,
                zorder=3, label="移動平均（20局）")

        patches = [
            mpatches.Patch(color="#4CAF50", label="≥ 2048（勝）"),
            mpatches.Patch(color="#FF9800", label="≥ 1024"),
            mpatches.Patch(color="#FFC107", label="≥ 512"),
            mpatches.Patch(color="#90A4AE", label="< 512"),
        ]
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=patches + h, labels=[p.get_label() for p in patches] + l,
                  fontsize=8, loc="upper left")

        ax.set_xlabel("局數", fontsize=9)
        ax.set_ylabel("最高方塊", fontsize=9)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v))))

        win_count = sum(1 for t in tiles if t >= 2048)
        fig.suptitle(f"每局最高方塊（共 {len(tiles)} 局，勝 {win_count} 局）", fontsize=10)
        fig.tight_layout()
        self._cv_tile.draw()

    # ── 分頁三：權重學習 ─────────────────────────────────────────

    def _draw_weights(self) -> None:
        wh = self.ai.weights_history
        if not wh:
            self._empty_plot(self._fig_weights, self._cv_weights,
                             "尚無數據（每 5 局批次更新一次）")
            return

        keys = list(wh[0].keys())
        batches = list(range(1, len(wh) + 1))
        palette = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

        fig = self._fig_weights
        fig.clear()
        ax = fig.add_subplot(111)

        for i, k in enumerate(keys):
            vals = [snap[k] for snap in wh]
            ax.plot(batches, vals, marker=".", markersize=4,
                    linewidth=1.6, color=palette[i % len(palette)], label=k)

        ax.axhline(y=0.05, color="#bbb", linestyle="--", linewidth=0.8)
        ax.set_xlabel("批次（每 5 局）", fontsize=9)
        ax.set_ylabel("權重值", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

        fig.suptitle(f"權重學習曲線（共 {len(wh)} 個批次）", fontsize=10)
        fig.tight_layout()
        self._cv_weights.draw()
