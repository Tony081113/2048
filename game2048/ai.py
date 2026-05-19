import json
import os
import random
from typing import Dict, Optional, Tuple

from .core import Board, Direction, Game2048


# 學習到的基準權重儲存路徑
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "artifacts", "weights.json")

DEFAULT_WEIGHTS = {
    "empty_cells": 2.7,
    "monotonicity": 1.0,
    "merge_potential": 1.5,
    "max_tile": 1.0,
}


class HeuristicExpectimaxAI:
    def __init__(self, game: Game2048, depth: int = 2) -> None:
        self.game = game
        self.depth = depth

        # Evolution Strategies 學習超參數
        self.sigma = 0.15          # 每局權重擾動的標準差
        self.lr = 0.03             # 梯度更新學習率
        self.baseline_alpha = 0.1  # 基準線指數移動平均係數

        # 學習狀態
        self.baseline = 0.0        # 獎勵基準線（EMA）
        self.last_reward = 0.0     # 上一局獎勵
        self.episode_count = 0     # 總訓練局數
        self._noise: Dict[str, float] = {}

        # 從檔案載入已學習的基準權重，或使用預設值
        self.base_weights = self._load_weights()
        # 套用初始擾動，得到本局實際使用的權重
        self._apply_perturbation()

    def _load_weights(self) -> Dict[str, float]:
        """從 JSON 載入已學習的基準權重"""
        try:
            with open(WEIGHTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return {k: float(data.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return DEFAULT_WEIGHTS.copy()

    def _save_weights(self) -> None:
        """持久化儲存基準權重"""
        os.makedirs(os.path.dirname(os.path.abspath(WEIGHTS_FILE)), exist_ok=True)
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.base_weights, f, indent=2, ensure_ascii=False)

    def _apply_perturbation(self) -> None:
        """對基準權重加高斯擾動，得到本局決策用的權重"""
        self._noise = {k: random.gauss(0, self.sigma) for k in self.base_weights}
        self.weights = {k: max(0.05, self.base_weights[k] + self._noise[k]) for k in self.base_weights}

    def on_episode_end(self, score: int, max_tile: int) -> float:
        """
        每局結束後根據獎懲更新基準權重（ES 梯度估計）。
        獎勵 = 分數 + 達到高方塊的里程碑加分
        """
        # 里程碑加分：鼓勵 AI 衝高方塊
        if max_tile >= 4096:
            milestone_bonus = 20000
        elif max_tile >= 2048:
            milestone_bonus = 8000
        elif max_tile >= 1024:
            milestone_bonus = 2000
        elif max_tile >= 512:
            milestone_bonus = 400
        else:
            milestone_bonus = 0

        reward = float(score + milestone_bonus)
        self.last_reward = reward

        # 計算優勢（本局超出基準線的程度）
        advantage = reward - self.baseline
        # 更新基準線（指數移動平均）
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward

        # ES 梯度估計：沿擾動方向調整基準權重
        for k in self.base_weights:
            grad = advantage * self._noise[k] / (self.sigma ** 2)
            self.base_weights[k] = max(0.05, self.base_weights[k] + self.lr * grad)

        self.episode_count += 1
        # 套用下一局的新擾動
        self._apply_perturbation()

        # 每 10 局持久化儲存一次
        if self.episode_count % 10 == 0:
            self._save_weights()

        return reward

    def choose_move(self, board: Board) -> Tuple[Optional[Direction], Dict[Direction, float]]:
        # 對每個方向執行 Expectimax，選出評估分最高的移動
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
        # 遞迴 Expectimax：chance 節點取期望值，max 節點取最大值
        if depth <= 0 or not self.game.can_move(board):
            return self.evaluate(board)

        if is_chance:
            # 機率節點：隨機新增方塊（2 佔 90%，4 佔 10%）
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

        # 最大化節點：選所有方向中的最高評估值
        best = float("-inf")
        for direction in ("up", "down", "left", "right"):
            result = self.game.simulate_move(board, direction)
            if not result.moved:
                continue
            best = max(best, self.expectimax(result.board, depth - 1, True))
        return best if best != float("-inf") else self.evaluate(board)

    def evaluate(self, board: Board) -> float:
        # 綜合評估棋盤狀態，回傳加權分數
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
        # 計算棋盤單調性：相鄰格差值越小越好
        score = 0.0
        for row in board:
            for i in range(3):
                score -= abs(row[i] - row[i + 1]) / 1024.0
        for c in range(4):
            for r in range(3):
                score -= abs(board[r][c] - board[r + 1][c]) / 1024.0
        return score

    def _merge_potential(self, board: Board) -> float:
        # 計算可合併的相鄰相同方塊數（橫向 + 縱向）
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
