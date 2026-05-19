import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from .core import Board, Direction, Game2048


# 學習到的基準權重儲存路徑
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "artifacts", "weights.json")

DEFAULT_WEIGHTS = {
    "empty_cells": 2.0,       # 略降：不過度保守
    "monotonicity": 1.0,
    "merge_potential": 0.9,
    "max_tile": 3.0,          # 最高：直接獎勵最大磚成長，是最重要的目標
    "snake": 2.0,             # 維持隊形但不壓過 max_tile
    "corner": 1.2,            # 低：不過度守成
}

# 蛇形梯度權重矩陣（左下角為最高優先位置）
# [3][0]=15（最大值理想位置）→ 沿蛇形路徑遞減到 [0][3]=0
SNAKE_WEIGHTS = [
    [ 3,  2,  1,  0],   # 第 0 列（遠離角落，低優先）
    [ 4,  5,  6,  7],   # 第 1 列
    [11, 10,  9,  8],   # 第 2 列
    [12, 13, 14, 15],   # 第 3 列（最接近左下角，高優先）
]
SNAKE_WEIGHT_SUM = 120  # 所有權重總和，用於正規化

# 機率節點最多採樣幾個空格（限制分支數，維持深度 4 的速度）
MAX_CHANCE_CELLS = 6

# 累積幾局後批次更新一次權重（降低 ES 單樣本噪聲）
BATCH_SIZE = 3


class HeuristicExpectimaxAI:
    def __init__(self, game: Game2048, depth: int = 4) -> None:
        self.game = game
        self.depth = depth

        # Evolution Strategies 學習超參數
        self.sigma = 0.15          # 每局權重擾動的標準差
        self.lr = 0.05             # 梯度更新學習率
        self.baseline_alpha = 0.1  # 基準線指數移動平均係數

        # 學習狀態
        self.baseline = 0.0        # 獎勵基準線（EMA）
        self.last_reward = 0.0     # 上一局獎勵（對數域）
        self.episode_count = 0     # 總訓練局數
        self._noise: Dict[str, float] = {}
        # 批次 buffer：儲存 (reward, noise) 等待批次更新
        self._episode_buffer: List[Tuple[float, Dict[str, float]]] = []

        # 歷史紀錄（供圖表觀察）
        self.score_history: List[int] = []
        self.tile_history: List[int] = []
        self.weights_history: List[Dict[str, float]] = []  # 每批次更新後快照

        # 從檔案載入已學習的基準權重，或使用預設值
        self.base_weights = self._load_weights()
        # 套用初始擾動，得到本局實際使用的權重
        self._apply_perturbation()

    def _load_weights(self) -> Dict[str, float]:
        """從 JSON 載入已學習的基準權重；若值域異常（舊版暴衝）則重置"""
        try:
            with open(WEIGHTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            weights = {k: float(data.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}
            # 任何權重超出 [0, 50] 視為損壞，重置
            if any(v < 0 or v > 50.0 for v in weights.values()):
                return DEFAULT_WEIGHTS.copy()
            return weights
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

    def _compute_reward(self, score: int, max_tile: int) -> float:
        """
        對數尺度獎勵 + 強化里程碑，讓 2048→4096→8192 的推力逐級增強。
        """
        log_score = math.log1p(score)
        if max_tile >= 8192:
            milestone = 30.0
        elif max_tile >= 4096:
            milestone = 20.0
        elif max_tile >= 2048:
            milestone = 10.0
        elif max_tile >= 1024:
            milestone = 6.0
        elif max_tile >= 512:
            milestone = 3.0
        else:
            milestone = 0.0
        return log_score + milestone

    def on_episode_end(self, score: int, max_tile: int) -> float:
        """
        每局結束後累積到 buffer，滿 BATCH_SIZE 局再做批次正規化梯度更新。
        """
        reward = self._compute_reward(score, max_tile)
        self.last_reward = reward

        # 更新基準線（EMA）
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward

        # 累積本局資料
        self._episode_buffer.append((reward, self._noise.copy()))
        self.episode_count += 1
        self._apply_perturbation()

        # 記錄歷史（供圖表觀察）
        self.score_history.append(score)
        self.tile_history.append(max_tile)

        # 批次更新
        if len(self._episode_buffer) >= BATCH_SIZE:
            self._batch_update()
            self._episode_buffer = []

        if self.episode_count % 10 == 0:
            self._save_weights()

        return reward

    def _batch_update(self) -> None:
        """
        批次 ES 梯度估計：對 buffer 內所有局標準化優勢後加總梯度，
        並截斷 ±1.0 防止暴衝，權重限制在 [0.05, 20.0]。
        """
        rewards = [r for r, _ in self._episode_buffer]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8

        for k in self.base_weights:
            grad = 0.0
            for r, noise in self._episode_buffer:
                advantage = (r - mean_r) / std_r  # 標準化優勢，消除量級影響
                grad += advantage * noise[k]
            grad /= len(self._episode_buffer) * self.sigma
            grad = max(-1.0, min(1.0, grad))       # 梯度截斷
            self.base_weights[k] = max(0.05, min(20.0, self.base_weights[k] + self.lr * grad))

        # 批次更新完成後快照一次權重（供圖表觀察）
        self.weights_history.append(self.base_weights.copy())

    def choose_move(self, board: Board) -> Tuple[Optional[Direction], Dict[Direction, float]]:
        # 對每個方向執行 Expectimax，選出評估分最高的移動
        # 即時合併得分（log 尺度）直接加入評估，避免搜尋深度不足時忽略大合併
        best_move: Optional[Direction] = None
        best_value = float("-inf")
        values: Dict[Direction, float] = {}

        for direction in ("up", "down", "left", "right"):
            result = self.game.simulate_move(board, direction)
            if not result.moved:
                values[direction] = float("-inf")
                continue
            immediate = math.log1p(result.score_gained) if result.score_gained > 0 else 0.0
            value = immediate + self.expectimax(result.board, self.depth - 1, is_chance=True)
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
            # 空格太多時隨機採樣，限制分支數以維持深度 4 的回應速度
            if len(empties) > MAX_CHANCE_CELLS:
                empties = random.sample(empties, MAX_CHANCE_CELLS)
            total = 0.0
            prob_per_cell = 1.0 / len(empties)
            for r, c in empties:
                for val, p in ((2, 0.9), (4, 0.1)):
                    next_board = [row[:] for row in board]
                    next_board[r][c] = val
                    total += prob_per_cell * p * self.expectimax(next_board, depth - 1, False)
            return total

        # 最大化節點：選所有方向中的最高評估值（含即時合併得分）
        best = float("-inf")
        for direction in ("up", "down", "left", "right"):
            result = self.game.simulate_move(board, direction)
            if not result.moved:
                continue
            immediate = math.log1p(result.score_gained) if result.score_gained > 0 else 0.0
            best = max(best, immediate + self.expectimax(result.board, depth - 1, True))
        return best if best != float("-inf") else self.evaluate(board)

    def evaluate(self, board: Board) -> float:
        # 綜合評估棋盤狀態，回傳加權分數
        empties = len(self.game.empty_cells(board))
        max_tile = self.game.max_tile(board)
        monotonicity = self._monotonicity(board)
        merge_potential = self._merge_potential(board)
        snake = self._snake_score(board)
        corner = self._corner_anchor(board, max_tile)
        danger = self._danger_penalty(empties)

        return (
            self.weights["empty_cells"] * empties
            + self.weights["monotonicity"] * monotonicity
            + self.weights["merge_potential"] * merge_potential
            + self.weights["max_tile"] * (max_tile.bit_length() - 1 if max_tile > 0 else 0)
            + self.weights["snake"] * snake
            + self.weights["corner"] * corner
            + danger  # 危機懲罰不參與 ES 學習，固定套用
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

    def _snake_score(self, board: Board) -> float:
        """
        蛇形梯度評分：依左下角蛇形權重矩陣計算盤面分，
        鼓勵大值沿固定路徑遞減排列，防止最大值在盤面漂移。
        回傳值已正規化（除以 SNAKE_WEIGHT_SUM）以維持量級一致性。
        """
        score = 0.0
        for r in range(4):
            for c in range(4):
                if board[r][c] > 0:
                    score += math.log2(board[r][c]) * SNAKE_WEIGHTS[r][c]
        return score / SNAKE_WEIGHT_SUM

    def _corner_anchor(self, board: Board, max_tile: int) -> float:
        """
        角落錨定懲罰（硬規則版）：
        - 最大值在左下角：給 log2(max_tile) 獎勵
        - 最大值在其他角落：給 -0.5*log2(max_tile) 懲罰
        - 最大值不在任何角落：給 -log2(max_tile) 重罰
        corner 權重現為 1.5，避免過度守成而不願冒險合成更大磚。
        """
        if max_tile == 0:
            return 0.0
        log_max = math.log2(max_tile)
        if board[3][0] == max_tile:
            return log_max
        for r, c in ((0, 0), (0, 3), (3, 3)):
            if board[r][c] == max_tile:
                return -log_max * 0.5
        return -log_max

    def _danger_penalty(self, empties: int) -> float:
        """
        空格稀少危機懲罰（固定規則，不參與 ES 學習）：
        空格 <= 2：重罰 -20（瀕死局面）
        空格 <= 4：中罰 -5（高壓局面）
        """
        if empties <= 2:
            return -20.0
        if empties <= 4:
            return -5.0
        return 0.0
