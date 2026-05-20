import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from .acceleration import AccelerationBackend, detect_acceleration_backend, load_cupy
from .core import Board, Direction, Game2048


# 學習到的基準權重儲存路徑
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "artifacts", "weights.json")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "artifacts", "training_history.json")

DEFAULT_WEIGHTS = {
    "empty_cells": 2.0,       # 略降：不過度保守
    "monotonicity": 1.0,
    "merge_potential": 0.9,
    "max_tile": 3.0,          # 最高：直接獎勵最大磚成長，是最重要的目標
    "snake": 2.0,             # 維持隊形但不壓過 max_tile
    "corner": 1.2,            # 低：不過度守成
}

# 蛇形梯度權重矩陣（沿 SNAKE_PATH 由高優先到低優先遞減）
# [3][0]=15（最大值理想位置）→ 沿蛇形路徑遞減到 [0][0]=0
SNAKE_WEIGHTS = [
    [ 0,  1,  2,  3],   # 第 0 列（遠離角落，低優先）
    [ 7,  6,  5,  4],   # 第 1 列
    [ 8,  9, 10, 11],   # 第 2 列
    [15, 14, 13, 12],   # 第 3 列（最接近左下角，高優先）
]
SNAKE_WEIGHT_SUM = 120  # 所有權重總和，用於正規化

# 蛇行順序（由高優先到低優先）與違規懲罰強度
SNAKE_PATH = [
    (3, 0), (3, 1), (3, 2), (3, 3),
    (2, 3), (2, 2), (2, 1), (2, 0),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (0, 3), (0, 2), (0, 1), (0, 0),
]
SNAKE_VIOLATION_COEF = 2.0
LATE_GAME_SNAKE_GUARD_COEF = 2.0
LATE_GAME_GUARD_TILE = 512
ANCHOR_BREAK_PENALTY = 18.0
STABILITY_DROP_PENALTY_COEF = 14.0
CRITICAL_EMPTY_CELLS = 4

# 機率節點最多採樣幾個空格（限制分支數，維持深度 4 的速度）
MAX_CHANCE_CELLS = 6

# 累積幾局後批次更新一次權重（降低 ES 單樣本噪聲）
BATCH_SIZE = 3

# ES 探索率：僅部分回合使用擾動權重，避免每局都因探索而壓低勝率
INITIAL_EXPLORATION_RATE = 0.35
MIN_EXPLORATION_RATE = 0.08
EXPLORATION_DECAY = 0.995

# 每一步的固定懲罰係數（預設關閉；可由建構子覆寫）
DEFAULT_STEP_PENALTY = 0.0
SNAKE_PATH_ROWS = [r for r, _ in SNAKE_PATH]
SNAKE_PATH_COLS = [c for _, c in SNAKE_PATH]


class HeuristicExpectimaxAI:
    def __init__(self, game: Game2048, depth: int = 4, step_penalty: float = DEFAULT_STEP_PENALTY, use_gpu: bool = True) -> None:
        self.game = game
        self.depth = depth
        self.step_penalty = max(0.0, float(step_penalty))
        self.use_gpu = bool(use_gpu)
        self._cupy = None
        self.backend: AccelerationBackend = AccelerationBackend("CPU", True, True, "初始化中")
        self._refresh_backend()

        # Evolution Strategies 學習超參數
        self.sigma = 0.15          # 每局權重擾動的標準差
        self.lr = 0.05             # 梯度更新學習率
        self.baseline_alpha = 0.1  # 基準線指數移動平均係數

        # 學習狀態
        self.baseline = 0.0        # 獎勵基準線（EMA）
        self.last_reward = 0.0     # 上一局獎勵（對數域）
        self.last_steps = 0        # 上一局實際步數
        self.episode_count = 0     # 總訓練局數
        self._noise: Dict[str, float] = {}
        self._episode_exploring = False
        self.exploration_rate = INITIAL_EXPLORATION_RATE
        # 批次 buffer：儲存 (reward, noise) 等待批次更新
        self._episode_buffer: List[Tuple[float, Dict[str, float]]] = []

        # 從檔案載入已學習的基準權重，或使用預設值
        self.base_weights = self._load_weights()
        # 歷史紀錄（供圖表觀察，支援跨重啟保留）
        self.score_history: List[int] = []
        self.tile_history: List[int] = []
        self.weights_history: List[Dict[str, float]] = []  # 每批次更新後快照
        self._load_history()
        # 初始化本局決策權重（探索或利用）
        self._start_episode_policy()

    @property
    def backend_summary(self) -> str:
        return self.backend.summary

    def _refresh_backend(self) -> None:
        self._cupy = load_cupy(prefer_gpu=self.use_gpu)
        self.backend = detect_acceleration_backend(prefer_gpu=self.use_gpu)

    def set_gpu_enabled(self, enabled: bool) -> None:
        self.use_gpu = bool(enabled)
        self._refresh_backend()

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

    def _load_history(self) -> None:
        """載入圖表歷史資料；格式不符時自動忽略。"""
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)

            scores = data.get("score_history", [])
            tiles = data.get("tile_history", [])
            weights = data.get("weights_history", [])

            self.score_history = [int(x) for x in scores if isinstance(x, (int, float))]
            self.tile_history = [int(x) for x in tiles if isinstance(x, (int, float))]

            valid_weights: List[Dict[str, float]] = []
            for snap in weights:
                if not isinstance(snap, dict):
                    continue
                row = {}
                ok = True
                for k in DEFAULT_WEIGHTS:
                    if k not in snap:
                        ok = False
                        break
                    try:
                        row[k] = float(snap[k])
                    except (TypeError, ValueError):
                        ok = False
                        break
                if ok:
                    valid_weights.append(row)
            self.weights_history = valid_weights
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            self.score_history = []
            self.tile_history = []
            self.weights_history = []

    def _save_history(self) -> None:
        """持久化圖表歷史資料，避免重啟後圖表歸零。"""
        os.makedirs(os.path.dirname(os.path.abspath(HISTORY_FILE)), exist_ok=True)
        payload = {
            "score_history": self.score_history,
            "tile_history": self.tile_history,
            "weights_history": self.weights_history,
        }
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _start_episode_policy(self) -> None:
        """
        決定新回合策略：
        - 探索回合：使用擾動權重（供 ES 估計梯度）
        - 利用回合：使用基準權重（提升即時勝率）
        """
        self._episode_exploring = random.random() < self.exploration_rate
        if self._episode_exploring:
            self._noise = {k: random.gauss(0, self.sigma) for k in self.base_weights}
            self.weights = {k: max(0.05, self.base_weights[k] + self._noise[k]) for k in self.base_weights}
            return

        # 利用回合不加噪聲，避免探索噪聲拖累實戰勝率
        self._noise = {k: 0.0 for k in self.base_weights}
        self.weights = self.base_weights.copy()

    def _compute_reward(self, score: int, max_tile: int, steps: int = 0) -> float:
        """
        對數尺度獎勵 + 里程碑獎勵/懲罰：
        - 高里程碑（>=1024）給正向推力
        - 低里程碑（<1024）給負向懲罰，避免模型停留在低品質策略
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
            milestone = -1.0
        elif max_tile >= 256:
            milestone = -3.0
        elif max_tile >= 128:
            milestone = -6.0
        else:
            milestone = -10.0
        step_cost = self._adaptive_step_penalty(max_tile) * max(0, int(steps))
        return log_score + milestone - step_cost

    def _adaptive_step_penalty(self, max_tile: int) -> float:
        """
        自適應步數懲罰：
        - 前期可施加完整步數壓力，避免無效拖步
        - 中後期逐步放鬆，避免為了省步數而破壞蛇行主架構
        """
        if self.step_penalty <= 0.0:
            return 0.0
        if max_tile >= 1024:
            return 0.0
        if max_tile >= 512:
            return 0.0
        if max_tile >= 256:
            return self.step_penalty * 0.35
        return self.step_penalty

    def on_episode_end(self, score: int, max_tile: int, steps: int = 0) -> float:
        """
        每局結束後累積到 buffer，滿 BATCH_SIZE 局再做批次正規化梯度更新。
        """
        reward = self._compute_reward(score, max_tile, steps)
        self.last_reward = reward
        self.last_steps = max(0, int(steps))

        # 更新基準線（EMA）
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward

        # 只用探索回合資料更新 ES，避免利用回合（零噪聲）稀釋梯度
        if self._episode_exploring:
            self._episode_buffer.append((reward, self._noise.copy()))

        self.episode_count += 1
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate * EXPLORATION_DECAY)
        self._start_episode_policy()

        # 記錄歷史（供圖表觀察）
        self.score_history.append(score)
        self.tile_history.append(max_tile)
        self._save_history()

        # 批次更新（以探索回合數量計）
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

        # 維持核心策略下限，避免學習過程過度貪分導致蛇行崩解
        self.base_weights["snake"] = max(1.5, self.base_weights["snake"])
        self.base_weights["corner"] = max(1.0, self.base_weights["corner"])

        # 批次更新完成後快照一次權重（供圖表觀察）
        self.weights_history.append(self.base_weights.copy())
        self._save_history()

    def choose_move(self, board: Board) -> Tuple[Optional[Direction], Dict[Direction, float]]:
        # 對每個方向執行 Expectimax，選出評估分最高的移動
        # 即時合併得分（log 尺度）直接加入評估，避免搜尋深度不足時忽略大合併
        best_move: Optional[Direction] = None
        best_value = float("-inf")
        values: Dict[Direction, float] = {}
        search_depth = self._effective_depth(board)

        for direction in ("up", "down", "left", "right"):
            result = self.game.simulate_move(board, direction)
            if not result.moved:
                values[direction] = float("-inf")
                continue
            immediate = math.log1p(result.score_gained) if result.score_gained > 0 else 0.0
            guard = self._late_game_move_guard(board, result.board)
            value = immediate + guard + self.expectimax(result.board, search_depth - 1, is_chance=True)
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
            guard = self._late_game_move_guard(board, result.board)
            best = max(best, immediate + guard + self.expectimax(result.board, depth - 1, True))
        return best if best != float("-inf") else self.evaluate(board)

    def evaluate(self, board: Board) -> float:
        # 綜合評估棋盤狀態，回傳加權分數
        if self._cupy is not None and self.backend.name == "CuPy" and self.backend.active:
            return self._evaluate_gpu(board)

        return self._evaluate_cpu(board)

    def _evaluate_cpu(self, board: Board) -> float:
        # CPU 路徑：保留原本啟發式評估

        empties = len(self.game.empty_cells(board))
        max_tile = self.game.max_tile(board)
        monotonicity = self._monotonicity(board)
        merge_potential = self._merge_potential(board)
        snake = self._snake_score(board)
        snake_penalty = self._snake_violation_penalty(board)
        snake_guard = self._late_game_snake_guard(max_tile, empties)
        corner = self._corner_anchor(board, max_tile)
        danger = self._danger_penalty(empties)

        return (
            self.weights["empty_cells"] * empties
            + self.weights["monotonicity"] * monotonicity
            + self.weights["merge_potential"] * merge_potential
            + self.weights["max_tile"] * (max_tile.bit_length() - 1 if max_tile > 0 else 0)
            + self.weights["snake"] * snake
            + (SNAKE_VIOLATION_COEF + snake_guard) * snake_penalty
            + self.weights["corner"] * corner
            + danger  # 危機懲罰不參與 ES 學習，固定套用
        )

    def _evaluate_gpu(self, board: Board) -> float:
        cp = self._cupy
        if cp is None:
            return self._evaluate_cpu(board)

        gpu_board = cp.asarray(board, dtype=cp.float32)
        log_board = cp.where(gpu_board > 0, cp.log2(gpu_board), 0.0)

        empties = int(cp.count_nonzero(gpu_board == 0).item())
        max_tile = int(cp.max(gpu_board).item())

        monotonicity = float(
            (
                -cp.sum(cp.abs(gpu_board[:, :-1] - gpu_board[:, 1:]))
                -cp.sum(cp.abs(gpu_board[:-1, :] - gpu_board[1:, :]))
            ).item()
            / 1024.0
        )

        merge_potential = float(
            (
                cp.count_nonzero((gpu_board[:, :-1] == gpu_board[:, 1:]) & (gpu_board[:, :-1] > 0))
                + cp.count_nonzero((gpu_board[:-1, :] == gpu_board[1:, :]) & (gpu_board[:-1, :] > 0))
            ).item()
        )

        snake_weights = cp.asarray(SNAKE_WEIGHTS, dtype=cp.float32)
        snake = float(cp.sum(log_board * snake_weights).item() / SNAKE_WEIGHT_SUM)

        path_values = gpu_board[SNAKE_PATH_ROWS, SNAKE_PATH_COLS]
        non_zero = path_values[path_values > 0]
        if int(non_zero.size) < 2:
            snake_penalty = 0.0
        else:
            non_zero_logs = cp.log2(non_zero)
            diffs = non_zero_logs[1:] - non_zero_logs[:-1]
            snake_penalty = -float(cp.sum(cp.where(diffs > 0, diffs, 0)).item())

        snake_guard = self._late_game_snake_guard(max_tile, empties)
        corner = self._corner_anchor(board, max_tile)
        danger = self._danger_penalty(empties)

        return (
            self.weights["empty_cells"] * empties
            + self.weights["monotonicity"] * monotonicity
            + self.weights["merge_potential"] * merge_potential
            + self.weights["max_tile"] * (max_tile.bit_length() - 1 if max_tile > 0 else 0)
            + self.weights["snake"] * snake
            + (SNAKE_VIOLATION_COEF + snake_guard) * snake_penalty
            + self.weights["corner"] * corner
            + danger
        )

    def _late_game_snake_guard(self, max_tile: int, empties: int) -> float:
        """
        後期蛇行守護：盤面進入中後期且空格變少時，
        逐步放大蛇行違規懲罰，降低「急著合併」造成的隊形破壞。
        """
        guard = 0.0
        if max_tile >= 512:
            guard += LATE_GAME_SNAKE_GUARD_COEF
        if empties <= 4:
            guard += 1.0
        return guard

    def _late_game_move_guard(self, prev_board: Board, next_board: Board) -> float:
        """
        後期移動護欄：
        - 大磚期（>=512）避免把最大磚從左下角錨點拉走
        - 空格緊張時重罰蛇行穩定度大幅下降的走法
        回傳非正值；數值越小代表風險越高。
        """
        prev_max = self.game.max_tile(prev_board)
        if prev_max < LATE_GAME_GUARD_TILE:
            return 0.0

        penalty = 0.0
        prev_anchor = prev_board[3][0] == prev_max
        next_max = self.game.max_tile(next_board)
        next_anchor = next_board[3][0] == next_max

        if prev_anchor and not next_anchor:
            penalty -= ANCHOR_BREAK_PENALTY

        prev_stability = self.snake_stability(prev_board)
        next_stability = self.snake_stability(next_board)
        stability_drop = max(0.0, prev_stability - next_stability)
        empties = len(self.game.empty_cells(next_board))

        if empties <= CRITICAL_EMPTY_CELLS and stability_drop > 0.0:
            penalty -= STABILITY_DROP_PENALTY_COEF * stability_drop

        if empties <= 2 and next_stability < 0.45:
            penalty -= 6.0

        return penalty

    def _effective_depth(self, board: Board) -> int:
        """
        後期動態加深搜尋：
        - 基本使用設定深度
        - 中後期且空格少時，加深 1~2 層降低短視錯判
        """
        depth = self.depth
        max_tile = self.game.max_tile(board)
        empties = len(self.game.empty_cells(board))

        if max_tile >= 512 and empties <= 6:
            depth += 1
        if max_tile >= 1024 and empties <= 4:
            depth += 1

        return min(6, depth)

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

    def _snake_violation_penalty(self, board: Board) -> float:
        """
        蛇行違規懲罰：沿蛇行路徑取出非零序列，若後一格大於前一格則扣分。
        回傳非正值；數值越小代表越偏離蛇行遞減隊形。
        """
        seq = [board[r][c] for r, c in SNAKE_PATH if board[r][c] > 0]
        if len(seq) < 2:
            return 0.0

        penalty = 0.0
        for prev, curr in zip(seq, seq[1:]):
            if curr > prev:
                penalty += math.log2(curr) - math.log2(prev)
        return -penalty

    def _corner_anchor(self, board: Board, max_tile: int) -> float:
        """
        角落錨定懲罰（硬規則版）：
        - 最大值在左下角：給 log2(max_tile) 獎勵
        - 最大值在其他角落：給 -0.5*log2(max_tile) 懲罰
        - 最大值不在任何角落：給 -log2(max_tile) 重罰
        corner 權重現為 1.2，避免過度守成而不願冒險合成更大磚。
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

    def snake_stability(self, board: Board) -> float:
        """
        回傳蛇行穩定度（0~1）：
        - 1.0 代表幾乎沒有蛇行違規
        - 越接近 0 代表隊形越混亂
        """
        violation = -self._snake_violation_penalty(board)
        return 1.0 / (1.0 + max(0.0, violation))
