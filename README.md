# 2048

模組化的 Python 2048 遊戲，支援：

- 手動模式（方向鍵操作）
- AI 訓練 / 自動遊玩模式（Expectimax + 啟發式）
- GUI 即時顯示棋盤、分數、Best Score、Episode、勝率、最佳方塊與 AI 評分資訊
- 可在 GUI 中切換模式與調整 AI 速度（延遲）

## 執行方式

```bash
python main.py
```

或：

```bash
python game_2048_ai.py
```

## 專案結構

- `game2048/core.py`：2048 核心規則與盤面狀態
- `game2048/ai.py`：AI 決策（Expectimax + heuristic）
- `game2048/gui.py`：Tkinter GUI 與即時訓練可視化
- `tests/`：核心與 AI 單元測試
