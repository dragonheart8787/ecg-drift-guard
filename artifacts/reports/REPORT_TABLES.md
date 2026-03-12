# ECG Drift Guard — 報告用表格與段落（可直接貼入期末報告）

以下內容皆來自本專案**最新一次完整 pipeline 產出**（`summary.json`、`test_metrics.json`、`external_summary.json`、`threshold_sensitivity.json`）。  
圖檔路徑：`artifacts/reports/figures/`。

---

## A. 每類別指標表（MIT-BIH Test Set）

### 表 1. Per-class metrics on MIT-BIH test set (AAMI 5 classes)

| Class | Support | Precision | Recall | F1-score | 說明 |
|-------|---------|-----------|--------|----------|------|
| N     | 44,242  | 0.9120    | 0.8571 | 0.8837   | 主類別，模型對 N 類整體表現穩定 |
| S     | 1,837   | 0.0068    | 0.0174 | 0.0098   | 少數類，幾乎都被判成 N 或其他類 |
| V     | 3,220   | 0.9254    | 0.3006 | 0.4538   | Ventricular，precision 高但 recall 偏低 |
| F     | 388     | 0.0005    | 0.0026 | 0.0008   | 極少數類，模型幾乎無法辨識 |
| Q     | 7       | 0.0028    | 0.1429 | 0.0056   | 樣本極少，指標不穩定 |

- **Overall**：Accuracy = 0.7832，Macro F1 = 0.2707。  
- **解讀**：整體 accuracy 約 78%，但由於類別高度不平衡，macro F1 僅約 0.27，顯示模型主要是把 N 類學得不錯，S/F/Q 等少數類仍表現不佳。這也呼應本專題「以安全層與風控為主，分類模型視為 baseline」的定位。

---

### 表 2. Confusion matrix on MIT-BIH test set（rows = true label, cols = predicted）

|            | Pred N | Pred S | Pred V | Pred F | Pred Q |
|------------|--------|--------|--------|--------|--------|
| **True N** | 37,918 | 3,969  | 36     | 1,983  | 336    |
| **True S** | 1,779  | 32     | 3      | 17     | 6      |
| **True V** | 1,554  | 677    | 968    | 11     | 10     |
| **True F** | 324    | 25     | 38     | 1      | 0      |
| **True Q** | 0      | 5      | 1      | 0      | 1      |

- **解讀**：N 類仍有相當比例被誤判為 S 或 F；V 類中約三分之一被預測為其他類；少數類 F/Q 幾乎完全被誤分。此混淆結構說明本模型更適合作為「安全層展示」用 baseline，而非最終臨床分類器。

---

## B. 校準結果（Temperature Scaling, H2）

### 表 3. Calibration metrics on MIT-BIH test set (before / after TS)

| 指標   | Before | After  | 說明 |
|--------|--------|--------|------|
| ECE    | 0.1060 | 0.0692 | 校準後 ECE 降低，信心較貼近真實正確率 |
| Brier  | 0.3191 | 0.3066 | 校準後 Brier score 略降，整體校準改善 |
| T 值   | —      | 0.7464 | 在 validation set 上以 NLL 擬合得到 |

- **圖**：`reliability_before.png`、`reliability_after.png`、`reliability_comparison.png`、`calibration_table.png`。  
- **結論**：在最新這一版模型上，Temperature Scaling **成功降低 ECE 與 Brier**，H2 標記為 **supported**。這說明在目前設定下，單一 global T 可以改善模型信心的可靠性。

---

## C. 風控結果（H3）— Coverage / Error / Action 比例

### 表 4. Risk control summary on MIT-BIH test set（Default 門檻）

| 項目                     | 數值    |
|--------------------------|---------|
| Overall error rate       | 0.2168  |
| Error after policy       | 0.1700  |
| Error reduction          | 0.0468  |
| Reject rate              | 0.0773  |
| Coverage (auto-decided)  | 0.9227  |
| Accept rate              | 0.8558  |
| Degrade rate             | 0.0227  |
| Warn rate                | 0.0442  |

- **解讀**：在拒判約 7.73% 的樣本下，風控策略把「自動決策樣本中的錯誤率」從 21.68% 降到 17.00%，錯誤率下降約 4.68 個百分點，覆蓋率仍維持 92.27%。這說明 H3「在可接受拒判率下降低錯誤率」是 **成立的**。  
- **圖**：`reject_curve.png`、`coverage_risk.png`、`confidence_distribution.png`。

---

### 表 4b. Policy threshold sensitivity（3 組 threshold 對 reject / coverage / error 的影響）

| 組別         | warning | critical | conf_low | conf_mid | Reject rate | Coverage | Error after policy |
|--------------|---------|----------|----------|----------|-------------|----------|--------------------|
| Conservative | 0.10    | 0.15     | 0.6      | 0.8      | 0.0984      | 0.9016   | 0.1572             |
| Default      | 0.15    | 0.22     | 0.5      | 0.7      | 0.0773      | 0.9227   | 0.1700             |
| Liberal      | 0.20    | 0.30     | 0.4      | 0.6      | 0.0217      | 0.9783   | 0.2040             |

- **解讀**：門檻越保守（conservative），reject 變多、coverage 降低，但錯誤率最低；門檻越寬鬆（liberal），coverage 接近 98%，但錯誤率升高到約 20.4%。Default 則提供一個「rejection、coverage、錯誤率」之間較平衡的折衷。  
- **產出**：執行 `py -3.11 scripts/threshold_sensitivity.py` 可更新 `artifacts/reports/threshold_sensitivity.json`。

---

## D. Internal vs External 摘要表（MIT-BIH vs SVDB）

### 表 5. Internal test (MIT-BIH) vs External (SVDB) summary

| 項目        | 內部測試 (MIT-BIH) | 外部 SVDB   |
|-------------|--------------------|-------------|
| Accuracy    | 0.7832             | 0.8532      |
| Macro F1    | 0.2707             | 0.3028      |
| ECE         | 0.1060 (before)    | 0.0943      |
| Brier       | 0.3191 (before)    | 0.1738      |
| Drift score | 0 (reference)      | 0.5827      |
| OOD rate    | 0% (reference)     | 37.30%      |
| 樣本數      | 49,694             | 27,720      |

- **解讀**：外部 SVDB 上的 accuracy 與 macro F1 略高於內部 test，主要來自類別分布差異（N 類比例更高）；同時 embedding drift score 與 OOD rate 顯著上升，顯示新資料確實與原訓練分布不同，而安全層成功捕捉到這些差異。  
- **圖**：`external_drift_vs_internal.png`、`ood_score_distribution.png`、`external_reliability.png`。

---

### 表 5b. External per-class metrics（SVDB）

| Class | Support | Precision | Recall | F1-score | 說明 |
|-------|---------|-----------|--------|----------|------|
| N     | 26,314  | 0.9756    | 0.8816 | 0.9262   | 主類別，N 類在外部資料上的表現相當穩定 |
| S     | 320     | 0.0093    | 0.0563 | 0.0159   | 少數類，仍然難以辨識 |
| V     | 1,081   | 0.9931    | 0.4015 | 0.5718   | precision 很高，recall 中等偏低 |
| F     | 5       | 0.0000    | 0.0000 | 0.0000   | 樣本極少，幾乎無法學習 |
| Q     | 0       | —         | —      | —        | SVDB 無 Q 類樣本 |

- **Overall**：Accuracy = 0.8532，Macro F1 = 0.3028。  
- **解讀**：在外部資料上，模型對 N 類維持高 precision 與 recall，對 V 類仍有一定能力，但 S/F 類依然是主要弱點。這也說明在多中心資料或轉換到新醫院時，需要重新檢討少數類的資料量與模型設計。

---

### 表 5c. Confusion matrix on external SVDB（rows = true, cols = predicted）

|            | Pred N | Pred S | Pred V | Pred F | Pred Q |
|------------|--------|--------|--------|--------|--------|
| **True N** | 23,199 | 1,648  | 1      | 2      | 1,464  |
| **True S** | 246    | 18     | 1      | 0      | 55     |
| **True V** | 330    | 275    | 434    | 17     | 25     |
| **True F** | 4      | 0      | 1      | 0      | 0      |
| **True Q** | 0      | 0      | 0      | 0      | 0      |

---

## E. 外部資料集上的風控統計（SVDB）

### 表 6. Risk policy applied to external SVDB

| 項目         | 數值    |
|--------------|---------|
| Reject rate  | 0.0338  |
| Degrade rate | 0.9662  |
| Warn rate    | 0.0000  |
| Accept rate  | 0.0000  |

- **解讀**：在外部 SVDB 上，大部分樣本被標為 critical 而進入 degrade，僅約 3.38% 被直接拒判，沒有樣本被標為 accept。這反映出目前 policy 對「跨資料集情境」採取偏保守策略，在有明顯漂移與 OOD 訊號時，避免系統過度自信地輸出多類分類結果。

---

## F. 自動化測試結果（可貼入報告附錄）

- 執行指令：`py -3.11 -m pytest tests -v`  
- 結果：**43 passed, 0 skipped, 0 failed**（約 5 秒）。  
- 涵蓋：determinism、schema（summary/decisions/benchmark/external）、split 無洩漏、NPZ 結構、policy 邊界、PSI/KS、ECE/Brier、AAMI 映射，以及**本文件所有表格對應的數值驗證**（`tests/test_report_claims.py`）。

可於報告中寫：「本系統所有自動化測試均通過，包含對報告中主要指標與表格數值的一致性檢查，確保 pipeline 可重現且輸出結果可機器驗證。」

---

## 使用方式

1. 將上述 **表 1～表 6** 複製到 Word / LaTeX 的 Results 章節，必要時微調欄位名稱或小數位。  
2. 在對應小節插入圖檔（路徑 `artifacts/reports/figures/` 下之檔名如上所述）。  
3. 段落文字可直接貼上或依學校格式略改。  
4. 若日後重跑 pipeline，可再執行以下腳本更新產出：  
   - `py -3.11 scripts/generate_report_tables.py` → `test_metrics.json`  
   - `py -3.11 scripts/07_external_validation.py` → `external_summary.json`（含 per_class、confusion_matrix）  
   - `py -3.11 scripts/threshold_sensitivity.py` → `threshold_sensitivity.json`  
   並重新跑 `py -3.11 -m pytest tests/test_report_claims.py -v` 確認新數字與報告一致。

