# Data Sheet — MIT-BIH Arrhythmia Database

> 參考 Gebru et al., "Datasheets for Datasets" (2021) 框架。

---

## 1. 動機（Motivation）

- **目的**：提供標準化的心律不整分類研究基準
- **建立者**：MIT-BIH Laboratory, Beth Israel Deaconess Medical Center
- **資助**：NIH / NIBIB

## 2. 組成（Composition）

| 項目 | 說明 |
|------|------|
| 記錄數 | 48 筆半小時雙導程 ambulatory ECG |
| 取樣率 | 360 Hz |
| 解析度 | 11-bit, 10 mV range |
| 導程 | MLII (大部分) + V1/V2/V4/V5 (部分) |
| 標註 | 逐拍標注（beat annotation）由至少兩位心臟科專家獨立標記 |
| 受試者 | 25 名門診患者（record 100–124）+ 25 名住院患者（record 200–234） |
| 年齡/性別 | 未完整公開（已知大部分為成年人） |

## 3. 預處理（Pre-processing）

本專案的預處理步驟：

1. **導程選擇**：使用第一導程（通常為 MLII）
2. **Beat 切片**：以 R-peak 為中心，前 0.2 秒 + 後 0.4 秒 = 216 samples
3. **正規化**：per-beat z-score normalisation
4. **標籤映射**：MIT-BIH symbols → AAMI 5 類（N, S, V, F, Q）
5. **排除規則**：
   - 無法映射到 AAMI 的 symbol（如 `+`, `~`, `|`）→ 排除
   - R-peak 過於靠近記錄邊界（切片超出範圍）→ 排除
   - 記錄 102, 104, 107, 217 因 paced rhythm 特殊性，部分研究排除

## 4. 切分（Split）

| 切分 | 記錄 | 依據 |
|------|------|------|
| DS1 (train + val) | 22 筆 | de Chazal et al. (2004) |
| DS2 (test) | 22 筆 | de Chazal et al. (2004) |

**Inter-patient split** 確保訓練與測試集無病患重疊。

## 5. 已知偏差與限制（Known Biases）

| 偏差來源 | 說明 |
|----------|------|
| **病患族群偏差** | 受試者主要為北美白人成年人；不代表全球人口 |
| **類別嚴重不平衡** | N 類占 ~89%，V 類 ~7%，S/F/Q 各 <3% |
| **設備偏差** | 資料收集於 1975-1979，使用 Del Mar Avionics 記錄器；現代設備特性不同 |
| **單一中心** | 僅來自 Beth Israel Hospital；缺乏多中心驗證 |
| **標註偏差** | 雖有兩位專家，但仍有 inter-observer disagreement |
| **時間偏差** | 資料收集至今已逾 40 年，心律不整的臨床定義可能有所變化 |

## 6. 分發與授權

- **來源**：[PhysioNet](https://physionet.org/content/mitdb/)
- **授權**：Open Data Commons Attribution License v1.0 (ODC-By)
- **引用**：Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database." IEEE Eng in Med and Biol 20(3):45-50 (2001).

## 7. 維護

- MIT-BIH 資料庫自 2005 年起由 PhysioNet 維護
- 本專案使用的 WFDB 版本與存取日期記錄於 `summary.json` 的 `environment` 欄位

## 8. 倫理考量

- 原始資料已去識別化（de-identified）
- 本專案僅使用公開資料集，不涉及人體試驗
- 任何使用本專案成果的後續研究應遵循所在機構的 IRB/倫理委員會規範
