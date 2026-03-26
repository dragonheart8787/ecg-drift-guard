# ECG Drift Guard

**醫療 AI 上線風控層（Safety Layer）** — 以 MIT-BIH 心電圖資料集為範例，展示從模型訓練、漂移偵測、OOD 檢測、可信度校準到風控決策的端到端 pipeline，並附外部資料集驗證。

> **Scope / 免責聲明**：本系統僅為個人製作成果，**非醫療器材、非臨床診斷工具**。任何醫療應用必須遵循當地法規（如 FDA、CE）並經臨床驗證。

---

## 系統架構圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ECG Drift Guard — System Architecture           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  WFDB    │──▶│  Beat    │──▶│  NPZ     │──▶│  1D-CNN Train    │ │
│  │  Download│   │  Cut +   │   │  Dataset │   │  + Embedder      │ │
│  │          │   │  Normalise│   │          │   │                  │ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘ │
│       ▲                                                 │           │
│       │              config/default.yaml                ▼           │
│       │              config/splits.yaml        ┌──────────────────┐ │
│  ┌────┴─────┐                                  │  artifacts/      │ │
│  │ External │                                  │  models/         │ │
│  │ Dataset  │                                  │  cnn1d.keras     │ │
│  │ (svdb)   │                                  │  embedder.keras  │ │
│  └──────────┘                                  └────────┬─────────┘ │
│                                                         │           │
│  ┌──────────────────────────────────────────────────────▼─────────┐ │
│  │                    INFERENCE PIPELINE                          │ │
│  │                                                                │ │
│  │  Input → Embedding → Drift Detection → OOD Check              │ │
│  │    │         │            │                │                   │ │
│  │    ▼         ▼            ▼                ▼                   │ │
│  │  Logits   Ref vs Cur   PSI/KS/Score   Mahalanobis             │ │
│  │    │                                      │                   │ │
│  │    ▼                                      │                   │ │
│  │  Temp Scaling (T) → Calibrated Proba      │                   │ │
│  │    │                                      │                   │ │
│  │    └──────────────┬───────────────────────┘                   │ │
│  │                   ▼                                            │ │
│  │          ┌─────────────────┐                                   │ │
│  │          │   Risk Policy   │                                   │ │
│  │          │  normal/warn/   │                                   │ │
│  │          │  critical       │                                   │ │
│  │          └────────┬────────┘                                   │ │
│  │                   ▼                                            │ │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌───────────────────┐  │ │
│  │  │ ACCEPT  │ │  WARN   │ │ DEGRADE  │ │     REJECT        │  │ │
│  │  │         │ │+ log    │ │→ binary  │ │→ human review     │  │ │
│  │  └─────────┘ └─────────┘ └──────────┘ └───────────────────┘  │ │
│  │                   │                                            │ │
│  │                   ▼                                            │ │
│  │          decisions.csv (audit trail)                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    OUTPUTS                                     │ │
│  │  summary.json │ decisions.csv │ 16+ figures │ benchmark.json   │ │
│  │  model_registry.json │ hypothesis_report.txt │ external_*      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 推論序列圖

```
User/Device          Model           Drift           OOD          Calibration      Policy         Audit
    │                  │               │               │               │              │              │
    │── ECG signal ──▶│               │               │               │              │              │
    │                  │── proba ─────▶│               │               │              │              │
    │                  │── embedding ──▶│               │               │              │              │
    │                  │               │── drift_score─▶│               │              │              │
    │                  │               │               │── ood_flag ──▶│              │              │
    │                  │── logits ────────────────────────────────────▶│              │              │
    │                  │               │               │  cal_proba ──▶│              │              │
    │                  │               │               │  confidence ──▶│              │              │
    │                  │               │               │               │── decision ─▶│              │
    │                  │               │               │               │              │── log row ──▶│
    │◀── result + action + reason_code ──────────────────────────────────────────────│              │
    │                  │               │               │               │              │              │
```

---

## 研究假設（Hypotheses）

| ID | 假設 | 驗證方式 |
|----|------|----------|
| **H1** | drift_score 上升時，模型效能（F1）顯著下降 | Spearman ρ + scatter plot |
| **H2** | Temperature Scaling 降低 ECE，使 confidence 與正確率更一致 | ECE/Brier before vs after |
| **H3** | 風控策略能在可接受拒判率下降低錯誤輸出 | Reject rate vs Error rate curve |

## Pipeline 流程（9 步）

```
01_make_splits           → DS1/DS2 inter-patient split + leakage check
02_build_dataset         → WFDB 下載 → R-peak 切 beat → NPZ + 類別分布
03_train_model           → 1D-CNN 訓練 → 模型 + embedder + bootstrap CI
04_drift_evaluate        → 3 scenarios × 3 intensities (S1/S2/S3)
                           → embedding drift + baseline-B + Spearman (H1)
05_calibrate_and_risk    → Temperature Scaling + ECE/Brier + uncertainty
                           → Selective prediction + Risk policy + Audit trail
06_hypothesis_report     → H1~H3 驗證報告
07_external_validation   → 外部資料集 (svdb) 推論 + drift + OOD + policy
08_model_registry_update → 版本管理 (model_registry.json)
09_benchmark             → Latency + Memory + Device info
```

## 快速開始

```bash
pip install -r requirements.txt

cd ecg_drift_guard
python scripts/01_make_splits.py --config config/default.yaml
python scripts/02_build_dataset.py
python scripts/03_train_model.py
python scripts/04_drift_evaluate.py
python scripts/05_calibrate_and_risk.py
python scripts/06_hypothesis_report.py
python scripts/07_external_validation.py --ext-db svdb
python scripts/08_model_registry_update.py
python scripts/09_benchmark.py
```

## 測試

```bash
python -m pytest tests/ -v
```

三類測試：
- **Smoke tests** (6 classes)：split 不重疊、NPZ shape、policy 邊界、PSI/KS、校準、AAMI mapping
- **Determinism tests**：同 seed 同 config 必產出相同結果
- **Schema tests**：summary.json / decisions.csv / benchmark.json / external_summary.json 欄位驗證

## 目錄結構

```
ecg_drift_guard/
  config/                        # YAML 設定
  data/                          # raw/ + processed/ + splits/
  artifacts/
    models/                      # cnn1d.keras, embedder.keras, model_registry.json
    calibration/                 # temperature.json
    reports/
      summary.json               # 完整指標 + 假設驗證 + 失效模式 + Model Card
      decisions.csv              # 逐筆 audit trail
      hypothesis_report.txt      # H1~H3 文字報告
      external_summary.json      # 外部驗證結果
      benchmark.json             # Latency / Memory / Device
      drift_results.json
      drift_correlation.json
      class_distribution.json
      figures/                   # 16+ 自動產出圖表
  src/
    common/                      # log, seed, io, metrics, stats, checks, versioning, benchmark
    dataset/                     # fetch, label, beat_cut, split, build_npz, external_loader
    models/                      # cnn1d, train, infer
    drift/                       # simulate, psi, ks, embedding_drift, baseline_drift, drift_eval, ood
    calibration/                 # ece (ECE+Brier), temperature_scaling, uncertainty, selective
    risk/                        # policy (reason codes + OOD), audit (decisions.csv), report
    viz/                         # 16 種圖表
  scripts/                       # 9 步驟腳本 (均支援 --config)
  tests/                         # 3 類測試 (smoke + determinism + schema)
  docs/                          # threat_model.md + datasheet.md
```

## 圖表清單（16+）

| # | 圖表 | 檔名 | 用途 |
|---|------|------|------|
| 1 | Reliability (Before) | `reliability_before.png` | H2 前測 |
| 2 | Reliability (After) | `reliability_after.png` | H2 後測 |
| 3 | Reliability 並排 | `reliability_comparison.png` | H2 對比 |
| 4 | Calibration 指標表 | `calibration_table.png` | ECE+Brier |
| 5 | Drift 曲線 | `drift_curve.png` | scenario×intensity |
| 6 | Intensity 曲線 | `intensity_curve.png` | S1→S3 + Baseline B |
| 7 | Correlation 散點 | `correlation_scatter.png` | H1 (Spearman ρ) |
| 8 | Performance vs Drift | `perf_vs_drift.png` | 效能趨勢 |
| 9 | Top Feature Shift | `top_feature_shift.png` | Embedding 維度 |
| 10 | Confidence 分布 | `confidence_distribution.png` | 正確 vs 錯誤 |
| 11 | Reject 曲線 | `reject_curve.png` | H3 驗證 |
| 12 | Coverage-Risk | `coverage_risk.png` | Selective prediction |
| 13 | Class 分布 | `class_distribution.png` | train/val/test |
| 14 | OOD 分布 | `ood_score_distribution.png` | Internal vs External |
| 15 | External Drift 對比 | `external_drift_vs_internal.png` | 模擬 vs 真實 |
| 16 | Latency 基準 | `latency_benchmark.png` | ms/beat vs batch |
| 17 | External Reliability | `external_reliability.png` | 外部校準 |
| 18 | External Conf 分布 | `external_confidence_dist.png` | 外部不確定度 |

## 五層防禦架構（Defence in Depth）

```
Layer 1: Drift Detection    → PSI / KS / Embedding drift (covariate shift)
Layer 2: OOD Detection      → Mahalanobis distance (out-of-distribution)
Layer 3: Calibration        → Temperature Scaling (confidence alignment)
Layer 4: Risk Policy        → normal / warning / critical + reason codes
Layer 5: Audit Trail        → decisions.csv (every inference, traceable)
```

## 失效模式表

| Failure Mode | Effect | Detection | Mitigation | Residual Risk |
|---|---|---|---|---|
| Sampling rate mismatch | Waveform distorted | drift_score ↑ | warning/critical → degrade | Extreme mismatch irreversible |
| Powerline noise | V/S misclassification ↑ | PSI/KS ↑ | reject or warn | Noise in QRS may evade |
| Gain/amplitude shift | Feature distortion | drift_score ↑ | warning at moderate | Saturated signals |
| Class prior shift | Biased recall | Prior monitor | Retrain trigger | Requires labels |
| Concept drift | Silent degradation | Performance monitor | Periodic retrain | Needs ground truth |
| OOD samples | Unpredictable output | Mahalanobis distance | reject (OOD_DETECTED) | Non-Gaussian OOD |

## 文件清單

| 文件 | 用途 |
|------|------|
| `docs/threat_model.md` | STRIDE 威脅模型 + 攻擊面 + 防禦 + 剩餘風險 |
| `docs/datasheet.md` | MIT-BIH 資料集 Data Sheet（Gebru et al. 框架） |
| `summary.json` → `model_card` | Model Card（用途、限制、倫理） |
| `summary.json` → `concept_drift_playbook` | Concept Drift 應對策略 |
| `model_registry.json` | 模型版本管理（hash + metrics + calibration） |
