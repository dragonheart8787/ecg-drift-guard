# Threat Model — ECG Drift Guard

> 本文件描述 ECG Drift Guard 系統的攻擊面、防禦機制與剩餘風險。
> 適用情境：醫療 AI 部署時的安全性評估（研究用途）。

---

## 1. 攻擊面（Attack Surface）

| # | 威脅 | 描述 | STRIDE 分類 |
|---|------|------|-------------|
| T1 | **資料投毒（Data Poisoning）** | 攻擊者在訓練資料中注入惡意樣本，使模型學到錯誤的決策邊界 | Tampering |
| T2 | **輸入擾動（Adversarial Perturbation）** | 微小的訊號修改使模型輸出錯誤分類，但人眼無法辨識差異 | Tampering |
| T3 | **分布操控（Distribution Manipulation）** | 攻擊者故意改變輸入資料的分布（如替換設備），繞過漂移偵測 | Spoofing |
| T4 | **觸發模式（Trigger/Backdoor）** | 在訊號中嵌入特定 pattern，觸發預埋的後門行為 | Tampering |
| T5 | **模型竊取（Model Extraction）** | 透過大量查詢推斷模型內部參數或決策邊界 | Information Disclosure |
| T6 | **Audit Trail 篡改** | 修改 decisions.csv 記錄以掩蓋異常決策 | Tampering / Repudiation |

## 2. 現有防禦機制

| 威脅 | 防禦 | 實作位置 |
|------|------|----------|
| T1 | Inter-patient split 限制洩漏；類別分布監控 | `split_records.py`, `checks.py` |
| T2 | Drift detection（PSI/KS/embedding）偵測異常輸入；OOD detection（Mahalanobis）拒絕異常樣本 | `embedding_drift.py`, `ood.py` |
| T3 | 多指標漂移偵測（embedding + signal feature 雙層）；intensity-graded 警報 | `drift_eval.py`, `baseline_drift.py` |
| T4 | Confidence calibration 降低過度自信；audit trail 記錄每筆決策 | `temperature_scaling.py`, `audit.py` |
| T5 | 不在本系統範圍內（需額外的 API rate limiting / watermarking） | N/A |
| T6 | decisions.csv 包含 deterministic reason_code；可重跑驗證（seed + config 固定） | `audit.py`, `seed.py` |

## 3. 剩餘風險（Residual Risk）

| 風險 | 說明 | 建議的額外防禦 |
|------|------|----------------|
| **Adaptive attacker** | 攻擊者了解 drift detector 的機制，精心設計繞過偵測的輸入 | 多樣化偵測指標（Wasserstein / MMD）、定期更新 reference distribution |
| **OOD blind spot** | Mahalanobis 假設高斯分布，非高斯分布的 OOD 可能漏報 | 結合 energy score 或 ensemble disagreement |
| **Concept drift 無標籤** | 無法在沒有 ground-truth 的情況下偵測 P(Y\|X) 變化 | 建立臨床回饋迴路，定期審計 delayed labels |
| **單一模型風險** | 所有決策依賴一個 1D-CNN，無 ensemble | 增加 ensemble / voting 機制以降低單點失效 |

## 4. 安全設計原則

1. **Defence in depth**：漂移偵測 → OOD 偵測 → 校準 → 風控策略 → Audit trail（五層防禦）
2. **Fail-safe default**：當不確定時 reject 或 degrade，而非默認 accept
3. **Auditability**：每筆決策可追溯（sample_idx + reason_code + drift_score + confidence）
4. **Reproducibility**：固定 seed + config，任何人可重跑相同結果
5. **Minimal privilege**：模型只輸出分類結果，風控策略獨立於模型之外

## 5. 合規聲明

本系統**僅供研究與教育用途**，不構成醫療器材或臨床診斷工具。
任何醫療場景的部署必須：
- 通過當地法規審查（如 FDA 510(k) / CE MDR）
- 完成臨床驗證（prospective study）
- 建立持續監控與回報機制（post-market surveillance）
