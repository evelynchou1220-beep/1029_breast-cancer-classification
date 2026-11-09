# 乳癌良性與惡性分類：LDA、QDA、Linear SVM、RBF-SVM

用 WDBC（Wisconsin Diagnostic Breast Cancer）資料的 30 項細胞核影像特徵，判斷腫瘤是良性（B）還是惡性（M），並比較四種分類方法。因為是醫療情境，分析特別關注 **False Negative（惡性被判成良性）**。


## 結果摘要

資料共 569 筆（B 357、M 212），沒有遺漏值。依類別分層切成 80% 訓練（454 筆）、20% 測試（115 筆）。四個模型都使用由訓練集估計的前 5 個主成分（累積解釋變異 84.54%）。

| 模型 | Accuracy | Sensitivity | Specificity | Precision | F1 | FN | FP |
|---|---|---|---|---|---|---|---|
| LDA | 0.983 | 0.953 | 1.000 | 1.000 | 0.976 | 2 | 0 |
| QDA | 0.983 | 0.977 | 0.986 | 0.977 | 0.977 | 1 | 1 |
| Linear SVM（cost = 10） | 0.983 | 0.977 | 0.986 | 0.977 | 0.977 | 1 | 1 |
| RBF-SVM（cost = 100, γ = 0.01） | **0.991** | 0.977 | 1.000 | 1.000 | **0.988** | 1 | 0 |

RBF-SVM 表現最好，但模型之間只差 1 到 2 個個案。

## 方法重點

- **先切分、再處理**：中位數填補、刪除零變異、標準化、PCA 的參數都只由訓練集估計，再套用到測試集。
- **固定切分**：第一次執行會寫出 `data/data_split.csv`，之後重跑都沿用同一批測試個案。
- **SVM 調參**：在訓練集內做分層五折交叉驗證，每一折都重新估計前處理，避免資料洩漏。
- **選參標準**：因類別不平衡，改用 Balanced Accuracy（Sensitivity 與 Specificity 的平均）選參數，同分時選 Sensitivity 較高、參數較小者。
- **補充分析**：計算各類別共變異數矩陣的條件數，並比較 QDA 使用不同 PC 數時的表現，說明高度共線特徵對 QDA 的影響。

## 專案結構

```
.
├── analysis_LDA_QDA_SVM.R    # 主程式
├── data/
│   ├── HW01_Breast_Cancer_Data.csv
│   └── data_split.csv        # 固定的訓練／測試切分
├── results/                  # 執行後產生的圖表與表格
└── report/                   # 書面報告（PDF / Word）
```

## 如何執行

需要 R（開發時使用 R 4.6.1）與兩個套件：

```r
install.packages(c("MASS", "e1071"))
```

在專案資料夾中執行：

```r
setwd("你的專案路徑")
source("analysis_LDA_QDA_SVM.R")
```

所有輸出（表 1–7、圖 1–6、`輸出結果.txt`）會寫到 `results/`。

## 資料來源

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

