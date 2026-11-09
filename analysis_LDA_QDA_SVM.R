# ==============================================================
# 作業 01：乳癌良性與惡性分類
# LDA、QDA、Linear SVM、RBF-SVM
# ==============================================================

# [0] 套件檢查 ---------------------------------------------------
required <- c("MASS", "e1071")
missing_pkg <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkg)) {
  stop("請先安裝套件：install.packages(c(",
       paste(shQuote(missing_pkg), collapse = ", "), "))")
}

dir.create("results", showWarnings = FALSE, recursive = TRUE)
dir.create("data", showWarnings = FALSE, recursive = TRUE)

# [1] 讀取資料 ---------------------------------------------------
data_candidates <- c(
  file.path("data", "HW01_Breast_Cancer_Data.csv"),
  "HW01_Breast_Cancer_Data.csv",
  file.path("HW", "HW01_Breast_Cancer_Data.csv")
)
data_file <- data_candidates[file.exists(data_candidates)][1]
if (is.na(data_file)) stop("找不到 HW01_Breast_Cancer_Data.csv")

raw <- read.csv(data_file, stringsAsFactors = FALSE, check.names = FALSE,
                na.strings = c("", "NA", "?"), fileEncoding = "UTF-8-BOM")
if (!"Class" %in% names(raw)) stop("資料缺少 Class 欄位")
# M（惡性）設為陽性類別，B（良性）為陰性類別。
raw$Class <- factor(raw$Class, levels = c("B", "M"))
if (anyNA(raw$Class)) stop("Class 含 B/M 以外的值")

feature_names <- setdiff(names(raw), c("Class", "ID", "PatientID"))
if (!all(vapply(raw[feature_names], is.numeric, logical(1)))) {
  stop("本作業的 30 個預測變數必須是數值型")
}
# 數值設計矩陣：只含 30 個預測變數，Class 與識別碼不進入模型。
X <- as.matrix(raw[feature_names])
y <- raw$Class

# [2] 資料探索 ---------------------------------------------------
# 2a. 樣本數、類別比例、遺漏值
class_tab <- table(y)
class_pct <- round(100 * prop.table(class_tab), 1)
n_missing <- sum(is.na(X))

# 2b. 重要預測變數在兩組中的分布（平均數、標準差、中位數、四分位距）
eda_vars <- c("mean_radius", "mean_concavity", "worst_radius", "worst_concave_points")
eda_table <- do.call(rbind, lapply(eda_vars, function(v) {
  do.call(rbind, lapply(levels(y), function(cl) {
    z <- X[y == cl, v]
    data.frame(Variable = v, Class = cl, n = length(z),
               Mean = mean(z), SD = sd(z), Median = median(z),
               Q1 = unname(quantile(z, 0.25)), Q3 = unname(quantile(z, 0.75)))
  }))
}))
write.csv(eda_table, file.path("results", "（表 2）重要變數分組摘要.csv"), row.names = FALSE)

# 2c. 特徵間相關：說明 30 個特徵高度共線
cor_mat <- cor(X)
upper <- which(upper.tri(cor_mat), arr.ind = TRUE)
cor_pairs <- data.frame(Var1 = rownames(cor_mat)[upper[, 1]],
                        Var2 = colnames(cor_mat)[upper[, 2]],
                        r = cor_mat[upper])
cor_pairs <- cor_pairs[order(-abs(cor_pairs$r)), ]
n_pairs_09 <- sum(abs(cor_pairs$r) > 0.9)
write.csv(head(cor_pairs, 15), file.path("results", "（表 1）相關係數最高的特徵.csv"), row.names = FALSE)

# 圖 A：兩組分布盒鬚圖
png(file.path("results", "（圖 1）盒鬚圖.png"), width = 1400, height = 520, res = 130)
op <- par(mfrow = c(1, 4), mar = c(3, 4, 3, 1))
for (v in eda_vars) {
  boxplot(X[, v] ~ y, col = c("#3D7EA6", "#D47A4A"), main = v,
          xlab = "", ylab = "", names = c("Benign B", "Malignant M"), cex.main = 1.05)
}
par(op); dev.off()

# 圖 B：30 個特徵的相關熱圖
png(file.path("results", "（圖 3）熱圖.png"), width = 900, height = 860, res = 130)
op <- par(mar = c(1, 1, 3, 1))
pal <- colorRampPalette(c("#3D7EA6", "white", "#D47A4A"))(41)
image(1:30, 1:30, cor_mat[, 30:1], col = pal, zlim = c(-1, 1), axes = FALSE,
      xlab = "", ylab = "", main = "")
title("Correlation among 30 features", line = 2)
abline(v = c(10.5, 20.5), h = c(10.5, 20.5), col = "grey40")
text(c(5.5, 15.5, 25.5), 31.3, c("mean", "error", "worst"), xpd = TRUE, cex = 0.9)
par(op); dev.off()

# [3] 訓練／測試切分 ---------------------------------------------
# 分層抽樣 80/20。第一次執行時寫出 data_split.csv，之後固定沿用，
# 使報告與程式使用完全相同的測試個案。
split_file <- file.path("data", "data_split.csv")
if (file.exists(split_file)) {
  split_info <- read.csv(split_file, stringsAsFactors = FALSE)
  if (!all(split_info$RowID == seq_len(nrow(raw)))) stop("data_split.csv 的 RowID 不符")
  train_idx <- split_info$RowID[split_info$Partition == "train"]
  test_idx <- split_info$RowID[split_info$Partition == "test"]
} else {
  set.seed(20261018)
  train_idx <- unlist(lapply(split(seq_len(nrow(raw)), y), function(ii) {
    sample(ii, floor(length(ii) * 0.8))
  }), use.names = FALSE)
  train_idx <- sort(train_idx)
  test_idx <- setdiff(seq_len(nrow(raw)), train_idx)
  write.csv(data.frame(RowID = seq_len(nrow(raw)),
                       Partition = ifelse(seq_len(nrow(raw)) %in% train_idx, "train", "test")),
            split_file, row.names = FALSE)
}

# [4] 前處理：填補 → 刪除零變異 → 標準化 → PCA ------------------------
# 所有參數只由「傳入的訓練資料」估計，再原封不動套用到驗證或測試資料。
fit_preprocessor <- function(x, n_pc = 5L) {
  med <- apply(x, 2, median, na.rm = TRUE)
  med[!is.finite(med)] <- 0
  for (j in seq_len(ncol(x))) x[is.na(x[, j]), j] <- med[j]
  variance <- apply(x, 2, var)
  keep <- is.finite(variance) & variance > 1e-10
  if (sum(keep) < 2L) stop("非定值特徵不足")
  x <- x[, keep, drop = FALSE]
  center <- colMeans(x)
  scale <- apply(x, 2, sd)
  z <- sweep(sweep(x, 2, center, "-"), 2, scale, "/")
  pca <- prcomp(z, center = FALSE, scale. = FALSE)
  n_pc <- min(n_pc, ncol(pca$rotation), nrow(x) - 2L)
  list(median = med, keep = keep, center = center, scale = scale,
       rotation = pca$rotation[, seq_len(n_pc), drop = FALSE],
       explained = (pca$sdev^2 / sum(pca$sdev^2))[seq_len(n_pc)])
}

apply_preprocessor <- function(x, prep) {
  for (j in seq_len(ncol(x))) x[is.na(x[, j]), j] <- prep$median[j]
  x <- x[, prep$keep, drop = FALSE]
  z <- sweep(sweep(x, 2, prep$center, "-"), 2, prep$scale, "/")
  ans <- z %*% prep$rotation
  colnames(ans) <- paste0("PC", seq_len(ncol(ans)))
  ans
}

# [5] 評估指標：M 為陽性 -----------------------------------------
confusion <- function(actual, predicted) {
  table(Actual = factor(actual, levels = c("B", "M")),
        Predicted = factor(predicted, levels = c("B", "M")))
}

metrics <- function(actual, predicted) {
  tab <- confusion(actual, predicted)
  TN <- tab["B", "B"]; FP <- tab["B", "M"]
  FN <- tab["M", "B"]; TP <- tab["M", "M"]
  # 分母為 0 或 NA（例如模型全部預測為 B，Precision 無定義）時回傳 NA。
  safe <- function(a, b) if (is.na(b) || b == 0) NA_real_ else as.numeric(a / b)
  sensitivity <- safe(TP, TP + FN)
  specificity <- safe(TN, TN + FP)
  precision <- safe(TP, TP + FP)
  f1 <- safe(2 * precision * sensitivity, precision + sensitivity)
  c(Accuracy = safe(TP + TN, sum(tab)), Sensitivity = sensitivity,
    Specificity = specificity, Precision = precision, F1 = f1,
    BalancedAccuracy = mean(c(sensitivity, specificity), na.rm = TRUE),
    TN = TN, FP = FP, FN = FN, TP = TP)
}

# [6] 最終前處理：只由完整訓練集估計 -------------------------------
prep_final <- fit_preprocessor(X[train_idx, , drop = FALSE], n_pc = 5L)
X_train_pc <- apply_preprocessor(X[train_idx, , drop = FALSE], prep_final)
X_test_pc <- apply_preprocessor(X[test_idx, , drop = FALSE], prep_final)
train_dat <- data.frame(Class = droplevels(y[train_idx]), X_train_pc)
test_dat <- data.frame(Class = factor(y[test_idx], levels = c("B", "M")), X_test_pc)
train_missing <- sum(is.na(X[train_idx, ]))
test_missing <- sum(is.na(X[test_idx, ]))

# 表 3：前五個主成分的解釋變異與各特徵的權重，只由訓練集估計。
# 第一列 ExplainedVariance 是解釋變異比例，其餘每一列是一個特徵在 PC1 到 PC5 的權重。
loading_mat <- rbind(prep_final$explained, prep_final$rotation)
colnames(loading_mat) <- paste0("PC", seq_len(ncol(loading_mat)))
pca_loadings <- data.frame(Feature = c("ExplainedVariance", rownames(prep_final$rotation)),
                           round(loading_mat, 4), row.names = NULL, check.names = FALSE)
write.csv(pca_loadings, file.path("results", "（表 3）主成分權重.csv"), row.names = FALSE)

# [7] 訓練集內分層五折，供 SVM 調參與 QDA 補充分析使用 ---------------
set.seed(20261019)
fold <- integer(length(train_idx))
for (cls in levels(train_dat$Class)) {
  ii <- which(train_dat$Class == cls)
  fold[ii] <- sample(rep(seq_len(5), length.out = length(ii)))
}

# 每一折都重新估計填補、標準化與 PCA，驗證折不參與前處理估計。
cv_one_svm <- function(kernel, cost, gamma) {
  oof <- rep(NA_character_, length(train_idx))
  for (v in seq_len(5)) {
    inner <- fold != v
    valid <- fold == v
    prep <- fit_preprocessor(X[train_idx[inner], , drop = FALSE], n_pc = 5L)
    inner_pc <- apply_preprocessor(X[train_idx[inner], , drop = FALSE], prep)
    valid_pc <- apply_preprocessor(X[train_idx[valid], , drop = FALSE], prep)
    inner_dat <- data.frame(Class = droplevels(y[train_idx[inner]]), inner_pc)
    model <- e1071::svm(Class ~ ., data = inner_dat,
                        type = "C-classification", kernel = kernel,
                        cost = cost, gamma = gamma, scale = FALSE)
    oof[valid] <- as.character(predict(model, newdata = data.frame(valid_pc)))
  }
  c(metrics(y[train_idx], oof), cost = cost, gamma = gamma)
}

# SVM 參數網格：cost 控制誤分類懲罰；gamma 控制 RBF 核的影響範圍。
linear_grid <- data.frame(kernel = "linear", cost = c(0.1, 1, 10, 100),
                          gamma = 1 / ncol(X_train_pc))
radial_grid <- expand.grid(kernel = "radial", cost = c(0.1, 1, 10, 100),
                           gamma = c(0.01, 0.1, 1), stringsAsFactors = FALSE)
grid <- rbind(linear_grid, radial_grid)

cv_result <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  score <- cv_one_svm(grid$kernel[i], grid$cost[i], grid$gamma[i])
  data.frame(kernel = grid$kernel[i], cost = grid$cost[i], gamma = grid$gamma[i],
             t(score[c("Accuracy", "Sensitivity", "Specificity", "Precision",
                       "F1", "BalancedAccuracy", "FN", "FP")]), check.names = FALSE)
}))

# 選擇規則：Balanced Accuracy 最高；同分時 Sensitivity 較高、模型較簡單者優先。
choose_best <- function(tbl) {
  tbl[order(-tbl$BalancedAccuracy, -tbl$Sensitivity, tbl$cost, tbl$gamma), , drop = FALSE][1, ]
}
best_linear <- choose_best(cv_result[cv_result$kernel == "linear", ])
best_rbf <- choose_best(cv_result[cv_result$kernel == "radial", ])
write.csv(cv_result, file.path("results", "SVM 超參數選擇.csv"), row.names = FALSE)

# 和老師範例程式的預設參數比較：Linear cost = 1；RBF cost = 1、gamma = 1/5。
# 同樣只在訓練集內做五折 CV，不使用測試集。
svm_compare <- do.call(rbind, lapply(list(
  list("Linear SVM teacher default", "linear", 1, 1 / ncol(X_train_pc)),
  list("RBF-SVM teacher default", "radial", 1, 1 / ncol(X_train_pc)),
  list("Linear SVM tuned", "linear", best_linear$cost, best_linear$gamma),
  list("RBF-SVM tuned", "radial", best_rbf$cost, best_rbf$gamma)
), function(z) {
  s <- cv_one_svm(z[[2]], z[[3]], z[[4]])
  data.frame(Setting = z[[1]], cost = z[[3]], gamma = z[[4]],
             t(s[c("BalancedAccuracy", "Accuracy", "Sensitivity", "Specificity", "FN", "FP")]),
             check.names = FALSE)
}))
write.csv(svm_compare, file.path("results", "預設參數和調參後的比較.csv"),
          row.names = FALSE)

# [8] 最終模型與獨立測試集預測 -----------------------------------
fit_lda <- MASS::lda(Class ~ ., data = train_dat)
fit_qda <- MASS::qda(Class ~ ., data = train_dat)
fit_linear <- e1071::svm(Class ~ ., data = train_dat, kernel = "linear",
                         cost = best_linear$cost, scale = FALSE)
fit_rbf <- e1071::svm(Class ~ ., data = train_dat, kernel = "radial",
                      cost = best_rbf$cost, gamma = best_rbf$gamma, scale = FALSE)

predictions <- list(
  LDA = predict(fit_lda, newdata = test_dat)$class,
  QDA = predict(fit_qda, newdata = test_dat)$class,
  SVM_Linear = predict(fit_linear, newdata = test_dat),
  SVM_RBF = predict(fit_rbf, newdata = test_dat)
)

score_table <- do.call(rbind, lapply(names(predictions), function(name) {
  data.frame(Method = name, t(metrics(test_dat$Class, predictions[[name]])),
             row.names = NULL, check.names = FALSE)
}))
write.csv(score_table, file.path("results", "（表 5）測試集表現.csv"), row.names = FALSE)

prediction_table <- data.frame(RowID = test_idx, Actual = as.character(test_dat$Class))
for (name in names(predictions)) prediction_table[[name]] <- as.character(predictions[[name]])
write.csv(prediction_table, file.path("results", "混淆矩陣的原始資料.csv"), row.names = FALSE)

# [9] 案例專屬分析：高度相關特徵對 QDA 的影響 -----------------------
# 9a. 各類別共變異數矩陣的條件數（最大／最小特徵值）。
#     條件數越大，矩陣越接近奇異，求反矩陣與行列式越不穩定。
z_train <- scale(X[train_idx, ], center = prep_final$center, scale = prep_final$scale)
cond_number <- function(m) { ev <- eigen(cov(m), symmetric = TRUE, only.values = TRUE)$values
                             max(ev) / min(ev) }
qda_condition <- data.frame(
  Space = c("Standardized raw 30 features", "First 5 PCs"),
  Params_per_class = c(30 * 31 / 2, 5 * 6 / 2),
  Cond_B = c(cond_number(z_train[train_dat$Class == "B", ]),
             cond_number(X_train_pc[train_dat$Class == "B", ])),
  Cond_M = c(cond_number(z_train[train_dat$Class == "M", ]),
             cond_number(X_train_pc[train_dat$Class == "M", ]))
)
write.csv(qda_condition, file.path("results", "（表 6）條件數.csv"), row.names = FALSE)

# 9b. 只在訓練集內以相同五折比較不同 PC 數的 QDA。
#     k = 30 等同使用全部標準化原始特徵（全秩旋轉不改變 QDA 結果）。
cv_qda_k <- function(k) {
  oof <- rep(NA_character_, length(train_idx))
  for (v in seq_len(5)) {
    inner <- fold != v; valid <- fold == v
    prep <- fit_preprocessor(X[train_idx[inner], , drop = FALSE], n_pc = k)
    d_in <- data.frame(Class = droplevels(y[train_idx[inner]]),
                       apply_preprocessor(X[train_idx[inner], , drop = FALSE], prep))
    d_va <- data.frame(apply_preprocessor(X[train_idx[valid], , drop = FALSE], prep))
    m <- MASS::qda(Class ~ ., data = d_in)
    oof[valid] <- as.character(predict(m, newdata = d_va)$class)
  }
  metrics(y[train_idx], oof)
}
qda_k <- c(2, 3, 5, 10, 15, 20, 30)
qda_cv <- do.call(rbind, lapply(qda_k, function(k) {
  s <- cv_qda_k(k)
  data.frame(n_PC = k, t(s[c("Accuracy", "Sensitivity", "Specificity",
                              "BalancedAccuracy", "FN", "FP")]), check.names = FALSE)
}))
write.csv(qda_cv, file.path("results", "（表 7）QDA 不同 PC 數結果.csv"), row.names = FALSE)

# [10] 文字報告 ---------------------------------------------------
report <- capture.output({
  cat("作業 01：乳癌良性與惡性分類\n")
  cat("資料：", data_file, "\n", sep = "")
  cat("總樣本：", nrow(raw), "；訓練：", length(train_idx),
      "；測試：", length(test_idx), "\n", sep = "")
  cat("類別分布：\n"); print(class_tab); print(class_pct)
  cat("遺漏值總數：", n_missing, "（訓練 ", train_missing, "，測試 ", test_missing, "）\n", sep = "")
  cat("\n重要變數分組摘要：\n"); print(eda_table, row.names = FALSE, digits = 4)
  cat("\n|r| > 0.9 的特徵對數：", n_pairs_09, "\n")
  print(head(cor_pairs, 5), row.names = FALSE, digits = 4)
  cat("\n前五個 PC 個別解釋變異：",
      paste(sprintf("%.2f%%", 100 * prep_final$explained), collapse = ", "), "\n")
  cat("前五個 PC 累積解釋變異：",
      sprintf("%.2f%%", 100 * sum(prep_final$explained)), "\n")
  cat("Linear SVM 最佳 cost：", best_linear$cost, "\n")
  cat("RBF-SVM 最佳 cost/gamma：", best_rbf$cost, "/", best_rbf$gamma, "\n")
  cat("\nSVM 五折 CV 結果（訓練集內）：\n"); print(cv_result, row.names = FALSE, digits = 4)
  cat("\n老師預設參數與調參後比較（訓練集五折 CV）：\n"); print(svm_compare, row.names = FALSE, digits = 4)
  cat("\n獨立測試集結果：\n"); print(score_table, row.names = FALSE, digits = 4)
  for (name in names(predictions)) {
    cat("\n", name, "\n", sep = "")
    print(confusion(test_dat$Class, predictions[[name]]))
  }
  cat("\nQDA 共變異數條件數：\n"); print(qda_condition, row.names = FALSE, digits = 4)
  cat("\nQDA 不同 PC 數的訓練集五折 CV：\n"); print(qda_cv, row.names = FALSE, digits = 4)
  cat("\nM 為陽性。False Negative 是實際惡性但預測為良性。\n")
  cat("此分析是統計教學，不能作為臨床診斷工具。\n")
})
writeLines(report, file.path("results", "輸出結果.txt"), useBytes = TRUE)
cat(paste(report, collapse = "\n"), "\n")

# [11] 圖表 -------------------------------------------------------
# 圖 C：訓練資料 PCA 投影（僅視覺化，模型使用前五個 PC）
png(file.path("results", "（圖 2）分布圖.png"), width = 1050, height = 760, res = 130)
plot(X_train_pc[, 1], X_train_pc[, 2],
     col = ifelse(train_dat$Class == "M", "#D47A4A", "#3D7EA6"), pch = 19, cex = 0.8,
     xlab = sprintf("PC1 (%.1f%%)", 100 * prep_final$explained[1]),
     ylab = sprintf("PC2 (%.1f%%)", 100 * prep_final$explained[2]),
     main = "Training data PCA projection")
legend("topright", legend = c("Benign B", "Malignant M"),
       col = c("#3D7EA6", "#D47A4A"), pch = 19, bty = "n")
dev.off()

# 圖 D：模型指標比較
png(file.path("results", "（圖 5）指標長條圖.png"), width = 1150, height = 760, res = 130)
bp <- barplot(t(as.matrix(score_table[, c("Accuracy", "Sensitivity", "Specificity", "F1")])),
        beside = TRUE, ylim = c(0, 1.18), names.arg = score_table$Method,
        col = c("#1F7A78", "#D47A4A", "#3D7EA6", "#8C8C8C"),
        ylab = "Metric", main = "Independent test performance (n = 115)")
text(bp, t(as.matrix(score_table[, c("Accuracy", "Sensitivity", "Specificity", "F1")])) + 0.03,
     sprintf("%.3f", t(as.matrix(score_table[, c("Accuracy", "Sensitivity", "Specificity", "F1")]))),
     cex = 0.6)
legend("top", legend = c("Accuracy", "Sensitivity", "Specificity", "F1"), horiz = TRUE,
       fill = c("#1F7A78", "#D47A4A", "#3D7EA6", "#8C8C8C"), bty = "n", cex = 0.85)
dev.off()

# 圖 E：四個模型的混淆矩陣（列為實際、欄為預測；左下格為 FN）
png(file.path("results", "（圖 4）混淆矩陣.png"), width = 1300, height = 400, res = 130)
op <- par(mfrow = c(1, 4), mar = c(3.5, 3.5, 3, 1))
for (name in names(predictions)) {
  tab <- confusion(test_dat$Class, predictions[[name]])
  plot(NA, xlim = c(0, 2), ylim = c(0, 2), axes = FALSE, xlab = "", ylab = "",
       main = name, asp = 1)
  fills <- c("#DCEBF5", "#F6DCCD", "#F6DCCD", "#DCEBF5")
  pos <- list(c(0, 1), c(1, 1), c(0, 0), c(1, 0))      # BB, BM, MB, MM
  vals <- c(tab["B", "B"], tab["B", "M"], tab["M", "B"], tab["M", "M"])
  labs <- c("TN", "FP", "FN", "TP")
  for (k in 1:4) {
    rect(pos[[k]][1], pos[[k]][2], pos[[k]][1] + 1, pos[[k]][2] + 1, col = fills[k], border = "white", lwd = 3)
    text(pos[[k]][1] + 0.5, pos[[k]][2] + 0.58, vals[k], cex = 1.6, font = 2)
    text(pos[[k]][1] + 0.5, pos[[k]][2] + 0.28, labs[k], cex = 0.8, col = "grey30")
  }
  axis(1, at = c(0.5, 1.5), labels = c("Pred B", "Pred M"), tick = FALSE, line = -0.8)
  axis(2, at = c(1.5, 0.5), labels = c("True B", "True M"), tick = FALSE, line = -0.8)
}
par(op); dev.off()

# 圖 F：QDA 的 PC 數與訓練集五折 CV 表現
png(file.path("results", "（圖 6）交叉驗證結果.png"), width = 1000, height = 620, res = 130)
matplot(qda_cv$n_PC, qda_cv[, c("BalancedAccuracy", "Sensitivity", "Specificity")],
        type = "b", pch = 19, lty = 1, col = c("#1F7A78", "#D47A4A", "#3D7EA6"),
        xlab = "Number of PCs used by QDA (30 = all standardized features)",
        ylab = "5-fold CV metric (training set)", main = "QDA vs. dimensionality",
        ylim = c(0.82, 1))
legend("bottomright", legend = c("Balanced accuracy", "Sensitivity", "Specificity"),
       col = c("#1F7A78", "#D47A4A", "#3D7EA6"), pch = 19, lty = 1, bty = "n")
dev.off()

cat("\n完成。所有輸出位於 results/。\n")
