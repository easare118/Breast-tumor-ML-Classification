# =============================================================================
# Comparative Evaluation of Machine-Learning Models for Distinguishing Benign
# and Malignant Breast Tumors Using Cell-Nucleus Morphological Features
# Data: Wisconsin Diagnostic Breast Cancer (WDBC), 569 samples, 30 features
# Run from the repository root:  Rscript scripts/wdbc_analysis.R
# =============================================================================

# install.packages(c("caTools","e1071","rpart","randomForest","pROC"))   # run once
library(caTools); library(class); library(e1071)
library(rpart); library(randomForest); library(pROC)

dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

# 1. Load data -----------------------------------------------------------------
wdbc <- read.csv("data/wdbc.data", header = FALSE)
names(wdbc) <- c("id", "diagnosis",
                 paste0(c("radius","texture","perimeter","area","smoothness","compactness",
                          "concavity","concave_points","symmetry","fractal_dim"),
                        rep(c("_mean","_se","_worst"), each = 10)))
wdbc$id <- NULL
wdbc$diagnosis <- factor(wdbc$diagnosis, levels = c("B","M"), labels = c("Benign","Malignant"))
print(table(wdbc$diagnosis))

# 2. Stratified 70/30 split (caTools); scale with training-set values only -----
set.seed(123)
split <- sample.split(wdbc$diagnosis, SplitRatio = 0.70)
train <- subset(wdbc, split == TRUE)
test  <- subset(wdbc, split == FALSE)

X   <- setdiff(names(wdbc), "diagnosis")
mu  <- colMeans(train[X]);  sdv <- apply(train[X], 2, sd)
train[X] <- scale(train[X], mu, sdv)
test[X]  <- scale(test[X],  mu, sdv)

# 3. Train models --------------------------------------------------------------
res <- list()

m <- suppressWarnings(glm(diagnosis ~ ., data = train, family = binomial))
p <- suppressWarnings(predict(m, test, type = "response"))
res$Logistic <- list(p = p, cl = ifelse(p > 0.5, "Malignant", "Benign"))

k  <- knn(train[X], test[X], train$diagnosis, k = 7, prob = TRUE)
pk <- ifelse(k == "Malignant", attr(k, "prob"), 1 - attr(k, "prob"))
res$KNN <- list(p = pk, cl = as.character(k))

m  <- svm(diagnosis ~ ., data = train, kernel = "radial", probability = TRUE)
pr <- predict(m, test, probability = TRUE)
res$SVM <- list(p = attr(pr, "probabilities")[, "Malignant"], cl = as.character(pr))

m <- rpart(diagnosis ~ ., data = train, method = "class")
p <- predict(m, test)[, "Malignant"]
res$DecisionTree <- list(p = p, cl = ifelse(p > 0.5, "Malignant", "Benign"))

m_rf <- randomForest(diagnosis ~ ., data = train, ntree = 500)
p <- predict(m_rf, test, type = "prob")[, "Malignant"]
res$RandomForest <- list(p = p, cl = ifelse(p > 0.5, "Malignant", "Benign"))

# 4. Evaluate on the held-out test set ----------------------------------------
evaluate <- function(name, r) {
  cl <- factor(r$cl, levels = c("Benign","Malignant"))
  tb <- table(Predicted = cl, Actual = test$diagnosis)
  TP <- tb["Malignant","Malignant"]; TN <- tb["Benign","Benign"]
  FP <- tb["Malignant","Benign"];    FN <- tb["Benign","Malignant"]
  data.frame(Model = name,
             Accuracy    = (TP+TN)/sum(tb),
             Sensitivity = TP/(TP+FN),
             Specificity = TN/(TN+FP),
             Precision   = TP/(TP+FP),
             F1          = 2*TP/(2*TP+FP+FN),
             AUC         = as.numeric(auc(roc(test$diagnosis, r$p, quiet = TRUE))))
}
results <- do.call(rbind, Map(evaluate, names(res), res))
results[-1] <- round(results[-1], 4)
results <- results[order(-results$Accuracy, -results$AUC), ]
print(results, row.names = FALSE)
write.csv(results, "results/model_performance.csv", row.names = FALSE)

best <- results$Model[1]
cm <- table(Predicted = factor(res[[best]]$cl, levels = c("Benign","Malignant")),
            Actual = test$diagnosis)
cat("\nConfusion matrix -", best, "\n"); print(cm)
write.csv(as.data.frame.matrix(cm), "results/confusion_matrix_best_model.csv")

# 5. ROC curves ----------------------------------------------------------------
cols <- c("blue","red","darkgreen","purple","orange")
plot_roc <- function() {
  plot(roc(test$diagnosis, res[[1]]$p, quiet = TRUE), col = cols[1], lwd = 2,
       main = "ROC curves (test set)")
  for (i in 2:5) plot(roc(test$diagnosis, res[[i]]$p, quiet = TRUE), col = cols[i], lwd = 2, add = TRUE)
  legend("bottomright", names(res), col = cols, lwd = 2)
}
plot_roc()
png("results/figures/roc_curves.png", width = 1300, height = 1300, res = 200); plot_roc(); dev.off()

# 6. Feature importance (random forest) ---------------------------------------
varImpPlot(m_rf, n.var = 10, main = "Top 10 important features")
png("results/figures/rf_feature_importance.png", width = 1300, height = 1300, res = 200)
varImpPlot(m_rf, n.var = 10, main = "Top 10 important features"); dev.off()

writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")
