rm(list=ls())

# ---------------------------------------- import library ----------------------------------------
library(readxl)
library(corrr)
library(ggcorrplot)
library(ggplot2)
library(reshape2)
library(factoextra)

# ---------------------------------------- import data ----------------------------------------
dataset = read.csv('../data/normalized_data.csv')
dataset = na.omit(dataset)        # handle NA values
remove.columns = c('date', 'daily_trading_value', 'UBB', 'LBB', 'BollMA', 'MA_5', 'MA_25', 'MACD', 'sig_9', 'diff', 'golden_cross', 'death_cross', 'return')
raw.data.normalized = dataset[ ,!(colnames(dataset) %in% remove.columns)]        # exclude date in dataset raw_data
head(as.data.frame(raw.data.normalized))

# ---------------------------------------- plot heatmap ----------------------------------------
# plot correlation heatmap on normalized data
normalized.data.corr.matrix <- cor(raw.data.normalized)
ggcorrplot(normalized.data.corr.matrix)

# get the daily changes of normalized dataframe 
normalized.data.diff = as.data.frame(apply(raw.data.normalized, 2, diff))
head(normalized.data.diff)

# plot correlation heatmap on normalized data diff
normalized.data.diff.corr.matrix <- cor(normalized.data.diff)
ggcorrplot(normalized.data.diff.corr.matrix)

# ---------------------------------------- PCA Method 1----------------------------------------
# Method1: PCA based on normalized raw data diff
# as.data.frame(apply(normalized.data.diff, 2, mean))
# PCA on normalzied.data.diff
out1 = princomp(normalized.data.diff)
summary(out1)

# visualization of PAC result
(PCA_componants <- as.data.frame(out1$loadings[,1:3]))
setwd('../data')
write.csv(PCA_componants, file = 'normalized_data_diff_PCA_componants.csv')

fviz_eig(out1, addlabels = TRUE)

# graph of the variables
fviz_pca_var(out1, col.var = "cos2", gradient.cols = c("black", "yellow", "red"), repel = TRUE)

# contribution of each variable
fviz_cos2(out1, choice = "var", axes = 1:2)

# ---------------------------------------- PCA Method 2 ----------------------------------------
# PCA on normalzied.data
out2 = princomp(raw.data.normalized)
summary(out2)

# visualization of PAC result
(PCA_componants <- as.data.frame(out2$loadings[, 1:3]))
setwd('../data')
write.csv(PCA_componants, file = 'normalized_data_PCA_componants.csv')


# visualization of PAC result
as.data.frame(out2$loadings[,1:4])

fviz_eig(out2, addlabels = TRUE)

# graph of the variables
fviz_pca_var(out2, col.var = "cos2", gradient.cols = c("black", "yellow", "red"), repel = TRUE)

# contribution of each variable
fviz_cos2(out2, choice = "var", axes = 1:2)
