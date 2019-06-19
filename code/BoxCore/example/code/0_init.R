
tfspath = strsplit(getwd(), "BOX", fixed = TRUE)[[1]][1]
.libPaths(c(paste0(tfspath,"BOX/Repository/Predictive_Modelling/BoxCore_Build"),
#.libPaths(c(paste0(tfspath,"BOX/Repository/Predictive_Modelling/BoxCore_Release/BoxCore-1.0"),
            paste0(tfspath,"BOX/CommonTools/R_Environment/R-3.3.0/library")))



#######################################################################################################################-
# Libraries + Parallel Processing Start ----
#######################################################################################################################-

library(Matrix)
library(plyr) #always load plyr before dplyr
# library(tidyverse)
library(ggplot2);library(dplyr);library(tidyr);library(readr);library(purrr);library(tibble);library(stringr);library(forcats)
library(readxl)

library(lubridate)
library(bindrcpp)
library(magrittr)
library(zeallot)
library(doParallel)

library(corrplot)
library(grid)
library(gridExtra)
library(waterfalls)
library(boot)
library(hexbin)

library(caret)
library(xgboost)
library(glmnet)
library(ranger)
library(ROCR)

#library(MicrosoftML)

#library(devtools); install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboostExplainer)

#library(devtools); options(devtools.install.args = "--no-multiarch"); install_github("Microsoft/LightGBM", subdir = "R-package")
library(lightgbm)

#library(h2o); h2o.init()
#library(keras)

library(BoxCore)


#######################################################################################################################-
# Parameters ----
#######################################################################################################################-

# Locations
dataloc = "data/"
plotloc = "output/"

# Colors
twocol = c("red","darkgreen")
threecol = c("#00BA38","#B79F00","#F8766D")

# OVerwrite BoxCore options
l.options = getOption("BoxCore_plot")
l.options$distr$class$colors = twocol
l.options$distr$multiclass$colors = threecol
options(BoxCore_plot = l.options)
#barplot(1:length(l.options$colors$many), col = l.options$colors$many)
#sevencol = l.options$colors$many[c(2,4,3,5,1,6,9)]
#hexcol = colorRampPalette(l.options$colors$hex)(100)






#######################################################################################################################-
# My Functions ----
#######################################################################################################################-

## Workaround for ggsave and marrangeGrob not to create first page blank
grid.draw.arrangelist <- function(x, ...) {
  for (ii in seq_along(x)) {
    if (ii > 1) grid.newpage()  # skips grid.newpage() call the first time around
    grid.draw(x[[ii]])
  }
}



#######################################################################################################################-
# Caret definition of non-caret algorithms ----
#######################################################################################################################-

## rxFastTrees (boosted trees)
ms_boosttree = list()
ms_boosttree$label = "MicrosoftML rxFastTrees"
ms_boosttree$library = c("MicrosoftML","RevoScaleR")
ms_boosttree$type = c("Regression","Classification")
ms_boosttree$parameters =
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             numTrees,numeric,numTrees
             numLeaves,numeric,numLeaves
             minSplit,numeric,minSplit
             learningRate,numeric,learningRate
             featureFraction,numeric,featureFraction
             exampleFraction,numeric,exampleFractione"
  )

ms_boosttree$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(numTrees = floor((1:len) * 50),
                       numLeaves = 2^seq(1, len),
                       minSplit = 10,
                       learningRate = .1,
                       featureFraction = 0.7,
                       exampleFraction = 0.7)
  } else {
    out <- data.frame(numTrees = floor(runif(len, min = 10, max = 5000)),
                      numLeaves = 2 ^ sample(1:6, replace = TRUE, size = len),
                      minSplit = 2 ^ sample(0:6, replace = TRUE, size = len),
                      learningRate = runif(len, min = .001, max = .6),
                      featureFraction = runif(len, min = .1, max = 1),
                      exampleFraction = runif(len, min = .1, max = 1))
    out <- out[!duplicated(out),]
  }
  out
}

ms_boosttree$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  #browser()
  theDots = list(...)
  #if (is.factor(y) && length(lev) == 2) {y = ifelse(y == lev[1], 1, 0)}
  #y = factor(y, levels = c(1,0))
  #x = as.matrix(x)
  if (is.factor(y)) type = "binary" else type = "regression"
  modArgs <- list(formula = paste("y~", paste0(names(x), collapse = "+")),
                  data = cbind(x, y),
                  numTrees = param$numTrees,
                  numLeaves = param$numLeaves,
                  minSplit = param$minSplit,
                  learningRate = param$learningRate,
                  featureFraction = param$featureFraction,
                  exampleFraction = param$exampleFraction,
                  type = type)
  if (length(theDots) > 0) modArgs <- c(modArgs, theDots)
  do.call(MicrosoftML::rxFastTrees, modArgs)
  #out = MicrosoftML::rxFastTrees(formula = modArgs$formula, data = modArgs$data, type = modArgs$type)
  #out
}

ms_boosttree$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (modelFit$problemType == "Classification") {
    out = RevoScaleR::rxPredict(modelFit, newdata)$Probability.Y
  } else {
    newdata$y = NA
    out = RevoScaleR::rxPredict(modelFit, newdata)$Score
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- ifelse(out >= 0.5, "Y", "N")
  }
  out
}

ms_boosttree$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  out = RevoScaleR::rxPredict(modelFit, newdata)[,"Probability.Y"]
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- c("Y","N")
  }
  out
}

ms_boosttree$levels = function(x) {c("N","Y")}

ms_boosttree$sort = function(x) {
  x[order(x$numTrees, x$numLeaves, x$learningRate), ]
}




## rxForest (random Forest)
ms_forest = list()
ms_forest$label = "MicrosoftML rxFastForest"
ms_forest$library = c("MicrosoftML","RevoScaleR")
ms_forest$type = c("Regression","Classification")
ms_forest$parameters =
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             numTrees,numeric,numTrees
             splitFraction,numeric,splitFraction"
  )

ms_forest$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(numTrees = floor((1:len) * 50),
                       splitFraction = seq(0.01, 1, length.out = len))
  } else {
    out <- data.frame(numTrees = floor(runif(len, min = 1, max = 5000)),
                      splitFraction = runif(len, min = 0.01, max = 1))
    out <- out[!duplicated(out),]
  }
  out
}

ms_forest$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  theDots = list(...)
  if (is.factor(y)) type = "binary" else type = "regression"
  modArgs <- list(formula = paste("y~", paste0(names(x), collapse = "+")),
                  data = cbind(x, y),
                  numTrees = param$numTrees,
                  splitFraction = param$splitFraction,
                  type = type)
  if (length(theDots) > 0) modArgs <- c(modArgs, theDots)
  do.call(MicrosoftML::rxFastForest, modArgs)
}

ms_forest$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (modelFit$problemType == "Classification") {
    out = RevoScaleR::rxPredict(modelFit, newdata)$Probability.Y
  } else {
    newdata$y = NA
    out = RevoScaleR::rxPredict(modelFit, newdata)$Score
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- ifelse(out >= 0.5, "Y", "N")
  }
  out
}

ms_forest$prob = function(modelFit, newdata, submodels = NULL) {
  out = RevoScaleR::rxPredict(modelFit, newdata)[,"Probability.Y"]
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- c("Y","N")
  }
  out
}

ms_forest$levels = function(x) {c("N","Y")}

ms_forest$sort = function(x) {
  x[order(x$numTrees, x$splitFraction), ]
}



## lightgbm (boosted trees)

lgbm = list()
lgbm$label = "lightgbm"
lgbm$library = c("lightgbm","plyr")
lgbm$type = c("Regression","Classification")
lgbm$parameters =
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             nrounds,numeric,nrounds
             num_leaves,numeric,num_leaves
             min_data_in_leaf,numeric,min_data_in_leaf
             learning_rate,numeric,learning_rate
             feature_fraction,numeric,feature_fraction
             bagging_fraction,numeric,bagging_fraction"
  )

lgbm$grid = function(x, y, len = NULL, search = "grid") {
  #browser()
  if (search == "grid") {
    out <- expand.grid(nrounds = floor((1:len) * 50),
                       num_leaves = 2^seq(1, len),
                       min_data_in_leaf = 10,
                       learning_rate = .1,
                       feature_fraction = 0.7,
                       bagging_fraction = 0.7)
  } else {
    out <- data.frame(nrounds = floor(runif(len, min = 10, max = 5000)),
                      num_leaves = 2 ^ sample(1:6, replace = TRUE, size = len),
                      min_data_in_leaf = 2 ^ sample(0:6, replace = TRUE, size = len),
                      learning_rate = runif(len, min = .001, max = .6),
                      feature_fraction = runif(len, min = .1, max = 1),
                      bagging_fraction = runif(len, min = .1, max = 1))
    out <- out[!duplicated(out),]
  }
  out
}

lgbm$loop = function(grid) {
  #browser()
  loop <- plyr::ddply(grid,
                c("learning_rate", "num_leaves", "feature_fraction", "min_data_in_leaf", "bagging_fraction"),
                function(x) c(nrounds = max(x$nrounds)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for (i in seq(along = loop$nrounds)) {
    index <- which(grid$learning_rate == loop$learning_rate[i] &
                   grid$num_leaves == loop$num_leaves[i] &
                   grid$feature_fraction == loop$feature_fraction[i] &
                   grid$min_data_in_leaf == loop$min_data_in_leaf[i] &
                   grid$bagging_fraction == loop$bagging_fraction[i])
    trees <- grid[index, "nrounds"]
    submodels[[i]] <- data.frame(nrounds = trees[trees != loop$nrounds[i]])
  }
  list(loop = loop, submodels = submodels)
}

lgbm$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  #browser()
  theDots = list(...)
  if (is.factor(y)) objective = "binary" else objective = "regression_l2"
  if (is.factor(y)) y = as.numeric(y) - 1
  modArgs <- list(data = lightgbm::lgb.Dataset(x, label = y),
                  nrounds = param$nrounds,
                  num_leaves = param$num_leaves,
                  min_data_in_leaf = param$min_data_in_leaf,
                  learning_rate = param$learning_rate,
                  feature_fraction = param$feature_fraction,
                  bagging_fraction = param$bagging_fraction,
                  objective = objective)
  if (length(theDots) > 0) modArgs <- c(modArgs, theDots)
  list("model" = do.call(lightgbm::lightgbm, modArgs)) #put it into list as it is a S4 object!
}

lgbm$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (modelFit$problemType == "Classification") {
    out = predict(modelFit$model, newdata)
  } else {
    out = predict(modelFit$model, newdata)
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- ifelse(out >= 0.5, "Y", "N")
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) +
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit$model, newdata, num_iteration = submodels$nrounds[j])
      if (modelFit$problemType == "Classification") {
        if (length(modelFit$obsLevels) == 2) {
          tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1],
                             modelFit$obsLevels[2])
        } else {
          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels),
                             byrow = TRUE)
          tmp_pred <- modelFit$obsLevels[apply(tmp_pred,
                                               1, which.max)]
        }
      }
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

lgbm$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  out = predict(modelFit$model, newdata)
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- c("Y","N")
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) + 1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit$model, newdata, num_iteration = submodels$nrounds[j])
      if (length(modelFit$obsLevels) == 2) {
        tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
        colnames(tmp_pred) <- c("Y","N")
      } else {
        tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels),
                           byrow = TRUE)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      tmp_pred <- as.data.frame(tmp_pred)
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

lgbm$levels = function(x) {c("N","Y")}

lgbm$sort = function(x) {
  #browser()
  x[order(x$nrounds, x$num_leaves, x$learning_rate, x$feature_fraction,
          x$bagging_fraction), ]
}


## glmnet with support for dgCMatrix

glmnet_custom = getModelInfo("glmnet")$glmnet

glmnet_custom$fit = function (x, y, wts, param, lev, last, classProbs, ...)
{
  #browser()
  numLev <- if (is.character(y) | is.factor(y))
    length(levels(y))
  else NA
  theDots <- list(...)
  if (all(names(theDots) != "family")) {
    if (!is.na(numLev)) {
      fam <- ifelse(numLev > 2, "multinomial", "binomial")
    }
    else fam <- "gaussian"
    theDots$family <- fam
  }
  if (!is.null(wts))
    theDots$weights <- wts
  if (!(class(x)[1] %in% c("matrix", "sparseMatrix", "dgCMatrix")))
    x <- Matrix::as.matrix(x)
  modelArgs <- c(list(x = x, y = y, alpha = param$alpha), theDots)
  out <- do.call(glmnet::glmnet, modelArgs)
  if (!is.na(param$lambda[1]))
    out$lambdaOpt <- param$lambda[1]
  out
}

glmnet_custom$predict = function (modelFit, newdata, submodels = NULL)
{
  #browser()
  if (!(class(newdata)[1] %in% c("matrix", "sparseMatrix", "dgCMatrix")))
    newdata <- Matrix::as.matrix(newdata)
  if (length(modelFit$obsLevels) < 2) {
    out <- predict(modelFit, newdata, s = modelFit$lambdaOpt, type = "response")
  }
  else {
    out <- predict(modelFit, newdata, s = modelFit$lambdaOpt, type = "class")
  }
  if (is.matrix(out))
    out <- out[, 1]
  if (!is.null(submodels)) {
    if (length(modelFit$obsLevels) < 2) {
      tmp <- as.list(as.data.frame(predict(modelFit, newdata,
                                           s = submodels$lambda, type = "response")))
    }
    else {
      tmp <- predict(modelFit, newdata, s = submodels$lambda, type = "class")
      tmp <- if (is.matrix(tmp))
        as.data.frame(tmp, stringsAsFactors = FALSE)
      else as.character(tmp)
      tmp <- as.list(tmp)
    }
    out <- c(list(out), tmp)
  }
  out
}

glmnet_custom$prob = function (modelFit, newdata, submodels = NULL)
{
  #browser()
  if (!(class(newdata)[1] %in% c("matrix", "sparseMatrix", "dgCMatrix")))
    newdata <- Matrix::as.matrix(newdata)
  obsLevels <- if ("classnames" %in% names(modelFit))
    modelFit$classnames
  else NULL
  probs <- predict(modelFit, newdata, s = modelFit$lambdaOpt,
                   type = "response")
  if (length(obsLevels) == 2) {
    probs <- as.vector(probs)
    probs <- as.data.frame(cbind(1 - probs, probs))
    colnames(probs) <- modelFit$obsLevels
  }
  else {
    probs <- as.data.frame(probs[, , 1, drop = FALSE])
    names(probs) <- modelFit$obsLevels
  }
  if (!is.null(submodels)) {
    tmp <- predict(modelFit, newdata,
                   s = submodels$lambda, type = "response")
    if (length(obsLevels) == 2) {
      tmp <- as.list(as.data.frame(tmp))
      tmp <- lapply(tmp, function(x, lev) {
        x <- as.vector(x)
        tmp <- data.frame(1 - x, x)
        names(tmp) <- lev
        tmp
      }, lev = modelFit$obsLevels)
    }
    else tmp <- apply(tmp, 3, function(x) data.frame(x))
    probs <- if (is.list(tmp))
      c(list(probs), tmp)
    else list(probs, tmp)
  }
  probs
}



## xgboost with alpha and lambda (boosted trees)

xgb = getModelInfo("xgb")$xgbTree

xgb$label = "xgbTree custom"

xgb$parameters =
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             nrounds,numeric,nrounds
             max_depth,numeric,Max Tree Depth
             eta,numeric,Shrinkage
             gamma,numeric,Minimum Loss Reduction
             colsample_bytree,numeric,Subsample Ratio of Columns
             min_child_weight,numeric,Minimum Sum of Instance Weight
             subsample,numeric,Subsample Percentage
             alpha,numeric,alpha
             lambda,numeric,lambda")

xgb$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(max_depth = seq(1, len), nrounds = floor((1:len) * 50),
                       eta = c(0.3, 0.4), gamma = 0, colsample_bytree = c(0.6,0.8),
                       min_child_weight = c(1), subsample = seq(0.5,1, length = len),
                       alpha = 0, lambda = 1)
  }
  else {
    out <- data.frame(nrounds = sample(1:1000, size = len, replace = TRUE),
                      max_depth = sample(1:10, replace = TRUE, size = len),
                      eta = runif(len, min = 0.001, max = 0.6),
                      gamma = runif(len, min = 0, max = 10),
                      colsample_bytree = runif(len, min = 0.3, max = 0.7),
                      min_child_weight = sample(0:20, size = len, replace = TRUE),
                      subsample = runif(len, min = 0.25, max = 1),
                      alpha = 0, lambda = 1)
    out$nrounds <- floor(out$nrounds)
    out <- out[!duplicated(out), ]
  }
  out
}

xgb$loop = function(grid) {
  loop <- plyr::ddply(grid, c("eta", "max_depth", "gamma",
                              "colsample_bytree", "min_child_weight", "subsample","alpha","lambda"),
                      function(x) c(nrounds = max(x$nrounds)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for (i in seq(along = loop$nrounds)) {
    index <- which(grid$max_depth == loop$max_depth[i] &
                     grid$eta == loop$eta[i] & grid$gamma == loop$gamma[i] &
                     grid$colsample_bytree == loop$colsample_bytree[i] &
                     grid$min_child_weight == loop$min_child_weight[i] &
                     grid$subsample == loop$subsample[i] &
                     grid$alpha == loop$alpha[i] &
                     grid$lambda == loop$lambda[i])
    trees <- grid[index, "nrounds"]
    submodels[[i]] <- data.frame(nrounds = trees[trees != loop$nrounds[i]])
  }
  list(loop = loop, submodels = submodels)
}

xgb$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  #browser()
  #theDots = list(...)

  # if (!inherits(x, "xgb.DMatrix"))
  #   x <- as.matrix(x)

  if (is.factor(y)) {
    if (length(lev) == 2) {
      y <- ifelse(y == lev[1], 1, 0)
      if (!inherits(x, "xgb.DMatrix"))
        x <- xgboost::xgb.DMatrix(x, label = y)
      else xgboost::setinfo(x, "label", y)
      if (!is.null(wts))
        xgboost::setinfo(x, "weight", wts)
      out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth,
                                     gamma = param$gamma, colsample_bytree = param$colsample_bytree,
                                     min_child_weight = param$min_child_weight, subsample = param$subsample,
                                     alpha = param$alpha, lambda = param$lambda),
                                data = x, nrounds = param$nrounds, objective = "binary:logistic",
                                ...)
    }
    else {
      y <- as.numeric(y) - 1
      if (!inherits(x, "xgb.DMatrix"))
        x <- xgboost::xgb.DMatrix(x, label = y)
      else xgboost::setinfo(x, "label", y)
      if (!is.null(wts))
        xgboost::setinfo(x, "weight", wts)
      out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth,
                                     gamma = param$gamma, colsample_bytree = param$colsample_bytree,
                                     min_child_weight = param$min_child_weight, subsample = param$subsample,
                                     alpha = param$alpha, lambda = param$lambda),
                                data = x, num_class = length(lev), nrounds = param$nrounds,
                                objective = "multi:softprob", ...)
    }
  }
  else {
    if (!inherits(x, "xgb.DMatrix"))
      x <- xgboost::xgb.DMatrix(x, label = y)
    else xgboost::setinfo(x, "label", y)
    if (!is.null(wts))
      xgboost::setinfo(x, "weight", wts)
    out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth,
                                   gamma = param$gamma, colsample_bytree = param$colsample_bytree,
                                   min_child_weight = param$min_child_weight, subsample = param$subsample,
                                   alpha = param$alpha, lambda = param$lambda),
                              data = x, nrounds = param$nrounds, objective = "reg:linear",
                              ...)
  }
  out
}

xgb$predict = function(modelFit, newdata, submodels = NULL) {
  if (!inherits(newdata, "xgb.DMatrix")) {
    #newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata)
  }
  out <- predict(modelFit, newdata)
  if (modelFit$problemType == "Classification") {
    if (length(modelFit$obsLevels) == 2) {
      out <- ifelse(out >= 0.5, modelFit$obsLevels[1],
                    modelFit$obsLevels[2])
    }
    else {
      out <- matrix(out, ncol = length(modelFit$obsLevels),
                    byrow = TRUE)
      out <- modelFit$obsLevels[apply(out, 1, which.max)]
    }
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) +
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
      if (modelFit$problemType == "Classification") {
        if (length(modelFit$obsLevels) == 2) {
          tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1],
                             modelFit$obsLevels[2])
        }
        else {
          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels),
                             byrow = TRUE)
          tmp_pred <- modelFit$obsLevels[apply(tmp_pred,
                                               1, which.max)]
        }
      }
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

xgb$prob = function(modelFit, newdata, submodels = NULL) {
  if (!inherits(newdata, "xgb.DMatrix")) {
    #newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata)
  }
  if (!is.null(modelFit$param$objective) && modelFit$param$objective ==
      "binary:logitraw") {
    p <- predict(modelFit, newdata)
    out <- binomial()$linkinv(p)
  }
  else {
    out <- predict(modelFit, newdata)
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- modelFit$obsLevels
  }
  else {
    out <- matrix(out, ncol = length(modelFit$obsLevels),
                  byrow = TRUE)
    colnames(out) <- modelFit$obsLevels
  }
  out <- as.data.frame(out)
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) +
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
      if (length(modelFit$obsLevels) == 2) {
        tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      else {
        tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels),
                           byrow = TRUE)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      tmp_pred <- as.data.frame(tmp_pred)
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}




## xgboost with alpha and lambda (boosted trees)

xgb_custom = getModelInfo("xgb")$xgbTree

xgb_custom$label = "xgbTree custom"

xgb_custom$parameters =
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             nrounds,numeric,nrounds
             max_depth,numeric,Max Tree Depth
             eta,numeric,Shrinkage
             gamma,numeric,Minimum Loss Reduction
             colsample_bytree,numeric,Subsample Ratio of Columns
             min_child_weight,numeric,Minimum Sum of Instance Weight
             subsample,numeric,Subsample Percentage
             alpha,numeric,alpha
             lambda,numeric,lambda,
             tweedie_variance_power,numeric,Tweedie Variance Power")

xgb_custom$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(max_depth = seq(1, len), nrounds = floor((1:len) * 50),
                       eta = c(0.3, 0.4), gamma = 0, colsample_bytree = c(0.6,0.8),
                       min_child_weight = c(1), subsample = seq(0.5,1, length = len),
                       alpha = 0, lambda = 1, tweedie_variance_power = 1.5)
  }
  else {
    out <- data.frame(nrounds = sample(1:1000, size = len, replace = TRUE),
                      max_depth = sample(1:10, replace = TRUE, size = len),
                      eta = runif(len, min = 0.001, max = 0.6),
                      gamma = runif(len, min = 0, max = 10),
                      colsample_bytree = runif(len, min = 0.3, max = 0.7),
                      min_child_weight = sample(0:20, size = len, replace = TRUE),
                      subsample = runif(len, min = 0.25, max = 1),
                      alpha = 0, lambda = 1, tweedie_variance_power = 1.5)
    out$nrounds <- floor(out$nrounds)
    out <- out[!duplicated(out), ]
  }
  out
}

xgb_custom$loop = function(grid) {
  loop <- plyr::ddply(grid, c("eta", "max_depth", "gamma",
                              "colsample_bytree", "min_child_weight", "subsample","alpha","lambda",
                              "tweedie_variance_power"),
                      function(x) c(nrounds = max(x$nrounds)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for (i in seq(along = loop$nrounds)) {
    index <- which(grid$max_depth == loop$max_depth[i] &
                     grid$eta == loop$eta[i] & grid$gamma == loop$gamma[i] &
                     grid$colsample_bytree == loop$colsample_bytree[i] &
                     grid$min_child_weight == loop$min_child_weight[i] &
                     grid$subsample == loop$subsample[i] &
                     grid$alpha == loop$alpha[i] &
                     grid$lambda == loop$lambda[i] &
                     grid$tweedie_variance_power == loop$tweedie_variance_power[i])
    trees <- grid[index, "nrounds"]
    submodels[[i]] <- data.frame(nrounds = trees[trees != loop$nrounds[i]])
  }
  list(loop = loop, submodels = submodels)
}

xgb_custom$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  theDots = list(...)

  # if (!inherits(x, "xgb.DMatrix"))
  #   x <- as.matrix(x)

  if (is.factor(y)) {
    if (length(lev) == 2) {
      y <- ifelse(y == lev[1], 1, 0)
      if (!inherits(x, "xgb.DMatrix")) x <- xgboost::xgb.DMatrix(x, label = y) else xgboost::setinfo(x, "label", y)
      if (!is.null(wts))
        xgboost::setinfo(x, "weight", wts)
      out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth,
                                     gamma = param$gamma, colsample_bytree = param$colsample_bytree,
                                     min_child_weight = param$min_child_weight, subsample = param$subsample,
                                     alpha = param$alpha, lambda = param$lambda),
                                data = x, nrounds = param$nrounds, objective = "binary:logistic",
                                ...)
    }
    else {
      y <- as.numeric(y) - 1
      if (!inherits(x, "xgb.DMatrix"))
        x <- xgboost::xgb.DMatrix(x, label = y)
      else xgboost::setinfo(x, "label", y)
      if (!is.null(wts))
        xgboost::setinfo(x, "weight", wts)
      out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth,
                                     gamma = param$gamma, colsample_bytree = param$colsample_bytree,
                                     min_child_weight = param$min_child_weight, subsample = param$subsample,
                                     alpha = param$alpha, lambda = param$lambda),
                                data = x, num_class = length(lev), nrounds = param$nrounds,
                                objective = "multi:softprob", ...)
    }
  }
  else {
    #browser()
    # Get Dots
    if ("loss" %in% names(theDots))
      loss = theDots[["loss"]] else loss = "reg:linear"
    # if ("use_weights_for_offset" %in% names(theDots))
    #   use_weights_for_offset = theDots[["use_weights_for_offset"]] else use_weights_for_offset = FALSE

    # # Rescale y for poisson (in case weights are not used for offset)
    # if (loss == "count:poisson" & !is.null(wts) & !use_weights_for_offset)
    #   y = y/wts

    # Set DMatrix label
    if (!inherits(x, "xgb.DMatrix")) x <- xgboost::xgb.DMatrix(x, label = y) else xgboost::setinfo(x, "label", y)

    # # Weight processing
    # if (!is.null(wts) & !use_weights_for_offset)
    #   xgboost::setinfo(x, "weight", wts)
    # if (loss == "count:poisson" & !is.null(wts) & use_weights_for_offset)
    #   xgboost::setinfo(x, "base_margin", log(wts))

    # # Reset theDots
    # theDots[["loss"]] = theDots[["use_weights_for_offset"]] = NULL

    if (!is.null(wts))
      xgboost::setinfo(x, "weight", wts)

    modArgs = list(list(eta = param$eta, max_depth = param$max_depth,
                        gamma = param$gamma, colsample_bytree = param$colsample_bytree,
                        min_child_weight = param$min_child_weight, subsample = param$subsample,
                        alpha = param$alpha, lambda = param$lambda,
                        tweedie_variance_power = param$tweedie_variance_power),
                   data = x, nrounds = param$nrounds, objective = loss)
    if (length(theDots) > 0) modArgs = c(modArgs, theDots)
    out = do.call(xgboost::xgb.train, modArgs)
  }
  out
}

xgb_custom$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (!inherits(newdata, "xgb.DMatrix")) {
    #newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata)
  }
  out <- predict(modelFit, newdata)
  if (modelFit$problemType == "Classification") {
    if (length(modelFit$obsLevels) == 2) {
      out <- ifelse(out >= 0.5, modelFit$obsLevels[1],
                    modelFit$obsLevels[2])
    }
    else {
      out <- matrix(out, ncol = length(modelFit$obsLevels),
                    byrow = TRUE)
      out <- modelFit$obsLevels[apply(out, 1, which.max)]
    }
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) +
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
      if (modelFit$problemType == "Classification") {
        if (length(modelFit$obsLevels) == 2) {
          tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1],
                             modelFit$obsLevels[2])
        }
        else {
          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels),
                             byrow = TRUE)
          tmp_pred <- modelFit$obsLevels[apply(tmp_pred,
                                               1, which.max)]
        }
      }
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

xgb_custom$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (!inherits(newdata, "xgb.DMatrix")) {
    #newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata)
  }
  if (!is.null(modelFit$param$objective) && modelFit$param$objective ==
      "binary:logitraw") {
    p <- predict(modelFit, newdata)
    out <- binomial()$linkinv(p)
  }
  else {
    out <- predict(modelFit, newdata)
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- modelFit$obsLevels
  }
  else {
    out <- matrix(out, ncol = length(modelFit$obsLevels),
                  byrow = TRUE)
    colnames(out) <- modelFit$obsLevels
  }
  out <- as.data.frame(out)
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) +
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
      if (length(modelFit$obsLevels) == 2) {
        tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      else {
        tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels),
                           byrow = TRUE)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      tmp_pred <- as.data.frame(tmp_pred)
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}




## Deep learning (from mlpKerasDecay)

deepLearn = list()

deepLearn$label = "Deep Learning"

deepLearn$library = "keras"

deepLearn$loop = NULL

deepLearn$type = c("Regression","Classification")

deepLearn$parameters =
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             size,character,Layer shape
             lambda,numeric,L2 L2 Regularization
             dropout,numeric,Dropout Rate
             batch_size,numeric,Batch Size
             lr,numeric,Learning Rate
             batch_normalization,boolean,Batch Normalization
             activation,character,Activation Function
             epochs,numeric,Epochs"
  )

deepLearn$grid = function(x, y, len = NULL, search = "grid") {
  afuncs <- c("sigmoid", "relu", "tanh")
  if (search == "grid") {
    out <- expand.grid(size = "10",
                       lambda = c(0, 10^seq(-1, -4, length = len - 1)),
                       batch_size = floor(nrow(x)/3),
                       lr = 2e-06,
                       dropout = 0,
                       batch_normalization = FALSE,
                       activation = "relu",
                       epochs = 10)
  }
  else {
    n <- nrow(x)
    out <- data.frame(size = "10",
                      lambda = 10^runif(len, min = -5, 1),
                      batch_size = floor(n * runif(len, min = 0.1)),
                      lr = runif(len),
                      dropout = 0,
                      batch_normalization = FALSE,
                      activation = sample(afuncs, size = len, replace = TRUE),
                      epochs = 10)
  }
  out
}

deepLearn$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  # browser()
  print(param)

  require(dplyr)
  K <- keras::backend()
  K$clear_session()
  if (!is.matrix(x))
    x <- as.matrix(x)
  model <- keras::keras_model_sequential()

  size = as.numeric(str_split(param$size,"-",simplify = TRUE)[1,])

  for (i in 1:length(size)) {
    model %>% keras::layer_dense(units = size[i], activation = as.character(param$activation),
                                 input_shape = ncol(x), kernel_initializer = keras::initializer_glorot_uniform(),
                                 kernel_regularizer = keras::regularizer_l2(param$lambda))
    if(param$batch_normalization) model %>% keras::layer_batch_normalization()
    if(param$dropout > 0) model %>% keras::layer_dropout(param$dropout)
  }
  if (is.factor(y)) {
    y <- class2ind(y)
    model %>% keras::layer_dense(units = length(lev), activation = "softmax",
                                 kernel_regularizer = keras::regularizer_l2(param$lambda)) %>%
      keras::compile(loss = "categorical_crossentropy",
                     optimizer = keras::optimizer_rmsprop(lr = param$lr),
                     metrics = "accuracy")
  }
  else {
    model %>% keras::layer_dense(units = 1, activation = "linear",
                                 kernel_regularizer = keras::regularizer_l2(param$lambda)) %>%
      compile(loss = "mean_squared_error",
              optimizer = keras::optimizer_rmsprop(lr = param$lr),
              metrics = "mean_squared_error")
  }
  model %>% keras::fit(x = x, y = y, batch_size = param$batch_size, epochs = param$epochs,
                       ...)
  if (last)
    model <- keras::serialize_model(model)
  list(object = model)
}

deepLearn$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (inherits(modelFit$object, "raw"))
    modelFit$object <- keras::unserialize_model(modelFit$object)
  if (!is.matrix(newdata))
    newdata <- as.matrix(newdata)
  out <- predict(modelFit$object, newdata)
  if (ncol(out) == 1) {
    out <- out[, 1]
  }
  else {
    out <- modelFit$obsLevels[apply(out, 1, which.max)]
  }
  out
}

deepLearn$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (inherits(modelFit$object, "raw"))
    modelFit$object <- keras::unserialize_model(modelFit$object)
  if (!is.matrix(newdata))
    newdata <- as.matrix(newdata)
  out <- predict(modelFit$object, newdata)
  colnames(out) <- modelFit$obsLevels
  as.data.frame(out)
}

deepLearn$sort = function(x) x[order(x$size, -x$lambda), ]

deepLearn$check = function(pkg) {
  testmod <- try(keras::keras_model_sequential(), silent = TRUE)
  if (inherits(testmod, "try-error"))
    stop("Could not start a sequential model. ", "`tensorflow` might not be installed. ",
         "See `?install_tensorflow`.", call. = FALSE)
  TRUE
}

