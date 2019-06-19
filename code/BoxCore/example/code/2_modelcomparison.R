rm(list = ls())

# Load libraries and functions
source("code/0_init.R")


# target types to be calculated
# -> REMOVE AND ADAPT AT APPROPRIATE LOCATIONS FOR A USE-CASE
TARGET_TYPES = c(
  "CLASS",
  "REGR",
  "MULTICLASS"
)


for (TARGET_TYPE in TARGET_TYPES) {
  tryCatch(
    {

      ##################################################################################################################-
      #|||| Initialize ||||----
      ##################################################################################################################-

      # Load result from exploration
      load(paste0(dataloc,TARGET_TYPE,"_1_explore.rdata"))

      # Initialize parallel processing
      closeAllConnections() #reset
      Sys.getenv("NUMBER_OF_PROCESSORS")
      cl = makeCluster(4)
      registerDoParallel(cl)
      customlibpaths = .libPaths()
      clusterExport(cl, "customlibpaths")
      clusterEvalQ(cl, .libPaths(customlibpaths))
      clusterEvalQ(cl, library(BoxCore))
      # stopCluster(cl); closeAllConnections() #stop cluster



      # Set metric for peformance comparison
      metric = switch(TARGET_TYPE, "CLASS" = "AUC", "REGR"  = "spearman", "MULTICLASS" = "AUC")
      classProbs = switch(TARGET_TYPE, "CLASS" = TRUE, "REGR"  = FALSE, "MULTICLASS" = TRUE)




      ##################################################################################################################-
      #|||| Test an algorithm (and determine parameter grid) ||||----
      ##################################################################################################################-

      # Sample data ----------------------------------------------------------------------------------------------------

      if (TARGET_TYPE %in% c("CLASS","MULTICLASS")) {
        # Sample from all data (take all but n_maxpersample at most)
        c(df.tune, b_sample, b_all) %<-%  (df %>% BoxCore::undersample_n(n_max_per_level = 5e3))
        # # Undersample only training data
        # c(df.tmp, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_max_per_level = 5e3))
        # df.tune = bind_rows(df.tmp, df %>% filter(fold == "test"))

        summary(df.tune$target); b_sample; b_all
      }
      if (TARGET_TYPE == "REGR") {
        # Sample from all data
        df.tune = df %>% sample_n(min(nrow(.),5e3))
      }




      # Define some controls -------------------------------------------------------------------------------------------

      l.index = list(i = which(df.tune$fold == "train"))
      #set.seed(998)
      #l.index = list(i = sample(1:nrow(df.tune), floor(0.8*nrow(df.tune))))

      # Index based test-set
      ctrl_idx = trainControl(method = "cv", number = 1, index = l.index,
                              returnResamp = "final", returnData = FALSE,
                              summaryFunction = BoxCore::performance_summary, classProbs = classProbs)

      # Dito but "fast" final fit: DO NOT USE in case of further application!!!
      ctrl_idx_fff = trainControl(method = "cv", number = 1, index = l.index,
                                  returnResamp = "final", returnData = FALSE,
                                  summaryFunction = BoxCore::performance_summary, classProbs = classProbs,
                                  indexFinal = sample(1:nrow(df.tune), 100)) #"Fast" final fit!!!

      # Dito but without parallel processing: Needed for "xgbTree" or DeepLearn
      ctrl_idx_nopar_fff = trainControl(method = "cv", number = 1, index = l.index,
                                        returnResamp = "final", returnData = FALSE,
                                        allowParallel = FALSE, #no parallel e.g. for "xgbTree" on big data or with DMatrix
                                        summaryFunction = BoxCore::performance_summary, classProbs = classProbs,
                                        indexFinal = sample(1:nrow(df.tune), 100)) #"Fast" final fit!!!

      # Dito as 5-fold cv
      ctrl_cv_fff = trainControl(method = "cv", number = 5,
                                 returnResamp = "final", returnData = FALSE,
                                 summaryFunction = BoxCore::performance_summary, classProbs = classProbs,
                                 indexFinal = sample(1:nrow(df.tune), 100)) #"Fast" final fit!!!


      # Fits -----------------------------------------------------------------------------------------------------------

      ## Lasso / Elastic Net
      fit = train(sparse.model.matrix(as.formula(formula_binned), df.tune[c("target",features_binned)]), df.tune$target,
                  method = glmnet_custom,
                  trControl = ctrl_idx_fff,
                  metric = metric,
                  tuneGrid = expand.grid(alpha = c(0,0.2,0.5,0.8,1),
                                         lambda = 2^(seq(5, -15, -2)))
                  #weights = exposure, family = "poisson"
      )
      #preProc = c("center","scale")) #no scaling needed due to dummy coding of all variables
      plot(fit)
      plot(fit, xlim = c(0,1))
      # -> keep alpha=1 to have a full Lasso



      ## Random Forest
      fit = train(df.tune[features], df.tune$target,
                  #fit = train(model.matrix(as.formula(formula), df.tune[c("target",features)]), df.tune$target,
                  method = "ranger",
                  trControl = ctrl_idx_fff,
                  metric = metric,
                  tuneGrid = expand.grid(mtry = seq(1,length(features),10),
                                         splitrule = switch(TARGET_TYPE, "CLASS" = "gini", "REGR" = "variance", "MULTICLASS" = "gini") ,
                                         min.node.size = c(1,5,10)),
                  num.trees = 500) #use the Dots (...) for explicitly specifiying randomForest parameter
      plot(fit)
      # -> keep around the recommended values: mtry(class) = sqrt(length(features), mtry(regr) = 0.3 * length(features))


      if (TARGET_TYPE != "MULTICLASS") {
        # fit = train(as.data.frame(df.tune[features]), df.tune$target,
        #             method = ms_forest,
        #             trControl = ctrl_idx_fff,
        #             metric = metric,
        #             tuneGrid = expand.grid(numTrees = c(100,300,500),
        #                                    splitFraction = c(0.1,0.3,0.5)),
        #             verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
        # plot(fit)
        # # -> keep around the recommended values: mtry = floor(sqrt(length(features))) or splitFraction = 0.3
      }



      ## Boosted Trees

      # Default xgbTree: no parallel processing possible with DMatrix (and using sparse matrix will result in nonsparse trafo)
      # fit = train(xgb.DMatrix(sparse.model.matrix(as.formula(formula), df.tune[c("target",features)])), df.tune$target,
      #             method = "xgbTree",
      #             trControl = ctrl_idx_nopar_fff, #no parallel for DMatrix
      #             metric = metric,
      #             tuneGrid = expand.grid(nrounds = seq(100,1100,200), eta = c(0.01),
      #                                    max_depth = c(3), min_child_weight = c(10),
      #                                    colsample_bytree = c(0.7), subsample = c(0.7),
      #                                    gamma = 0))
      # plot(fit)

      # Overwritten xgbTree: additional alpha and lambda parameter. Possible to use sparse matrix and parallel processing
      fit = train(sparse.model.matrix(as.formula(formula), df.tune[c("target",features)]), df.tune$target,
                  method = xgb,
                  trControl = ctrl_idx_fff, #parallel for overwritten xgb
                  metric = metric,
                  tuneGrid = expand.grid(nrounds = seq(100,3100,200), eta = c(0.01),
                                         max_depth = c(3), min_child_weight = c(10),
                                         colsample_bytree = c(0.7), subsample = c(0.7),
                                         gamma = 0, alpha = 0, lambda = 1))
      plot(fit)
      BoxCore::plot_caret_result(fit, metric = metric, x = "nrounds",
                                 color = "max_depth", linetype = "eta", shape = "min_child_weight",
                                 facet = "min_child_weight ~ subsample + colsample_bytree")

      if (TARGET_TYPE == "CLASS") {
        # Overwritten xgbTree: support for tweedie + poisson
        exposure = runif(nrow(df.tune), min = 1/12, max = 1)
        formula_poisson = paste("target_num", "~ -1 +", paste(features, collapse = " + "))
        set.seed(123)
        fit = train(sparse.model.matrix(as.formula(formula_poisson), df.tune[c("target_num",features)]), df.tune$target_num/exposure,
                    method = xgb_custom,
                    trControl = ctrl_idx_nopar_fff, #parallel for overwritten xgb
                    metric = "spearman",
                    tuneGrid = expand.grid(nrounds = seq(100,3100,200), eta = c(0.01),
                                           max_depth = c(3), min_child_weight = c(10),
                                           colsample_bytree = c(0.7), subsample = c(0.7),
                                           gamma = 0, alpha = 0, lambda = 1,
                                           tweedie_variance_power = c(1.5)),
                    weights = exposure, loss = "count:poisson"
                    #loss = "reg:tweedie"
        )
        plot(fit)
      }




      if (TARGET_TYPE != "MULTICLASS") {
        # # MicrosoftML: numTrees is not a sequential parameter (like in xgbTree) !!!
        # fit = train(df.tune[features], df.tune$target,
        #             trControl = ctrl_idx_fff, metric = metric,
        #             method = ms_boosttree,
        #             tuneGrid = expand.grid(numTrees = seq(100,3100,500), learningRate = c(0.01,0.05),
        #                                    numLeaves = c(10,20), minSplit = c(10),
        #                                    featureFraction = c(0.7), exampleFraction = c(0.7)),
        #             verbose = 0)
        # plot(fit)

        # # Lightgbm
        # fit = train(sparse.model.matrix(as.formula(formula), df.tune[c("target",features)]), df.tune$target,
        #             method = lgbm,
        #             trControl = ctrl_idx_fff,
        #             metric = metric,
        #             tuneGrid = expand.grid(nrounds = seq(100,3100,200), learning_rate = c(0.01, 0.05),
        #                                    num_leaves = c(16,64), min_data_in_leaf = c(10),
        #                                    feature_fraction = c(0.7), bagging_fraction = c(0.7)),
        #             max_depth = 3, #use for small data
        #             verbose = 0)
        # plot(fit)
        # plot_caret_result(fit, metric = "AUC", x = "numTrees",
        #                   color = "numLeaves", linetype = "learningRate",
        #                   shape = "minSplit", facet = "minSplit ~ exampleFraction + featureFraction") #lightgbm
      }



      ## DeepLearning
      # fit = train(as.formula(formula_notree), df.tune[c("target",features_notree)],
      #             method = deepLearn,
      #             trControl = ctrl_idx_nopar_fff,
      #             metric = metric,
      #             tuneGrid = expand.grid(size = c("10","10-10","10-10-10"),
      #                                    lambda = c(0,2^-1), dropout = 0.5,
      #                                    batch_size = c(100), lr = c(1e-3),
      #                                    batch_normalization = TRUE,
      #                                    activation = c("relu","elu"),
      #                                    epochs = 10),
      #             preProc = c("center","scale"),
      #             verbose = 0)
      # plot(fit)




      ##################################################################################################################-
      #|||| Evaluate generalization gap ||||----
      ##################################################################################################################-

      # Sample data (usually undersample training data)
      df.gengap = df.tune

      # Tune grid to loop over
      tunepar = expand.grid(nrounds = seq(100,3100,200), eta = c(0.01),
                            max_depth = c(3,6), min_child_weight = c(10),
                            colsample_bytree = c(0.7), subsample = c(0.7),
                            gamma = c(0,10), alpha = c(0,1), lambda = c(1,20))

      # Calc generalization gap
      df.gengap_result = BoxCore::calc_gengap(df.gengap, formula, sparse = TRUE,
                                              method = xgb, tune_grid = tunepar,
                                              cluster = cl)

      # Plot generalization gap
      (plots = BoxCore::plot_gengap(df.gengap_result, metric = metric, x = "nrounds",
                                    color = "max_depth", shape = "gamma", facet = "min_child_weight ~ alpha + lambda"))
      ggsave(paste0(plotloc,TARGET_TYPE,"_generalization_gap.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1),
             width = 12, height = 8)




      ##################################################################################################################-
      #|||| Simulation: compare algorithms ||||----
      ##################################################################################################################-

      # Basic data sampling
      df.sim = df.tune

      # Define methods to run in simulation
      l.xgb = list(method = xgb, formula_string = formula, sparse = TRUE,
                   tune_grid = expand.grid(nrounds = 2100, eta = c(0.01),
                                           max_depth = c(3), min_child_weight = c(10),
                                           colsample_bytree = c(0.7), subsample = c(0.7),
                                           gamma = 0, alpha = 0, lambda = 1))
      l.glmnet = list(method = "glmnet", formula_string = formula_binned, sparse = TRUE,
                      tune_grid = expand.grid(alpha = 0,
                                              lambda = 2^(seq(4, -10, -2))))

      # Simulate
      df.sim_result = BoxCore::calc_simulation(df.sim, n_sim = 3, metric = metric,
                                               sample_frac_train = 0.5, sample_frac_test = 0.5,
                                               l_methods = list(xgb = l.xgb, glmnet = l.glmnet))
      (plot = BoxCore::plot_simulation(df.sim_result, metric = metric))
      ggsave(paste0(plotloc,TARGET_TYPE,"_model_comparison.pdf"), plot, width = 12, height = 8)




      ##################################################################################################################-
      #|||| Learning curve for winner algorithm ||||----
      ##################################################################################################################-

      # Basic data sampling (do NOT undersamle as this is done in calc_learningcurve;
      # in fact you finally should not sample at all)
      df.lc = df %>% sample_n(min(nrow(.),5e3))

      # Tunegrid
      tunepar = expand.grid(nrounds = seq(100,2100,500), eta = c(0.01),
                            max_depth = c(3), min_child_weight = c(10),
                            colsample_bytree = c(0.7), subsample = c(0.7),
                            gamma = 0, alpha = 0, lambda = 1)

      # Calc lc
      df.lc_result = BoxCore::calc_learningcurve(df.lc, formula_string = formula, sparse = TRUE,
                                                 method = xgb, tune_grid = tunepar,
                                                 chunks_pct = c(seq(5,10,1), seq(20,100,10)),
                                                 balanced = TRUE, metric = metric)
      (p = BoxCore::plot_learningcurve(df.lc_result, metric = metric))
      ggsave(paste0(plotloc,TARGET_TYPE,"_learningCurve.pdf"), p, width = 8, height = 6)

    },
    error = function (e) {
      message("ERROR in 2_modelcomparison.R for TARGET_TYPE '", TARGET_TYPE, "'")
    }
  )
}
