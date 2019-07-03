rm(list = ls())


# target types to be calculated
# -> REMOVE AND ADAPT AT APPROPRIATE LOCATIONS FOR A USE-CASE
TARGET_TYPES = c(
  "CLASS",
  "REGR",
  "MULTICLASS"
)


for (TARGET_TYPE in TARGET_TYPES) {
  cat(paste0("\n\n******************* Compare models with '", TARGET_TYPE, "' target. *******************\n\n"))
  tryCatch(
    {
      #TARGET_TYPE = "CLASS"

      #################################################################################################################-
      #|||| Initialize ||||----
      #################################################################################################################-

      # Load result from exploration
      load(paste0("data/",TARGET_TYPE,"_1_explore.rdata"))

      # Load libraries and functions
      source("code/0_init.R")

      # Initialize parallel processing
      closeAllConnections() #reset
      Sys.getenv("NUMBER_OF_PROCESSORS")
      cl = makeCluster(4)
      registerDoParallel(cl)
      # stopCluster(cl); closeAllConnections() #stop cluster

      # Set metric for peformance comparison
      metric = switch(TARGET_TYPE, "CLASS" = "AUC", "REGR"  = "spearman", "MULTICLASS" = "AUC")
      classProbs = switch(TARGET_TYPE, "CLASS" = TRUE, "REGR"  = FALSE, "MULTICLASS" = TRUE)




      #################################################################################################################-
      #|||| Test an algorithm (and determine parameter grid) ||||----
      #################################################################################################################-

      # Sample data ----------------------------------------------------------------------------------------------------

      if (TARGET_TYPE %in% c("CLASS","MULTICLASS")) {
        # Sample from all data (take all but n_maxpersample at most)
        #c(df.tune, b_sample, b_all) %<-%  (df %>% hmsPM::undersample_n(n_max_per_level = 5e3))

        # Undersample only training data
        c(df.tmp, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_max_per_level = 5e3))
        df.tune = bind_rows(df.tmp, df %>% filter(fold == "test"))

        summary(df.tune$target); b_sample; b_all
      }
      if (TARGET_TYPE == "REGR") {
        # Sample from all data
        df.tune = df %>% sample_n(min(nrow(.),5e3))
      }




      # Define some controls -------------------------------------------------------------------------------------------

      l.index = list(i = which(df.tune$fold == "train"))
      #set.seed(998)
      #l.index = list(i = sample(1:nrow(df.tune), floor(0.8*nrow(df.tune)))) #random sample

      # Index based test-set
      ctrl_idx = trainControl(method = "cv", number = 1, index = l.index,
                              returnResamp = "final", returnData = FALSE,
                              summaryFunction = hmsPM::performance_summary, classProbs = classProbs)

      # Dito but "fast" final fit: DO NOT USE in case of further application!!!
      ctrl_idx_fff = ctrl_idx
      ctrl_idx_fff$indexFinal = sample(1:nrow(df.tune), 100) #"Fast" final fit!!!

      # Dito but without parallel processing: Needed for DeepLearn or H2o
      ctrl_idx_nopar_fff = ctrl_idx_fff
      ctrl_idx_nopar_fff$allowParallel = FALSE

      # FFF as 5-fold cv
      ctrl_cv_fff = trainControl(method = "cv", number = 5,
                                 returnResamp = "final", returnData = FALSE,
                                 summaryFunction = hmsPM::performance_summary, classProbs = classProbs,
                                 indexFinal = sample(1:nrow(df.tune), 100)) #"Fast" final fit!!!


      # Fits -----------------------------------------------------------------------------------------------------------

      ## Overwritten Lasso / Elastic Net: Possible to use sparse matrix
      fit = train(x         = sparse.model.matrix(as.formula(formula_binned),
                                                  df.tune[c("target",features_binned)]),
                  y         = df.tune$target,
                  method    = glmnet_custom,
                  trControl = ctrl_idx_fff,
                  metric    = metric,
                  tuneGrid  = expand.grid(alpha = c(0,0.2,0.5,0.8,1),
                                          lambda = 2^(seq(5, -15, -2)))
                  #weights = exposure, family = "poisson"
      )
      #preProc = c("center","scale")) #no scaling needed due to dummy coding of all variables
      plot(fit)
      plot(fit, xlim = c(0,1))
      # -> keep alpha=1 to have a full Lasso



      ## Random Forest
      fit = train(x         = df.tune[features],
                  y         = df.tune$target,
                  #fit = train(x = model.matrix(as.formula(formula), df.tune[c("target",features)]), y = df.tune$target,
                  method    = "ranger",
                  trControl = ctrl_idx_fff,
                  metric    = metric,
                  tuneGrid  = expand.grid(mtry = seq(1,length(features),10),
                                          splitrule = switch(TARGET_TYPE,
                                                             "CLASS" = "gini", "REGR" = "variance",
                                                             "MULTICLASS" = "gini") ,
                                          min.node.size = c(1,5,10)),
                  num.trees = 500) #use the Dots (...) for explicitly specifiying randomForest parameter
      plot(fit)
      # -> keep around the recommended values: mtry(class) = sqrt(length(features), mtry(regr) = 0.3 * length(features))



      ## Boosted Trees

      # Default xgbTree: no parallel processing possible with DMatrix (and using sparse matrix
      # will result in nonsparse trafo)
      # fit = train(x         = xgb.DMatrix(sparse.model.matrix(as.formula(formula),
      #                                                         df.tune[c("target",features)])),
      #             y         = df.tune$target,
      #             method    = "xgbTree",
      #             trControl = ctrl_idx_nopar_fff, #no parallel for DMatrix
      #             metric    = metric,
      #             tuneGrid  = expand.grid(nrounds = seq(100,1100,200), eta = c(0.01),
      #                                     max_depth = c(3), min_child_weight = c(10),
      #                                     colsample_bytree = c(0.7), subsample = c(0.7),
      #                                     gamma = 0))
      # plot(fit)

      # Overwritten xgbTree: additional alpha and lambda parameter. Possible to use sparse matrix
      # and parallel processing
      fit = train(x         = sparse.model.matrix(as.formula(formula),
                                                  df.tune[c("target",features)]),
                  y         = df.tune$target,
                  method    = xgb_custom,
                  trControl = ctrl_idx_fff, #parallel for overwritten xgb
                  metric    = metric,
                  tuneGrid  = expand.grid(nrounds = seq(100,3100,200), eta = c(0.01),
                                          max_depth = c(3), min_child_weight = c(10),
                                          colsample_bytree = c(0.7), subsample = c(0.7),
                                          gamma = 0, alpha = 0, lambda = 1))
      plot(fit)
      hmsPM::plot_caret_result(fit = fit, metric = metric, x = "nrounds",
                                 color = "max_depth", linetype = "eta", shape = "min_child_weight",
                                 facet = "min_child_weight ~ subsample + colsample_bytree")



      if (TARGET_TYPE != "MULTICLASS") {
        # Lightgbm
        fit = train(x         = df.tune[features],
                    y         = df.tune$target,
        #fit = train(sparse.model.matrix(as.formula(formula), df.tune[c("target",features)]), df.tune$target,
                    method    = lgbm,
                    trControl = ctrl_idx_nopar_fff,
                    metric    = metric,
                    tuneGrid  = expand.grid(nrounds = seq(100,2100,200), learning_rate = c(0.01),
                                            num_leaves = 32, min_data_in_leaf = c(10),
                                            feature_fraction = c(0.7), bagging_fraction = c(0.7)),
                    #max_depth = 3, #use for small data
                    verbose   = -1)

        plot(fit)
        hmsPM::plot_caret_result(fit, metric = metric, x = "nrounds",
                                 color = "num_leaves", linetype = "learning_rate",
                                 shape = "min_data_in_leaf", facet = " ~ bagging_fraction + feature_fraction")
      }



      # DeepLearning
      fit = train(form      = as.formula(formula_notree),
                  data      = df.tune[c("target",features_notree)],
                  method    = deepLearn,
                  trControl = ctrl_idx_nopar_fff,
                  metric    = metric,
                  tuneGrid  = expand.grid(size = c("10","10-10"),
                                          lambda = c(0), dropout = 0.5,
                                          batch_size = c(100), lr = c(1e-3),
                                          batch_normalization = TRUE,
                                          activation = c("relu","elu"),
                                          epochs = 10),
                  preProc = c("center","scale"),
                  verbose = 0)
      plot(fit)




      #################################################################################################################-
      #|||| Evaluate generalization gap ||||----
      #################################################################################################################-

      # Sample data (usually undersample training data)
      df.gengap = df.tune

      # Tune grid to loop over
      tunepar = expand.grid(nrounds = seq(100,3100,500), eta = c(0.01),
                            max_depth = c(3,6), min_child_weight = c(10),
                            colsample_bytree = c(0.7), subsample = c(0.7),
                            gamma = c(0), alpha = c(0), lambda = c(1))

      # Calc generalization gap
      df.gengap_result = hmsPM::calc_gengap(df_data        = df.gengap,
                                            formula_string = formula,
                                            sparse         = TRUE,
                                            method         = xgb_custom,
                                            tune_grid      = tunepar,
                                            cluster        = cl)

      # Plot generalization gap
      (plots = hmsPM::plot_gengap(df_gengap = df.gengap_result,
                                  metric    = metric,
                                  x         = "nrounds",
                                  color     = "max_depth",
                                  shape     = "gamma",
                                  facet     = "min_child_weight ~ alpha + lambda"))
      grobs = marrangeGrob(plots, ncol = 2, nrow = 1)
      ggsave(paste0(plotloc,TARGET_TYPE,"_generalization_gap.pdf"),
             grobs,
             width = 12, height = 8)




      #################################################################################################################-
      #|||| Simulation: compare algorithms ||||----
      #################################################################################################################-

      # Basic data sampling
      df.sim = df.tune

      # Define methods to run in simulation
      l.xgb = list(method         = xgb_custom,
                   formula_string = formula,
                   sparse         = TRUE,
                   tune_grid      = expand.grid(nrounds = 2100, eta = c(0.01),
                                                max_depth = c(3), min_child_weight = c(10),
                                                colsample_bytree = c(0.7), subsample = c(0.7),
                                                gamma = 0, alpha = 0, lambda = 1))
      l.glmnet = list(method         = glmnet_custom,
                      formula_string = formula_binned,
                      sparse         = TRUE,
                      tune_grid      = expand.grid(alpha = 0,
                                                   lambda = 2^(seq(4, -10, -2))))

      # Simulate
      df.sim_result = hmsPM::calc_simulation(df_data           = df.sim,
                                             n_sim             = 3,
                                             metric            = metric,
                                             sample_frac_train = 0.8,
                                             sample_frac_test  = 0.5,
                                             l_methods         = list(xgb = l.xgb,
                                                                      glmnet = l.glmnet))
      (plot = hmsPM::plot_simulation(df_simulation = df.sim_result,
                                     metric        = metric))
      ggsave(paste0(plotloc,TARGET_TYPE,"_model_comparison.pdf"),
             plot,
             width = 12, height = 8)




      #################################################################################################################-
      #|||| Learning curve for winner algorithm ||||----
      #################################################################################################################-

      # Basic data sampling (do NOT undersamle as this is done in calc_learningcurve;
      # in fact you finally should not sample at all)
      df.lc = df %>% sample_n(min(nrow(.),5e3))

      # Tunegrid
      tunepar = expand.grid(nrounds = seq(100,2100,500), eta = c(0.01),
                            max_depth = c(3), min_child_weight = c(10),
                            colsample_bytree = c(0.7), subsample = c(0.7),
                            gamma = 0, alpha = 0, lambda = 1)

      # Calc lc
      df.lc_result = hmsPM::calc_learningcurve(df_data        = df.lc,
                                               formula_string = formula,
                                               sparse         = TRUE,
                                               method         = xgb_custom,
                                               tune_grid      = tunepar,
                                               chunks_pct     = c(seq(5,10,1), seq(20,100,10)),
                                               balanced       = TRUE,
                                               metric         = metric)
      (p = hmsPM::plot_learningcurve(df_lc  = df.lc_result,
                                           metric = metric))
      ggsave(paste0(plotloc,TARGET_TYPE,"_learningCurve.pdf"),
             p,
             width = 8, height = 6)

    },
    error = function (e) {
      message("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR in 2_modelcomparison.R for TARGET_TYPE '", TARGET_TYPE, "'")
    }
  )
}
