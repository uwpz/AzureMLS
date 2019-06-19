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
VERBOSE = FALSE


for (TARGET_TYPE in TARGET_TYPES) {
  tryCatch(
    {

      ##################################################################################################################-
      #|||| Initialize: settings are used in all subsequent steps ||||----
      ##################################################################################################################-

      # Load libraries and functions
      source("code/0_init.R")

      # Load result from exploration
      load(paste0(dataloc,TARGET_TYPE,"_1_explore.rdata"))

      # Adapt some default parameter different for target types -> probably also different for a new use-case
      pred_type = switch(TARGET_TYPE, "CLASS" = "prob", "REGR" = "raw", "MULTICLASS" = "prob") #do not change this one
      b_all = b_sample = NULL #do not change this one (as it is default in regression case)
      color_switch = switch(TARGET_TYPE, "CLASS" = twocol, "REGR" = c("blue","red"), "MULTICLASS" = threecol)
      ylim_perf = switch(TARGET_TYPE, "CLASS" = c(0,1), "REGR"  = c(0,5e5), "MULTICLASS" = c(0,1)) #adapt Regr slot
      ylim_res = switch(TARGET_TYPE, "CLASS" = c(-1,1), "REGR"  = c(-5e4,5e4), "MULTICLASS" = c(0,1)) #adapt Regr slot
      ylim_pd = switch(TARGET_TYPE, "CLASS" = c(0.2,0.7), "REGR"  = c(1.5e5,2.5e5), "MULTICLASS" = c(0,1)) #need to adapt
      ylim_xgbexpl = ylim_perf
      topn_switch = switch(TARGET_TYPE, "CLASS" = 8, "REGR" = 20, "MULTICLASS" = 20) #adapt
      id_name = switch(TARGET_TYPE, "CLASS" = "id", "REGR"  = "PID", "MULTICLASS" = "PID") #adapt to name of id columns in your data

      n.sample = 1e3 #sampling rate


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




      ## Tuning parameter to use (for xgb)
      if (TARGET_TYPE == "CLASS") {
        tunepar = expand.grid(nrounds = 2100, eta = c(0.01),
                              max_depth = c(3), min_child_weight = c(10),
                              colsample_bytree = c(0.7), subsample = c(0.7),
                              gamma = 0, alpha = 0, lambda = 1)
      }
      if (TARGET_TYPE %in% c("REGR","MULTICLASS")) {
        tunepar = expand.grid(nrounds = 2100, eta = c(0.01),
                              max_depth = c(3), min_child_weight = c(10),
                              colsample_bytree = c(0.7), subsample = c(0.7),
                              gamma = 0, alpha = 0, lambda = 1)
      }




      # Sample data ----------------------------------------------------------------------------------------------------

      ## Training data
      set.seed(999)
      if (TARGET_TYPE %in% c("CLASS","MULTICLASS")) {
        # Just take data from train fold (take all but n_maxpersample at most)
        summary(df[df$fold == "train", "target"])
        c(df.train, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>%
                                              BoxCore::undersample_n(n_max_per_level = n.sample))
        summary(df.train$target); b_sample; b_all

        # Set metric for peformance comparison
        metric = "AUC"
      }
      if (TARGET_TYPE == "REGR") {
        # Just take data from train fold
        df.train = df %>% filter(fold == "train") %>% sample_n(min(nrow(.), n.sample))

        # Set metric for peformance comparison
        metric = "spearman"
      }



      ## Test data
      set.seed(999)
      df.test = df %>% filter(fold == "test") %>% sample_n(min(nrow(.), n.sample)) #ATTENTION: Do not sample in final run!!!



      ## Folds for crossvalidation
      set.seed(123)
      c(l.train, l.test) %<-% create_folds(nrow(df), df$fold, method = "byfold")

      # Check
      map(l.train, ~ summary(df$fold[.]))
      map(l.test, ~ summary(df$fold[.]))




      ##################################################################################################################-
      #|||| Performance ||||----
      ##################################################################################################################-

      #---- Do the full fit and predict on test data -------------------------------------------------------------------

      # Fit
      tmp = Sys.time()
      m.train = sparse.model.matrix(as.formula(formula), data = df.train[c("target",features)])
      fit = train(m.train, df.train$target,
                  trControl = trainControl(method = "none", returnData = FALSE),
                  method = xgb,
                  tuneGrid = tunepar)
      Sys.time() - tmp

      # Predict
      yhat_test = predict(fit, sparse.model.matrix(as.formula(formula), df.test[c("target",features)]), type = pred_type) %>%
        BoxCore::scale_predictions(b_sample, b_all) #rescale
      # # Scoring in chunks in parallel in case of high memory consumption of xgboost
      # l.split = split(df.test[c("target",features)], (1:nrow(df.test)) %/% 1e3)
      # yhat_test = foreach(df.split = l.split, .combine = bind_rows, .packages = c("Matrix","xgboost")) %dopar% {
      #   predict(fit, sparse.model.matrix(as.formula(formula), df.split), type = pred_type)
      # } %>% BoxCore::scale_predictions(b_sample, b_all) #rescale

      # Plot performance
      BoxCore::performance_summary(data.frame(y = df.test$target, yhat = yhat_test))
      plots = BoxCore::plot_all_performances(y = df.test$target, yhat = yhat_test, colors = color_switch, ylim = ylim_perf, verbose = VERBOSE)
      ggsave(paste0(plotloc,TARGET_TYPE,"_performance.pdf"),
             suppressMessages(marrangeGrob(plots, ncol = length(plots)/2, nrow = 2, top = NULL)),
             w = 18, h = 12)

      # Training performance (to estimate generalization gap)
      yhat_train = predict(fit, m.train, type = pred_type) # no rescaling here!
      BoxCore::performance_summary(data.frame(y = df.train$target, yhat = yhat_train))




      #---- Check performance for crossvalidated fits ---------------------------------------------------------------------

      ## Fit
      l.cv = foreach(i = 1:length(l.train), .combine = c,
                     .packages = c("caret","ROCR","xgboost","Matrix","dplyr","purrr","zeallot","BoxCore")
      ) %dopar% {

        # Get cv data (ATTENTION: basically do the same as above for the full run)
        if (TARGET_TYPE %in% c("CLASS","MULTICLASS")) {
          c(df.train_cv, b_sample_cv, b_all_cv) %<-% (df[l.train[[i]],] %>%
                                                        BoxCore::undersample_n(n_max_per_level = n.sample))
        }
        if (TARGET_TYPE == "REGR") {
          df.train_cv = df[l.train[[i]],] %>% sample_n(min(nrow(.), n.sample))
          b_sample_cv = b_all_cv = NULL
        }
        df.test_cv = df[l.test[[i]],] %>% sample_n(min(nrow(.), n.sample)) #ATTENTION: Do not sample in final run!!!

        # Fit and calc performance
        fit_cv = train(sparse.model.matrix(as.formula(formula), df.train_cv), df.train_cv$target,
                       trControl = trainControl(method = "none", returnData = FALSE),
                       method = xgb,
                       tuneGrid = tunepar)
        yhat_cv = predict(fit_cv, sparse.model.matrix(as.formula(formula), df.test_cv), type = pred_type) %>%
          BoxCore::scale_predictions(b_sample_cv, b_all_cv)
        perf_cv = BoxCore::performance_summary(data.frame(yhat = yhat_cv, y = df.test_cv$target))
        return(setNames(list(fit_cv, perf_cv, b_sample_cv, b_all_cv),
                        c(paste0("fit_",i), paste0("perf_",i), paste0("b_sample_",i), paste0("b_all_",i))))}

      # Performance
      map_dbl(l.cv[grep("perf",names(l.cv))], ~ .[metric])

      # Copy for later usage
      l.fits = l.cv[grep("fit",names(l.cv))]
      l.b_sample = l.cv[grep("b_sample",names(l.cv))]
      l.b_all = l.cv[grep("b_all",names(l.cv))]




      #--- Most important variables (importance_cum < 95) model fit ----------------------------------------------------

      # Variable importance (on train data!)
      df.varimp_train = BoxCore::calc_varimp_by_permutation(df.train, fit, formula, sparse = TRUE,
                                                            metric = metric, b_sample = b_sample, b_all = b_all)

      # Top features (importances sum up to 95% of whole sum)
      features_top = df.varimp_train %>% filter(importance_cum < 95) %>% .$feature
      formula_top = paste("target", "~ -1 + ", paste(features_top, collapse = " + "))

      # Fit again only on features_top
      fit_top = train(sparse.model.matrix(as.formula(formula_top), df.train[c("target",features_top)]), df.train$target,
                      trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE),
                      method = xgb,
                      tuneGrid = tunepar)

      # Plot performance
      yhat_top = predict(fit_top,
                         sparse.model.matrix(as.formula(formula_top), df.test[c("target",features_top)]),
                         type = pred_type) %>%
        BoxCore::scale_predictions(b_sample, b_all)
      plots = BoxCore::plot_all_performances(y = df.test$target, yhat = yhat_top, colors = color_switch, ylim = ylim_perf, verbose = VERBOSE)
      ggsave(paste0(plotloc,TARGET_TYPE,"_performance_top",length(features_top),"features.pdf"),
             suppressMessages(marrangeGrob(plots, ncol = length(plots)/2, nrow = 2, top = NULL)),
             w = 18, h = 12)




      ##################################################################################################################-
      #|||| Diagnosis ||||----
      ##################################################################################################################-

      #---- Check residuals --------------------------------------------------------------------------------------------

      ## Residuals
      if (TARGET_TYPE == "CLASS") {
        df.test$yhat = yhat_test[,2]
        df.test$residual = df.test$target_num - df.test$yhat
      }
      if (TARGET_TYPE == "MULTICLASS")  {
        # Dynamic reference member per obs, i.e. the true label gets a "1" and the residual is 1-predicted_prob ...
        # ... in case of binary classifiaction this is equal to the absolute residual
        df.test$yhat = rowSums(yhat_test * model.matrix(~ -1 + df.test$target, data.frame(df.test$target)))
        df.test$residual = 1 - df.test$yhat
      }
      if (TARGET_TYPE == "REGR") {
        df.test$yhat = yhat_test
        df.test$residual = df.test$target - df.test$yhat
      }
      summary(df.test$residual)
      df.test$abs_residual = abs(df.test$residual)

      # For non-regr tasks one might want to plot the following for each target level (df.test %>% filter(target == "level"))
      plots = c(map(metr, ~ BoxCore::plot_distr(df.test[[.]], df.test$residual, .,  nbins = 20, ylim = ylim_res, verbose = VERBOSE)),
                map(cate, ~ BoxCore::plot_distr(df.test[[.]], df.test$residual, .,  nbins = 20, ylim = ylim_res, verbose = VERBOSE)))
      ggsave(paste0(plotloc, TARGET_TYPE, "_diagnosis_residual.pdf"),
             suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 3)), width = 18, height = 12)



      ## Absolute residuals
      if (TARGET_TYPE %in% c("CLASS","REGR")) {
        summary(df.test$abs_residual)
        plots = c(map(metr, ~ BoxCore::plot_distr(df.test[[.]], df.test$abs_residual, .,  nbins = 20,
                                                  ylim = c(0,ylim_res[2], verbose = VERBOSE))),
                  map(cate, ~ BoxCore::plot_distr(df.test[[.]], df.test$abs_residual, .,  nbins = 20,
                                                  ylim = c(0,ylim_res[2], verbose = VERBOSE))))
        ggsave(paste0(plotloc, TARGET_TYPE, "_diagnosis_absolute_residual.pdf"),
               suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 3)), width = 18, height = 12)
      }




      #---- Explain bad predictions ------------------------------------------------------------------------------------

      if (TARGET_TYPE != "MULTICLASS") {

        ## Get explainer data
        df.explainer = BoxCore::calc_xgb_explainer(fit, formula, sparse = TRUE, df_train = df.train)



        ## Get n_worst most false predicted ids
        n_worst = 30
        df.test_explain = df.test %>% arrange(desc(residual)) %>% top_n(n_worst, abs_residual)
        yhat_explain = predict(fit, sparse.model.matrix(as.formula(formula), df.test_explain[c("target",features)]), type = pred_type) %>%
          BoxCore::scale_predictions(b_sample, b_all)



        ## Get explanations
        df.weights = BoxCore::calc_xgb_weights(fit, formula, sparse = TRUE, df_test = df.test_explain, yhat = yhat_explain,
                                               df_explainer = df.explainer, b_sample = b_sample, b_all = b_all,
                                               id_name = id_name)

        # # For Multiclass target take only explanations for target (and not other levels)
        # if (TARGET_TYPE == "MULTICLASS") {
        #   df.weights = df.weights %>% inner_join(df.test_explain %>% select_("target", id_name, "yhat"))
        # }



        ## Plot
        plots = BoxCore::plot_all_xgb_explanations(df.weights, id_name = id_name, top_n = topn_switch,
                                                   target_type = detect_target_type(df.test_explain$target),
                                                   ylim = ylim_xgbexpl, colors = alpha(color_switch,0.5), verbose = VERBOSE)
        ggsave(paste0(plotloc,TARGET_TYPE,"_fails.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2),
               w = 18, h = 12)

      }




      ##################################################################################################################-
      #|||| Variable Importance ||||----
      ##################################################################################################################-

      #--- Default Variable Importance: uses gain sum of all trees -----------------------------------------------------

      # Default plot
      plot(varImp(fit))



      #--- Variable Importance by permuation argument ------------------------------------------------------------------

      ## Importance for "total" fit (on test data!)
      df.varimp = BoxCore::calc_varimp_by_permutation(df.test, fit, formula, sparse = TRUE,
                                                      metric = metric, b_sample = b_sample, b_all = b_all)

      # Visual check how many variables needed
      ggplot(df.varimp) +
        geom_bar(aes(x = reorder(feature, importance), y = importance), stat = "identity") +
        coord_flip()
      topn_features = df.varimp[1:topn_switch, "feature"]

      # Add other information (e.g. special coloring): color variable is needed -> fill with "dummy" if it should be ommited
      df.varimp %<>% mutate(color = cut(importance, c(-1,10,50,101), labels = c("low","middle","high")))



      ## Crossvalidate Importance: ONLY for topn_vars
      # Get cv values
      df.varimp_cv = c()
      for (i in 1:length(l.test)) {
        df.varimp_cv %<>%
          bind_rows(BoxCore::calc_varimp_by_permutation(df[l.test[[i]],], l.fits[[i]], formula, sparse = TRUE,
                                                        metric = metric,
                                                        b_sample = l.b_sample[[i]], b_all = l.b_all[[i]]) %>%
                      mutate(run = i))
      }

      # Plot
      plots = list(BoxCore::plot_varimp(df.varimp, topn_switch, df_varimp_cv = df.varimp_cv, verbose = VERBOSE),
                   BoxCore::plot_varimp_cum(df.varimp, topn_switch, verbose = VERBOSE))
      ggsave(paste0(plotloc,TARGET_TYPE,"_variable_importance.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1), w = 12, h = 8)




      #--- Compare variable importance for train and test (hints to variables prone to overfitting) --------------------

      plots = map(c("importance","importance_sumnormed"), ~ {
        df.tmp = df.varimp %>% select_("feature",.x) %>% mutate(type = "test") %>%
          bind_rows(df.varimp_train %>% select_("feature",.x) %>% mutate(type = "train")) %>%
          filter(feature %in% topn_features)
        ggplot(df.tmp, aes_string("feature", .x)) +
          geom_bar(aes(fill = type), position = "dodge", stat = "identity") +
          scale_fill_discrete(limits = c("train","test")) +
          coord_flip() +
          labs(title = .x)
      })
      ggsave(paste0(plotloc,TARGET_TYPE,"_variable_importance_comparison.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1),
             w = 12, h = 8)




      ##################################################################################################################-
      #|||| Partial Dependance ||||----
      ##################################################################################################################-

      ## Partial depdendance for "total" fit
      df.partialdep = BoxCore::calc_partial_dependencies(df.test, df.test, fit, formula, sparse = TRUE,
                                                         feature_names = topn_features,
                                                         b_sample = b_sample, b_all = b_all)
      # Visual check whether all fits
      plots = BoxCore::plot_all_partial_dependencies(df.partialdep, df.test, topn_features,
                                                     ylim = ylim_pd, colors = color_switch, verbose = VERBOSE)
      ggsave(paste0(plotloc,TARGET_TYPE,"_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2),
             w = 18, h = 12)



      ## Partial dependance cv models
      # Get crossvalidated values
      df.partialdep_cv = c()
      for (i in 1:length(l.test)) {
        df.partialdep_cv %<>%
          bind_rows(BoxCore::calc_partial_dependencies(df[l.test[[i]],], df.test, l.fits[[i]], formula, sparse = TRUE,
                                                       feature_names = topn_features,
                                                       b_sample = l.b_sample[[i]], b_all = l.b_all[[i]]) %>%
                      mutate(run = i))
      }



      ## Plot
      plots = suppressWarnings(suppressMessages(
        BoxCore::plot_all_partial_dependencies(df.partialdep, df.test, topn_features,
                                               df_partialdep_cv = df.partialdep_cv,
                                               ylim = ylim_pd, colors = color_switch,
                                               add_cv_lines = TRUE, add_cv_conf_int = TRUE, verbose = VERBOSE)))
      ggsave(paste0(plotloc,TARGET_TYPE,"_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2),
             w = 18, h = 12)




      ##################################################################################################################-
      #|||| xgboost Explainer ||||----
      ##################################################################################################################-

      if (TARGET_TYPE != "MULTICLASS") {

        # Subset test data (explanations are only calculated for this subset)
        i.top = order(df.test$yhat, decreasing = TRUE)[1:20]
        i.bottom = order(df.test$yhat)[1:20]
        i.random = sample(1:length(df.test$yhat), 20)
        i.explain = sample(unique(c(i.top, i.bottom, i.random)))
        df.test_explain = df.test[i.explain,]
        yhat_explain = predict(fit, sparse.model.matrix(as.formula(formula), df.test_explain[c("target",features)]), type = pred_type) %>%
          BoxCore::scale_predictions(b_sample, b_all)

        # Get explanations
        df.weights = BoxCore::calc_xgb_weights(fit, formula, sparse = TRUE, df_test = df.test_explain, yhat = yhat_explain,
                                               df_explainer = df.explainer, b_sample = b_sample, b_all = b_all,
                                               id_name = id_name)

        # # For Multiclass target take only explanations for target (and not other levels)
        # if (TARGET_TYPE == "MULTICLASS") {
        #   df.weights = df.weights %>% inner_join(df.test_explain %>% select_("target", id_name, "yhat"))
        # }

        # Plot
        plots = BoxCore::plot_all_xgb_explanations(df.weights, id_name = id_name, top_n = topn_switch,
                                                   target_type = detect_target_type(df.test_explain$target),
                                                   ylim = ylim_xgbexpl, colors = alpha(color_switch,0.5), verbose = VERBOSE)
        ggsave(paste0(plotloc,TARGET_TYPE,"_explanations.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2),
               w = 18, h = 12)

      }


      rm(plots)
      #save.image(paste0(TARGET_TYPE,"_3_interpret.rdata"))
      #load(paste0(TARGET_TYPE,"_3_interpret.rdata"))

    },
    error = function (e) {
      message("ERROR in 3_interpret.R for TARGET_TYPE '", TARGET_TYPE, "'")
    }
  )
}
