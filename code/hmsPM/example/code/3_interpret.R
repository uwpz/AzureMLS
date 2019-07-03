rm(list = ls())

# target types to be calculated
# -> REMOVE AND ADAPT AT APPROPRIATE LOCATIONS FOR A USE-CASE
TARGET_TYPES = c(
  "CLASS",
  "REGR",
  "MULTICLASS"
)
VERBOSE = TRUE


for (TARGET_TYPE in TARGET_TYPES) {
  cat(paste0("\n\n******************* Interpret model with '", TARGET_TYPE, "' target. *******************\n\n"))
  tryCatch(
    { #TARGET_TYPE = "CLASS"

      #################################################################################################################-
      #|||| Initialize: settings are used in all subsequent steps ||||----
      #################################################################################################################-

      # Load result from exploration
      load(paste0("data/",TARGET_TYPE,"_1_explore.rdata"))

      # Load libraries and functions
      source("code/0_init.R")

      # Adapt some default parameter different for target types -> probably also different for a new use-case
      pred_type = switch(TARGET_TYPE, "CLASS" = "prob", "REGR" = "raw", "MULTICLASS" = "prob") #do not change this one
      b_all = b_sample = NULL #do not change this one (as it is default in regression case)
      color_switch = switch(TARGET_TYPE, "CLASS" = twocol, "REGR" = c("blue","red"), "MULTICLASS" = threecol) #adapt
      if(TARGET_TYPE == "MULTICLASS") color_explain = c("blue","red") else color_explain = color_switch #do not change
      ylim_perf = switch(TARGET_TYPE, "CLASS" = c(0,1), "REGR"  = c(0,5e5), "MULTICLASS" = c(0,1)) #adapt REGR slot
      ylim_res = switch(TARGET_TYPE, "CLASS" = c(-1,1), "REGR"  = c(-5e4,5e4), "MULTICLASS" = c(0,1)) #adapt REGR slot
      ylim_pd = switch(TARGET_TYPE, "CLASS" = c(0.2,0.7), "REGR"  = c(1.5e5,2.5e5), "MULTICLASS" = c(0,1)) #adapt
      ylim_xgbexpl = switch(TARGET_TYPE, "CLASS" = c(0,1), "REGR"  = c(0,5e5), "MULTICLASS" = c(0,1)) #adapt
      topn_switch = switch(TARGET_TYPE, "CLASS" = 8, "REGR" = 20, "MULTICLASS" = 20) #adapt
      id_name = switch(TARGET_TYPE, "CLASS" = "id", "REGR"  = "PID", "MULTICLASS" = "PID") #adapt (id column name)

      n.sample = 3e3 #sampling rate


      # Initialize parallel processing
      closeAllConnections() #reset
      Sys.getenv("NUMBER_OF_PROCESSORS")
      cl = makeCluster(4)
      registerDoParallel(cl)
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
                                              hmsPM::undersample_n(n_max_per_level = n.sample))
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
      df.test = df %>% filter(fold == "test") #%>% sample_n(min(nrow(.), n.sample)) #!!!: Do not sample in final run



      ## Folds for crossvalidation
      set.seed(123)
      c(l.train, l.test) %<-% hmsPM::create_folds(nrow(df), df$fold, method = "byfold")

      # Check
      map(l.train, ~ summary(df$fold[.]))
      map(l.test, ~ summary(df$fold[.]))




      #################################################################################################################-
      #|||| Performance ||||----
      #################################################################################################################-

      #---- Do the full fit and predict on test data -------------------------------------------------------------------

      # Fit
      tmp = Sys.time()
      m.train = sparse.model.matrix(as.formula(formula), data = df.train[c("target",features)])
      fit = train(x         = m.train,
                  y         = df.train$target,
                  trControl = trainControl(method = "none", returnData = FALSE),
                  method    = xgb_custom,
                  tuneGrid  = tunepar)
      Sys.time() - tmp

      # Predict
      yhat_test = predict(fit,
                          sparse.model.matrix(as.formula(formula), df.test[c("target",features)]),
                          type = pred_type) %>%
        hmsPM::scale_predictions(b_sample, b_all) #rescale
      # # Scoring in chunks in parallel in case of high memory consumption of xgboost
      # l.split = split(df.test[c("target",features)], (1:nrow(df.test)) %/% 1e3)
      # yhat_test = foreach(df.split = l.split, .combine = bind_rows, .packages = c("Matrix","xgboost")) %dopar% {
      #   predict(fit, sparse.model.matrix(as.formula(formula), df.split), type = pred_type)
      # } %>% hmsPM::scale_predictions(b_sample, b_all) #rescale

      # Plot performance
      hmsPM::performance_summary(data.frame(y = df.test$target, yhat = yhat_test))
      plots =  hmsPM::plot_all_performances(y        = df.test$target,
                                            yhat     = yhat_test,
                                            colors   = color_switch,
                                            ylim     = ylim_perf,
                                            oneclass = TRUE)
      grobs = suppressMessages(marrangeGrob(plots, ncol = length(plots)/2, nrow = 2, top = NULL))
      ggsave(paste0(plotloc,TARGET_TYPE,"_performance.pdf"),
             grobs,
             w = 18, h = 12)

      # Training performance (to estimate generalization gap)
      yhat_train = predict(fit,
                           m.train,
                           type = pred_type) # no rescaling here!
      hmsPM::performance_summary(data.frame(y = df.train$target, yhat = yhat_train))




      #---- Check performance for crossvalidated fits -----------------------------------------------------------------

      ## Fit
      l.cv = foreach(i = 1:length(l.train), .combine = c,
                     .packages = c("caret","ROCR","xgboost","Matrix","dplyr","purrr","zeallot","hmsPM")
      ) %dopar% {

        # Get cv data (ATTENTION: basically do the same as above for the full run)
        if (TARGET_TYPE %in% c("CLASS","MULTICLASS")) {
          c(df.train_cv, b_sample_cv, b_all_cv) %<-% (df[l.train[[i]],] %>%
                                                        hmsPM::undersample_n(n_max_per_level = n.sample))
        }
        if (TARGET_TYPE == "REGR") {
          df.train_cv = df[l.train[[i]],] %>% sample_n(size = min(nrow(.), n.sample))
          b_sample_cv = b_all_cv = NULL
        }
        df.test_cv = df[l.test[[i]],] #%>% sample_n(min(nrow(.), n.sample)) #ATTENTION: Do not sample in final run!!!

        # Fit and calc performance
        fit_cv = train(x         = sparse.model.matrix(as.formula(formula), df.train_cv),
                       y         = df.train_cv$target,
                       trControl = trainControl(method = "none", returnData = FALSE),
                       method    = xgb_custom,
                       tuneGrid  = tunepar)
        yhat_cv = predict(fit_cv,
                          sparse.model.matrix(as.formula(formula), df.test_cv),
                          type = pred_type) %>%
          hmsPM::scale_predictions(b_sample_cv, b_all_cv)
        perf_cv = hmsPM::performance_summary(data.frame(yhat = yhat_cv, y = df.test_cv$target))
        return(setNames(list(fit_cv, perf_cv, b_sample_cv, b_all_cv),
                        c(paste0("fit_",i), paste0("perf_",i), paste0("b_sample_",i), paste0("b_all_",i))))}

      # Performance
      map_dbl(l.cv[grep("perf",names(l.cv))],
              ~ .[metric])

      # Copy for later usage
      l.fits = l.cv[grep("fit",names(l.cv))]
      l.b_sample = l.cv[grep("b_sample",names(l.cv))]
      l.b_all = l.cv[grep("b_all",names(l.cv))]




      #--- Most important variables (importance_cum < 95) model fit ----------------------------------------------------

      # Variable importance (on train data!)
      df.varimp_train = hmsPM::calc_varimp_by_permutation(df_data        = df.train,
                                                          fit            = fit,
                                                          formula_string = formula, sparse = TRUE,
                                                          metric         = metric,
                                                          b_sample = b_sample, b_all = b_all)

      # Top features (importances sum up to 95% of whole sum)
      features_top = df.varimp_train %>% filter(importance_cum < 95) %>% .$feature
      formula_top = paste("target", "~ -1 + ", paste(features_top, collapse = " + "))

      # Fit again only on features_top
      fit_top = train(x         = sparse.model.matrix(as.formula(formula_top), df.train[c("target",features_top)]),
                      y         = df.train$target,
                      trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE),
                      method    = xgb_custom,
                      tuneGrid  = tunepar)

      # Plot performance
      yhat_top = predict(fit_top,
                         sparse.model.matrix(as.formula(formula_top), df.test[c("target",features_top)]),
                         type = pred_type) %>%
        hmsPM::scale_predictions(b_sample, b_all)
      plots = hmsPM::plot_all_performances(y      = df.test$target,
                                           yhat   = yhat_top,
                                           colors = color_switch,
                                           ylim   = ylim_perf)
      grobs = suppressMessages(marrangeGrob(plots, ncol = length(plots)/2, nrow = 2,
                                            top = paste0("Top ",length(features_top)," Features Fit")))
      ggsave(paste0(plotloc,TARGET_TYPE,"_performance_top","_","features.pdf"),
             grobs,
             w = 18, h = 12)




      #################################################################################################################-
      #|||| Diagnosis ||||----
      #################################################################################################################-

      #---- Check residuals --------------------------------------------------------------------------------------------

      ## Residuals
      if (TARGET_TYPE == "CLASS") {
        df.test$yhat = yhat_test[,2]
        df.test$residual = df.test$target_num - df.test$yhat
      }
      if (TARGET_TYPE == "MULTICLASS")  {
        # Dynamic reference member per obs, i.e. the true label gets a "1" and the residual is 1-predicted_prob ...
        # ... in case of binary classifiaction this is equal to the absolute residual
        df.test$yhat = rowSums(yhat_test * model.matrix(~ -1 + df.test$target,
                                                        data.frame(df.test$target)))
        df.test$residual = 1 - df.test$yhat
      }
      if (TARGET_TYPE == "REGR") {
        df.test$yhat = yhat_test
        df.test$residual = df.test$target - df.test$yhat
      }
      summary(df.test$residual)
      df.test$abs_residual = abs(df.test$residual)

      # For non-regr tasks one might want to plot it for each target level (df.test %>% filter(target == "level"))
      plots = map(features, ~ hmsPM::plot_distr(x            = df.test[[.]],
                                                y            = df.test$residual,
                                                feature_name = .,
                                                nbins        = 20,
                                                ylim         = ylim_res,
                                                verbose      = VERBOSE))
      grobs = suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 3))
      ggsave(paste0(plotloc, TARGET_TYPE, "_diagnosis_residual.pdf"),
             grobs,
             width = 18, height = 12)



      ## Absolute residuals
      if (TARGET_TYPE %in% c("CLASS","REGR")) {
        summary(df.test$abs_residual)
        plots = map(features, ~ hmsPM::plot_distr(x            = df.test[[.]],
                                                y            = df.test$abs_residual,
                                                feature_name = .,
                                                nbins        = 20,
                                                ylim         = c(0,ylim_res[2]),
                                                verbose = VERBOSE))
        grobs = suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 3))
        ggsave(paste0(plotloc, TARGET_TYPE, "_diagnosis_absolute_residual.pdf"),
               grobs,
               width = 18, height = 12)
      }




      #---- Explain bad predictions ------------------------------------------------------------------------------------

      ## Get explainer data
      df.explainer = hmsPM::calc_xgb_explainer(fit            = fit,
                                               formula_string = formula, sparse = TRUE,
                                               df_train       = df.train)



      ## Get n_worst most false predicted ids
      n_worst = 10
      df.test_explain = df.test %>%
        arrange(desc(residual)) %>%
        top_n(n_worst, abs_residual)
      yhat_explain = predict(fit,
                             sparse.model.matrix(as.formula(formula), df.test_explain[c("target",features)]),
                             type = pred_type) %>%
        hmsPM::scale_predictions(b_sample, b_all)



      ## Get explanations
      df.explanations = hmsPM::calc_xgb_weights(fit            = fit,
                                                formula_string = formula, sparse = TRUE,
                                                df_test        = df.test_explain,
                                                yhat           = yhat_explain,
                                                df_explainer   = df.explainer,
                                                b_sample = b_sample, b_all = b_all,
                                                id_name = id_name,
                                                top_n = 5)



      ## Plot
      plots = plot_all_xgb_explanations(df_explanations = df.explanations,
                                        id_name         = id_name,
                                        ylim            = ylim_xgbexpl,
                                        colors          = alpha(color_explain, 0.5),
                                        verbose         = VERBOSE)
      grobs = marrangeGrob(plots, ncol = 2, nrow = 2)
      ggsave(paste0(plotloc,TARGET_TYPE,"_fails.pdf"),
             grobs,
             w = 18, h = 12)





      #################################################################################################################-
      #|||| Variable Importance ||||----
      #################################################################################################################-

      #--- Default Variable Importance: uses gain sum of all trees -----------------------------------------------------

      # Default plot
      plot(varImp(fit))



      #--- Variable Importance by permuation argument ------------------------------------------------------------------

      ## Importance for "total" fit (on test data!)
      df.varimp = hmsPM::calc_varimp_by_permutation(df_data        = df.test,
                                                    fit            = fit,
                                                    formula_string = formula, sparse = TRUE,
                                                    metric         = metric,
                                                    b_sample = b_sample, b_all = b_all)

      # Visual check how many variables needed
      ggplot(df.varimp) +
        geom_bar(aes(x = reorder(feature, importance),
                     y = importance),
                 stat = "identity") +
        coord_flip()
      topn_features = df.varimp[1:topn_switch, "feature"]

      # Add other information (e.g. special coloring): color variable is needed -> fill with "dummy" if no coloring
      df.varimp %<>% mutate(color = cut(x = importance,
                                        breaks = c(-1,10,50,101),
                                        labels = c("low","middle","high")))



      ## Crossvalidate Importance: ONLY for topn_vars
      # Get cv values
      df.varimp_cv = c()
      for (i in 1:length(l.test)) {
        df.varimp_cv %<>%
          bind_rows(hmsPM::calc_varimp_by_permutation(df_data        = df[l.test[[i]],],
                                                      fit            = l.fits[[i]],
                                                      formula_string = formula, sparse = TRUE,
                                                      metric         = metric,
                                                      b_sample = l.b_sample[[i]], b_all = l.b_all[[i]]) %>%
                      mutate(run = i))
      }

      # Plot
      plots = list(hmsPM::plot_varimp(df_varimp      = df.varimp,
                                      n_top_features = topn_switch,
                                      df_varimp_cv   = df.varimp_cv),
                   hmsPM::plot_varimp_cum(df_varimp      = df.varimp,
                                          n_top_features = topn_switch))
      grobs = marrangeGrob(plots, ncol = 2, nrow = 1, top = NULL)
      ggsave(paste0(plotloc,TARGET_TYPE,"_variable_importance.pdf"),
             grobs,
             w = 12, h = 8)




      #--- Compare variable importance for train and test (hints to variables prone to overfitting) --------------------

      plots = map(c("importance","importance_sumnormed"), ~ {
        df.tmp = df.varimp %>%
          select_at(c("feature",.x)) %>%
          mutate(type = "test") %>%
          bind_rows(df.varimp_train %>%
                      select_("feature",.x) %>%
                      mutate(type = "train")) %>%
          filter(feature %in% topn_features)
        ggplot(df.tmp, aes_string("feature", .x)) +
          geom_bar(aes(fill = type),
                   position = "dodge",
                   stat = "identity") +
          scale_fill_discrete(limits = c("train","test")) +
          coord_flip() +
          labs(title = .x)
      })
      grobs = marrangeGrob(plots, ncol = 2, nrow = 1, top = NULL)
      ggsave(paste0(plotloc,TARGET_TYPE,"_variable_importance_comparison.pdf"),
             plot = grobs,
             w = 12, h = 8)




      #################################################################################################################-
      #|||| Partial Dependance ||||----
      #################################################################################################################-

      ## Partial depdendance for "total" fit
      df.partialdep = hmsPM::calc_partial_dependencies(df_data        = df.test,
                                                       df_data_orig   = df.test,
                                                       fit            = fit,
                                                       formula_string = formula,
                                                       sparse         = TRUE,
                                                       feature_names  = topn_features,
                                                       b_sample = b_sample, b_all = b_all)


      # Visual check whether all fits
      plots = hmsPM::plot_all_partial_dependencies(df_partialdep = df.partialdep,
                                                   df_data       = df.test,
                                                   feature_names = topn_features,
                                                   ylim          = ylim_pd,
                                                   colors        = color_switch,
                                                   verbose       = VERBOSE)
      grobs = marrangeGrob(plots, ncol = 4, nrow = 2)
      ggsave(paste0(plotloc,TARGET_TYPE,"_partial_dependence.pdf"),
             grobs,
             w = 18, h = 12)



      ## Partial dependance cv models
      # Get crossvalidated values
      df.partialdep_cv = c()
      for (i in 1:length(l.test)) {
        df.partialdep_cv %<>%
          bind_rows(hmsPM::calc_partial_dependencies(df_data        = df[l.test[[i]],],
                                                     df_data_orig   = df.test,
                                                     fit            = l.fits[[i]],
                                                     formula_string = formula, sparse = TRUE,
                                                     feature_names  = topn_features,
                                                     b_sample = l.b_sample[[i]], b_all = l.b_all[[i]]) %>%
                      mutate(run = i))
      }



      ## Plot
      plots = suppressWarnings(suppressMessages(
        hmsPM::plot_all_partial_dependencies(df_partialdep    = df.partialdep,
                                             df_data          = df.test,
                                             feature_names    = topn_features,
                                             df_partialdep_cv = df.partialdep_cv,
                                             ylim             = ylim_pd,
                                             colors           = color_switch,
                                             add_cv_lines     = TRUE,
                                             add_cv_conf_int   = TRUE,
                                             verbose          = VERBOSE)))
      grobs = marrangeGrob(plots, ncol = 4, nrow = 2)
      ggsave(paste0(plotloc,TARGET_TYPE,"_partial_dependence.pdf"),
             grobs,
             w = 18, h = 12)




      #################################################################################################################-
      #|||| xgboost Explainer ||||----
      #################################################################################################################-

      ## Subset test data (explanations are only calculated for this subset)
      i.top = order(df.test$yhat, decreasing = TRUE)[1:5]
      i.bottom = order(df.test$yhat)[1:5]
      i.random = sample(1:length(df.test$yhat), 5)
      i.explain = sample(unique(c(i.top, i.bottom, i.random)))
      df.test_explain = df.test[i.explain,]
      yhat_explain = predict(fit,
                             sparse.model.matrix(as.formula(formula), df.test_explain[c("target",features)]),
                             type = pred_type) %>%
        hmsPM::scale_predictions(b_sample, b_all)



      ## Get explanations
      df.weights = hmsPM::calc_xgb_weights(fit            = fit,
                                           formula_string = formula, sparse = TRUE,
                                           df_test        = df.test_explain,
                                           yhat           = yhat_explain,
                                           df_explainer   = df.explainer,
                                           b_sample = b_sample, b_all = b_all,
                                           id_name        = id_name,
                                           top_n          = 5)



      ## Plot
      plots = hmsPM::plot_all_xgb_explanations(df_explanations = df.weights,
                                               id_name         = id_name,
                                               ylim            = ylim_xgbexpl,
                                               colors          = alpha(color_explain,0.5),
                                               verbose         = VERBOSE)
      ggsave(paste0(plotloc,TARGET_TYPE,"_explanations.pdf"),
             marrangeGrob(plots, ncol = 2, nrow = 2),
             w = 18, h = 12)




      #################################################################################################################-
      #|||| Individual Dependencies ||||----
      #################################################################################################################-

      set.seed(123)
      ids = sample(unique(df.test[[id_name]]), 10)
      df.indidep = hmsPM::calc_individual_dependencies(df_data        = df.test %>% filter(.data[[id_name]] %in% ids),
                                                       df_data_orig   = df %>% filter(fold == "train"),
                                                       fit            = fit,
                                                       formula_string = formula, sparse = TRUE,
                                                       feature_names  = topn_features,
                                                       b_sample = b_sample, b_all = b_all,
                                                       id_name        = id_name)
      id_select = ids[3]
      plots = hmsPM::plot_all_individual_dependencies(df_individualdep     = df.indidep %>%
                                                                                filter(.data[[id_name]] == id_select),
                                                      df_data              = df %>% filter(fold == "train"),
                                                      feature_names        = topn_features,
                                                      individual_flag_name = "orig_data_flag",
                                                      ylim                 = ylim_perf,
                                                      colors               = color_switch,
                                                      verbose              = VERBOSE)
      grobs = marrangeGrob(plots, ncol = 4, nrow = 2,
                           top = paste0("Individual dependency for id ", id_select))
      ggsave(paste0(plotloc,TARGET_TYPE,"_individual_dependence.pdf"),
             grobs,
             w = 18, h = 12)

      #rm(grobs, plots)
      #save.image(paste0(TARGET_TYPE,"_3_interpret.rdata"))
      #load(paste0(TARGET_TYPE,"_3_interpret.rdata"))


    },
    error = function (e) {
      message("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR in 3_interpret.R for TARGET_TYPE '", TARGET_TYPE, "'")
    }
  )
}
