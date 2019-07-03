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
VERBOSE = TRUE


for (TARGET_TYPE in TARGET_TYPES) {
  cat(paste0("\n\n******************* Explore data with '", TARGET_TYPE, "' target. *******************\n\n"))
  tryCatch(
    { #TARGET_TYPE = "CLASS"

      #################################################################################################################-
      #|||| Initialize ||||----
      #################################################################################################################-

      # Adapt some default parameter differing per target types (NULL results in default value usage)
      cutoff_switch = switch(TARGET_TYPE, "CLASS" = 0.1, "REGR"  = 0.8, "MULTICLASS" = 0.8) #adapt
      ylim_switch = switch(TARGET_TYPE, "CLASS" = NULL, "REGR"  = c(0,250e3), "MULTICLASS" = NULL) #adapt REGR opt
      color_switch = switch(TARGET_TYPE, "CLASS" = twocol, "REGR"  = NULL, "MULTICLASS" = threecol)
      cutoff_varimp = 0.51




      #################################################################################################################-
      #|||| ETL ||||----
      #################################################################################################################-

      # Read data ------------------------------------------------------------------------------------------------------

      if (TARGET_TYPE == "CLASS") df.orig = suppressMessages(read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE))
      if (TARGET_TYPE %in% c("REGR","MULTICLASS")) {
        df.orig = suppressMessages(read_delim(paste0(dataloc,"AmesHousing.txt"), delim = "\t", col_names = TRUE,
                                              guess_max = 1e5))
        colnames(df.orig) = str_replace_all(colnames(df.orig), " ", "_")
      }

      skip = function() {
        # Quick check
        df.tmp = df.orig %>% mutate_if(is.character, as.factor)
        summary(df.tmp)
        if (TARGET_TYPE == "CLASS") table(df.tmp$survived) / nrow(df.tmp)
        if (TARGET_TYPE == "REGR") { hist(df.tmp$SalePrice, 30); hist(log(df.tmp$SalePrice), 30) }
      }

      # "Save" original data
      df = df.orig




      # Read metadata (Project specific) -------------------------------------------------------------------------------

      if (TARGET_TYPE == "CLASS")
        df.meta = suppressMessages(read_excel(paste0(dataloc,"datamodel_","titanic.xlsx"), skip = 1))
      if (TARGET_TYPE %in% c("REGR","MULTICLASS"))
        df.meta = suppressMessages(read_excel(paste0(dataloc,"datamodel_","AmesHousing.xlsx"), skip = 1))

      # Check
      setdiff(colnames(df), df.meta$variable)
      setdiff(df.meta %>% filter(category == "orig") %>% .$variable,
              colnames(df))

      # Filter on "ready" and "derive"(see below)
      df.meta_sub = df.meta %>% filter(status %in%  c("ready","derive"))




      # Feature engineering --------------------------------------------------------------------------------------------

      if (TARGET_TYPE == "CLASS") {
        df$deck = as.factor(str_sub(df$cabin, 1, 1)) #deck as first character of cabin
        df$familysize = df$sibsp + df$parch + 1 #add number of siblings and spouses to number of parents and children
        df %<>% group_by(ticket) %>% mutate(fare_pp = fare/n()) %>% ungroup() #fare per person
        summary(df[c("deck","familysize","fare_pp")])
      }
      if (TARGET_TYPE %in% c("REGR","MULTICLASS")) {
        # number of rooms, sqm_per_room, ...
      }

      # Check
      setdiff(df.meta %>% .$variable,
              colnames(df))




      # Define target and train/test-fold ------------------------------------------------------------------------------

      # Target
      if (TARGET_TYPE == "CLASS") {
        df = df %>% mutate(target = factor(ifelse(survived == 0, "N", "Y"),
                                           levels = c("N","Y")),
                           target_num = ifelse(target == "N", 0 ,1))
        summary(df$target_num)
      }
      if (TARGET_TYPE == "REGR") df$target = df$SalePrice
      if (TARGET_TYPE == "MULTICLASS") {
        df$target = as.factor(paste0("Cat_",
                                     as.numeric(cut(df.orig$SalePrice,
                                                    c(-Inf,quantile(df.orig$SalePrice, c(.3,.95)),Inf)))))
      }
      summary(df$target)

      # Train/Test fold: usually split by time
      df$fold = factor("train", levels = c("train", "test"))
      set.seed(123)
      df[sample(1:nrow(df), floor(0.3*nrow(df))),"fold"] = "test" #70/30 split
      summary(df$fold)

      # Define the id
      df$id = 1:nrow(df)





      #################################################################################################################-
      #|||| Metric variables: Explore and adapt ||||----
      #################################################################################################################-

      # Define metric covariates ---------------------------------------------------------------------------------------

      metr = df.meta_sub %>% filter(type == "metr") %>% .$variable
      df[metr] = map(df[metr],
                     ~ hmsPM::convert_scale(., "metr"))
      if (TARGET_TYPE %in% c("REGR","MULTICLASS")) {
        df[metr] = map(df[metr],
                       ~ na_if(., 0)) #zeros represent missings in Ames housing data
      }
      summary(df[metr])




      # Create nominal variables for all metric variables (for linear models)  -----------------------------------------

      metr_binned = paste0(metr,"_BINNED_")
      df[metr_binned] = map(df[metr], ~ {
        # Hint: Adapt sequence increment in case you have lots of data
        cut(., unique(quantile(., seq(0, 1, 0.1), na.rm = TRUE)), include.lowest = TRUE)
      })

      # Convert missings to own level ("(Missing)")
      df[metr_binned] = map(df[metr_binned],
                            ~ fct_explicit_na(., na_level = "(Missing)"))
      summary(df[metr_binned],11)

      # Remove binned variables with just 1 bin
      (onebin = metr_binned[map_lgl(metr_binned,
                                    ~ length(levels(df[[.]])) == 1)])




      # Missings + Outliers + Skewness ---------------------------------------------------------------------------------

      # Remove covariates with too many missings from metr
      misspct = map_dbl(df[metr],
                        ~ round(sum(is.na(.)/nrow(df)), 3)) #misssing percentage
      misspct[order(misspct, decreasing = TRUE)] #view in descending order
      (remove = names(misspct[misspct > 0.99])) #vars to remove
      metr = setdiff(metr, remove) #adapt metadata
      metr_binned = setdiff(metr_binned,
                            paste0(remove,"_BINNED_")) #keep "binned" version in sync

      # Check for outliers and skewness
      summary(df[metr])
      plots = map(metr, ~ hmsPM::plot_distr(x            = df[[.]],
                                            y            = df$target,
                                            feature_name = .,
                                            colors       = color_switch,
                                            ylim         = ylim_switch,
                                            verbose      = VERBOSE))
      grobs = suppressWarnings(suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2))) #need for first npn-blank page
      ggsave(paste0(plotloc, TARGET_TYPE, "_distr_metr.pdf"),
             grobs,
             width = 18, height = 12)

      # Winsorize
      df[metr] = map(df[metr],
                     ~ hmsPM::winsorize(x     = .,
                                        lower = 0.01,
                                        upper = 0.99)) #hint: one might plot again before deciding for log-trafo

      # Log-Transform
      if (TARGET_TYPE == "CLASS") tolog = c("fare")
      if (TARGET_TYPE %in% c("REGR","MULTICLASS")) tolog = c("Lot_Area")
      df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {
        if(min(., na.rm=TRUE) <= 0) log(. - min(., na.rm = TRUE) + 1) else log(.)})
      metr = map_chr(metr,
                     ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metadata (keep order)




      # Final variable information -------------------------------------------------------------------------------------

      # Univariate variable importance: with random imputation!
      (varimp_metr = filterVarImp(x       = map_df(df[metr], ~ hmsPM::impute(x = ., strategy = "random")),
                                  y       = df$target,
                                  nonpara = TRUE) %>%
         rowMeans() %>%
         .[order(., decreasing = TRUE)] %>%
         round(2))
      (varimp_metr_binned = filterVarImp(x       = df[metr_binned],
                                         y       = df$target,
                                         nonpara = TRUE) %>%
          rowMeans() %>%
          .[order(., decreasing = TRUE)] %>%
          round(2))

      # Plot
      plots1 = map(metr, ~ hmsPM::plot_distr(x            = df[[.]],
                                             y            = df$target,
                                             feature_name = .,
                                             varimps      = varimp_metr,
                                             colors       = color_switch,
                                             ylim         = ylim_switch,
                                             verbose      = VERBOSE))
      plots2 = map(metr_binned, ~ hmsPM::plot_distr(x            = df[[.]],
                                                    y            = df$target,
                                                    feature_name = .,
                                                    varimps      = varimp_metr_binned,
                                                    colors       = color_switch,
                                                    ylim         = ylim_switch,
                                                    verbose      = VERBOSE))
      plots = mapply(list, plots1, plots2, SIMPLIFY = TRUE) #zip plots
      grobs = suppressWarnings(suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2)))
      ggsave(paste0(plotloc, TARGET_TYPE, "_distr_metr_final.pdf"),
             grobs,
             width = 21, height = 14)




      # Removing variables ---------------------------------------------------------------------------------------------

      # Remove leakage features
      metr = setdiff(metr, c("xxx","xxx"))
      metr_binned = setdiff(metr_binned,
                            paste0(remove,"_BINNED_")) #keep "binned" version in sync

      # Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
      summary(df[metr])
      plot = hmsPM::plot_corr(df_plot = df[metr],
                              cutoff  = cutoff_switch,
                              verbose = VERBOSE)
      ggsave(paste0(plotloc, TARGET_TYPE, "_corr_metr.pdf"),
             plot,
             width = 8, height = 8)
      remove = c("xxx","xxx")
      metr = setdiff(metr, remove) #remove
      metr_binned = setdiff(metr_binned,
                            paste0(remove,"_BINNED_")) #keep "binned" version in sync




      # Time/fold depedency --------------------------------------------------------------------------------------------

      # Hint: In case of having a detailed date variable this can be used as regression target here as well!

      # Univariate variable importance (again ONLY for non-missing observations!)
      df$fold_test = factor(ifelse(df$fold == "test", "Y", "N"))
      (varimp_metr_fold = filterVarImp(x       = df[metr],
                                       y       = df$fold_test,
                                       nonpara = TRUE) %>%
          rowMeans() %>%
          .[order(., decreasing = TRUE)] %>%
          round(2))

      # Plot: only variables with with highest importance
      metr_toprint = names(varimp_metr_fold)[varimp_metr_fold >= cutoff_varimp]
      plots = map(metr_toprint, ~ hmsPM::plot_distr(x            = df[[.]],
                                                    y            = df$fold_test,
                                                    feature_name = .,
                                                    target_name  = "fold_test",
                                                    varimps      = varimp_metr_fold,
                                                    colors       = c("blue","red"),
                                                    verbose      = VERBOSE))
      grobs = marrangeGrob(plots, ncol = 4, nrow = 2)
      ggsave(paste0(plotloc, TARGET_TYPE, "_distr_metr_final_folddependency.pdf"),
             grobs,
             width = 18, height = 12)




      # Missing indicator and imputation (must be done at the end of all processing)------------------------------------

      # Create mising indicators
      (miss = metr[map_lgl(df[metr],
                           ~ any(is.na(.)))])
      df[paste0("MISS_",miss)] = map(df[miss],
                                     ~ as.factor(ifelse(is.na(.x), "miss", "no_miss")))
      summary(df[,paste0("MISS_",miss)])

      # Impute missings with randomly sampled value (or median, see below)
      df[miss] = map(df[miss],
                     ~ hmsPM::impute(., "random"))
      summary(df[metr])




      #################################################################################################################-
      #|||| Categorical variables: Explore and adapt ||||----
      #################################################################################################################-

      # Define categorical covariates ----------------------------------------------------------------------------------

      # Nominal variables
      nomi = df.meta_sub %>% filter(type == "nomi") %>% .$variable
      df[nomi] = map(df[nomi],
                     ~ hmsPM::convert_scale(., "nomi")) #map to nominal
      summary(df[nomi])

      # Ordinal variables
      ordi = df.meta_sub %>% filter(type == "ordi") %>% .$variable
      df[ordi] = map(df[ordi],
                     ~ hmsPM::convert_scale(., "ordi")) #map to ordinal
      summary(df[ordi])

      # Merge categorical variable (keep order)
      cate = c(intersect(df.meta_sub %>% filter(type %in% c("nomi","ordi")) %>% .$variable, c(nomi, ordi)),
               paste0("MISS_",miss))




      # Handling factor values -----------------------------------------------------------------------------------------

      # Get "too many members" columns and copy these for additional encoded features (for tree based models)
      topn_toomany = 10
      (levinfo = map_int(df[cate],
                         ~ length(levels(.))) %>%
          .[order(., decreasing = TRUE)]) #number of levels
      (toomany = names(levinfo)[which(levinfo > topn_toomany)])
      (toomany = setdiff(toomany, c("xxx","xxx"))) #set exception for important variables
      df[paste0(toomany,"_ENCODED")] = df[toomany]



      ## Convert categorical variables
      # Convert "standard" features: map missings to own level
      l.tmp_enc = map(df[setdiff(cate, toomany)], ~ hmsPM::cate_encoding(x          = .,
                                                                         method     = "self",
                                                                         enc_others = TRUE,
                                                                         enc_nas    = TRUE,
                                                                         na_value   = "(Missing)"))
      df[setdiff(cate, toomany)] = map(setdiff(cate, toomany), ~ hmsPM::encode(x        = df[[.]],
                                                                               encoding = l.tmp_enc[[.]]))
      summary(df[setdiff(cate, toomany)])

      # Convert toomany features: lump levels and map missings to own level
      l.tmp_enc = map(df[toomany], ~ hmsPM::cate_encoding(x            = .,
                                                          method       = "self_topn",
                                                          n_top_levels = topn_toomany,
                                                          other_value  = "_OTHER_",
                                                          na_value     = "(Missing)"))
      df[toomany] = map(toomany, ~ hmsPM::encode(x        = df[[.]],
                                                 encoding = l.tmp_enc[[.]]))
      summary(df[toomany], 20)

      # Create encoded features (for tree based models), i.e. numeric representation
      df[paste0(toomany,"_ENCODED")] = map(df[paste0(toomany,"_ENCODED")], ~
                                             hmsPM::encode(x        = .,
                                                           encoding = hmsPM::cate_encoding(x            = .,
                                                                                           method       = "count_others",
                                                                                           n_top_levels = topn_toomany,
                                                                                           na_value     = 0)))
      summary(df[paste0(toomany,"_ENCODED")], 20)



      # Univariate variable importance
      (varimp_cate = filterVarImp(x       = df[cate],
                                  y       = df$target,
                                  nonpara = TRUE) %>%
          rowMeans() %>%
          .[order(., decreasing = TRUE)] %>%
          round(2))

      # Check
      plots = map(cate, ~ hmsPM::plot_distr(x            = df[[.]],
                                            y            = df$target,
                                            feature_name = .,
                                            varimps      = varimp_cate,
                                            colors       = color_switch,
                                            ylim         = ylim_switch,
                                            verbose      = VERBOSE))
      grobs = suppressWarnings(marrangeGrob(plots, ncol = 3, nrow = 2))
      ggsave(paste0(plotloc,TARGET_TYPE,"_distr_cate.pdf"),
             grobs,
             width = 18, height = 12)




      # Removing variables ---------------------------------------------------------------------------------------------

      # Remove Self-features
      if (TARGET_TYPE == "CLASS") {
        cate = setdiff(cate, "boat")
        toomany = setdiff(toomany, "boat")
      }
      if (TARGET_TYPE %in% c("REGR","MULTICLASS"))
        cate = setdiff(cate, c("xxx","xxx"))

      # Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
      plot = hmsPM::plot_corr(df_plot        = df[setdiff(cate, paste0("MISS_",miss))],
                              feature_scale  = "nomi",
                              cutoff         = cutoff_switch,
                              verbose        = FALSE)
      ggsave(paste0(plotloc,TARGET_TYPE,"_corr_cate.pdf"),
             plot,
             width = 9, height = 9)
      if (TARGET_TYPE %in% c("REGR","MULTICLASS")) {
        plot = hmsPM::plot_corr(df_plot       = df[ paste0("MISS_",miss)],
                                feature_scale = "nomi",
                                cutoff        = 0,
                                verbose       = VERBOSE)
        ggsave(paste0(plotloc,TARGET_TYPE,"_corr_cate_MISS.pdf"),
               plot,
               width = 9, height = 9)
        cate = setdiff(cate, c("MISS_Garage_Yr_Blt"))
      }



      # Time/fold depedency --------------------------------------------------------------------------------------------

      # Hint: In case of having a detailed date variable this can be used as regression target here as well!

      # Univariate variable importance
      (varimp_cate_fold = filterVarImp(x       = df[cate],
                                       y       = df$fold_test,
                                       nonpara = TRUE) %>%
         rowMeans() %>%
         .[order(., decreasing = TRUE)] %>%
         round(2))

      # Plot (Hint: one might want to filter just on variable importance with highest importance)
      cate_toprint = names(varimp_cate_fold)[varimp_cate_fold >= cutoff_varimp]
      plots = map(cate_toprint, ~ hmsPM::plot_distr(x            = df[[.]],
                                                    y            = df$fold_test,
                                                    feature_name = .,
                                                    target_name  = "fold_test",
                                                    varimps      = varimp_cate_fold,
                                                    colors       = c("blue","red"),
                                                    verbose      = VERBOSE))
      grobs = marrangeGrob(plots, ncol = 4, nrow = 3)
      ggsave(paste0(plotloc,TARGET_TYPE,"_distr_cate_folddependency.pdf"),
             grobs,
             width = 18, height = 12)




      #################################################################################################################-
      #|||| Prepare final data ||||----
      #################################################################################################################-

      # Define final features ------------------------------------------------------------------------------------------

      features_notree = c(metr, cate)
      formula_notree = paste("target", "~ -1 +", paste(features_notree, collapse = " + "))
      features = c(metr, cate, paste0(toomany,"_ENCODED"))
      formula = paste("target", "~ -1 +", paste(features, collapse = " + "))
      features_binned = c(setdiff(metr_binned, onebin), setdiff(cate, paste0("MISS_",miss))) #do not need indicators
      formula_binned = paste("target", "~ -1 +", paste(features_binned, collapse = " + "))

      # Check
      setdiff(features_notree, colnames(df))
      setdiff(features, colnames(df))
      setdiff(features_binned, colnames(df))




      # Save image -----------------------------------------------------------------------------------------------------
      rm(df.orig, grobs, plots, plots1, plots2)
      save.image(paste0(dataloc,TARGET_TYPE,"_1_explore.rdata"))
      #load(paste0(dataloc,TARGET_TYPE,"_1_explore.rdata"))

    },
    error = function (e) {
      message("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR in 1_explore for TARGET_TYPE '", TARGET_TYPE, "'\n\n")
    }
  )
}
