
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
      #|||| ETL ||||----
      ##################################################################################################################-

      # Read data and transform ----------------------------------------------------------------------------------------

      # ABT
      df = suppressMessages(read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE))

      # Feature Engineering
      df$deck = as.factor(str_sub(df$cabin, 1, 1)) #deck as first character of cabin
      df$familysize = df$sibsp + df$parch + 1 #add number of siblings and spouses to the number of parents and children
      df %<>% group_by(ticket) %>% mutate(fare_pp = fare/n()) %>% ungroup() #fare per person (one ticket might comprise several persons)

      # Read column metadata
      df.meta = suppressMessages(read_excel(paste0(dataloc,"datamodel_","titanic.xlsx"), skip = 1) %>%
        filter(status %in% c("ready","derive")))

      # Define target
      df$target = factor(ifelse(df$survived == 0, "N", "Y"), levels = c("N","Y"))
      summary(df$target)




      # Adapt categorical variables ------------------------------------------------------------------------------------

      # Define nominal and ordinal features
      nomi = df.meta %>% filter(type == "nomi") %>% .$variable
      ordi = df.meta %>% filter(type == "ordi") %>% .$variable

      # Make them factors
      df[nomi] = map(df[nomi], ~ BoxCore::convert_scale(., "nomi")) #map to nominal
      df[ordi] = map(df[ordi], ~ BoxCore::convert_scale(., "ordi")) #map to ordinal

      # Merge nomi and ordi
      cate = union(nomi, ordi)
      summary(df[cate],20)

      # Derive "toomanys" and copy them
      topn_toomany = 10
      levinfo = map_int(df[cate], ~ length(levels(.)))
      data.frame(n = levinfo[order(levinfo, decreasing = TRUE)])
      (toomany = names(levinfo)[which(levinfo > topn_toomany)])
      toomany = setdiff(toomany, c("xxx")) #Set exception for important variables
      df[paste0(toomany,"_ENCODED")] = df[toomany]

      # Create encoding for all categorical variables
      l.encoding = c(map(df[setdiff(cate, toomany)], ~
                           BoxCore::cate_encoding(., "self", other_value = "_OTHER_", na_value = "(Missing)")),
                     map(df[toomany], ~
                           BoxCore::cate_encoding(., "self_topn", n_top_levels = topn_toomany,
                                                  other_value = "_OTHER_", na_value = "(Missing)")),
                     map(df[paste0(toomany,"_ENCODED")], ~
                           BoxCore::cate_encoding(., "count_others", n_top_levels = topn_toomany, na_value = 0)))

      # Apply encoding (and map to factors if needed)
      df[names(l.encoding)] = map(names(l.encoding), ~ BoxCore::encode(df[[.]], l.encoding[[.]]))

      # Map cate to factors
      #df[cate] = map(cate, ~ factor(df[[.]], levels = l.encoding[[.]]))

      # Check
      summary(df[names(l.encoding)], 20)

      # Catch information about encoding (needed for scoring)
      l.metacate = list(toomany = toomany, encoding = l.encoding)




      # Adapt metric variables -----------------------------------------------------------------------------------------

      # Define metric features
      metr = df.meta %>% filter(type == "metr") %>% .$variable
      summary(df[metr])

      # Impute 0
      mins = map_dbl(df[metr], ~ min(., na.rm = TRUE)) #get minimum for these vars
      df[metr] = map(metr, ~ df[[.]] - mins[.] + 1) #shift
      miss = metr[map_lgl(df[metr], ~ any(is.na(.)))]
      df[miss] = map(df[miss], ~ impute(., "zero"))
      l.metametr = list(mins = mins)




      ##################################################################################################################-
      #|||| Train model ||||----
      ##################################################################################################################-

      # Fit ------------------------------------------------------------------------------------------------------------

      # Define features
      features = c(metr, cate, paste0(toomany,"_ENCODED"))

      # Undersample
      c(df.train, b_sample, b_all) %<-%  (df %>% BoxCore::undersample_n(n_max_per_level = 1e7))
      summary(df.train$target); b_sample; b_all

      # Save Metainformation (needed for scoring)
      l.metasample = list(b_all = b_all, b_sample = b_sample)

      # Final Fit
      Sys.time()
      m.train = sparse.model.matrix(as.formula(paste("~ -1 +", paste(features, collapse = " + "))),
                                    data = df.train[features])
      fit = train(m.train, df.train$target,
                  trControl = trainControl(method = "none", returnData = FALSE),
                  method = xgb,
                  tuneGrid = expand.grid(nrounds = 2100, eta = 0.01,
                                         max_depth = 3, min_child_weight = 2,
                                         colsample_bytree = 0.7, subsample = 0.7,
                                         gamma = 0, alpha = 0, lambda = 1))
      Sys.time()





      # Save Metadata --------------------------------------------------------------------------------------------------

      l.metadata = list("cate" = l.metacate, metr = l.metametr, "features" = list("metr" = metr, "cate" = cate),
                        "sample" = l.metasample, "fit" = fit)
      save(l.metadata, file = paste0(dataloc,"METADATA.RData"))

    },
    error = function (e) {
      message("ERROR in productive_score.R for TARGET_TYPE '", TARGET_TYPE, "'")
    }
  )
}
