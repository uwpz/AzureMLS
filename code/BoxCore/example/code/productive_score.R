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

      # Read data ------------------------------------------------------------------------------------------------------

      # ABT
      df = suppressMessages(read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE))
      df$id = 1:nrow(df)

      # Feature Engineering
      df$deck = as.factor(str_sub(df$cabin, 1, 1)) #deck as first character of cabin
      df$familysize = df$sibsp + df$parch + 1 #add number of siblings and spouses to the number of parents and children
      df %<>% group_by(ticket) %>% mutate(fare_pp = fare/n()) %>% ungroup() #fare per person (one ticket might comprise several persons)

      # Metadata
      load(file = paste0(dataloc,"METADATA.RData"))




      # Adapt categorical variables: Order of following steps is very important! ---------------------------------------

      # Get metadata
      cate = l.metadata$features$cate
      toomany = l.metadata$cate$toomany
      l.encoding = l.metadata$cate$encoding

      # Copy "toomanys"
      df[paste0(toomany,"_ENCODED")] = df[toomany]

      # Apply encoding (and map to factors if needed)
      df[names(l.encoding)] = map(names(l.encoding), ~ BoxCore::encode(df[[.]], l.encoding[[.]]))

      # Map cate to factors
      #df[cate] = map(cate, ~ factor(df[[.]], levels = l.encoding[[.]]))





      # Adapt metric variables -----------------------------------------------------------------------------------------

      metr = l.metadata$features$metr

      # Impute
      mins = l.metadata$metr$mins
      if (length(mins)) {
        df[names(mins)] = map(names(mins), ~ ifelse(df[[.]] < mins[.], mins[.], df[[.]])) #set lower values to min
        df[names(mins)] = map(names(mins), ~ df[[.]] - mins[.] + 1) #shift
      }
      df[metr] = map(df[metr], ~ impute(., "zero"))




      ##################################################################################################################-
      #|||| Score ||||----
      ##################################################################################################################-

      # Define features
      features = c(metr, cate, paste0(toomany,"_ENCODED"))

      # Score and rescale
      Sys.time()
      m.score = sparse.model.matrix(as.formula(paste("~ -1 +", paste(features, collapse = " + "))),
                                    data = df[features])
      yhat_score = predict(l.metadata$fit, m.score, type = "prob") %>%
        BoxCore::scale_predictions(l.metadata$sample$b_sample, l.metadata$sample$b_all)
      Sys.time()

      # Write scored data
      df.score = bind_cols(df[c("id")], "score" = round(yhat_score[,2], 5))
      write_delim(df.score, paste0(dataloc,"scoreddata.psv"), delim = "|")

    },
    error = function (e) {
      message("ERROR in productive_score.R for TARGET_TYPE '", TARGET_TYPE, "'")
    }
  )
}
