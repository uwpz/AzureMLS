
#######################################################################################################################-
#|||| Create model for webservice ||||----
#######################################################################################################################-

# Load model and additional information created during training
load(file = "data/METADATA.RData")


# Init will be called from rpy2 when container is started
init = function() {
  # Score specific libraries
  library(Matrix)
  library(dplyr)
  library(stringr)
  library(purrr)
  library(caret)
  library(xgboost)
  library(config)
  library(BoxCore)

  # Webservice specific libraries
  library(jsonlite)
  #library(rjson)
  #library(readr)
}


# Run will be called from rpy2 for scoring
run = function(inputJsonString) {
  # library(readr); inputJsonString = read_file("data/records.json")

  # get input
  inputJson = fromJSON(inputJsonString)#, unexpected.escape = "keep")

  # Convert to data frame
  df = map_df(inputJson, function(x) data.frame(map(x, ~ ifelse(is.null(.),NA,.)), stringsAsFactors = FALSE))

  # Transform and Score
  yhat = score(df, l.metadata)[,2] #l.metadata is known during run time

  # return result
  outputJsonString = toJSON(yhat)
  return(outputJsonString)
}


# Used by run
score = function(df, l.metadata) {
  # Id
  df$id = 1:nrow(df)

  # Feature Engineering
  df$deck = as.factor(str_sub(df$cabin, 1, 1)) #deck as first character of cabin
  df$familysize = df$sibsp + df$parch + 1 #add number of siblings and spouses to the number of parents and children
  df = df %>% group_by(ticket) %>% mutate(fare_pp = fare/n()) %>% ungroup() #fare per person (one ticket might comprise several persons)


  # Adapt categorical variables: Order of following steps is very important! -----------------------------------------------

  # Get metadata
  cate = l.metadata$features$cate
  toomany = l.metadata$cate$toomany
  l.encoding = l.metadata$cate$encoding

  # Copy "toomanys"
  df[paste0(toomany,"_ENCODED")] = df[toomany]

  # Apply encoding
  df[names(l.encoding)] = map(names(l.encoding), ~ BoxCore::encode(df[[.]], l.encoding[[.]]))

  # Map cate to factors
  df[cate] = map(cate, ~ factor(df[[.]], levels = l.encoding[[.]]))


  # Adapt metric variables ----------------------------------------------------------------------------------

  metr = l.metadata$features$metr

  # Impute
  mins = l.metadata$metr$mins
  if (length(mins)) {
    df[names(mins)] = map(names(mins), ~ ifelse(df[[.]] < mins[.], mins[.], df[[.]])) #set lower values to min
    df[names(mins)] = map(names(mins), ~ df[[.]] - mins[.] + 1) #shift
  }
  df[metr] = map(df[metr], ~ BoxCore::impute(., "zero"))


  # Score ----------------------------------------------------------------------------------

  # Define features
  features = c(metr, cate, paste0(toomany,"_ENCODED"))

  # Score and rescale
  m.score = sparse.model.matrix(as.formula(paste("~ -1 +", paste(features, collapse = " + "))),
                                data = df[features])
  yhat_score = predict(l.metadata$fit, m.score, type = "prob") %>%
    BoxCore::scale_predictions(l.metadata$sample$b_sample, l.metadata$sample$b_all)

  return(yhat_score)
}



# Save all we need during scoring time
save("l.metadata", "init", "score", "run", file = "data/model.RData")



#######################################################################################################################-
#|||| Testing: Simulate rpy2 calls ||||----
#######################################################################################################################-

# clean workspace
rm(list=ls())

# Load model
load("data/model.RData")

# invoke init() method
init()

# invoke run() method
library(readr)
cat(run(read_file("data/records.json")))

