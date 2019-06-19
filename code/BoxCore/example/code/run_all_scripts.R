# 'source' this file!

scripts = c(
  "1_explore.R",
  "2_modelcomparison.R",
  "3_interpret.R",
  "productive_score.R",
  "productive_train.R"
)

for (script in scripts) {
  current_dir = dirname(sys.frame(1)$ofile)
  setwd(file.path(current_dir, "../"))

  script_path = file.path(current_dir, script)
  if (!file.exists(script_path)) {
    message("File '", script_path, "' does not exist.")
  } else {
    message("Run script '", script, "'.")
    source(script_path)
  }
}
