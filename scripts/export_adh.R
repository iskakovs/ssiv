# Export the ADH list (from ShiftShareSE's data/ADH.rda) to CSV files.
# Uses base R only: the .rda file is deserialised; no package code is run.
# Usage: Rscript scripts/export_adh.R <path/to/ADH.rda> <output dir>
args <- commandArgs(trailingOnly = TRUE)
e <- new.env()
load(args[1], envir = e)
adh <- get("ADH", envir = e)
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
write.csv(adh$reg, file.path(out, "adh_reg.csv"), row.names = FALSE)
write.csv(as.matrix(adh$W), file.path(out, "adh_W.csv"), row.names = FALSE)
write.csv(data.frame(sic = adh$sic), file.path(out, "adh_sic.csv"), row.names = FALSE)
cat("ADH data written to", out, "\n")
