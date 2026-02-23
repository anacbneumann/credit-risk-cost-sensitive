'''
================
PREPROCESSING DATA - DRAFT

Noting the steps being taken in the EDA (Enterprise Data Analysis) process for later construction of a data preprocessing script.
================
'''

'''
1) Renaming columns
'''
# Renaming ID column
names(cs_train)[1] <- "id"

# Renaming map -> Consistency in names
rename_map <- c(
  "SeriousDlqin2yrs"                    = "default_2y",
  "RevolvingUtilizationOfUnsecuredLines"= "util_unsecured",
  "age"                                 = "age_years",
  "NumberOfTime30.59DaysPastDueNotWorse"= "dpd_30_59_cnt",
  "DebtRatio"                           = "debt_ratio",
  "MonthlyIncome"                       = "monthly_income",
  "NumberOfOpenCreditLinesAndLoans"     = "open_credit_cnt",
  "NumberOfTimes90DaysLate"             = "dpd_90p_cnt",
  "NumberRealEstateLoansOrLines"        = "real_estate_cnt",
  "NumberOfTime60.89DaysPastDueNotWorse"= "dpd_60_89_cnt",
  "NumberOfDependents"                  = "dependents_cnt"
)

# Apply renaming
names(cs_train) <- ifelse(names(cs_train) %in% names(rename_map),
                          rename_map[names(cs_train)],
                          names(cs_train))

# Checking final column names ----
names(cs_train)


'''
2) Duplicate rows removal
'''
# Duplicated rows ignoring the ID column
cat('\nCounting how many complete rows (ignoring ID) are repeated (all columns are the same):\n')

dup_mask_no_id <- duplicated(cs_train[, setdiff(names(cs_train), "id")])

dup_n <- sum(dup_mask_no_id)
dup_pct <- 100 * dup_n / nrow(cs_train)

cat(sprintf("Repeated rows (ignoring ID): %d (%.2f%% of dataset)\n", dup_n, dup_pct))

dup_rows_no_id <- cs_train[
  dup_mask_no_id | duplicated(cs_train[, setdiff(names(cs_train), "id")], fromLast = TRUE),
]

dup_rows_no_id_sorted <- dup_rows_no_id[order(-dup_rows_no_id$age_years), ]
head(dup_rows_no_id_sorted, 20)
# Removing duplicated rows - keeping 1 of the duplicated rows
cs_train_dedup <- cs_train[!duplicated(cs_train[, setdiff(names(cs_train), "id")]), ]