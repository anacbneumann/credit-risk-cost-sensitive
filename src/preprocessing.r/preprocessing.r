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


'''
3) preenchendo nan com mediana
'''
# Filling in null values ​​with the median
df$monthly_income[is.na(df$monthly_income)] <- median(df$monthly_income, na.rm = TRUE)
df$dependents_cnt[is.na(df$dependents_cnt)] <- median(df$dependents_cnt, na.rm = TRUE)
head(df, 5)

'''
4) criar flags
'''
income_missing:  = nan
dependents_missing:  = nan

util_gt1: util_unsecured > 1
util_gt10: util_unsecured > 10


# Flag: debt ratio too high + missing income (suspicious)
df$dr_unreliable <- df$income_missing & (df$debt_ratio > 100)

# Flag: debt_ratio greater than 100
df$dr_gt100 <- df$debt_ratio > 100


'''
5) Removendo linha onde age = 0
'''
# remove rows where age_years == 0
df <- df[df$age_years != 0, ]

# checagem rápida
cat("Rows after removing age_years == 0:", nrow(df), "\n")

'''
6) Capar valores de debt_ratio
'''
# Transform the column values ​​into logarithms to normalize the gradient.
df$debt_ratio <- log1p(df$debt_ratio)
