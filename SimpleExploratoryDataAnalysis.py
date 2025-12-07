# Numerical Bivariate Analysis
import pandas as pd
import time
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors


def numerical_bivariate_analysis(
    data, target, exclude=None, norm_alpha_threshold=0.05, corr_alpha_threshold=0.05
):
    start_time = time.time()

    if exclude:
        exclude_list = [exclude] if isinstance(exclude, str) else exclude
        data_1 = data.drop(columns=exclude_list)
    else:
        data_1 = data.copy()

    if data_1.isnull().any().any():
        raise ValueError("The data contain some missing values")
    else:
        print("No missing values found.")

    nsa_table = (
        data_1.describe()
        .T.reset_index()
        .rename(columns={"index": "feature", "50%": "median"})
    )
    num_col_list = nsa_table["feature"].tolist()

    sum_list, mode_list, var_list = [], [], []
    p_val_norm_list, isnorm_list = [], []
    (
        corr_coef_list,
        corr_category_list,
        p_val_corr_list,
        is_corr_list,
        corr_test_meth_list,
    ) = [], [], [], [], []

    def normality_test(var):
        sample_size = nsa_table["count"].max()

        if sample_size > 2000:
            stat_val, p_val_norm_val = stats.jarque_bera(var)
            test_done = "Jarque-Bera test"
        elif sample_size > 200:
            stat_val, p_val_norm_val = stats.normaltest(var)
            test_done = "D’Agostino’s K-squared test"
        elif sample_size > 50:
            stat_val, p_val_norm_val = lilliefors(var)
            test_done = "Lilliefors test"
        else:
            stat_val, p_val_norm_val = stats.shapiro(var)
            test_done = "Shapiro-Wilk test"

        return stat_val, p_val_norm_val, test_done

    def correlation_test(var1, var2, test_type):
        if test_type == "Spearman":
            corr_coef, p_val_corr = stats.spearmanr(var1, var2)
        else:
            corr_coef, p_val_corr = stats.pearsonr(var1, var2)

        is_corr = int(p_val_corr < corr_alpha_threshold)
        abs_corr = abs(corr_coef)

        if abs_corr >= 0.8:
            corr_category = "Very Strong Correlation"
        elif abs_corr >= 0.6:
            corr_category = "Strong Correlation"
        elif abs_corr >= 0.4:
            corr_category = "Moderate Correlation"
        elif abs_corr >= 0.2:
            corr_category = "Weak Correlation"
        else:
            corr_category = "Very Weak Correlation"

        return p_val_corr, corr_coef, corr_category, is_corr

    for col in num_col_list:
        col_data = data_1[col]
        sum_list.append(col_data.sum())
        mode_list.append(col_data.mode()[0])
        var_list.append(col_data.var())

        stat_val, p_val_norm_val, norm_test_done = normality_test(col_data)
        p_val_norm_list.append(p_val_norm_val)

        if p_val_norm_val <= norm_alpha_threshold:
            isnorm_list.append(0)
            corr_test_type = "Spearman"
        else:
            isnorm_list.append(1)
            if stats.jarque_bera(data_1[target])[1] < 0.05:
                corr_test_type = "Spearman"
            else:
                corr_test_type = "Pearson"

        p_val_corr, corr_coef, corr_category, is_corr = correlation_test(
            col_data, data_1[target], corr_test_type
        )
        corr_coef_list.append(corr_coef)
        corr_category_list.append(corr_category)
        p_val_corr_list.append(p_val_corr)
        is_corr_list.append(is_corr)
        corr_test_meth_list.append(corr_test_type)

    result_table = pd.concat(
        [
            nsa_table[["feature", "count"]],
            pd.Series(sum_list, name="sum"),
            nsa_table[["min", "mean", "median"]],
            pd.Series(mode_list, name="mode"),
            nsa_table[["max", "std"]],
            pd.DataFrame(
                {
                    "var": var_list,
                    "p_val_norm": p_val_norm_list,
                    "is_norm": isnorm_list,
                    "corr_coef": corr_coef_list,
                    "corr_category": corr_category_list,
                    "p_val_corr": p_val_corr_list,
                    "is_corr": is_corr_list,
                    "corr_test_method": corr_test_meth_list,
                }
            ),
        ],
        axis=1,
    )

    print(f"The target variable is {target}")
    print(norm_test_done)

    elapsed_time = time.time() - start_time
    print(f"The table has been created successfully in {elapsed_time:.6f} seconds")

    return result_table


# Categorical Bivariate Analysis
import pandas as pd
import numpy as np
import time
from scipy.stats import chi2_contingency


def categorical_bivariate_analysis(
    data, target, include=None, exclude=None, alpha_threshold=0.05
):
    start_time = time.time()

    if data.isnull().any().any():
        raise ValueError("The data contain some missing values")
    else:
        print("No missing values found.")

    if data[target].dtype != "int64":
        raise ValueError(
            "The target must be a categorical or discrete categorical variable. Please convert the target first!"
        )

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = list(set(data.columns) - set(num_cols))

    if target not in cat_cols:
        cat_cols.append(target)

    if include:
        include = [include] if isinstance(include, str) else include
        cat_cols.extend([col for col in include if col not in cat_cols])

    if exclude:
        exclude = [exclude] if isinstance(exclude, str) else exclude
        for col in exclude:
            if col not in cat_cols:
                raise ValueError(f"The column {col} you want to exclude does not exist")
            cat_cols.remove(col)

    csa_table = data[cat_cols]

    results = {
        "feature": [],
        "count": [],
        "unique": [],
        "mode": [],
        "freq": [],
        "p_val_independency": [],
        "is_dependent": [],
    }

    contingency_table = pd.DataFrame()

    for col in cat_cols:
        results["feature"].append(col)
        results["count"].append(csa_table[col].count())
        results["unique"].append(csa_table[col].nunique())
        mode_val = csa_table[col].mode()[0]
        results["mode"].append(mode_val)
        results["freq"].append((csa_table[col] == mode_val).sum())

        contingency = pd.crosstab(csa_table[col], csa_table[target])
        chi2, p, dof, expected = chi2_contingency(contingency)
        results["p_val_independency"].append(p)
        results["is_dependent"].append(int(p <= alpha_threshold))

        contingency_reset = (
            contingency.reset_index()
            .melt(id_vars=col, var_name="target", value_name="count")
            .rename(columns={col: "categories"})
        )
        contingency_reset["feature"] = col
        contingency_table = pd.concat([contingency_table, contingency_reset])

    results_df = pd.DataFrame(results)
    contingency_table.reset_index(drop=True, inplace=True)

    elapsed_time = time.time() - start_time
    print(f"The tables have been created successfully in {elapsed_time:.6f} seconds")

    return results_df, contingency_table[["feature", "categories", "target", "count"]]


# Multivariate Analysis of Categorical Variables
import pandas as pd
import time
from scipy.stats import chi2_contingency


def categorical_var_independency_test(
    data, include=None, exclude=None, is_dependent=False, alpha_threshold=0.05
):
    start_time = time.time()

    if data.isnull().any().any():
        raise ValueError("The data contain some missing values")
    else:
        print("No missing values found.")

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = list(set(data.columns) - set(num_cols))

    if include:
        include = [include] if isinstance(include, str) else include
        cat_cols.extend([col for col in include if col not in cat_cols])

    if exclude:
        exclude = [exclude] if isinstance(exclude, str) else exclude
        for col in exclude:
            if col not in cat_cols:
                raise ValueError(f"The column {col} you want to exclude does not exist")
            cat_cols.remove(col)

    csa_table = data[cat_cols]
    table = pd.DataFrame({"feature": cat_cols})

    for col1 in cat_cols:
        col1_results = []
        for col2 in cat_cols:
            contingency_table = pd.crosstab(csa_table[col1], csa_table[col2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            if is_dependent:
                col1_results.append(1 if p <= alpha_threshold else 0)
            else:
                col1_results.append(p)
        table[col1] = col1_results

    elapsed_time = time.time() - start_time
    print(f"The table has been created successfully in {elapsed_time:.6f} seconds")

    return table
