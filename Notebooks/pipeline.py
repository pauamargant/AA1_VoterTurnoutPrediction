from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.under_sampling import RandomUnderSampler

# ------------------
# Global parameters
# ------------------


rng = 42

# Columns used from the dataset
used_cols = [
    "GQTYPE",
    "STATEFIP",
    "METRO",
    "FAMINC",
    "NFAMS",
    "INTTYPE",
    "RELATE",
    "AGE",
    "SEX",
    "RACE",
    "MARST",
    "VETSTAT",
    "FAMSIZE",
    "NCHILD",
    "NCHLT5",
    "NSIBS",
    "BPL",
    "YRIMMIG",
    "CITIZEN",
    "MBPL",
    "FBPL",
    "NATIVITY",
    "HISPAN",
    "EMPSTAT",
    "CLASSWKR",
    "UHRSWORKT",
    "EDUC",
    "DIFFANY",
    "VOTED",
    "COVIDUNAW",
    "EMPSTAT_HEAD",
    "EDUC_HEAD",
    "COVIDPAID",
    "VOTERES"
]

# We divide the columns into diferent subsets in order to apply different preprocessing to each subset

num_cols_imputate = ["UHRSWORKT"]

num_cols_basic = [
    "NFAMS",
    "FAMSIZE",
    "NCHILD",
    "NCHLT5",
    "NSIBS",
    "YRIMMIG",

]
cat_cols_onehot = [
    "GQTYPE",
    "METRO",
    "INTTYPE",
    "SEX",
    "VETSTAT",
    "CITIZEN",
    "DIFFANY",
    "COVIDUNAW",
    "NATIVITY",
    "COVIDPAID",
]

cat_cols_many = [
    "STATEFIP",
    "RELATE",
    "RACE",
    "MARST",
    "BPL",
    "MBPL",
    "FBPL",
    "HISPAN",
    "EMPSTAT",
    "CLASSWKR",
    "EMPSTAT_HEAD",
]

cat_cols = cat_cols_onehot + cat_cols_many

cat_cols_ord = ["FAMINC", "AGE", "EDUC", "EDUC_HEAD", "VOTERES"]

# Functions used to recode the variables


def country_mapper(x):
    if x == 11000:
        return 1  # puerto rico
    if x < 15000:
        return 0  # us and territories
    elif x == 15000:
        return 2  # canada
    elif x <= 20000:
        return 3  # rest of north america and some islands
    elif x <= 30000:
        return 4  # central america and caribbean
    elif x <= 31000:
        return 5  # south america
    elif x < 50000:
        return 6  # europe
    elif x < 60000:
        return 7  # asia (including turkey)
    elif x < 70000:
        return 8  # africa
    elif x < 80000:
        return 9  # oceania
    else:
        return 10  # unknown


def race_mapper(x):
    if x == 100:
        return 1  # white
    elif x == 200:
        return 2  # black
    elif x == 300:
        return 3  # american indian
    elif x in [650, 651, 652]:
        return 4  # asian
    elif x == 801:
        return 5  # black and white
    elif x in [802, 803, 804]:
        return 6  # white mix
    elif x in [805, 806, 807]:
        return 7  # black mix
    elif x in [808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819]:
        return 8  # asian and american mixes
    else:
        return 9


faminc_mapper = {
    100: 0,
    210: 5000,
    300: 7500,
    430: 10000,
    470: 12500,
    500: 15000,
    600: 20000,
    710: 25000,
    720: 30000,
    730: 35000,
    740: 40000,
    820: 50000,
    830: 60000,
    841: 75000,
    842: 100000,
    843: 150000
}


def select_and_filter(
    dataset_fname,
    columns,
    catcols,
    output_file=None,
    resample=False,
    random_state=rng,
):
    """
    This function reads the raw dataset and applied the data cleaning steps.

    Parameters
    ----------
    dataset_fname : str
        The name of the file containing the raw dataset.
    columns : list
        The list of columns to keep.
    catcols : list
        The list of categorical columns.
    output_file : str, optional
        The name of the file where the cleaned dataset will be saved. The default is "prepared_data.csv".
    resample : bool, optional
        Whether to resample the dataset in order to balance it. The default is False.
    random_state : int, optional
        The random state to use for reproducibility. The default is rng.
    """

    try:
        df = pd.read_csv(dataset_fname)
    except:
        print("The file does not exist")
        return
    try:
        df = df[columns]
        print(set(df.columns) - set(columns))
    except:
        print("Some columns are missing")
        return

    # We only keep the rows where the target variable is 1 or 2
    df = df[df["VOTED"].isin([1, 2])]
    # We remove the rows where NATIVITY is 0 (nan)
    df = df[df["NATIVITY"] != 0]

    # We recode
    df["VOTED"] = df["VOTED"].replace({1: 0, 2: 1})
    df["UHRSWORKT"] = df["UHRSWORKT"].replace({999: 0, 997: np.NaN})
    df["COVIDPAID"] = df["COVIDPAID"].replace({99: 0})

    # We transform yrimmig into percentage of life in the US
    df["YRIMMIG"] = df.apply(lambda x: 1 if x["YRIMMIG"] == 0 else (
        2020-x["YRIMMIG"])/x["AGE"], axis=1)

    # apply country mapper to BPL, FBPL, MBPL
    df["BPL"] = df["BPL"].apply(country_mapper)
    df["FBPL"] = df["FBPL"].apply(country_mapper)
    df["MBPL"] = df["MBPL"].apply(country_mapper)
    df["RACE"] = df["RACE"].apply(race_mapper)

    # apply faminc mapper
    df["FAMINC"] = df["FAMINC"].apply(
        lambda x: faminc_mapper[x] if x in faminc_mapper else x)

    # in GQTYPE map 5 8 9 10 to 5 (very uncommon categories)
    df["GQTYPE"] = df["GQTYPE"].replace({5: 8, 8: 5, 9: 5, 10: 5})

    # UHRSWORKT if more than 80 hours per week, we set it to 80
    df["UHRSWORKT"] = df["UHRSWORKT"].apply(lambda x: 80 if x > 80 else x)

    # Convert voteres into ordered categorical
    df["VOTERES"] = df["VOTERES"].replace(
        {999: 0, 10: 0, 20: 1, 31: 3, 33: 5})

    df[catcols] = df[catcols].astype("object")
    # save the dataframe
    if resample == True:
        under_sampling = RandomUnderSampler(
            random_state=random_state, sampling_strategy=pctg
        )
        df, _ = under_sampling.fit_resample(df, df["VOTED"])
    if output_file != None:
        try:
            df.to_csv(output_file, index=False)
        except:
            print("Could not save the file")
    return df


# make pipeline scale and impute
pipeline_scale_impute = Pipeline(
    [
        ("scaler", RobustScaler()),
        ("imputer", IterativeImputer(max_iter=10, random_state=rng)),
    ]
)


def get_train_test(fname: str = "dataset_v1.csv", balanced=False, seed=rng):
    """
    This function reads the raw dataset and applied the data cleaning steps.
    Then it splits the dataset into train and test sets.

    Parameters
    ----------
    fname : str, optional
        The name of the file containing the raw dataset. The default is "dataset_v1.csv".
    balanced : bool, optional
        Whether to resample the dataset in order to balance it. The default is False.
    seed : int, optional
        The random state to use for reproducibility. The default is rng.

    Returns
    -------
    X_train : pandas.DataFrame
        The training set.
    X_test : pandas.DataFrame
        The test set.
    y_train : pandas.DataFrame 
        The training set target variable.
    y_test : pandas.DataFrame   
        The test set target variable.
    """
    df = select_and_filter(
        fname,
        used_cols,
        cat_cols,
        None,
        resample=balanced,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["VOTED"]), df["VOTED"], test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test


preprocessing_oh = ColumnTransformer(
    transformers=[
        ("scaler", RobustScaler(), num_cols_basic),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                min_frequency=25,
            ),
            cat_cols_onehot + cat_cols_many,
        ),
        ("imputer", pipeline_scale_impute, num_cols_imputate),
    ],
    remainder="passthrough",
)

preprocessing_oh_target = ColumnTransformer(
    transformers=[
        ("scaler", RobustScaler(), num_cols_basic),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols_onehot),
        ("target_econder", TargetEncoder(), cat_cols_many),
        ("imputer", pipeline_scale_impute, num_cols_imputate),
    ],
    remainder="passthrough",
)


preprocessing_num = ColumnTransformer(
    transformers=[
        ("scaler", RobustScaler(), num_cols_basic),
        ("imputer", IterativeImputer(max_iter=10, random_state=rng), num_cols_imputate),
    ],
    remainder="passthrough",
)
