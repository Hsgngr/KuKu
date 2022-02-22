"""
This scripts allows the user to create train, validation, test datasets by:
    * Dropping unnecessary columns
    * Removing Duplicates
    * Creating time window pairs
    * Splitting with Session UID to prevent any data leakages


This file can be imported as a module to use following functions:
    * prepare_datasets()
    * get_dfs()

Usage:
    from src.data_prep import prepare_datasets, get_dfs
    train_df, val_df, test_df = prepare_datasets(data_path, single_val_cols, multi_val_cols)


"""
from audioop import mul
from genericpath import exists
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os.path as op
import os

# to create subsequence data samples
time_windows = {
    "5": [(0, 5), (5, 10), (10, 15)],
    "10": [(0, 10), (5, 15)],
    "15": [(0, 15), (15, 30)],
    "30": [(0, 30), (30, 60), (60, 90), (90, 120)],
    "60": [(0, 60), (60, 120)]
}


def clean_dataframe_dct(data, single_val_cols, multi_val_cols):

    """
    Removes rows that has no forecast samples
    Selects rows that contain relevant forecast sample for each (session_uid, timestamp) tuple

    Parameters
    ----------
    data : DataFrame
        The dataframe which is created from weather.csv file
    single_val_cols : list
        Columns which has only single values. Example: "M_TOTAL_LAPS"
    multi_val_cols  : list
        Columns which might have multiple values. Example: "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",

    Returns
    -------
    dct: Dictionary
        that maps (session_uid, timestamp) tuples to tables that has corresponding forecast sample rows
    """
    dct = {}
    columns = ["M_SESSION_UID","TIMESTAMP"] + single_val_cols + multi_val_cols
    data = data[data["M_NUM_WEATHER_FORECAST_SAMPLES"]>0]
    for (sid, ts), data_sid_ts in tqdm(data.groupby(["M_SESSION_UID", "TIMESTAMP"])):
            num_samples = list(data_sid_ts["M_NUM_WEATHER_FORECAST_SAMPLES"])[0]
            sess_col = "M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE"
            num_nans = data_sid_ts[sess_col].isna().sum()
            for sess_type, data_sid_ts_sess in data_sid_ts.iloc[num_nans:num_nans+num_samples].groupby(sess_col):
                dct[(sid, ts, sess_type)] = data_sid_ts_sess[columns]
    return dct


def create_processed_frame(dct, single_val_cols, multi_val_cols):
    """
    Creates a table where each row corresponds to a single (session_uid, timestamp) tuple and its all possible future forecasts
    Puts NaN values for forecasts that are not given

    Parameters
    ----------
    dct : Dictionary
        Gets the dataframe Dictionary
    single_val_cols : list
        Columns which has only single values. Example: "M_TOTAL_LAPS"
    multi_val_cols  : list
        Columns which might have multiple values. Example: "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",

    Returns
    -------
    df: DataFrame
        Generated a tabular form.
    """
    times = ["0", "5", "10", "15", "30", "45", "60", "90", "120"]
    multi_val_cols_timed = [f"{el}_{time}" for time in times for el in multi_val_cols]
    rows = []
    for table in tqdm(dct.values()):
        nans = [np.nan]*(len(times)-len(table))
        single_vals = list(table[["M_SESSION_UID", "TIMESTAMP"] +  single_val_cols].iloc[0])
        multi_vals = np.array([list(table[col])+nans for col in multi_val_cols]).T.flatten()
        row = single_vals + list(multi_vals)
        rows.append(row)
    columns = ["M_SESSION_UID", "TIMESTAMP"] + \
        single_val_cols + multi_val_cols_timed
    df = pd.DataFrame(columns = columns, data=rows)
    return df

#adds flag information columns to given processed dset
def add_flag_info(original_dset, processed_dset):
    ls = []
    for i in range(len(processed_dset)):
        sess_uid, ts = processed_dset[["M_SESSION_UID", "TIMESTAMP"]].iloc[i]
        flags = set(original_dset[(original_dset["M_SESSION_UID"] == sess_uid) & (
            original_dset["TIMESTAMP"] == ts)]["M_ZONE_FLAG"].dropna())
        ls.append([1 if f in flags else 0 for f in [1, 2, 3, 4]])
    processed_dset[["IS_GREEN_FLAG_UP", "IS_BLUE_FLAG_UP",
                    "IS_YELLOW_FLAG_UP", "IS_RED_FLAG_UP"]] = ls
    return processed_dset

# calls clean_dataframe_dct, create_processed_frame for the weather.csv
# and then splits the cleaned df into train, val, test partition considering session uids
def prepare_datasets(dataset_path, single_val_cols, multi_val_cols, train_ratio = 0.7, val_ratio = 0.2, use_flag_info=True):
    """
    Main function which calls clean_dataframe_dct and create_processed_frame functions
    Splits them into Train, Validation Test set by session uids

    Parameters
    ----------
    dataset_path    : str
        Path for preprocessed_data
    single_val_cols : list
        Columns which has only single values. Example: "M_TOTAL_LAPS"
    multi_val_cols  : list
        Columns which might have multiple values. Example: "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",
    train_ratio,val_ratio,test_ratio  : 
        Ratio of session_uids for the given dataset
    Returns
    -------
    train_df: DataFrame
    val_df: Dataframe
    test_df: Dataframe

    Note: Splitting by session_uids do not guarantee that data will split exactly as the given ratio 
    since each session_uid have different amount of rows.
    """
    data = pd.read_csv(dataset_path)
    if "Unnamed: 58" in data.columns:
        data = data.drop(["Unnamed: 58"],axis=1)

    print("Creating (session_uid, timestamp) pairs:")
    cleaned_dct = clean_dataframe_dct(data, single_val_cols, multi_val_cols)
    print("Converting into dataframe:")
    processed_df = create_processed_frame(cleaned_dct, single_val_cols, multi_val_cols)
    
    # drops duplicates ignoring NA
    temp_na_token = -999
    processed_df[processed_df.isna()] = temp_na_token
    ignored_cols = ["M_SESSION_UID", "TIMESTAMP"]
    processed_df = processed_df.drop_duplicates(
        subset=[col for col in processed_df.columns if col not in ignored_cols])
    processed_df[processed_df==temp_na_token] = pd.NA

    session_uids = list(set(processed_df["M_SESSION_UID"]))
    random.shuffle(session_uids)
    train_uids, val_uids, test_uids = np.split(session_uids, [int(len(session_uids)*train_ratio),
                                                              int(len(session_uids)*(train_ratio+val_ratio))])
    train_df = processed_df[[uid in train_uids for uid in processed_df["M_SESSION_UID"]]]
    val_df = processed_df[[
        uid in val_uids for uid in processed_df["M_SESSION_UID"]]]
    test_df = processed_df[[uid in test_uids for uid in processed_df["M_SESSION_UID"]]]

    if use_flag_info:
        train_df = add_flag_info(data, train_df)
        val_df = add_flag_info(data, val_df)
        test_df = add_flag_info(data, test_df)

    train_df = train_df.drop(["M_SESSION_UID", "TIMESTAMP"], axis=1)
    val_df = val_df.drop(["M_SESSION_UID", "TIMESTAMP"], axis=1)
    test_df = test_df.drop(["M_SESSION_UID", "TIMESTAMP"], axis=1)

    train_df.to_csv(op.join("data","train.csv"), index=False)
    val_df.to_csv(op.join("data","val.csv"), index=False)
    test_df.to_csv(op.join("data","test.csv"), index=False)

    return train_df, val_df, test_df

# for given time offset creates a table that has all relevant input features and outputs
def create_dataset(dset_dct, time_offset, single_val_cols, multi_val_cols, drop_duplicates=False):
    flag_cols = [col for col in dset_dct["train"].columns if "FLAG" in col]
    columns = single_val_cols + multi_val_cols + flag_cols + ["TARGET_WEATHER", "TARGET_RAIN_PERCENTAGE"]
    windows = time_windows[time_offset]
    processed_dset_dct = {}
    os.makedirs(op.join("data", str(time_offset)), exist_ok=True)
    for typ, dset in dset_dct.items():
        tables = []
        for w in windows:
            y_cols = [f"M_WEATHER_FORECAST_SAMPLES_M_WEATHER_{w[1]}", f"M_RAIN_PERCENTAGE_{w[1]}"]
            tmp_cols = single_val_cols + [f"{c}_{w[0]}" for c in multi_val_cols] + flag_cols + y_cols
            dset_tmp = dset[tmp_cols]
            dset_tmp = dset_tmp.dropna()
            tables.append(dset_tmp.__array__())
        rows = [row for table in tables for row in table]
        df = pd.DataFrame(columns=columns, data=rows)
        df["TARGET_WEATHER"] = df["TARGET_WEATHER"].astype("int64")
        # drop duplicates only from train
        if drop_duplicates and typ=="train":
            df = df.drop_duplicates()
        df.to_csv(op.join("data", str(time_offset), typ+".csv"), index=False)
        processed_dset_dct[typ] = df
    return processed_dset_dct

# calls create_dataset if the dataset is not saved otherwise reads it
def get_df(df_dct, time_offset, single_val_cols, multi_val_cols, force_recreate=False):
    """
    Gets single dataframes for single Time Offset
    Parameters
    ----------
    df_dct    : Dictionary
        Dictionary which holds main train,val,test dataset.
    single_val_cols : list
        Columns which has only single values. Example: "M_TOTAL_LAPS"
    multi_val_cols  : list
        Columns which might have multiple values. Example: "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",
    force_recreate : Optional, Default: False

    Returns
    -------
    df_dct for a single offset
    """

    if force_recreate or not op.exists(op.join("data", time_offset, "train.csv")) or \
        not op.exists(op.join("data", time_offset, "val.csv")) or not op.exists(op.join("data", time_offset, "test.csv")):
        df_t_min_dct = create_dataset(
            df_dct, time_offset, single_val_cols, multi_val_cols)
    else:
        train_df = pd.read_csv(op.join("data", time_offset, "train.csv"))
        val_df = pd.read_csv(op.join("data", time_offset, "val.csv"))
        test_df = pd.read_csv(op.join("data", time_offset, "test.csv"))
        df_t_min_dct = {"train": train_df, "val": val_df, "test": test_df}
    return df_t_min_dct

# calls get_df for all possible time_offset values
def get_dfs(df_dct, single_val_cols, multi_val_cols):
    """
    Main function to get all the dataframes 
    Parameters
    ----------
    df_dct    : Dictionary
        Dictionary which holds main train,val,test dataset.
    single_val_cols : list
        Columns which has only single values. Example: "M_TOTAL_LAPS"
    multi_val_cols  : list
        Columns which might have multiple values. Example: "M_WEATHER_FORECAST_SAMPLES_M_WEATHER",

    Returns
    -------
    df_timed_dct: Dictionary of Time Offset Dictionary. 
    """
    df_timed_dct = {}
    for time_offset in ["5","10","15","30","60"]:
        print("Creating dataset for time_offset={}".format(time_offset))
        df_timed_dct[time_offset] = get_df(
            df_dct, time_offset, single_val_cols, multi_val_cols)
    return df_timed_dct
        
