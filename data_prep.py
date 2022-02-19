import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

forecast_sample_cols = [
    'M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE',
    'M_WEATHER_FORECAST_SAMPLES_M_WEATHER',
    'M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE',
    'M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE',
    'M_TIME_OFFSET',
    'M_TRACK_TEMPERATURE_CHANGE',
    'M_AIR_TEMPERATURE_CHANGE',
    'M_RAIN_PERCENTAGE']

additional_cols = [
    "M_TRACK_ID",
    "M_FORECAST_ACCURACY"
]

def clean_dataframe(data):
    dct = {}
    for sid in tqdm(set(data["M_SESSION_UID"])):
        data_sid = data[data["M_SESSION_UID"] ==sid]
        for ts in set(data_sid["TIMESTAMP"]):
            data_sid_ts = data_sid[data_sid["TIMESTAMP"] == ts]
            num_samples = list(data_sid_ts["M_NUM_WEATHER_FORECAST_SAMPLES"])[0]
            if num_samples > 0:
                sess_col = "M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE"
                num_nans = data_sid_ts[sess_col].isna().sum()
                data_sid_ts = data_sid_ts.iloc[num_nans:num_nans+num_samples]
                for sess_type in set(data_sid_ts[sess_col]):
                    data_sid_ts_sess = data_sid_ts[data_sid_ts[sess_col]==sess_type]
                    data_sid_ts_sess = data_sid_ts_sess[forecast_sample_cols+additional_cols]
                    dct[(sid,ts,sess_type)] = data_sid_ts_sess
    return dct


def create_processed_frame(dct, name=None):
    times = ["0", "5", "10", "15", "30", "45", "60", "90", "120"]

    single_val_cols = ["M_WEATHER_FORECAST_SAMPLES_M_SESSION_TYPE", "M_TRACK_ID", "M_FORECAST_ACCURACY"]
    multi_val_cols = ["M_WEATHER_FORECAST_SAMPLES_M_WEATHER", "M_WEATHER_FORECAST_SAMPLES_M_TRACK_TEMPERATURE",
                      "M_WEATHER_FORECAST_SAMPLES_M_AIR_TEMPERATURE",
                      "M_TRACK_TEMPERATURE_CHANGE", "M_AIR_TEMPERATURE_CHANGE", "M_RAIN_PERCENTAGE"]
    multi_val_cols_timed = [f"{el}_{time}" for time in times for el in multi_val_cols]
    rows = []
    for (sid,ts,sess_type), table in tqdm(dct.items()):
        nans = [np.nan]*(len(times)-len(table))
        single_vals = list(table[single_val_cols].iloc[0])
        multi_vals = np.array([[el for el in list(table[col])+nans] for col in multi_val_cols]).T.flatten()
        row = single_vals + list(multi_vals)
        rows.append(row)
    columns = single_val_cols + multi_val_cols_timed
    columns = [col.replace("M_WEATHER_FORECAST_SAMPLES_M_","").replace("M_","") for col in columns]
    df = pd.DataFrame(columns = columns, data=rows)
    if name is not None:
        df.to_csv(name)
    return df

def prepare_datasets(dataset_path, train_ratio = 0.7, val_ratio = 0.2, test_ratio = 0.1):
    data = pd.read_csv("weather.csv")
    if "Unnamed: 58" in data.columns:
        data = data.drop(["Unnamed: 58"],axis=1)
    session_uids = list(set(data["M_SESSION_UID"]))
    random.shuffle(session_uids)
    train_uids, val_uids, test_uids = np.split(session_uids, [int(len(session_uids)*train_ratio),
                                                              int(len(session_uids)*(train_ratio+val_ratio))])
    train_data = data[[uid in train_uids for uid in data["M_SESSION_UID"]]]
    val_data = data[[uid in val_uids for uid in data["M_SESSION_UID"]]]
    test_data = data[[uid in test_uids for uid in data["M_SESSION_UID"]]]
    len(train_data), len(val_data), len(test_data)
    
    print("Cleaning Data")
    train_dct_cleaned = clean_dataframe(train_data)
    val_dct_cleaned = clean_dataframe(val_data)
    test_dct_cleaned = clean_dataframe(test_data)
    
    print("Creating Dataframes")
    train_df = create_processed_frame(train_dct_cleaned, "train.csv")
    val_df = create_processed_frame(val_dct_cleaned, "val.csv")
    test_df = create_processed_frame(test_dct_cleaned, "test.csv")
    
    
