"""
preprocessing.py
---------------

    Preprocess data for training and testing
"""

# Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from helpers import *
from SharedPandasDF import SharedDF, SharedNumpyArray
from scipy import stats
import threading
import multiprocessing
from multiprocessing import Process, Manager, Pool
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time
from tqdm import tqdm


############################## Global Variables ###############################
# containers for dataframe construction and manipulation
features_combined = ["speed", "altitude", "verticalSpeed", "range", "bearing", 
"rangeRate", "signalToNoiseRatio", "sCov1", "sCov2", "sCov3", "sCov4", "sCov5",
"sCov6", "sCov7", "sCov8", "sCov9", "sCov10", "sCov11", "sCov12", "sCov13", "sCov14", 
"sCov15", "sCov16", "sCov17", "sCov18", "sCov19", "sCov20", "sCov21", "sCov22", "sCov23", 
"sCov24", "sCov25", "sCov26", "sCov27", "sCov28", "sCov29", "sCov30", "sCov31", "sCov32", 
"sCov33", "sCov34", "sCov35", "sCov36", "rCov1", "rCov2", "rCov3", "rCov4", "rCov5", 
"rCov6", "rCov7", "rCov8", "rCov9", "Class", "Subclass", "Type", "Subtype"]
dict_one_hot = {"2200": [0,0,0,0,1], "5200": [0,0,0,1,0], "1122": [0,0,1,0,0], 
"1112": [0,1,0,0,0], "1111": [1,0,0,0,0]}
dict_specific_data = {"2200": [2,2,0,0], "5200": [5,2,0,0], "1122": [1,1,2,2], 
"1112": [1,1,1,2], "1111": [1,1,1,1]}

# multiprocessing variables
lock = multiprocessing.Lock()
###############################################################################


def convert_to_df(args, csv_dir, time_series):
    """
    Create labels for data
    """
    shared_df, file = args
    local_df = shared_df.read()

    # reads it in so that first row gets read in as header and then replaced with labels (first row is convoluted)
    df = pd.read_csv(f"{csv_dir}/{file}", delimiter=",")
    df.columns = features_combined

    # if total samples is not a multiple of the time series step, remove the last few samples
    if time_series:
        if len(df) % time_series != 0:
            df = df[:-(len(df) % time_series)]

    local_df = pd.concat([local_df, df], ignore_index=True)

    return SharedDF(local_df)



def threaded_time_series_clean(df):
    None

def clean_data(df, time_series):
    """
    Clean data
    """
    df_clean = pd.DataFrame()
    features_subset = features_combined[:2] + features_combined[3:]

    # time series dataframes must handles differently
    if time_series:
        # loop through dataframe in increments of time series
        cnt = 0
        for i in range(0, len(df), time_series):
            # slice dataframe into rows of size time_series
            df_slice = df.iloc[i:i+time_series]
            df_slice = df_slice[df_slice["verticalSpeed"] != 0]
            df_slice = df_slice[df_slice["signalToNoiseRatio"] != 0]
            df_slice = df_slice[df_slice["range"] >= 300]
            df_slice = df_slice.drop_duplicates(inplace=False, keep='first', subset=features_subset)
            if len(df_slice) < time_series:
                continue # this series is corrupted, so skip it
            else:
                # concat this slice onto the clean dataframe
                df_clean = pd.concat([df_clean, df_slice], ignore_index=True)
            cnt += 1
            print("df_clean: ", len(df_clean), " df_slices: ", cnt)
        return df_clean

    # exclude vertical velocity from the data, it is falsy calculated when afsim reports duplicates
    df_clean = df.drop_duplicates(inplace=False, keep='first', subset=features_subset)

    df_clean = df_clean[df_clean["verticalSpeed"] != 0]
    df_clean = df_clean[df_clean["signalToNoiseRatio"] != 0]

    # removing outliers and shuffling changes dataset every time, only use during dataset creation
    if __name__ == "__main__":
        # outlier removal
        separated_df = sep_data(df_clean, dict_specific_data)
        for label in separated_df:
            print(separated_df[label].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))
            separated_df[label] = remove_outliers(separated_df[label])
            print(separated_df[label].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))
        df_clean = pd.concat(separated_df)

        # shuffle the data
        df_clean = df_clean.sample(frac=1).reset_index(drop=True)

    return df_clean

def retrieve(data_dir, time_series):
    if data_dir:
        remove_version_1_and_3(data_dir)
        files = os.listdir(data_dir)
        # initialize dataframe with 1s
        temp_df = pd.DataFrame(np.ones((1, len(features_combined))))
        shared_df = SharedDF(temp_df)
        st = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            tasks = ((shared_df, file) for file in (files))
            result = executor.map(partial(convert_to_df, csv_dir=data_dir, time_series=time_series), tasks)
            for res in tqdm(result, total=len(files)):
                res.unlink()
            shared_df.unlink()
        print(time.perf_counter() - st)
        # for file in os.listdir(data_dir):
            # if file.endswith(".csv"):
                # convert_to_df(data_dir + "/" + file, df_container, time_series)
        return shared_df.copy()
    
    return [pd.DataFrame()]

def create_train_sets(general_data='', specific_data='', specific_data_type=[],
                      specific_data_amount=[], x_train_path=[], y_train_path='',
                      complete_csv='', time_series=0):
    """
    Create ML/DL training/testing sets
        All data sources (general_data, specific_data(can be altered using spec data params), and complete_csv)
        are cleaned and combined into single dataframe that is returned 
            Must provide at least one of these data sources

    Params:
        general_data: data directory from Guardian Batch run
        specific_data: data directory from Guardian Batch run
        specific_data_type: data type to extract from specific data; can be list (ie. [5200, 2200])
        specific_data_amount: amount of data type to extract from specific data; can be list (ie. [100000, 25000])
        x_train: list of output paths to store data features (only runs in main, and is strictly used for optional viewing/analysis)
        y_train: singe output path to store data labels (only runs in main, and is strictly used for optional viewing/analysis)
        complete_csv: path to csv containing previous output of guardpreprocessing.py
        time_series: boolean to determine if data is time series (ie. if it is a sequence of data)
        time_series_steps: number of steps in time series; used if time_series is True
    Returns:
        x_df: dataframe of x data
        y_df: dataframe of labels
    """
    
    gen_df = retrieve(general_data, time_series)
    spec_df = retrieve(specific_data, time_series)
    spec_df = pd.concat(spec_df); gen_df = pd.concat(gen_df)

    if specific_data:
        separated_df = sep_data(spec_df, specific_data_type)
        pos = 0
        for label in separated_df:
            try:
                if time_series:
                    print(f"Extracting {specific_data_amount[pos] // time_series} time series from {label}")
                    separated_df[label] = separated_df[label].iloc[:(specific_data_amount[pos] // time_series * time_series)]
                else:
                    print(f"Extracting {specific_data_amount[pos]} data samples from {label}") 
                    separated_df[label] = separated_df[label].sample(n=specific_data_amount[pos])
            except:
                print(f"\nError: specific data amount for {label} is too large or not provided, taking as much as possible\n")
            pos += 1
        spec_df = pd.concat(separated_df)

    df = pd.concat([gen_df, spec_df])

    if complete_csv:
        df_complete = pd.read_csv(complete_csv, delimiter=",")
        df_complete.columns = features_combined
        df = pd.concat([df, df_complete])

    print("Size of df before cleaning: " + str(df.shape)) 
    df = clean_data(df, time_series)
    print("Size of df after necessary cleaning: " + str(df.shape))

    # slice dataset into its respective parts
    x_num_df, x_sCov_df, x_rCov_df = slice_x(df)
    x_df = df.loc[:, 'speed':'rCov9']
    y_df = df.loc[:, 'Class':'Subtype']

    if __name__ == "__main__":
        # display the amount of training examples in each class
        one_hot_df = one_hot_encode(y_df)
        print("\nAmount of training examples in each class:")
        for col in one_hot_df.columns:
            print(f"{col[0]}: {one_hot_df[col].sum()}")

        df.to_csv(f"current_complete.csv", index=False)
        x_num_df.to_csv(x_train_path[0], index=False)
        x_sCov_df.to_csv(x_train_path[1], index=False)
        x_rCov_df.to_csv(x_train_path[2], index=False)
        y_df.to_csv(y_train_path, index=False)

    return x_df, y_df


def main():
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument("-gd","--general_data_dir", type=str, default='',
                        help="directory for collecting all data returned from guardian <dir>")
    parser.add_argument("-sd","--specific_data_dir", type=str, default='',
                        help="directory for collecting specific data returned from guardian <dir>")
    parser.add_argument("-type","--specific_data_type", nargs='+', default=[],
                        help="list of types of data to extract from specific_data_dir <class (i.e. 5200)>")
    parser.add_argument("-amount", "--specific_data_amount", nargs='+', default=[],
                        help=("list of amounts of data to extract from specific_data_dir of specific_data_type>(correlates with type)." 
                        "If time series is set, this number must be multiple of time series steps." 
                        "If not, it will be rounded down to the closest multiple of time series steps."))
    parser.add_argument("-c","--complete_csv", type=str, default='',
                        help="path to previous csv file output from this application")
    parser.add_argument("-ts", "--time_series", type=int, default=0,
                        help="number of time series steps; if set, it will enable time series data construction.")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print("\nCall with --help or -h for more information\n")
        sys.exit(1)

    valid_data = False
    if args.specific_data_dir:
        if not os.path.exists(args.specific_data_dir):
            print("\nSpecific Data Directory does not exist")
            sys.exit(1)
        if args.specific_data_type:
            if args.specific_data_amount:
                if len(args.specific_data_amount) != len(args.specific_data_type):
                    print("\nNumber of amounts of specific data to extract does not match number of types")
                    sys.exit(1)
                for i in range(len(args.specific_data_type)):
                    if args.specific_data_type[i] not in dict_specific_data.keys():
                        print("Proper Specific Data Types not provided")
                        sys.exit(1)
                    if float(args.specific_data_amount[i]) < 0:
                        print("Amount of specific data to extract cannot be negative")
                        sys.exit(1)
                    if args.time_series and float(args.specific_data_amount[i]) % args.time_series != 0:
                        print("Amount of specific data to extract must be a multiple of time series steps")
                        sys.exit(1)
                args.specific_data_amount = [int(i) for i in args.specific_data_amount]
        else:
            print("\nMust use -type if using -sd")
            sys.exit(1)
        valid_data = True
    if args.general_data_dir:
        if not os.path.exists(args.general_data_dir):
            print("\nMain Data Directory does not exist")
            sys.exit(1)
        valid_data = True
    if args.complete_csv:
        if not os.path.exists(args.complete_csv):
            print("\nComplete Data File path does not exist")
            sys.exit(1)
        valid_data = True

    # output file names
    y_train_path = "train_y_set.csv"
    x_train_path = ["train_x_set_num.csv", "train_x_set_sCov.csv", "train_x_set_rCov.csv"]
    print("[INFO] Input args processed...\n")
    # create training sets
    if valid_data:
        create_train_sets(args.general_data_dir, args.specific_data_dir, args.specific_data_type, 
                          args.specific_data_amount, x_train_path, y_train_path, args.complete_csv,
                          args.time_series)

    
if __name__ == "__main__":
    main()                    
        