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
from scipy import stats
import threading
import multiprocessing
from multiprocessing.pool import ThreadPool
import queue as q
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time


############################## Global Variables ###############################
# containers for dataframe construction and manipulation
features_combined = ["speed", "altitude", "verticalSpeed", "range", "bearing", 
"rangeRate", "signalToNoiseRatio", "sCov1", "sCov2", "sCov3", "sCov4", "sCov5",
"sCov6", "sCov7", "sCov8", "sCov9", "sCov10", "sCov11", "sCov12", "sCov13", "sCov14", 
"sCov15", "sCov16", "sCov17", "sCov18", "sCov19", "sCov20", "sCov21", "sCov22", "sCov23", 
"sCov24", "sCov25", "sCov26", "sCov27", "sCov28", "sCov29", "sCov30", "sCov31", "sCov32", 
"sCov33", "sCov34", "sCov35", "sCov36", "rCov1", "rCov2", "rCov3", "rCov4", "rCov5", 
"rCov6", "rCov7", "rCov8", "rCov9", "Class", "Subclass", "Type", "Subtype"]
dict_one_hot = {"2200": [0,0,0,0,0,1], "5200": [0,0,0,0,1,0], "1122": [0,0,0,1,0,0], 
"1112": [0,0,1,0,0,0], "1111": [0,1,0,0,0,0], "1113":[1,0,0,0,0,0]}
specific_data_dict = {"2200": [2,2,0,0], "5200": [5,2,0,0], "1122": [1,1,2,2], 
"1112": [1,1,1,2], "1111": [1,1,1,1], "1113": [1,1,1,3]}

# multiprocessing variables
procs = multiprocessing.cpu_count() * 2 // 3
###############################################################################


def combine_data(csv_file, data_dir, time_series):
    """
    Create labels for data
    """
    # reads it in so that first row gets read in as header and then replaced with labels (first row is convoluted)
    try:
        df = pd.read_csv(data_dir + "/" + csv_file, delimiter=",")
        df.columns = features_combined
    except Exception as e:
        print("[WARNING]", e)
        return pd.DataFrame()

    # if total samples is not a multiple of the time series step, remove the last few samples
    if time_series:
        try:
            sep_df = sep_data(df, specific_data_dict)
            for label in sep_df:
                if len(sep_df[label]) % time_series != 0:
                    sep_df[label] = sep_df[label].iloc[:-(len(sep_df[label]) % time_series)]
            df = pd.concat(sep_df, ignore_index=True)
        except Exception as e:
            print("[WARNING]", e)
            return pd.DataFrame()
    return df

def sep_data(df, class_container):
    # separate data into dataframes based on classes specified in class_container
    class_frames = {}
    for label in class_container:
        df_class = df[df["Class"] == specific_data_dict[label][0]]
        df_class = df_class[df_class["Subclass"] == specific_data_dict[label][1]]
        df_class = df_class[df_class["Type"] == specific_data_dict[label][2]]
        df_class = df_class[df_class["Subtype"] == specific_data_dict[label][3]]
        if len(df_class) > 0:
            class_frames[label] = df_class

    return class_frames

def slice_x(container):
    # if dataframe, use pandas slice; else, use numpy slice
    if isinstance(container, pd.DataFrame):
        x_num_df = container[features_combined[0:7]].copy(deep=True)
        x_sCov_df = container[features_combined[7:43]].copy(deep=True)
        x_rCov_df = container[features_combined[43:52]].copy(deep=True)
        return x_num_df, x_sCov_df, x_rCov_df
    else:
        x_num_np = container[:, 0:7]
        x_sCov_np = container[:, 7:43]
        x_rCov_np = container[:, 43:52]
        return x_num_np, x_sCov_np, x_rCov_np

def remove_outliers(df):
    # loop through only the columns that are not the class
    for col in df.columns[:-4]:
        mean = df[col].mean()
        sd = df[col].std()
        if col == "range":
            df = df[df[col] >= 300]
        else:
            # this helps center the data to what happens most of the time in scenarios
            df = df[(df[col] <= mean+(4*sd))] #removes top 1% of data
            df = df[(df[col] >= mean-(4*sd))] #removes bottom 1% of data

    return df

def one_hot_encode(df):
    """
    One hot encode the dataframe
    """
    df_one_hot = df.copy(deep=True)
    df_one_hot = df_one_hot.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
    df_one_hot = df_one_hot.values.reshape(-1, 1)

    # one hot encode the data
    enc = OneHotEncoder(sparse=False)
    enc.fit(df_one_hot)
    df_one_hot = enc.transform(df_one_hot)
    df_one_hot = pd.DataFrame(df_one_hot, columns=enc.categories_)
    
    return df_one_hot

def time_series_split(x_df, y_df, train_size, ts_steps):
    """
    Params
    train_size: percentage of data to be used for training (test size is whatever is remaining)
    ts_steps: number of time steps to be used for each sample

    Returns
    x_train, y_train, x_test, y_test    
    """
    separated_y_df = sep_data(y_df, specific_data_dict)
    separated_x_df_train = {}
    separated_x_df_test = {}
    separated_y_df_train = {}
    separated_y_df_test = {}
    for label in separated_y_df:
        total_class_size = len(separated_y_df[label])
        train_class_size = int(total_class_size * train_size)
        train_class_size = train_class_size // ts_steps * ts_steps

        # cut x and y dataframe
        starting_index = separated_y_df[label].index[0]
        ending_index = separated_y_df[label].index[total_class_size-1] + 1
        separated_x_df_train[label] = x_df[starting_index:starting_index+train_class_size]
        separated_y_df_train[label] = y_df[starting_index:starting_index+train_class_size]
        separated_x_df_test[label] = x_df[starting_index+train_class_size:ending_index]
        separated_y_df_test[label] = y_df[starting_index+train_class_size:ending_index]

    # release dataframes
    del separated_y_df, x_df, y_df

    x_train = pd.concat(separated_x_df_train, ignore_index=True)
    x_test = pd.concat(separated_x_df_test, ignore_index=True)
    y_train = pd.concat(separated_y_df_train, ignore_index=True)
    y_test = pd.concat(separated_y_df_test, ignore_index=True)

    return x_train, x_test, y_train, y_test

def clean_time_series(df_slice, features_subset, time_series):
    df_slice = df_slice[df_slice["verticalSpeed"] != 0]
    df_slice = df_slice[df_slice["signalToNoiseRatio"] != 0]
    df_slice = df_slice[df_slice["range"] >= 300]
    df_slice = df_slice.drop_duplicates(inplace=False, keep='first', subset=features_subset)
    if len(df_slice) < time_series:
        return pd.DataFrame()
    else:
        return df_slice

def clean_data(df, time_series):
    """
    Clean data
    """
    features_subset = features_combined[:2] + features_combined[3:] # verticalSpeed ignored

    # time series dataframes must be handled differently
    if time_series:
        df_slices_list = []
        for i in range(0, len(df), time_series):
            df_slices_list.append(df.iloc[i:i+time_series])

        st = time.perf_counter()
        print(f"[INFO] Launching {procs} processes to clean each time series...")
        with multiprocessing.Pool(processes=procs) as pool:
            result = pool.map(partial(clean_time_series, features_subset=features_subset, 
                              time_series=time_series), df_slices_list)
            print("[INFO] Cleaning time series took (s):", time.perf_counter() - st)
            df_clean = pd.concat(result)

            return df_clean

    # exclude vertical velocity from the data, it is falsy calculated when afsim reports duplicates
    df_clean = df.drop_duplicates(inplace=False, keep='first', subset=features_subset)

    # particular removal for improved accuracy on demo version
    ## remove all data that has both altitude above 3000 and Subtype equal to 1
    # df_clean = df_clean[~((df_clean["altitude"] >= 3000) & (df_clean["Subtype"] == 1))]

    df_clean = df_clean[df_clean["verticalSpeed"] != 0]
    df_clean = df_clean[df_clean["signalToNoiseRatio"] != 0]
 
    ########### NO NEED FOR THIS IF DATA IS RECORDED WELL ##############
    # outlier removal
    # separated_df = sep_data(df_clean, specific_data_dict)
    # for label in separated_df:
    #     print(separated_df[label].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))
    #     separated_df[label] = remove_outliers(separated_df[label])
    #     print(separated_df[label].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))
    # df_clean = pd.concat(separated_df)

    # shuffle the data
    df_clean = df_clean.sample(frac=1).reset_index(drop=True)

    return df_clean

def retrieve(data_dir, time_series):
    remove_version_1_and_3(data_dir)
    files = os.listdir(data_dir)

    print(f"[INFO] Launching {procs} processes to load data into memory...")
    st = time.perf_counter()
    with multiprocessing.Pool(processes=procs) as pool:
        results = pool.map(partial(combine_data, data_dir=data_dir, time_series=time_series), files)                
        print("[INFO] Loading data into memory took (s):", time.perf_counter() - st)

    return results

def create_train_sets(data_dir='', dataset='', add=False, erase=False, specific_data_type=[],
                      specific_data_amount=[], x_train_path=[], y_train_path='',
                      time_series=0):

    df = pd.DataFrame() # initialize return dataframe
    
    if data_dir:
        df = retrieve(data_dir, time_series)
        print(f"[INFO] Concatenating data returned from each processor...", end="", flush=True)
        df = pd.concat(df, ignore_index=True); 
        print("done")

    if add:
        separated_df = sep_data(df, specific_data_type)
        pos = 0
        for label in separated_df:
            try:
                if time_series:
                    print(f"[INFO] Extracting {specific_data_amount[pos] // time_series} time series from {label}")
                    separated_df[label] = separated_df[label].iloc[:(specific_data_amount[pos] // time_series * time_series)]
                else:
                    print(f"[INFO] Extracting {specific_data_amount[pos]} data samples from {label}") 
                    separated_df[label] = separated_df[label].sample(n=specific_data_amount[pos])
            except:
                print(f"[WARNING] Specific data amount for {label} is too large or not provided, taking as much as possible")
            pos += 1
        df = pd.concat(separated_df, ignore_index=True) # indices become ("class", index)

    if dataset:
        print(f"[INFO] Loading data from dataset...")
        df_complete = pd.read_csv(dataset, delimiter=",")
        df_complete.columns = features_combined
        df = pd.concat([df, df_complete]) # concat if this is main process, else just split and return
        if not __name__ == "__main__":
            x_df = df.loc[:, 'speed':'rCov9']
            y_df = df.loc[:, 'Class':'Subtype']
            return x_df, y_df

    if erase:
        #### TODO: implement time series erasure
        separated_df = sep_data(df, specific_data_dict)
        pos = 0
        for label in specific_data_type:
            try:
                print(f"[INFO] Erasing {specific_data_amount[pos]} data samples from {label}")
                random_indices = separated_df[label].sample(n=specific_data_amount[pos]).index
                separated_df[label] = separated_df[label].drop(random_indices)
            except:
                print(f"[WARNING] Specific data amount for {label} is too large or not provided, erasing all")
                separated_df.pop(label)
            pos += 1
        df = pd.concat(separated_df, ignore_index=True) # indices become ("class", index)

    print("[INFO] Size of df before cleaning: " + str(df.shape)) 
    df = clean_data(df, time_series)
    print("[INFO] Size of df after necessary cleaning: " + str(df.shape))

    # slice dataset into its respective parts
    x_num_df, x_sCov_df, x_rCov_df = slice_x(df)
    x_df = df.loc[:, 'speed':'rCov9']
    y_df = df.loc[:, 'Class':'Subtype']
        
    # display the amount of training examples in each class
    one_hot_df = one_hot_encode(y_df)
    print("\nAmount of training examples in each class:")
    for col in one_hot_df.columns:
        print(f"{col[0]}: {one_hot_df[col].sum()}")

    df.to_csv("current_dataset.csv", index=False)
    # x_num_df.to_csv(x_train_path[0], index=False)
    # x_sCov_df.to_csv(x_train_path[1], index=False)
    # x_rCov_df.to_csv(x_train_path[2], index=False)
    # y_df.to_csv(y_train_path, index=False)


def main():
    """
    Bring in user inputs and begin preprocessing
    """
    args = parse_args_preprocessing()

    if len(sys.argv) < 2:
        print("\nCall with --help or -h for more information\n")
        sys.exit(1)

    valid_run = False
    if args.data_dir:
        if not os.path.exists(args.data_dir):
            print("\nData Directory does not exist")
            sys.exit(1)
        valid_run = True
    if args.dataset:
        if not os.path.exists(args.dataset):
            print("\nComplete Data File path does not exist")
            sys.exit(1)
        valid_run = True

    # source data provided
    if valid_run:
        if args.add or args.erase:
            if args.add and args.erase:
                print("\n[ERROR] Cannot add and erase at the same time\n")
                sys.exit(1)
            if args.specific_data_type:
                if args.specific_data_amount:
                    for i in range(len(args.specific_data_type)):
                        if args.specific_data_type[i] not in specific_data_dict.keys():
                            print("Proper Specific Data Types not provided")
                            sys.exit(1)
                        try:
                            if float(args.specific_data_amount[i]) < 0:
                                print("Amount of specific data to extract cannot be negative")
                                sys.exit(1)
                        except:
                            continue
                    args.specific_data_amount = [int(i) for i in args.specific_data_amount]
            else:
                print("\nMust use -type if using -a or -e")
                sys.exit(1)

        # output file names
        y_train_path = "prepared_y_set.csv"
        x_train_path = ["prepared_x_set_num.csv", "prepared_x_set_sCov.csv", "prepared_x_set_rCov.csv"]

        print("[INFO] Input args processed...")
        # create training sets

        create_train_sets(args.data_dir, args.dataset, args.add, args.erase, 
                        args.specific_data_type, args.specific_data_amount,
                        x_train_path, y_train_path, args.time_series)
    else:
        print("\nMust use either data directory or dataset. Call -h for more information")

    
if __name__ == "__main__":
    main()                    
        