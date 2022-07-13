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
import matplotlib.pyplot as plt


# globals
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


def convert_to_df(csv_file, df_container):
    """
    Create labels for data
    """
    # cleaning and restructuring data
    df = pd.read_csv(csv_file, delimiter=",")
    df.columns = features_combined

    df_container.append(df)

def sep_data(df, class_container):
    # filter dataframe by class
    class_frames = {}
    for label in class_container:
        df_class = df[df["Class"] == dict_specific_data[label][0]]
        df_class = df_class[df_class["Subclass"] == dict_specific_data[label][1]]
        df_class = df_class[df_class["Type"] == dict_specific_data[label][2]]
        df_class = df_class[df_class["Subtype"] == dict_specific_data[label][3]]
        class_frames[label] = df_class

    return class_frames

def slice_x(container):
    # verify container is dataframe
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
            df = df[(df[col] <= mean+(4*sd))] #removes top 1% of data
            df = df[(df[col] >= mean-(4*sd))] #removes bottom 1% of data

    return df

def clean_data(df):
    """
    Clean data
    """
    # exclude vertical velocity from the data, it is falsy calculated when afsim reports duplicates
    features_subset = features_combined[:2] + features_combined[3:]
    df_clean = df.drop_duplicates(inplace=False, keep='first', subset=features_subset)

    # remove outliers
    # separated_df = sep_data(df_clean, dict_specific_data)
#     for label in separated_df:
#         print(separated_df[label].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))
#         separated_df[label] = remove_outliers(separated_df[label])
#         print(separated_df[label].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))
    # df_clean = pd.concat(separated_df)

    # remove duplicates
    

    # shuffle the data during preparation phase, not model development phase
    if __name__ == "__main__":
        df_clean = df_clean.sample(frac=1).reset_index(drop=True)

    return df_clean

def one_hot_encode(df):
    """
    One hot encode the dataframe
    """
    df_one_hot = df.copy(deep=True)

    # join each column of each row 
    df_one_hot = df_one_hot.apply(lambda x: ''.join(x.values.astype(str)), axis=1)

    # reshape for single feature
    df_one_hot = df_one_hot.values.reshape(-1, 1)

    # one hot encode the data
    enc = OneHotEncoder(sparse=False)
    enc.fit(df_one_hot)
    df_one_hot = enc.transform(df_one_hot)
    df_one_hot = pd.DataFrame(df_one_hot, columns=enc.categories_)
    
    return df_one_hot

def retrieve(data_dir, df_container):
    if data_dir:
        remove_version_1_and_3(data_dir)
        count = 0
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                convert_to_df(data_dir + "/" + file, df_container)
                count += 1

def create_train_sets(general_data='', specific_data='', specific_data_type=[],
                      specific_data_amount=[], x_train_path=[], y_train_path='',
                      complete_csv=''):
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
    Returns:
        x_df: dataframe of x data
        y_df: dataframe of labels
    """
    spec_df = [pd.DataFrame()]
    gen_df = [pd.DataFrame()]
    
    gen_thread = threading.Thread(target=retrieve, args=(general_data, gen_df))
    spec_thread = threading.Thread(target=retrieve, args=(specific_data, spec_df))
    gen_thread.start(); spec_thread.start()
    gen_thread.join(); spec_thread.join()
    spec_df = pd.concat(spec_df); gen_df = pd.concat(gen_df)

    if specific_data:
        separated_df = sep_data(spec_df, specific_data_type)
        pos = 0
        for label in separated_df:
            try:
                separated_df[label] = separated_df[label].sample(n=specific_data_amount[pos])
            except:
                print("\nError: specific data amount is too large or not provided, taking as much as possible\n")
            pos += 1
        spec_df = pd.concat(separated_df)

    df = pd.concat([gen_df, spec_df])

    if complete_csv:
        df_complete = pd.read_csv(complete_csv, delimiter=",")
        df_complete.columns = features_combined
        df = pd.concat([df, df_complete])

    print("Size of df: " + str(df.shape)) 
    df = clean_data(df)
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
                        help="list of amounts of data to extract from specific_data_dir of specific_data_type> (correlates with type)")
    parser.add_argument("-c","--complete_csv", type=str, default='',
                        help="path to previous csv file output from this application")
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

    # create training sets
    if valid_data:
        create_train_sets(args.general_data_dir, args.specific_data_dir, args.specific_data_type, 
                          args.specific_data_amount, x_train_path, y_train_path, args.complete_csv)

    
if __name__ == "__main__":
    main()                    
        