"""
Remove all files in directory that are version 3 and version 1
"""
import os
import pandas as pd
import argparse

guardian_dir = "Missile_Capture-Classification-Guard_Missile"

def remove_version_1_and_3(guardian_dir):
    for file in os.listdir(os.path.join(os.getcwd(), guardian_dir)):
        if file.endswith("3.csv"):
            os.remove(os.path.join(os.getcwd(), guardian_dir, file))
        if file.endswith("1.csv"):
            os.remove(os.path.join(os.getcwd(), guardian_dir, file))

def parse_args_preprocessing():
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument("-dir","--data_dir", type=str, default='',
                        help="directory for collecting all data returned from guardian <dir>")
    parser.add_argument("-e", "--erase", type=bool, default=False,
                        help="erase type and amount from complete_csv")
    parser.add_argument("-a", "--add", type=bool, default=False,
                        help="add type and amount to complete_csv")
    parser.add_argument("-type","--specific_data_type", nargs='+', default=[],
                        help="list of types of data to extract from specific_data_dir <class (i.e. 5200)>")
    parser.add_argument("-amount", "--specific_data_amount", nargs='+', default=[],
                        help=("list of amounts of data to extract from specific_data_dir of specific_data_type>(correlates with type)." 
                        "If time series is set, this number must be multiple of time series steps." 
                        "If not, it will be rounded down to the closest multiple of time series steps."))
    parser.add_argument("-d","--dataset", type=str, default='',
                        help="path to previous csv file output from this application")
    parser.add_argument("-ts", "--time_series", type=int, default=0,
                        help="number of time series steps; if set, it will enable time series data construction.")
    
    args = parser.parse_args()
    return args

def parse_args_classification():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, required=True,
        help="path to input dataset from guardian")
    ap.add_argument("-m", "--model", type=str, default="dense",
        help="type of model to train (merge, dense, knn, gb, rf, nb, ts)")
    ap.add_argument("-lm", "--load_model", type=str, default="",
        help="path to load model from")
    ap.add_argument("-ls", "--load_scaler", type=str, default="",
        help="path to load scaler from")
    ap.add_argument("-p", "--perf_check", type=bool, default=False,
        help="make prediction only (true or false)")
    ap.add_argument("-gs", "--grid_search", type=bool, default=False,
        help="grid search (true or false)") 
    ap.add_argument("-tss", "--ts_steps", type=int, default=0,
        help="number of time steps per series to use on ts data")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    remove_version_1_and_3(guardian_dir)