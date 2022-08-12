"""
GuardClassification.py
-----------------------
    Driver for target classification model training
"""

# Imports
from GuardPreprocessing import create_train_sets
from helpers import parse_args_classification
from models import NN_Dense_Model, NN_Merged_Model, NN_Time_Series_Model, KNN_Model, RF_Model, GB_Model
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os
import sys
import pandas as pd
import time
import datetime
import random
import numpy as np

stringToModel = {'dense': NN_Dense_Model, 'merge': NN_Merged_Model, 'ts': NN_Time_Series_Model,
                 'knn': KNN_Model, 'gb': GB_Model, 'rf': RF_Model}

def dataset(ds):
    # check if dataset exists
    if os.path.exists(ds):
        x_df, y_df = create_train_sets(dataset=ds)
    else:
        print("\nDataset does not exist")
        sys.exit(1)

    return x_df, y_df

def loadModelFromDisk(model_path, scaler_path):
    model_loaded = False
    model = None
    scaler = None
    model_type = ''
    if model_path and scaler_path:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print("[INFO] Loading trained model...")
            if 'knn' in model_path:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                model_type = 'knn'
            elif 'rf' in model_path:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                model_type = 'rf'
            elif 'gb' in model_path:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                model_type = 'gb'
            elif 'dense' in model_path:
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                model_type = 'dense'
            elif 'merge' in model_path:
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                model_type = 'merge'
            elif 'ts' in model_path:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                model_type = 'ts'
        else:
            print("[ERROR] Model or scaler not found")
            sys.exit(1)

    return model_loaded, model, model_type, scaler

def main():
    args = parse_args_classification()

    ts_steps = args.ts_steps
    gs = args.grid_search
    model_type = args.model
    loaded_model = args.load_model
    loaded_scaler = args.load_scaler
    perf_check = args.perf_check

    if model_type == 'ts' and ts_steps <= 0:
        print("[ERROR] ts_steps must be greater than 0, if using ts data")
        sys.exit(1)

    load_success = False
    if loaded_model and loaded_scaler:
        load_success, model, model_type, scaler = loadModelFromDisk(loaded_model, loaded_scaler)

    x_df, y_df = dataset(args.dataset)

    if not load_success:
        if model_type == 'ts':
            model = stringToModel[model_type](x_df, y_df, ts_steps=ts_steps)
        else:
            model = stringToModel[model_type](x_df, y_df)
        print("[INFO] Preprocessing data...")
        model.preprocess()
        print("[INFO] Training model...")
        model.train(gs=gs)
        print("[INFO] Evaluating model...")
        model.evaluate()
        print("[INFO] Saving model...")
        model.save()
    else:
        if model_type == 'ts':
            model = stringToModel[model_type](x_df, y_df, ts_steps=ts_steps,
                                                model=model, scaler=scaler)
        else:
            model = stringToModel[model_type](x_df, y_df, model=model, scaler=scaler)
        print("[INFO] Preprocessing data...")
        model.preprocess()
        print("[INFO] Evaluating model...")
        model.evaluate()

    if perf_check:
        model.measure_performance()

if __name__ == "__main__":
    main()

