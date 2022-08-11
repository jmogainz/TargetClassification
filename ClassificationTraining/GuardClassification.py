"""
GuardClassification.py
-----------------------
    Driver for target classification model training
"""

# Imports
from GuardPreprocessing import create_train_sets, one_hot_encode, slice_x, time_series_split
from helpers import parse_args_classification
from models import dense_model, merge_model, cnn2D_model, GS_model, time_series_model 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import os
import sys
import seaborn as sns
import pandas as pd
import time
import datetime
import random
import numpy as np


global model_type, scaler, gs, acc, ts_steps


def dataset(ds):
    # check if dataset exists
    if os.path.exists(ds):
        x_df, y_df = create_train_sets(dataset=ds)
    else:
        print("\nDataset does not exist")
        sys.exit(1)

    return x_df, y_df

def preprocessor(x_df, y_df):
    global scaler
    train = {}; dev = {}; test = {}
    print("[INFO] Preprocessing data...")

    # classic data preprocessing procedure, 3 way split
    if model_type == 'ts':
        (x_train, x_temp, y_train, y_temp) = time_series_split(x_df, y_df, train_size=.7, 
                                                            ts_steps=ts_steps)
        (x_dev, x_test, y_dev, y_test) = time_series_split(x_temp, y_temp, train_size=.99,
                                                        ts_steps=ts_steps)
    else:
        (x_train, x_temp, y_train, y_temp) = train_test_split(x_df, y_df, 
                                                          train_size=0.9, random_state=42)
        (x_dev, x_test, y_dev, y_test) = train_test_split(x_temp, y_temp,
                                                        train_size=.5, random_state=42)

    # if ML algo, do not one-hot encode; else, NN, do one-hot encode
    if (model_type == 'knn' or model_type == 'gb' or model_type == 'rf' or
        model_type == 'nb'):
        y_train = y_train.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        y_dev = y_dev.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        y_test = y_test.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
    else:
        y_train = one_hot_encode(y_train)
        y_dev = one_hot_encode(y_dev)
        y_test = one_hot_encode(y_test)

    # slice data up into optional subsets 
    (x_train_num, x_train_scov, x_train_rcov) = slice_x(x_train)
    (x_dev_num, x_dev_scov, x_dev_rcov) = slice_x(x_dev)
    (x_test_num, x_test_scov, x_test_rcov) = slice_x(x_test)

    # join num and scov for num+scov subset dataframe
    x_train_num_scov = pd.concat([x_train_num, x_train_scov], axis=1)
    x_dev_num_scov = pd.concat([x_dev_num, x_dev_scov], axis=1)
    x_test_num_scov = pd.concat([x_test_num, x_test_scov], axis=1)

    # scaler has been imported, scale desired subset and return; we are only predicting
    if scaler:
        scaler_names = scaler['imported'].feature_names_in_
        possible_dfs = {'num_scov': x_test_num_scov, 
                        'scov': x_test_scov,
                        'all': x_test, 
                        'rcov': x_test_rcov}
        for df in possible_dfs:
            df_found = True
            for col in scaler_names:
                if col not in possible_dfs[df].columns:
                    df_found = False
                    break
            if df_found:
                test[f'x_test_{df}'] = scaler['imported'].transform(possible_dfs[df])
                test['y_test'] = y_test
                break

        return train, dev, test

    # building subsets for training models and predicting
    scaler['all'] = MinMaxScaler()
    x_train_all = scaler['all'].fit_transform(x_train)
    x_dev_all = scaler['all'].transform(x_dev)
    x_test_all = scaler['all'].transform(x_test)

    scaler['num'] = MinMaxScaler()
    x_train_num = scaler['num'].fit_transform(x_train_num)
    x_dev_num = scaler['num'].transform(x_dev_num)
    x_test_num = scaler['num'].transform(x_test_num)

    if model_type == 'merge':
        # normalize each row of dataframe to [0, 1]
        # we do not know enough about covariance matrices relate any sample to each other
        # there is also not a known maximum for a covariance data point (such as 255 for image)
        for i in range(len(x_train_scov)):
            x_train_scov.iloc[i] = x_train_scov.iloc[i] / x_train_scov.iloc[i].max()
        for i in range(len(x_dev_scov)):
            x_dev_scov.iloc[i] = x_dev_scov.iloc[i] / x_dev_scov.iloc[i].max()
        for i in range(len(x_test_scov)):
            x_test_scov.iloc[i] = x_test_scov.iloc[i] / x_test_scov.iloc[i].max()
        for i in range(len(x_train_rcov)):
            x_train_rcov.iloc[i] = x_train_rcov.iloc[i] / x_train_rcov.iloc[i].max()
        for i in range(len(x_dev_rcov)):
            x_dev_rcov.iloc[i] = x_dev_rcov.iloc[i] / x_dev_rcov.iloc[i].max()
        for i in range(len(x_test_rcov)):
            x_test_rcov.iloc[i] = x_test_rcov.iloc[i] / x_test_rcov.iloc[i].max()

        # must be converted to numpy array
        x_train_scov = x_train_scov.values.reshape(-1, 6, 6, 1)
        x_dev_scov = x_dev_scov.values.reshape(-1, 6, 6, 1)
        x_test_scov = x_test_scov.values.reshape(-1, 6, 6, 1)
        x_train_rcov = x_train_rcov.values.reshape(-1, 3, 3, 1)
        x_dev_rcov = x_dev_rcov.values.reshape(-1, 3, 3, 1)
        x_test_rcov = x_test_rcov.values.reshape(-1, 3, 3, 1)

        # dict of all data
        train = {'x_train_all': x_train_all, 'x_train_num': x_train_num, 'x_train_scov': x_train_scov,
                'x_train_rcov': x_train_rcov, 'y_train': y_train}
        dev = {'x_dev_all': x_dev_all, 'x_dev_num': x_dev_num, 'x_dev_scov': x_dev_scov, 'x_dev_rcov': 
                x_dev_rcov, 'y_dev': y_dev}
        test = {'x_test_all': x_test_all, 'x_test_num': x_test_num, 'x_test_scov': x_test_scov, 
                'x_test_rcov': x_test_rcov, 'y_test': y_test}

    else:
        scaler['num_scov'] = MinMaxScaler()
        x_train_num_scov = scaler['num_scov'].fit_transform(x_train_num_scov)
        x_dev_num_scov = scaler['num_scov'].transform(x_dev_num_scov)
        x_test_num_scov = scaler['num_scov'].transform(x_test_num_scov)

        scaler['scov'] = MinMaxScaler()
        x_train_scov = scaler['scov'].fit_transform(x_train_scov)
        x_dev_scov = scaler['scov'].transform(x_dev_scov)
        x_test_scov = scaler['scov'].transform(x_test_scov)

        scaler['rcov'] = MinMaxScaler()
        x_train_rcov = scaler['rcov'].fit_transform(x_train_rcov)
        x_dev_rcov = scaler['rcov'].transform(x_dev_rcov)
        x_test_rcov = scaler['rcov'].transform(x_test_rcov)

        train = {'x_train_num': x_train_num, 'x_train_scov': x_train_scov, 
                'x_train_rcov': x_train_rcov, 'y_train': y_train, 
                'x_train_num_scov': x_train_num_scov, 'x_train_all': x_train_all}
        dev = {'x_dev_num': x_dev_num, 'x_dev_scov': x_dev_scov, 'x_dev_rcov': x_dev_rcov, 
               'y_dev': y_dev, 'x_dev_num_scov': x_dev_num_scov, 'x_dev_all': x_dev_all}
        test = {'x_test_num': x_test_num, 'x_test_scov': x_test_scov, 'x_test_rcov': x_test_rcov,
                'y_test': y_test, 'x_test_num_scov': x_test_num_scov, 'x_test_all': x_test_all}

        if model_type == 'ts':
            print("[INFO] Performing time series reshaping...", end='', flush=True)
            train['x_train_all'] = train['x_train_all'].reshape(-1, ts_steps, 52)
            dev['x_dev_all'] = dev['x_dev_all'].reshape(-1, ts_steps, 52)
            test['x_test_all'] = test['x_test_all'].reshape(-1, ts_steps, 52)
            train['x_train_num_scov'] = train['x_train_num_scov'].reshape(-1, ts_steps, 43)
            dev['x_dev_num_scov'] = dev['x_dev_num_scov'].reshape(-1, ts_steps, 43)
            test['x_test_num_scov'] = test['x_test_num_scov'].reshape(-1, ts_steps, 43)
            train['y_train'] = y_train.iloc[::ts_steps]
            dev['y_dev'] = y_dev.iloc[::ts_steps]
            test['y_test'] = y_test.iloc[::ts_steps]
            print("done")


    return train, dev, test

def trainer(train, dev):
    print("[INFO] Training model...")

    if model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(train['x_train_num_scov'], train['y_train'])

        return model

    if model_type == 'nb':
        model = GaussianNB()
        model.fit(train['x_train_num_scov'], train['y_train'])

        return model

    if model_type == 'gb':
        if gs:
            model = XGBClassifier(n_jobs=6)
            n_estimators = [1000, 3500, 5000, 10000]
            max_depth = [3, 6, 9, 12, 15]
            learning_rate = [0.3, 0.7, 0.9]
            param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=3, verbose=1)
            grid_result = grid_search.fit(train['x_train_num_scov'], train['y_train'])
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            model = XGBClassifier(n_estimators=3500, max_depth=6, learning_rate=0.3, n_jobs=12)
            model.fit(train['x_train_num_scov'], train['y_train'])
            print(model)

        return model

    if model_type == 'rf':
        if gs:
            model = RandomForestClassifier(n_jobs=6, n_estimators=1000)
            max_depth = [26, 29, 32, 35, 38]
            criterion = ['entropy']
            min_samples_split = [2]
            max_features = [.2]
            param_grid = dict(max_depth=max_depth, criterion=criterion, 
                            min_samples_split=min_samples_split, max_features=max_features)
            grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=3, verbose=10)
            grid_result = grid_search.fit(train['x_train_num_scov'], train['y_train'])
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            model = RandomForestClassifier(criterion='entropy', max_depth=32, min_samples_split=2, max_features=.2, n_estimators=1000, n_jobs=12)
            model.fit(train['x_train_num_scov'], train['y_train'])

        return model

    if model_type == 'dense':
        if gs:
            seed = 7
            np.random.seed(seed)
            model = KerasClassifier(build_fn=GS_model, verbose=0)
            batch_size = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
            epochs = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 
                    325, 350, 375, 400]
            final_activ = ['softmax', 'sigmoid', 'tanh']
            optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
            activ = ['relu', LeakyReLU(alpha=0.1), 'softmax', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            weight_init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
            drop_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]       
            hl_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            hl_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            learn_rate = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]    
            decay = [0, .0005, .001, .002, .003, .004, .005, .006, .007, .008, .009, .01]
            momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
            grid_result = grid.fit(train['x_train_all'], train['y_train'])
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            model = dense_model(52)
            opt = Adam(learning_rate=0.001)
            es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
            es_2 = EarlyStopping(monitor='val_loss', mode='min', patience=20)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            history = model.fit(train['x_train_all'], train['y_train'], 
                                validation_data=(dev['x_dev_all'], dev['y_dev']), 
                                epochs=1000, batch_size=512, callbacks=[es, es_2])

    if model_type == 'merge':
        ##### for testing purposes #####

        """Merge Model"""
        model = merge_model()
        opt = Adam(learning_rate=0.001)
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
        es_2 = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # concatenate all x data
        history = model.fit([train['x_train_all'], train['x_train_scov']], train['y_train'], 
                   validation_data=([dev['x_dev_all'], dev['x_dev_scov']], dev['y_dev']), 
                   epochs=1000, batch_size=128, callbacks=[es, es_2])

    if model_type == 'ts':
        model = time_series_model(ts_steps, 43, 6) 
        opt = Adam(learning_rate=0.001)
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
        es_2 = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        history = model.fit(train['x_train_num_scov'], train['y_train'],
                     validation_data=(dev['x_dev_num_scov'], dev['y_dev']),
                        epochs=1000, batch_size=128, callbacks=[es, es_2])

    print(history.history.keys())
    # plot loss and accuracy for train and validation set on same graph
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
        
    return model

def predictor(model, test):
    global acc
    print("[INFO] Predicting...")

    # ML algos
    if (model_type == 'knn' or model_type == 'gb' or model_type == 'rf' or
        model_type == 'nb'):
        pred = model.predict(test['x_test_num_scov'])
        print(confusion_matrix(test['y_test'], pred))
        acc = accuracy_score(test['y_test'], pred)
        class_report = classification_report(test['y_test'], pred)
        print("[INFO] classification accuracy: ", acc)
        print("[INFO] classification report: ", class_report)

        return
    
    # Neural Networks
    if model_type == 'dense':
        pred = model.predict(test['x_test_all'])
    if model_type == 'merge':
        pred = model.predict([test['x_test_num'], test['x_test_scov']])
    if model_type == 'ts':
        pred = model.predict(test['x_test_num_scov'])

    y_test_np = test['y_test'].values

    # calculate accuracy of the model on the testing set
    # loop through row of y_dev_np and compare to pred
    correct = 0
    for i in range(len(y_test_np)):
        if np.argmax(pred[i]) == np.argmax(y_test_np[i]):
            correct += 1
    
    acc = correct / len(y_test_np)

    print("[INFO] classification accuracy: {:.2f}%".format(
        (acc) * 100.0))

def measure(model, test):
    print("[INFO] Measuring prediction performance...")
    prediction_performance = 0
    
    if (model_type == 'knn' or model_type == 'gb' or model_type == 'rf'
        or model_type == 'nb'):
        sample = random.randint(0, len(test['x_test_num_scov']))
        x_test_num_scov_one = test['x_test_num_scov'][sample][0:].reshape(1, -1)
        y_test_one = test['y_test'].values[sample]

        start = time.perf_counter()
        pred = model.predict(x_test_num_scov_one)
        prediction_performance = time.perf_counter() - start

        if isinstance(pred, np.ndarray):
            pred = int(pred)

    if model_type == 'dense':
        # predict random sample
        sample = random.randint(0, len(test['x_test_all']))
        x_test_all_one = test['x_test_all'][sample][0:].reshape(1, -1)
        y_test_one = test['y_test'].values[sample][0:]

        start = time.perf_counter()
        pred = model.predict(x_test_all_one)[0]
        prediction_performance = time.perf_counter() - start
        # print the type of pred
        print(type(pred))
        largest = np.argmax(pred)
        for idx in range(len(pred)):
            if idx == largest:
                pred[idx] = 1
            else:
                pred[idx] = 0

    if model_type == 'merge':
        # predict random sample
        sample = random.randint(0, len(test['x_test_all']))
        x_test_num_one = test['x_test_num'][sample][0:].reshape(1, -1)
        x_test_scov_one = test['x_test_scov'][sample][0:].reshape(1, -1)
        y_test_one = test['y_test'].values[sample][0:]

        start = time.perf_counter()
        pred = model.predict([x_test_num_one, x_test_scov_one])
        prediction_performance = time.perf_counter() - start

    if model_type == 'ts':
        # predict random sample
        sample = random.randint(0, len(test['x_test_num_scov']))
        x_test_num_scov_one = test['x_test_num_scov'][sample][0:].reshape(1, -1)
        y_test_one = test['y_test'].values[sample][0:]
        
        start = time.perf_counter()
        pred = model.predict(x_test_num_scov_one)
        prediction_performance = time.perf_counter() - start
    
    print("prediction performance:", prediction_performance)
    print("prediction:", pred)
    print("truth:", y_test_one)

def saver(model, ds, test):
    global acc
    print("[INFO] Serializing Model...")

    scaler_used = ''
    current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    acc = round(acc, 4) * 100 # round accuracy to 2 decimal places

    model_path = f"Models/{model_type}_model_{acc}_{current_dt}_ds_size_{len(ds)}_tested_with_{len(test['y_test'])}.h5"
    scaler_path = f"Models/{model_type}_scaler_{acc}_{current_dt}_ds_size_{len(ds)}_tested_with_{len(test['y_test'])}.save"

    if (model_type == 'knn' or model_type == 'gb' or model_type == 'rf' or 
        model_type == 'nb'):
        joblib.dump(model, model_path)
        scaler_used = 'num_scov'
    if model_type == 'dense':
        model.save(model_path)
        scaler_used = 'all'
    if model_type == 'merge':
        model.save(model_path)
        scaler_used = 'num'
    if model_type == 'ts':
        model.save(model_path)
        scaler_used = 'num_scov'

    # save scaler model 
    joblib.dump(scaler[scaler_used], scaler_path)

def main():
    global model_type, gs, scaler, ts_steps

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

    model_loaded = False
    scaler = {}
    if os.path.exists(loaded_model) and os.path.exists(loaded_scaler):
        print("[INFO] Loading trained model...")
        if 'knn' in loaded_model:
            model = joblib.load(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'knn'
        elif 'nb' in loaded_model:
            model = joblib.load(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'nb'
        elif 'rf' in loaded_model:
            model = joblib.load(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'rf'
        elif 'gb' in loaded_model:
            model = joblib.load(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'gb'
        elif 'dense' in loaded_model:
            model = load_model(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'dense'
        elif 'merge' in loaded_model:
            model = load_model(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'merge'
        elif 'ts' in loaded_model:
            model = joblib.load(loaded_model)
            scaler['imported'] = joblib.load(loaded_scaler)
            model_loaded = True
            model_type = 'ts'

    x_df, y_df = dataset(args.dataset)
    train, dev, test = preprocessor(x_df, y_df) 
    
    if model_loaded == False:
        model = trainer(train, dev)
        predictor(model, test)
        saver(model, x_df, test)
    elif not perf_check:
        predictor(model, test)
    else:
        measure(model, test)


if __name__ == "__main__":
    main()

