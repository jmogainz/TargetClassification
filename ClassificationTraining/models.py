"""
Models
"""
# Imports
from GuardPreprocessing import *
from DLArchitectures import dense_model, merge_model, GS_model, time_series_model 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import time
import datetime
import random
import numpy as np

class ML_Model:
    def __init__():
        pass

    def display_summary(self, confusion, acc, report):
        print("[INFO] confusion matrix:\n", confusion)
        print("[INFO] classification accuracy: ", acc)
        print("[INFO] classification report: ", report)

    def display_performance(self, perf, pred, truth):
        print("prediction performance:", perf)
        print("prediction:", pred)
        print("truth:", truth)

class GB_Model(ML_Model):
    def __init__(self, x_df, y_df, scaler=MinMaxScaler(), model=XGBClassifier(n_estimators=3500, 
                                                                              max_depth=6, learning_rate=0.3, 
                                                                              n_jobs=12)):
        (self.x_train, x_temp, self.y_train, y_temp) = train_test_split(x_df, y_df, 
                                                          train_size=0.9, random_state=42)
        (self.x_dev, self.x_test, self.y_dev, self.y_test) = train_test_split(x_temp, y_temp,
                                                        train_size=.5, random_state=42)
        self.scaler = scaler
        self.model = model

    def preprocess(self):
        self.y_train = self.y_train.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        self.y_dev = self.y_dev.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        self.y_test = self.y_test.apply(lambda x: ''.join(x.values.astype(str)), axis=1)

        # eliminate rcov
        self.x_train = self.x_train.iloc[:, :-9]
        self.x_dev = self.x_dev.iloc[:, :-9]
        self.x_test = self.x_test.iloc[:, :-9]

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_dev = self.scaler.transform(self.x_dev)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self, gs=False):
        if gs:
            self.model = XGBClassifier(n_jobs=6)
            n_estimators = [1000, 3500, 5000, 10000]
            max_depth = [3, 6, 9, 12, 15]
            learning_rate = [0.3, 0.7, 0.9]
            param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=3, verbose=1)
            grid_result = grid_search.fit(self.x_train, self.y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        pred = self.model.predict(self.x_test)
        self.confusion = confusion_matrix(self.y_test, pred)
        self.acc = accuracy_score(self.y_test, pred)
        self.class_report = classification_report(self.y_test, pred)
        self.display_summary(self.confusion, self.acc, self.class_report)

    def measure_performance(self):
        sample = random.randint(0, len(self.x_train))
        x_train_one = self.x_test[sample][0:].reshape(1, -1)
        self.y_test_one = self.y_test.values[sample]

        start = time.perf_counter()
        self.pred = int(self.model.predict(x_train_one))
        self.prediction_performance = time.perf_counter() - start
        self.display_performance(self.prediction_performance, self.pred, self.y_test_one)

    def save(self):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        acc = round(self.acc, 4) * 100 # round accuracy to 2 decimal places

        model_path = f"Models/gb_model_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.h5"
        scaler_path = f"Models/gb_scaler_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.save"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)


class KNN_Model(ML_Model):
    def __init__(self, x_df, y_df, scaler=MinMaxScaler(), model=KNeighborsClassifier(n_neighbors=1)):
        (self.x_train, x_temp, self.y_train, y_temp) = train_test_split(x_df, y_df, 
                                                          train_size=0.9, random_state=42)
        (self.x_dev, self.x_test, self.y_dev, self.y_test) = train_test_split(x_temp, y_temp,
                                                        train_size=.5, random_state=42)
        self.scaler = scaler
        self.model = model

    def preprocess(self):
        self.y_train = self.y_train.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        self.y_dev = self.y_dev.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        self.y_test = self.y_test.apply(lambda x: ''.join(x.values.astype(str)), axis=1)

        # eliminate rcov
        self.x_train = self.x_train.iloc[:, :-9]
        self.x_dev = self.x_dev.iloc[:, :-9]
        self.x_test = self.x_test.iloc[:, :-9]

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_dev = self.scaler.transform(self.x_dev)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self, gs=False):
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        pred = self.model.predict(self.x_test)
        self.confusion = confusion_matrix(self.y_test, pred)
        self.acc = accuracy_score(self.y_test, pred)
        self.class_report = classification_report(self.y_test, pred)
        self.display_summary(self.confusion, self.acc, self.class_report)

    def measure_performance(self):
        sample = random.randint(0, len(self.x_train))
        x_train_one = self.x_test[sample][0:].reshape(1, -1)
        self.y_test_one = self.y_test.values[sample]

        start = time.perf_counter()
        self.pred = int(self.model.predict(x_train_one))
        self.prediction_performance = time.perf_counter() - start
        self.display_performance(self.prediction_performance, self.pred, self.y_test_one)

    def save(self):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        acc = round(self.acc, 4) * 100 # round accuracy to 2 decimal places

        model_path = f"Models/knn_model_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.h5"
        scaler_path = f"Models/knn_scaler_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.save"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)


class RF_Model(ML_Model):
    def __init__(self, x_df, y_df, scaler=MinMaxScaler(), model=RandomForestClassifier(criterion='entropy', 
                                                                max_depth=32, min_samples_split=2,
                                                                max_features=.2, n_estimators=1000, n_jobs=12)):
        (self.x_train, x_temp, self.y_train, y_temp) = train_test_split(x_df, y_df, 
                                                          train_size=0.9, random_state=42)
        (self.x_dev, self.x_test, self.y_dev, self.y_test) = train_test_split(x_temp, y_temp,
                                                        train_size=.5, random_state=42)
        self.scaler = scaler
        self.model = model

    def preprocess(self):
        self.y_train = self.y_train.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        self.y_dev = self.y_dev.apply(lambda x: ''.join(x.values.astype(str)), axis=1)
        self.y_test = self.y_test.apply(lambda x: ''.join(x.values.astype(str)), axis=1)

        # eliminate rcov
        self.x_train = self.x_train.iloc[:, :-9]
        self.x_dev = self.x_dev.iloc[:, :-9]
        self.x_test = self.x_test.iloc[:, :-9]

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_dev = self.scaler.transform(self.x_dev)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self, gs=False):
        if gs:
            self.model = RandomForestClassifier(n_jobs=6, n_estimators=1000)
            max_depth = [26, 29, 32, 35, 38]
            criterion = ['entropy']
            min_samples_split = [2]
            max_features = [.2]
            param_grid = dict(max_depth=max_depth, criterion=criterion, 
                            min_samples_split=min_samples_split, max_features=max_features)
            grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=3, verbose=10)
            grid_result = grid_search.fit(self.x_train, self.y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        pred = self.model.predict(self.x_test)
        self.confusion = confusion_matrix(self.y_test, pred)
        self.acc = accuracy_score(self.y_test, pred)
        self.class_report = classification_report(self.y_test, pred)
        self.display_summary(self.confusion, self.acc, self.class_report)

    def measure_performance(self):
        sample = random.randint(0, len(self.x_train))
        x_train_one = self.x_test[sample][0:].reshape(1, -1)
        self.y_test_one = self.y_test.values[sample]

        start = time.perf_counter()
        self.pred = int(self.model.predict(x_train_one))
        self.prediction_performance = time.perf_counter() - start
        self.display_performance(self.prediction_performance, self.pred, self.y_test_one)

    def save(self):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        acc = round(self.acc, 4) * 100 # round accuracy to 2 decimal places

        model_path = f"Models/rf_model_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.h5"
        scaler_path = f"Models/rf_scaler_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.save"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

class DL_Model:
    def __init__():
        pass

    def display_history(self, history):
        print(history.history.keys())
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

    def display_performance(self, perf, pred, truth):
        print("prediction performance:", perf)
        print("prediction:", pred)
        print("truth:", truth)

class NN_Dense_Model(DL_Model):
    def __init__(self, x_df, y_df, scaler=MinMaxScaler(), model=dense_model(52)):
        (self.x_train, x_temp, self.y_train, y_temp) = train_test_split(x_df, y_df, 
                                                          train_size=0.9, random_state=42)
        (self.x_dev, self.x_test, self.y_dev, self.y_test) = train_test_split(x_temp, y_temp,
                                                        train_size=.5, random_state=42)
        self.scaler = scaler
        self.model = model

    def preprocess(self):
        self.y_train = one_hot_encode(self.y_train)
        self.y_dev = one_hot_encode(self.y_dev)
        self.y_test = one_hot_encode(self.y_test)

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_dev = self.scaler.transform(self.x_dev)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self, gs=False):
        if gs:
            seed = 7
            np.random.seed(seed)
            self.model = KerasClassifier(build_fn=GS_model, verbose=0)
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
            grid = GridSearchCV(estimator=self.model, param_grid=param_grid, n_jobs=1, cv=3)
            grid_result = grid.fit(self.x_train, self.y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            opt = Adam(learning_rate=0.001)
            es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
            es_2 = EarlyStopping(monitor='val_loss', mode='min', patience=20)
            self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            self.history = self.model.fit(self.x_train, self.y_train, 
                                validation_data=(self.x_dev, self.y_dev), 
                                epochs=1000, batch_size=512, callbacks=[es, es_2])
        self.display_history(self.history)

    def evaluate(self):
        pred = self.model.predict(self.x_test)
        y_test_np = self.y_test.values

        correct = 0
        for i in range(len(y_test_np)):
            if np.argmax(pred[i]) == np.argmax(y_test_np[i]):
                correct += 1
    
        self.acc = correct / len(y_test_np)
        print("[INFO] accuracy: {:.2f}%".format(self.acc * 100))

    def measure_performance(self):
        sample = random.randint(0, len(self.x_test))
        x_test_one = self.x_test[sample][0:].reshape(1, -1)
        self.y_test_one = self.y_test.values[sample][0:]

        start = time.perf_counter()
        self.pred = self.model.predict(x_test_one)[0]
        self.prediction_performance = time.perf_counter() - start

        largest = np.argmax(self.pred)
        for idx in range(len(self.pred)):
            if idx == largest:
                self.pred[idx] = 1
            else:
                self.pred[idx] = 0

        self.display_performance(self.prediction_performance, self.pred, self.y_test_one)

    def save(self):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        acc = round(self.acc, 4) * 100 # round accuracy to 2 decimal places

        model_path = f"Models/dense_model_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.h5"
        scaler_path = f"Models/dense_scaler_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.save"

        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)


class NN_Merged_Model(DL_Model):
    def __init__(self, x_df, y_df, scaler=MinMaxScaler(), model=merge_model()):
        (self.x_train, x_temp, self.y_train, y_temp) = train_test_split(x_df, y_df, 
                                                          train_size=0.9, random_state=42)
        (self.x_dev, self.x_test, self.y_dev, self.y_test) = train_test_split(x_temp, y_temp,
                                                        train_size=.5, random_state=42)
        self.scaler = scaler
        self.model = model

    def preprocess(self):
        self.y_train = one_hot_encode(self.y_train)
        self.y_dev = one_hot_encode(self.y_dev)
        self.y_test = one_hot_encode(self.y_test)

        self.x_train_scov = self.x_train[features_combined[7:43]].copy(deep=True)
        self.x_dev_scov = self.x_dev[features_combined[7:43]].copy(deep=True)
        self.x_test_scov = self.x_test[features_combined[7:43]].copy(deep=True)

        # normalize relative to each sample (used in cnn)
        for i in range(len(self.x_train_scov)):
            self.x_train_scov.iloc[i] = self.x_train_scov.iloc[i] / self.x_train_scov.iloc[i].max()
        for i in range(len(self.x_dev_scov)):
            self.x_dev_scov.iloc[i] = self.x_dev_scov.iloc[i] / self.x_dev_scov.iloc[i].max()
        for i in range(len(self.x_test_scov)):
            self.x_test_scov.iloc[i] = self.x_test_scov.iloc[i] / self.x_test_scov.iloc[i].max()

        # must be converted to numpy array
        self.x_train_scov = self.x_train_scov.values.reshape(-1, 6, 6, 1)
        self.x_dev_scov = self.x_dev_scov.values.reshape(-1, 6, 6, 1)
        self.x_test_scov = self.x_test_scov.values.reshape(-1, 6, 6, 1)

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_dev = self.scaler.transform(self.x_dev)
        self.x_test = self.scaler.transform(self.x_test)

    def train(self, gs=False):
        opt = Adam(learning_rate=0.001)
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
        es_2 = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # concatenate all x data
        self.history = self.model.fit([self.x_train, self.x_train_scov], self.y_train, 
                   validation_data=([self.x_dev, self.x_dev_scov], self.y_dev), 
                   epochs=1000, batch_size=128, callbacks=[es, es_2])
        self.display_history(self.history)

    def evaluate(self):
        pred = self.model.predict([self.x_test, self.x_test_scov])
        y_test_np = self.y_test.values

        correct = 0
        for i in range(len(y_test_np)):
            if np.argmax(pred[i]) == np.argmax(y_test_np[i]):
                correct += 1
    
        self.acc = correct / len(y_test_np)
        print("[INFO] accuracy: {:.2f}%".format(self.acc * 100))

    # UNTESTED
    def measure_performance(self):
        sample = random.randint(0, len(self.x_test))
        x_test_one = self.x_test[sample][0:].reshape(1, -1)
        x_test_scov_one = self.x_test_scov[sample][0:].reshape(1, 6, 6, 1)
        self.y_test_one = self.y_test.values[sample][0:]

        start = time.perf_counter()
        self.pred = self.model.predict([x_test_one, x_test_scov_one])[0]
        self.prediction_performance = time.perf_counter() - start

        largest = np.argmax(self.pred)
        for idx in range(len(self.pred)):
            if idx == largest:
                self.pred[idx] = 1
            else:
                self.pred[idx] = 0

        self.display_performance(self.prediction_performance, self.pred, self.y_test_one)

    def save(self):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        acc = round(self.acc, 4) * 100 # round accuracy to 2 decimal places

        model_path = f"Models/merged_model_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.h5"
        scaler_path = f"Models/merged_scaler_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.save"

        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)


class NN_Time_Series_Model(DL_Model):
    def __init__(self, x_df, y_df, scaler=MinMaxScaler(), model=None, ts_steps=4):
        self.ts_steps = ts_steps

        (self.x_train, x_temp, self.y_train, y_temp) = time_series_split(x_df, y_df, 
                                                          train_size=0.9, ts_steps=self.ts_steps)
        (self.x_dev, self.x_test, self.y_dev, self.y_test) = time_series_split(x_temp, y_temp,
                                                        train_size=.5, ts_steps=self.ts_steps)
        self.scaler = scaler
        self.model = model

    def preprocess(self):
        self.y_train = one_hot_encode(self.y_train)
        self.y_dev = one_hot_encode(self.y_dev)
        self.y_test = one_hot_encode(self.y_test)

        # eliminate rcov
        self.x_train = self.x_train.iloc[:, :-9]
        self.x_dev = self.x_dev.iloc[:, :-9]
        self.x_test = self.x_test.iloc[:, :-9]

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_dev = self.scaler.transform(self.x_dev)
        self.x_test = self.scaler.transform(self.x_test)

        self.x_train = self.x_train.reshape(-1, self.ts_steps, 43)
        self.x_dev = self.x_dev.reshape(-1, self.ts_steps, 43)
        self.x_test = self.x_test.reshape(-1, self.ts_steps, 43)
        self.y_train = self.y_train.iloc[::self.ts_steps]
        self.y_dev = self.y_dev.iloc[::self.ts_steps]
        self.y_test = self.y_test.iloc[::self.ts_steps]

    def train(self, gs=False):
        model = time_series_model(self.ts_steps, 43, 6) 
        opt = Adam(learning_rate=0.001)
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
        es_2 = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.history = model.fit(self.x_train, self.y_train,
                     validation_data=(self.x_dev, self.y_dev),
                        epochs=1000, batch_size=128, callbacks=[es, es_2])
        self.display_history(self.history)

    def evaluate(self):
        pred = self.model.predict(self.x_test)
        y_test_np = self.y_test.values

        correct = 0
        for i in range(len(y_test_np)):
            if np.argmax(pred[i]) == np.argmax(y_test_np[i]):
                correct += 1
    
        self.acc = correct / len(y_test_np)
        print("[INFO] accuracy: {:.2f}%".format(self.acc * 100))

    # UNTESTED
    def measure_performance(self):
        sample = random.randint(0, len(self.x_test))
        x_test_one = self.x_test[sample][0:].reshape(1, -1)
        self.y_test_one = self.y_test.values[sample][0:]

        start = time.perf_counter()
        self.pred = self.model.predict(x_test_one)[0]
        self.prediction_performance = time.perf_counter() - start

        largest = np.argmax(self.pred)
        for idx in range(len(self.pred)):
            if idx == largest:
                self.pred[idx] = 1
            else:
                self.pred[idx] = 0

        self.display_performance(self.prediction_performance, self.pred, self.y_test_one)

    def save(self):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        acc = round(self.acc, 4) * 100 # round accuracy to 2 decimal places

        model_path = f"Models/ts_model_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.h5"
        scaler_path = f"Models/ts_scaler_{acc}_{current_dt}_ds_size_{len(self.x_train)}_tested_with_{len(self.y_test)}.save"

        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)


