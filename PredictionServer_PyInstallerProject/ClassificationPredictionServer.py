"""
ClassificationPredictionServer.py
--------------------------------
    Server used solely to predict the class of a Guardian target.

    Possible Classes:
        5200 - MMB
        2200 - MSL
        1112 - Slow UAV
        1122 - SRecon UAV
        1111 - Fast UAV
        1113 - MQ9R UAV

    Executable: ClassificationPredictionServer.exe
"""

import os
import socket
import pika
import time
from yaml import safe_load as yaml_load
import json
from tensorflow.keras.models import load_model
import sklearn
import joblib
import pickle
import numpy as np
from enum import Enum
from os.path import exists, join, abspath
import sys
from tkinter import Tk
from tkinter import filedialog as fd
import pandas as pd


# simple class to convert yaml and json files to pythonic objects
class RecursiveNamespace:
  @staticmethod
  def map_entry(entry):
    if isinstance(entry, dict):
        return RecursiveNamespace(**entry)
    return entry
  def __init__(self, **kwargs):
    for key, val in kwargs.items():
        if type(val) == dict:
            setattr(self, key, RecursiveNamespace(**val))
        elif type(val) == list:
            setattr(self, key, list(map(self.map_entry, val)))
        else:
            setattr(self, key, val)


class COMMS_TYPE(Enum):
    UDP = 1
    RABBITMQ = 2


class ClassificationPredictionServer:
    def __init__(self):
        self.__buffer_size = 4096
        self.__comms_config = None
        self.__my_socket = None
        self.__udp_host = "127.0.0.1"
        self.__udp_port = 12002
        self.__rabbitMQ_Connection = None
        self.__rabbitMQ_Channel = None
        self.__rabbitMQ_Queue = None
        self.__user_name = "guest"
        self.__password = 'guest'
        self.__rabbitmq_address = 'localhost'
        self.__rabbitmq_port = 5672
        self.__exchange_name = "sim-exchange"
        self.__routing_key_prediction_server = 'classificationpredictionserver.cmd.primary'
        self.__routing_key_requestor = 'classification.cmd.primary'
        self.__communication_type = COMMS_TYPE.UDP
        self.__classification_by_index = {0: 5200, 1: 2200, 2: 1112, 3: 1122, 4: 1111, 5: 1113}

        self.__debugging = False
        self.__track_TimeToPredict = False
        self.__total_Time = 0
        self.__total_Time_Count = 0

        self.__ml_config = None
        self.__normalize_model = None
        self.__dimensionality_reduction_model = None
        self.__prediction_model = None
        self.__feature_list = None
        self.__target_list = None

    def loadConfig(self, config = None):
        if config is not None and config.comms_configuration is not None:
            self.__comms_config = config.comms_configuration
            if self.__comms_config.protocol_choice and self.__comms_config.protocol_choice == "UDP":
                self.__communication_type = COMMS_TYPE.UDP
                if self.__comms_config.udp_host and self.__comms_config.udp_port:
                    self.__udp_host = self.__comms_config.udp_host
                    self.__udp_port = self.__comms_config.udp_port
            if self.__comms_config.protocol_choice and self.__comms_config.protocol_choice == "CDXMESH":
                self.__communication_type = COMMS_TYPE.RABBITMQ
                if self.__comms_config.rabbitmq_user_name and self.__comms_config.rabbitmq_password and self.__comms_config.rabbitmq_address and \
                        self.__comms_config.rabbitmq_port and self.__comms_config.exchange_name and \
                        self.__comms_config.routing_key_prediction_server and self.__comms_config.routing_key_requestor:
                    self.__user_name = self.__comms_config.rabbitmq_user_name
                    self.__password = self.__comms_config.rabbitmq_password
                    self.__rabbitmq_address = self.__comms_config.rabbitmq_address
                    self.__rabbitmq_port = self.__comms_config.rabbitmq_port
                    self.__exchange_name = self.__comms_config.exchange_name
                    self.__routing_key_prediction_server = self.__comms_config.routing_key_prediction_server
                    self.__routing_key_requestor = self.__comms_config.routing_key_requestor
        if config is not None and config.machine_learning_configuration:
            self.__ml_config = config.machine_learning_configuration
            if self.__ml_config.feature_names_list and self.__ml_config.target_names_list:
                self.__feature_list = self.__ml_config.feature_names_list
                self.__target_list = self.__ml_config.target_names_list
            else:
                print("Warning - Missing Features and Target List in Configuration")
            if self.__ml_config.normalize_model_loading:
                temp_config = self.__ml_config.normalize_model_loading
                if temp_config.load_normalization_model_flag and temp_config.input_path and temp_config.input_file_name:
                    if temp_config.load_normalization_model_flag:
                        self.__normalize_model = self.__loadTrainedModel(os.path.join(temp_config.input_path, temp_config.input_file_name))
            if self.__ml_config.dimensionality_reduction_model_loading:
                temp_config = self.__ml_config.dimensionality_reduction_model_loading
                if temp_config.load_dimensionality_reduced_model_flag and temp_config.input_path and temp_config.input_file_name:
                    if temp_config.load_dimensionality_reduced_model_flag:
                        self.__dimensionality_reduction_model = self.__loadTrainedModel(os.path.join(temp_config.input_path, temp_config.input_file_name))
            if self.__ml_config.prediction_model_loading:
                temp_config = self.__ml_config.prediction_model_loading
                if temp_config.load_prediction_model_flag and temp_config.input_path and temp_config.input_file_name:
                    if temp_config.load_prediction_model_flag:
                        self.__prediction_model = self.__loadTrainedModel(os.path.join(temp_config.input_path, temp_config.input_file_name))
            else:
                print("Warning - No Prediction Model Provided in Configuration")
        else:
            print("Error with configuration file")
        print("Finished Loading Configuration... Starting Service")

    def __loadTrainedModel(self, file_name):
        try:
            return joblib.load(file_name)
        except:
            try:
                return pickle.load(open(file_name, 'rb'))
            except:
                try:
                    return load_model(file_name)
                except:
                    print("Error with Loading Model " + file_name)
                    return None

    def __extractFeatures(self, received_message_decoded):
        features = pd.DataFrame() # ensures that the features are in the same order as the training data
        for feature in self.__feature_list:
            if feature in received_message_decoded['Parameters']:
                features.loc[0, feature] = received_message_decoded['Parameters'][feature]
        return features

    def __predict(self, features):
        from copy import deepcopy
        features_predict = deepcopy(features)
        if self.__normalize_model is not None:
            features_predict = self.__normalize_model.transform(features_predict)
            features_predict = features_predict.reshape(1, -1)
        else:
            features_predict = features_predict.values.reshape(1, -1) # still a dataframe
        if self.__dimensionality_reduction_model is not None:
            features_predict = self.__dimensionality_reduction_model.transform(features_predict)
        if self.__prediction_model is not None:
            return self.__prediction_model.predict(features_predict)
        else:
            return None

    def __createResponse(self, received_message_decoded, prediction):
        response_message = {}
        response_message['AppName'] = received_message_decoded['AppName']
        response_message['ReplyForCommand'] = received_message_decoded['Command']
        if 'ReplyId' in received_message_decoded:
            response_message['ReplyId'] = received_message_decoded['ReplyId']
        response_message['ReplyStatus'] = 'Normal'
        response_message_parameters = {}
        response_message_parameters[self.__target_list[0]] = self.__convert_to_native_type(prediction)  ## just one predicted value for right now
        response_message_parameters['version'] = received_message_decoded['Parameters']['version']
        response_message['Parameters'] = response_message_parameters
        json_response = json.dumps(response_message)
        json_response_encoded = json_response.encode('utf-8')
        return json_response_encoded

    def __convert_to_native_type(self, value):
        def isInteger(n):
            try:
                float(n)
            except ValueError:
                return False
            else:
                return float(n).is_integer()

        if isinstance(value, np.ndarray):
            if value.size > 1:
                return self.__classification_by_index(np.argmax(value))
            elif isInteger(value[0]):
                return int(value)
            else:
                try:
                    return float(value)
                except:
                    return value
        elif isinstance(value, int):
            return int(value)
        elif isinstance(value, float):
            return float(value)
        else:
            return value

    def startServer(self):
        if self.__communication_type == COMMS_TYPE.UDP:
            self.__my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # udp
            self.__my_socket.bind((self.__udp_host, self.__udp_port))
            print("Classification Prediction Server Online and Listening")
            while True:
                data_received, _ = self.__my_socket.recvfrom(self.__buffer_size)
                received_message_decoded = json.loads(data_received.decode('utf-8'))
                if self.__debugging:
                    print("Received Message Body : ")
                    for key, value in received_message_decoded.items():
                        print(key, ' : ', value)
                features = self.__extractFeatures(received_message_decoded)
                t = time.perf_counter()
                prediction = self.__predict(features)
                elapsed = time.perf_counter() - t
                if self.__track_TimeToPredict:
                    self.__total_Time = self.__total_Time + elapsed
                    self.__total_Time_Count = self.__total_Time_Count + 1
                    print("Average Elapsed Time: " + str(self.__total_Time / self.__total_Time_Count))
                if self.__debugging:
                    print(self.__routing_key_prediction_server + " Predicted = " + str(prediction))
                if prediction is not None:
                    json_response_encoded = self.__createResponse(received_message_decoded, prediction)
                    if received_message_decoded['ReplyInfo']:
                        response_address = (received_message_decoded['ReplyInfo'].split(":")[1], int(received_message_decoded['ReplyInfo'].split(":")[2]))
                        self.__my_socket.sendto(json_response_encoded, response_address)
        if self.__communication_type == COMMS_TYPE.RABBITMQ:
            credentials = pika.PlainCredentials(self.__user_name, self.__password)
            self.__rabbitMQ_Connection = pika.BlockingConnection(pika.ConnectionParameters(self.__rabbitmq_address, self.__rabbitmq_port, '/', credentials))
            self.__rabbitMQ_Channel = self.__rabbitMQ_Connection.channel()
            self.__rabbitMQ_Channel.exchange_declare(exchange=self.__exchange_name, exchange_type=pika.exchange_type.ExchangeType.topic)
            self.__rabbitMQ_Queue = self.__rabbitMQ_Channel.queue_declare(self.__routing_key_prediction_server, exclusive=True)
            self.__rabbitMQ_Channel.queue_bind(exchange=self.__exchange_name, queue=self.__rabbitMQ_Queue.method.queue, routing_key=self.__routing_key_prediction_server)
            self.__rabbitMQ_Channel.basic_consume(queue=self.__rabbitMQ_Queue.method.queue, on_message_callback=self.rabbitMQCallback, auto_ack=True)
            print("AIML Prediction Server Online and Listening")
            print("Waiting for RabbitMQ Messages")
            self.__rabbitMQ_Channel.start_consuming()
            self.__rabbitMQ_Connection.close()

    def rabbitMQCallback(self, ch, method, properties, data_received):
        received_message_decoded = json.loads(data_received.decode('utf-8'))
        if self.__debugging:
            print("Received Message Body : ")
            for key, value in received_message_decoded.items():
                print(key, ' : ', value)
        features = self.__extractFeatures(received_message_decoded)
        t = time.time()
        prediction = self.__predict(features)
        elapsed = time.time() - t
        if self.__track_TimeToPredict:
            self.__total_Time = self.__total_Time + elapsed
            self.__total_Time_Count = self.__total_Time_Count + 1
            print("Average Elapsed Time: " + str(self.__total_Time/self.__total_Time_Count))
        if self.__debugging:
            print(self.__routing_key_prediction_server + " Predicted = " + str((prediction[0])))
        if prediction is not None:
            json_response_encoded = self.__createResponse(received_message_decoded, (prediction[0]))
            if self.__debugging:
                print("Sending Response of " + str(json.loads(json_response_encoded.decode('utf-8'))) + " on " + method.routing_key)
            self.__rabbitMQ_Channel.basic_publish(exchange=self.__exchange_name, routing_key=self.__routing_key_requestor, body=json_response_encoded)


def startSystem(config):
    prediction_server = ClassificationPredictionServer()
    prediction_server.loadConfig(config)
    prediction_server.startServer()

if __name__ == "__main__":

    def get_path(wildcard):
        root = Tk()
        root.withdraw()
        file_path = \
            fd.askopenfilename(parent=None, defaultextension='.json, .yml, .yaml',
                               initialdir=os.getcwd(),
                               title="Choose ML Test Python Input Configuration (Server)",
                               filetypes=[("JSON OR YAML Config", wildcard)])
        root.update()
        root.destroy()
        return file_path

    configFile = get_path("*.json *.yaml *.yml")

    if configFile is not None and not exists(configFile):
        if not os.path.exists(abspath(join(os.getcwd(), 'config', configFile))):
            raise Exception("Config file : {} : Not found!".format(configFile))

    if configFile.lower().endswith("yaml") or configFile.lower().endswith("yml"):
        config = yaml_load(open(configFile, 'r'))
        config = RecursiveNamespace(**config)
    elif configFile.lower().endswith("json"):
        config = json.load(open(configFile))
        config = RecursiveNamespace(**config)
    else:
        print("Invalid Configuration File")
        sys.exit()

    startSystem(config)