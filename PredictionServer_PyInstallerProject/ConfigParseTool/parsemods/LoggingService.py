import csv
import logging
import os
import Decorators
from os.path import abspath, join, dirname, exists


@Decorators.singleton
class LoggingService:

    #
    # ####################################################
    # Logging Service
    # ####################################################
    #
    def __init__(self):
        self.__logger_dictionary = {}

    def setup(self, directory: str = "/", logger_name: str = "default"):
        if logger_name not in self.__logger_dictionary:
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Setup Logging
            # create logger with 'logger_name'
            new_logger = logging.getLogger(logger_name)
            new_logger.setLevel(logging.DEBUG)
            # create file handler which logs even debug messages
            fh = logging.FileHandler(os.path.join(directory, logger_name + ".log"))
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to the logger
            new_logger.addHandler(fh)
            new_logger.addHandler(ch)
            self.__logger_dictionary[logger_name] = new_logger

    def error(self, message: str = None, logger_name: str = "default"):
        if logger_name in self.__logger_dictionary and message is not None:
            self.__logger_dictionary[logger_name].error(message)

    def info(self, message: str = None, logger_name: str = "default"):
        if logger_name in self.__logger_dictionary and message is not None:
            self.__logger_dictionary[logger_name].info(message)

    def warning(self, message: str = None, logger_name: str = "default"):
        if logger_name in self.__logger_dictionary and message is not None:
            self.__logger_dictionary[logger_name].warning(message)

    def debug(self, message: str = None, logger_name: str = "default"):
        if logger_name in self.__logger_dictionary and message is not None:
            self.__logger_dictionary[logger_name].debug(message)
