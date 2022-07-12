from os.path import exists
from ConfigParseTool.parsemods import Decorators, DictObj
from ConfigParseTool.parsemods.LoggingService import LoggingService
import json


@Decorators.singleton
class JSONParser:

    #
    # ####################################################
    # JSON Parser Singleton
    # ####################################################
    #
    def __init__(self):
        pass

    def loadConfigFile(self, configFile: str = None) -> dict:

        if configFile is not None and exists(configFile):
            # Opening JSON file
            json_file = open(configFile, "r")
            json_dict = json.loads(json_file.read())
            LoggingService().debug("Loaded Config file : {}".format(configFile))
            return DictObj(json_dict)
        else:
            LoggingService().error("Config file : {} : Not found!".format(configFile))
            raise Exception("Config file : {} : Not found!".format(configFile))
            sys.exit()
