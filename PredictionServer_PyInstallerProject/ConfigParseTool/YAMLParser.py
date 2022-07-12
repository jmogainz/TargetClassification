from os.path import exists
from ConfigParseTool.parsemods import Decorators, DictObj
from ConfigParseTool.parsemods.LoggingService import LoggingService
import yaml


@Decorators.singleton
class YAMLParser:

    #
    # ####################################################
    # Yaml Parser Singleton
    # ####################################################
    #
    def __init__(self):
        pass

    def loadConfigFile(self, configFile: str = None) -> dict:

        if configFile is not None and exists(configFile):
            # Opening YAML file
            yaml_file = open(configFile, "r")
            yaml_dict = yaml.safe_load(yaml_file.read())
            LoggingService().debug("Loaded Config file : {}".format(configFile))
            return DictObj(yaml_dict)
        else:
            LoggingService().error("Config file : {} : Not found!".format(configFile))
            raise Exception("Config file : {} : Not found!".format(configFile))
            sys.exit()
