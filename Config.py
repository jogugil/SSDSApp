#################################################
### José Javier Gutierrez Gil
### jogugil@gmail.com
#################################################

import os
import yaml
from dotmap import DotMap

class Config ():
    config      = None
    config_dict = None

    def __init__ (self,config_file):
        self.process_config (config_file)
        
    def get_config_from_yaml (self, yaml_file):
        config = None
        config_dict = None
         
        try:
             
            with open (yaml_file, 'r') as config_file:
                config_dict = yaml.safe_load (config_file)
            # Convert the dictionary to a DotMap
             
            config = DotMap (config_dict)
             
        except Exception  as error:
            print(error)
        finally:
             return config, config_dict
            
    def process_config (self, yaml_file):
        config, config_dict = self.get_config_from_yaml (yaml_file)
        self.config = config
        self.config_dict = config_dict
        return config 
    