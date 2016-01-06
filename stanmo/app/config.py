import os
import traceback, logging

# Those are all options for end user to customize. Other internal configurations may be found in different classes.
def get_stanmo_home_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
# os.path.abspath(os.path.join(os.path.dirname(__file__),'..')) #
STANMO_CONFIG_DICT = {"stanmo": { "stanmo_home": get_stanmo_home_path(),
                                  },
                      "logging": {"log_file": "stanmo.log",
                                    "sqla_log_file": "stanmo_sqlalchemy.log",
                                    "root_log_file": "stanmo_others.log",
                                    "log_dir":"log",
                                    "log_level":"ERROR" # DEBUG,INFO,WARNING,ERROR,CRITICAL
                                  }
                      }
LOG_DIR = os.path.join(STANMO_CONFIG_DICT['stanmo']['stanmo_home'],
                       STANMO_CONFIG_DICT['logging']['log_dir'])
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

current_logfile_path = os.path.join(LOG_DIR,
                                    STANMO_CONFIG_DICT['logging']['log_file'])
class MyStreamHandler(logging.StreamHandler):
    def format(self, record):
        try:
            return logging.StreamHandler.format(self, record)
        except TypeError:
            # Print a stack trace that includes the original log call
            traceback.print_stack()
    def handleError(self, record):
        raise

logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(lineno)d}  - %(message)s',
                    level=logging.getLevelName(STANMO_CONFIG_DICT['logging']['log_level']),
                    filename=current_logfile_path)

log=logging.getLogger('stanmo_logger')
handler = MyStreamHandler()
log.addHandler(handler)
# logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.engine.base.Engine').setLevel(logging.ERROR)

sqla_logger = logging.getLogger('sqlalchemy')
sqla_logger.propagate = False
sqla_logger.addHandler(logging.FileHandler(os.path.join(LOG_DIR,STANMO_CONFIG_DICT['logging']['sqla_log_file'])))
logging.getLogger('sqlalchemy.engine').addHandler(logging.FileHandler(os.path.join(LOG_DIR,STANMO_CONFIG_DICT['logging']['sqla_log_file'])))
logging.getLogger().addHandler(logging.FileHandler(os.path.join(LOG_DIR,STANMO_CONFIG_DICT['logging']['root_log_file'])))

#log.error("Now try passing a string to an int: %d", 'abc')
#log.error("Try interpolating an int correctly: %i", 1)
#log.error("Now try passing a string to an int: %d", 'abc')
#log.error("And then a string to a string %s", 'abc')
logging.getLogger('stanmo_logger').info('Logging initialization finished.') # will not print anything

class StanmoError(Exception):
    pass
class StanmoParameterError(Exception):
    pass
class StanmoErrorNoInstanceID(Exception):
    pass

'''
log=logging.getLogger('stanmo_logger')
handler = MyStreamHandler()
log.addHandler(handler)
log.error("Now try passing a string to an int: %d", 'abc')
'''
'''
        config_file_path = os.path.join(self.stanmo_home, self.CONFIG_FILE_LOCATION)
        config_parser = SafeConfigParser()
        # print 'using log config file: ' + config_file_path
        config_parser.read(config_file_path)
        # log_dir  =  config_parser.get('logging', 'log_dir')
        log_file =  config_parser.get('logging', 'log_file')
        log_level =  config_parser.get('logging', 'log_level')

        # LOG_FILENAME = 'logging_rotatingfile_example.out'
        # logging.basicConfig(format='%(asctime)s %(message)s')
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(lineno)d}  - %(message)s',
                            level=logging.getLevelName(log_level),
                            filename=log_file)
        # Add the log message handler to the logger
        # logger_level =logging.getLevelName(log_level)
        # logger.setLevel(logger_level)
'''
'''
def get_stanmo_home_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
    try:
        stanmo_home = os.environ['STANMO_HOME']
    except KeyError:
        stanmo_home = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        # print('Environment variable STANMO_HOME is not found, so I use current directory as STANMO_HOME:' + stanmo_home)
    return stanmo_home
'''

