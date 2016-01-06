from .config import STANMO_CONFIG_DICT, StanmoError
import os
import time
import logging
import simplejson
from .basemodelspec import BaseMiningModel
import sys

def load_class(dottedpath):
    """Load a class from a module in dotted-path notation.
    E.g.: load_class("package.module.class").
    Based on recipe 16.3 from Python Cookbook, 2ed., by Alex Martelli,
    Anna Martelli Ravenscroft, and David Ascher (O'Reilly Media, 2005)
   """
    assert dottedpath is not None, "dottedpath must not be None"
    splitted_path = dottedpath.split('.')
    modulename = '.'.join(splitted_path[:-1])
    classname = splitted_path[-1]
    # print(sys.path)

    try:
        try:
            module = __import__(modulename, globals(), locals(), [classname])
        except ValueError: # Py < 2.5
            if not modulename:
                module = __import__(__name__.split('.')[0],
                    globals(), locals(), [classname])
    except ImportError:
        # properly log the exception information and return None
        # to tell caller we did not succeed
        logging.exception('tg.utils: Could not import %s because an exception occurred', dottedpath)
        return None
    try:
        return getattr(module, classname)
    except AttributeError:
        logging.exception('tg.utils: Could not import %s because the class was not found', dottedpath)
        return None


class StanmoApp:
    """ The runtime platform for running all mining models
    """
    MODEL_DIR = 'model'
    MODEL_DATAFRAME_DIR = 'data'
    MODEL_INSTANCE_DIR = 'instance'
    MODEL_SPEC_DIR = 'spec' # Save the specs of a specific model, including file mining_model.json
    MODEL_SPEC_PATH = 'spec' # Save all model_spec programs. One spec may serve multiple models
    MODEL_SPEC_FILENAME = 'mining_model.json'

    TIME_FORMATER = "%Y-%m-%d %H:%M:%S"
    def __init__(self):
        self.stanmo_home = STANMO_CONFIG_DICT['stanmo']['stanmo_home']
        # Add the plugin (model specs) home to sys path for dynamic loading all model specs defined under $STANMO_HOME/spec
        sys.path.append(os.path.join(self.stanmo_home, self.MODEL_SPEC_PATH))

        self.model_storage_path = os.path.join(self.stanmo_home,self.MODEL_DIR) # ConfigAttribute('model_storage_path')
        if not os.path.exists(self.model_storage_path):
            os.mkdir(self.model_storage_path)
        self.load_model_dict()

        logger = logging.getLogger('stanmo_logger')
        logger.debug('stanmo platform is started with debug option.') # will not print anything

    def load_model(self, model_name = None):
        model_spec_name =  self.model_dict[model_name]["spec_name"]
        logger = logging.getLogger('stanmo_logger')
        logger.debug('will load model from model path: ' + model_spec_name )
        the_model_class = load_class(model_spec_name)
        the_model = the_model_class(stanmoapp, model_name)
        return the_model

    def create_model(self, model_name = None, spec_name = None):
        if model_name in self.model_dict.keys():
            raise StanmoError('Model name is already used.')
        the_model_class = load_class(spec_name)
        if the_model_class is None:
            raise StanmoError('Model Spec is not found, or corrupted.')

        new_model_storage_path = os.path.join(self.model_storage_path,model_name) # ConfigAttribute('model_storage_path')
        try:
            os.mkdir(new_model_storage_path)
        except:
            raise StanmoError('Can not create folder {0}.'.format(new_model_storage_path))
        os.mkdir(os.path.join(new_model_storage_path,self.MODEL_DATAFRAME_DIR))
        os.mkdir(os.path.join(new_model_storage_path,self.MODEL_INSTANCE_DIR))

        model_spec_path = os.path.join(new_model_storage_path,self.MODEL_SPEC_DIR)
        os.mkdir(model_spec_path)
        model_spec = { "name": "churn_mining_model",
                       "model_instances": {},
                       "champion_instance": "-1",
                       "last_modify_date": "NA",
                       "spec_name":  spec_name
                       }
        model_spec['last_modify_date'] = time.strftime(self.TIME_FORMATER,time.gmtime())
        with open(os.path.join(model_spec_path,self.MODEL_SPEC_FILENAME),'w') as f:
             simplejson.dump(model_spec,f)

        # self.add_model_dict(model_name=model_name,spec_name=spec_name)
        self.load_model_dict() # Refresh model dict to include the new model
        return


    def get_model_spec_path(self, model_name = None):
        logger = logging.getLogger('stanmo_logger')
        logger.debug('will load model from model path: ' + self.model_storage_path )
        model_spec_storage_path = os.path.join(self.model_storage_path,  model_name, self.MODEL_SPEC_DIR, self.MODEL_SPEC_FILENAME)
        return model_spec_storage_path

    def get_model_instance_list_path(self, model_name = None):
        logger = logging.getLogger('stanmo_logger')
        logger.debug('will load model from model path: ' + self.model_storage_path )
        model_spec_storage_path = os.path.join(self.model_storage_path,  model_name, self.MODEL_INSTANCE_DIR)
        return model_spec_storage_path

    def get_model_instance_path(self, model_name = None, model_instance_id = None):
        logger = logging.getLogger('stanmo_logger')
        logger.debug('will give model path as: ' + self.model_storage_path )
        model_spec_storage_dir = os.path.join(self.model_storage_path,  model_name, self.MODEL_INSTANCE_DIR, str(model_instance_id))
        if not os.path.exists(model_spec_storage_dir):
            os.makedirs(model_spec_storage_dir)

        model_spec_storage_path = os.path.join(model_spec_storage_dir, 'model_inst.pkl' )
        return model_spec_storage_path


    def get_dataframe_spec_path(self, model_name = None, dataframe_name = None):
        storage_path = os.path.join(self.stanmo_home, self.MODEL_SPEC_PATH,
                     self.model_dict[model_name]['spec_name'].split('.')[0],
                     dataframe_name + '.json')
        return storage_path

    def get_dataframe_path(self, model_name = None, dataframe_name = None):
        storage_path = os.path.join(self.model_storage_path,  model_name, self.MODEL_DATAFRAME_DIR, dataframe_name +  '.csv')
        return storage_path

    def load_model_dict(self):
        '''
        This function is kept only for back compatibility.
        '''
        self.model_dict = {}

        for item in os.listdir(self.model_storage_path):
            if os.path.isdir(os.path.join(self.model_storage_path, item)):
                with open(os.path.join(self.model_storage_path, item,self.MODEL_SPEC_PATH, self.MODEL_SPEC_FILENAME)) as f:
                    a_model_spec = simplejson.load(f)
                    self.model_dict[item] = a_model_spec

        return self.model_dict
    '''
        def add_model_dict(self,model_name=None,spec_name=None):
            # model_list = []
            # I will use model dir for list of models and will not save dedicated json file anymore.
            self.model_dict[model_name] =  {"spec_name":spec_name}
            model_list_storage_path = os.path.join(self.stanmo_home, self.MODEL_LIST_FILE_LOCATION)
            with open(model_list_storage_path,'w') as f:
                simplejson.dump(self.model_dict,f)
    '''
    def list_models(self):
        logger = logging.getLogger('stanmo_logger')
        logger.debug('looking for model path: ' + self.model_storage_path)
        model_list = []
        models = []
        # for root, dirnames, filenames in os.walk(self.model_storage_path):
        # for subdirname in dirnames:
        # for model in self.model_dict:  model["name"]
        for model_name in self.model_dict.keys():
            try:
                new_model = BaseMiningModel(self, model_name)
                model_list.append(model_name)
                models.append(new_model)
            except:
                continue;
        if len(models) > 0:
            print 'Model Name:           Spec Name:             Instances:            Champion:         Last Modify Date:  '
            for a_model in models:
                if len(a_model._loaded_model['model_instances'].keys()) > 3:
                    new_inst_list = a_model._loaded_model['model_instances'].keys()[0:2]
                    new_inst_list.append('...')
                else:
                    new_inst_list = a_model._loaded_model['model_instances'].keys()
                print('{0:20} {spec_name:15}  {inst_list:19}        {champ:10}        {modify:20}  '.format(a_model.model_name,
                                                                                            spec_name = self.model_dict[a_model.model_name]["spec_name"],
                                                                                            inst_list = str(new_inst_list),
                                                                                            champ = a_model._loaded_model['champion_instance'],
                                                                                            modify =  a_model._loaded_model['last_modify_date']
                                                                                           )
                      )
        else:
            print 'No models found!'

        logging.getLogger('stanmo_logger').debug('discovered models: ' + model_list.__str__())
        return model_list

    def list_specs(self):
        model_spec_path = os.path.join(self.stanmo_home, self.MODEL_SPEC_PATH)
        logger = logging.getLogger('stanmo_logger')
        logger.debug('looking for model spec in path: ' + model_spec_path)
        spec_list = []

        for item in os.listdir(model_spec_path):
            if os.path.isdir(os.path.join(model_spec_path, item)):
                try:
                    spec_name = load_class(item+'.__model_spec_name__')
                    spec_list.append({"name":spec_name, "path":os.path.join(model_spec_path,item)} )
                except:
                    continue;

        return spec_list

stanmoapp = StanmoApp()

