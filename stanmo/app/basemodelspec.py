import os
import time
import logging
import simplejson
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sqlalchemy.orm import sessionmaker

from .config import StanmoError
from .dbmodel import PredictionHistory, engine

try:
   import cPickle as pickle
except:
   import pickle

class StanmoDataFrame:
    """ Abstraction of Spark & Pandas DataFrame, acting as the input and output of a Mining Model
    It is used only as a spec to another dataframe.
    """
    source_name = None
    apply_name = None
    column_name_types = None
    target_column_name = None
    pk_column_name = None
    storage_filename = None
    df_type = 'Pandas'
    df = None

    '''
    def load_dataframe_spec(self, spec_file_name=None):
        self.load_dataframe_spec(spec_file_name)

    def set_dataframe_spec(self, column_types=None, target_column = None, pk_column = None, df_storage_name = None):
        """
            :param dtypes:
                the data frame types for all columns. The definition follows PySpark DataFrame.dtypes definition.
            :param target_column:
                the name of the target column, which must be one of the names in dtypes.

        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
        """
        self.column_name_types = column_types
        self.target_column = target_column
        self.pk_column = pk_column
        self.df_storage_name = df_storage_name
    '''

    def __init__(self, stanmoapp = None,model_name = None, df_name=None, load_data = False):
        self.stanmoapp = stanmoapp
        self.model_name = model_name
        self.df_name = df_name

        spec_file_name = stanmoapp.get_dataframe_spec_path(model_name, df_name)
        logging.getLogger('stanmo_logger').debug('loading dataframe from file : ' + spec_file_name)
        with open(spec_file_name) as f:
            loaded_spec = simplejson.load(f)

            self.source_name = loaded_spec.get('source_name')
            self.apply_name = loaded_spec.get('apply_name')
            self.column_name_types = loaded_spec.get('column_name_types')
            self.target_column = loaded_spec.get('target_column')
            self.pk_column = loaded_spec.get('pk_column')
            self.storage_filename = loaded_spec.get('storage_filename')
            logging.getLogger('stanmo_logger').debug('loaded spec file: ' + spec_file_name)

        if load_data:
            logging.getLogger('stanmo_logger').warning('do not use direct load, use set_df() instead ... ' )
            self.df_type = 'Pandas'
            self.df = pd.read_csv(stanmoapp.get_dataframe_path(model_name, self.source_name))

    def set_df(self,new_df=None,new_df_type = 'Pandas'):
        self.df=new_df
        self.df_type=new_df_type

    def from_csv(self,csv_file=None):
        self.df=pd.read_csv(csv_file)
        self.df_type='Pandas'



    def save_dataframe(self, output_filename = None):
        if output_filename is None:
            output_filename = self.stanmoapp.get_dataframe_path(self.model_name, self.df_name)

        logging.getLogger('stanmo_logger').debug('looking for model instance at path' + output_filename)
        self.df.to_csv(output_filename)
        logging.getLogger('stanmo_logger').debug('saved dataframe: ' + output_filename)

class MiningModelInstance(dict):
    ''' This is an umbrella to put encoder and classifier into a single object, for the convenience purpose.
        Then the single object can be saved by pickle onto the disk.
    '''
    '''def __init__(self, encoder = None, model = None, testing_precision = None, input_stanmodataframe = None):
        self.encoder = encoder
        self.model = model
        self.testing_precision = testing_precision
        self.input_stanmodataframe = input_stanmodataframe
        '''
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class BaseMiningModel(object):
    """
        It represent a mining model definition. Customers may create more Mining Model Instance based on a Mining Model.
        Each mining model is executed in the following orders:
        1.
        2.

        The baseminingmodel will load the model by the model_name from disk, according to a predefined structure.
        Then real miningmodel_implementation can do the rest mining work.
    """
    input_dataframe_names = []
    output_dataframe_names = []

    def __init__(self, stanmoapp = None, model_name=None):
        """
            Constructor.

            :param model_instance_id:
                To uniquely identify the mining model, it is from the database PK.
            :param model_storage_path:
                Base model_storage_path
        """
        self.model_name = model_name
        self.stanmoapp = stanmoapp
        self.model_instances = {}  # Initialize the model instance cache.
        self.load_model()
        # self.load_dataframes()

        # self.list_model_instance_ids()

    def list_model_instance_ids(self):
        self.model_instance_ids =  []
        self.model_instances = {}
        inst_path = self.stanmoapp.get_model_instance_list_path(self.model_name)
        logging.getLogger('stanmo_logger').debug('looking for model instance at path' + inst_path)
        for root, dirnames, filenames in os.walk(inst_path):
            for subdirname in dirnames:
                self.model_instance_ids.append(subdirname)
                self.model_instances[subdirname] = None
        logging.getLogger('stanmo_logger').debug('discovered models: ' + self.model_instances.__str__())

    def get_current_model_instance_id(self):
        return self._loaded_model['current_instance_id']

    def get_new_model_instance_id(self):
        return self._loaded_model['current_instance_id']

    def get_model_instance(self, model_instance_id = None):

        if model_instance_id in self.model_instances:
            # cache hit
            logging.getLogger('stanmo_logger').debug('cache hit: ' + str(model_instance_id))
            return self.model_instances[model_instance_id]

        inst_path = self.stanmoapp.get_model_instance_path(self.model_name, model_instance_id)
        with open(inst_path, 'rb') as fid:
            mi = pickle.load(fid)
            self.model_instances[model_instance_id] = mi
            return mi

    def set_model_instance(self, model_instance=None, model_instance_id = None):
        inst_path = self.stanmoapp.get_model_instance_path(self.model_name, model_instance_id)
        with open(inst_path, 'wb') as fid:
            pickle.dump(model_instance, fid)
            self._loaded_model['model_instances'][model_instance_id] = 'trained'
            self._loaded_model['champion_instance'] = model_instance_id
            self.model_instances[model_instance_id] = model_instance
            self._loaded_model['last_modify_date'] = time.strftime(self.stanmoapp.TIME_FORMATER ,time.gmtime())  # time.gmtime()
        self.save_mining_model_state()


    def new_model_instance_id(self):
        # create and Insert one version for each model instance, this will be set as the default version
        # new_instance_id = http_rest_api(get new model instance id)
        mi_dict = self._loaded_model['model_instances']
        for mi_key in mi_dict.keys():
            if mi_dict[mi_key] == 'NA':
                return mi_key
        return None

    def set_champion_version_id(self, model_instance=None, model_instance_id = None):
        print "set new version: " + str(model_instance_id)
        # caluclate by feedback, set campaign

    def add_prediction_history(self,model_name=None, prediction_df = None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()

        for i, row in prediction_df.iterrows():
            #print row['Date']
            recid = str(row['record_id'])
            res = '{"result":"' + str(row['prediction_result']) + '","probability":"' \
                  + str(row['prediction_probability']) + '"}'
            session.merge(PredictionHistory(model_name=model_name, record_id=recid, prediction_result=res))

        # session.add_all(all_preds)
        session.commit()
        session.close()


    def set_feedback_json(self,model_name=None, feedback_json = None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()

        for feedback in feedback_json:
            #print row['Date']
            recid = str(feedback['record_id'])
            res = str(feedback['feedback_result'])
            session.merge(PredictionHistory(model_name=model_name, record_id=recid, feedback_result=res))

        # session.add_all(all_preds)
        session.commit()
        session.close()
        logger = logging.getLogger('stanmo_logger')
        logger.debug('saved feedback of : ' + feedback_json[0]['record_id'] )

    def get_roc_curve_all(self,model_name=None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()

        q = session.query(PredictionHistory).filter(PredictionHistory.model_name==model_name).filter(PredictionHistory.feedback_result!=None)
        df = pd.read_sql(q.statement, q.session.bind)
        session.close()
        # logging.getLogger('stanmo_logger').debug('calculated prediction hisotry for model_name: ' + model_name + '---' + df.to_json())

        if len(df.index) < 1:
            # for chart.js, it is not used anymore.
            ''' adata = {
                'labels' : ["0","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
                'datasets' : [
                    {
                        'label' : "My First dataset",
                        'fillColor' : "rgba(220,220,220,0.2)",
                        'strokeColor' : "rgba(220,220,220,1)",
                        'pointColor' : "rgba(220,220,220,1)",
                        'pointStrokeColor' : "#fff",
                        'pointHighlightFill' : "#fff",
                        'pointHighlightStroke' : "rgba(220,220,220,1)",
                        'data' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    }
                ]
            }
            '''
            # This data is according to the NVD3 format.
            # Empty data
            adata = [  {
                'values': [{'x':0, 'y': 0},{'x':1, 'y': 1}],
                'key': "Random Benchmark, Model ROC Not Available",
                'color': "#ff7f0e"
            } ]
        else:
            df['proba'] = df['prediction_result'].apply(lambda x:simplejson.loads(x)['probability'])
            fpr, tpr, thresholds = roc_curve(df['feedback_result'].astype(int), df['proba'].astype(float))
            model_roc  = [{"y": 0.0, "x": 0.0}]
            random_roc = [{"y": 0.0, "x": 0.0}]
            for i in range(0, len(fpr)):
                model_roc.append({'x':fpr[i], 'y': tpr[i] })
                random_roc.append({'x':fpr[i], 'y': fpr[i] })

            adata = [  {
                'values': random_roc,
                'key': "Random Benchmark",
                'color': "#ff7f0e"
            },{
                'values': model_roc,
                'key': "Model ROC",
                'color': "#2ca02c"
            } ]
        return adata

    def get_daily_prediction_count(self,model_name=None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()

        q = session.query(PredictionHistory).filter(PredictionHistory.model_name==model_name)
        df = pd.read_sql(q.statement, q.session.bind)
        session.close()

        df['ymd'] = df['prediction_date'].apply(lambda x: x.strftime('%Y%m%d'))
        df = df.groupby('ymd')['prediction_date'].count()
        logging.getLogger('stanmo_logger').debug('calculated prediction hisotry for model_name: ' + model_name + '---' + df.to_json())
        return df


    def get_overall_statistics(self,model_name=None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()
        print("get for model : " + model_name)

        overall_stat = {}
        q = session.query(PredictionHistory).filter(PredictionHistory.model_name==model_name)
        df = pd.read_sql(q.statement, q.session.bind)
        session.close()

        overall_stat['total_prediction_count'] = len(df.index)
        overall_stat['total_feedback_count'] = df['feedback_result'].count()

        def convert_hit_score (x):
            predict0 = simplejson.loads(x[0])['result']
            predict1 = x[1] #simplejson.load(x[1]).result
            hit_ratio = 1 if predict0 == predict1 else 0 # abs(int(x[0]) - int(x[1]))
            return hit_ratio
        df['hit_score'] = df[['prediction_result','feedback_result']].apply(convert_hit_score, axis=1)
        overall_stat['total_precision'] = "{0:.1f}%".format(df['hit_score'].mean() * 100) # format like 65.8%


        df['ymd'] = df['prediction_date'].apply(lambda x: x.strftime('%Y%m%d'))
        df_count = df.groupby('ymd')['prediction_date'].count()
        overall_stat['daily_prediction_count'] = df_count

        # logging.getLogger('stanmo_logger').debug('calculated overall for model_name: ' + model_name + '---' + overall_stat.to_json())
        return overall_stat


    def get_number_of_predictions(self,model_name=None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()

        q = session.query(PredictionHistory).filter(PredictionHistory.model_name==model_name).filter(PredictionHistory.feedback_result!=None)
        df = pd.read_sql(q.statement, q.session.bind)
        session.close()
        return 16

    def calculate_daily_precision(self,model_name=None):
        if model_name is None:
            model_name = self.model_name
        Session = sessionmaker(bind=engine)
        session = Session()

        q = session.query(PredictionHistory).filter(PredictionHistory.model_name==model_name).filter(PredictionHistory.feedback_result!=None)
        df = pd.read_sql(q.statement, q.session.bind)
        session.close()

        def convert_hit_score (x):
            predict0 = simplejson.loads(x[0])['result']
            predict1 = x[1] #simplejson.load(x[1]).result
            hit_ratio = 1 if predict0 == predict1 else 0 # abs(int(x[0]) - int(x[1]))
            return hit_ratio
        df['hit_score'] = df[['prediction_result','feedback_result']].apply(convert_hit_score, axis=1)
        df['ymd'] = df['prediction_date'].apply(lambda x: x.strftime('%Y%m%d'))
        df = df.groupby('ymd')['hit_score'].mean()
        logging.getLogger('stanmo_logger').debug('calculated prediction hisotry for model_name: ' + model_name + '---' + df.to_json())
        return df

    def load_model(self):
        model_spec_storage_path = self.stanmoapp.get_model_spec_path(self.model_name)
        logging.getLogger('stanmo_logger').debug('will load model from: '+  model_spec_storage_path )
        with open(model_spec_storage_path) as f:
            self._loaded_model = simplejson.load(f)
            self._champion_instance = self._loaded_model.get('champion_instance')

    def get_champion_instance(self):
        if self._loaded_model.get('champion_instance') is None or self._loaded_model.get('champion_instance') == -1:
            raise StanmoError('The requested model has no champion instance to run prediction')
        mmi = self.get_model_instance(self._loaded_model.get('champion_instance'))
        return mmi

    def create_default_record(self):
        mmi = self.get_champion_instance()
        attr_list = mmi.encoder.curr_sdf.keeping_columns.keys()
        new_customer = {}
        for customer_attr in attr_list:
            new_customer[customer_attr] = 0
        new_customer['target'] = 0
        return new_customer

    def load_dataframes(self, dataframe_names=None):
        if self.stanmoapp == None:
            raise StanmoError('stanmoapp can not be None.')

        input_dataframe_names=dataframe_names['input_df_names']
        output_dataframe_names=dataframe_names['output_df_names']


        self.input_dataframes = {}
        for df_name in input_dataframe_names:
            logging.getLogger('stanmo_logger').debug('loading dataframe: ' + df_name)
            sdf = StanmoDataFrame(self.stanmoapp, self.model_name, df_name, False)
            self.input_dataframes[df_name] = sdf # append(df)

        self.output_dataframes = {}
        for df_name in output_dataframe_names:
            # Table_instance_id is auto-generated.
            df = StanmoDataFrame(self.stanmoapp,self.model_name, df_name, False)
            self.output_dataframes[df_name] = df # append(df)

    def save_mining_model_state(self):
        model_spec_storage_path = self.stanmoapp.get_model_spec_path(self.model_name)
        logging.getLogger('stanmo_logger').debug('will load model from: '+  model_spec_storage_path )
        with open(model_spec_storage_path,'w') as f:
             simplejson.dump(self._loaded_model,f)


    def save_dataframe_specs(self):
        self.output_data_instance_location = 'a'
        self.output_data_instance_df.to_csv(self.output_data_instance_location)

    def fit(self, input_file = None):
        raise NotImplementedError("BaseMiningModel is an abstract Model and should not be used directly!")
    def predict(self, input_file = None, output_file = None):
        raise NotImplementedError("BaseMiningModel is an abstract Model and should not be used directly!")
    def run(self):
        raise NotImplementedError("BaseMiningModel is an abstract Model and should not be used directly!")
    def show(self):
        """ Show the model result in a predefined dashboard.
        :return:
        """
        raise NotImplementedError("BaseMiningModel is an abstract Model and should not be used directly!")
    def encode(self):
        raise NotImplementedError("BaseMiningModel is an abstract Model and should not be used directly!")


class BaseInputDataEncoder:
    def __init__(self,cat_columns = None):
        pass
    def etl(self, input_files = None):
        """  The ETL logic from customer sources into the input dataframes as defined in input_dataframe_spec.
             If you use input dataframe as it is, you may simply implement it by single 'pass' command.
        :param input_files: A dictionary of name/file_path pairs for all input dataframes
        :return:
        """
        raise NotImplementedError("BaseMiningModel is an abstract Model and should not be used directly!")
    def fit(self,X,y=None):
        raise NotImplementedError("Not implemented!")
    def transform(self,X):
        raise NotImplementedError("Not implemented!")
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    def slim(self):
        if self.curr_sdf is not None:
            self.curr_sdf.df=None
            self.curr_sdf.df_type=None
            self.curr_sdf.stanmoapp=None
        self.stanmoapp=None
