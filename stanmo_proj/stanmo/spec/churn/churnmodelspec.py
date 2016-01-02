import pandas as pd
import numpy as np
import simplejson
import logging

from stanmo.app.basemodelspec import BaseMiningModel, MiningModelInstance, BaseInputDataEncoder
from stanmo.app import StanmoErrorNoInstanceID
from .churnmodelflask import ChurnModelFlask

ALLOWED_NUMERICAL_DTYPES = [np.dtype('int64'), np.dtype('int32'),  np.dtype('float32'),  np.dtype('float64')]

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV

CLASSIFICATION_ALGORITHMS = (SVC,RF,KNN)
ATTRIBUTE_CADINALITY_THRESHOLD = 5
INPUT_APPLY_DF_NAME = 'churn_apply'

# MODEL_NAME = "churn1"
INPUT_DF_NAME = 'churn_source'
OUTPUT_DF_NAME = 'churn_result'
CATEGORICAL_ATTRIBUTE_CADINALITY_THRESHOLD = 8


class ChurnMiningModelInstance (MiningModelInstance):
    pass


class ChurnInputDataEncoder(BaseInputDataEncoder):
    def __init__(self,curr_sdf=None, stanmoapp=None):
        self.curr_sdf = curr_sdf # The curr_sdf should contain the stanmodataframe definition.
        self.stanmoapp = stanmoapp
        self.encoder = None
        self.scaler = None

    def etl(self):
        pass
    def fit(self,input_df=None):
        """  The ETL logic from customer sources into the input dataframes as defined in input_dataframe_spec.
             If you use input dataframe as it is, you may simply implement it by single 'pass' command.
             The fit command must be called with self.curr_sdf.df properly setup.
        :param input_file:
        :return:
        """
        '''
       '''
        if self.curr_sdf is None:
            return
        if self.curr_sdf.df is None:
            return

        # churn_source_filename =  input_files['churn_source']
        curr_sdf = self.curr_sdf

        current_input_attr_names = input_df.columns.tolist()
        current_dtypes = input_df.dtypes
        """ Now we inspect each table attributes and decide exclude or include.
            End user may change by clicking checkbox on the webpage.
        """
        #

        #load all attributes for predefined baseline data soruce
        baseline_name_types =curr_sdf.column_name_types
        baseline_input_attr_names = baseline_name_types.keys()

        to_keep_cat_columns = []
        to_keep_num_columns = []
        keeping_columns = {}

        for curr_attr_name in current_input_attr_names:
            """ Loop through all known attributes,
              if they are in provided files, then setup include flag.
          """
            if curr_attr_name in baseline_input_attr_names:
                if baseline_name_types[curr_attr_name] == 'numerical':
                    # new_df.append (new column )
                    if current_dtypes[curr_attr_name] in  ALLOWED_NUMERICAL_DTYPES:
                        # I save all names first into a list, and then slice the dataframe in one go.
                        keeping_columns[curr_attr_name] = 'numerical'
                        to_keep_num_columns.append(curr_attr_name)
                    else:
                        logging.getLogger('stanmo_logger').debug('Expecting numerical for column %s , but got %s, so column is removed', curr_attr_name, current_dtypes[curr_attr_name])
                elif curr_attr_name in set([curr_sdf.pk_column, curr_sdf.target_column]):
                    to_keep_cat_columns.append(curr_attr_name)
                    keeping_columns[curr_attr_name] = 'pk_or_target'
                elif baseline_name_types[curr_attr_name] == 'categorical':
                    cadinality = len(np.unique(input_df[curr_attr_name]))
                    if cadinality < CATEGORICAL_ATTRIBUTE_CADINALITY_THRESHOLD :
                        to_keep_cat_columns.append(curr_attr_name)
                        keeping_columns[curr_attr_name] = 'categorical'
                    else:
                        logging.getLogger('stanmo_logger').debug( 'categorical attribute with cardinality > 5 is excluded: ' + curr_attr_name)
            else:
                if current_dtypes[curr_attr_name] in  ALLOWED_NUMERICAL_DTYPES:
                    # I save all names first into a list, and then slice the dataframe in one go.
                    keeping_columns[curr_attr_name] = 'numerical'
                    to_keep_num_columns.append(curr_attr_name)
                elif len(np.unique(input_df[curr_attr_name])) < CATEGORICAL_ATTRIBUTE_CADINALITY_THRESHOLD :
                        to_keep_cat_columns.append(curr_attr_name)
                        keeping_columns[curr_attr_name] = 'categorical'
                else:
                    logging.getLogger('stanmo_logger').debug('Expecting numerical for column %s , but got %s, so column is removed', curr_attr_name, current_dtypes[curr_attr_name])

                logging.getLogger('stanmo_logger').debug( 'Not in baseline, for now i drop them ,but i can include them later by dynamicallly determine data distributrion  ' + curr_attr_name)
                """ Actionally I do not have to consider baseline attributes, just tell from data dynamically.
                if pandas (current column is numerical):
                    to_add_num_columns.
              """

        curr_sdf.to_keep_cat_columns = to_keep_cat_columns
        curr_sdf.to_keep_num_columns = to_keep_num_columns
        curr_sdf.keeping_columns = keeping_columns
        new_df = input_df[keeping_columns.keys()]
        # curr_sdf.df = new_df

        all_attribue_names =curr_sdf.keeping_columns.keys()
        all_source_names  = list(set(all_attribue_names)  -  set([curr_sdf.pk_column, curr_sdf.target_column ]))
        all_source_categorical_names  = list(set( curr_sdf.to_keep_cat_columns)  -  set([ curr_sdf.pk_column, curr_sdf.target_column ]))
        all_source_numerical_names  = list(set( curr_sdf.to_keep_num_columns)  -  set([ curr_sdf.pk_column, curr_sdf.target_column ]))

        # To train the dict vectorizer for all categorical values.
        cat_train =  input_df[all_source_categorical_names]
        x_cat_train = cat_train.T.to_dict().values()
        vectorizer = DV(sparse=False)
        self.cat_encoder = vectorizer.fit( x_cat_train )

        # To train the scaler for all numerical values.
        num_train1 = input_df[all_source_numerical_names] #                   X.drop( self.cat_columns , axis = 1)
        num_train = num_train1.as_matrix().astype(np.float64)
        s_scaler = StandardScaler()
        self.scaler =  s_scaler.fit(num_train)
        return self

    def transform(self,input_df=None):
        '''
        It transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        X should be a pandas data frame.
        '''
        all_source_categorical_names  = list(set( self.curr_sdf.to_keep_cat_columns)  -  set([ self.curr_sdf.pk_column, self.curr_sdf.target_column ]))
        all_source_numerical_names  = list(set( self.curr_sdf.to_keep_num_columns)  -  set([ self.curr_sdf.pk_column, self.curr_sdf.target_column ]))

        # Now transform all categorical values.
        cat_train =  input_df[all_source_categorical_names]
        x_cat_train = cat_train.T.to_dict().values()
        trans_cat_train = self.cat_encoder.transform(x_cat_train)


        # Now transform all numerical values.
        num_train1 = input_df[all_source_numerical_names] #                   X.drop( self.cat_columns , axis = 1)
        num_train = num_train1.as_matrix().astype(np.float64)
        trans_num_train = self.scaler.transform(num_train)

        # Concatenate numerical and categorical into a single matrix for training.
        X = np.hstack(( trans_num_train, trans_cat_train ))

        y = input_df[self.curr_sdf.target_column]

        return X,y



class ChurnMiningModel(BaseMiningModel):
    """
        It represent a mining model definition. Customers may create more Mining Model Instance based on a Mining Model.
        Each mining model is executed in the following orders:
    """

    mining_model_inst = None

    def __init__(self, stanmoapp = None, model_name=None):
        """
            Constructor.

            :param model_instance_id:
                To uniquely identify the mining model, it is from the database PK.
            :param model_storage_path:
                Base model_storage_path
        """

        super(ChurnMiningModel, self).__init__(stanmoapp=stanmoapp, model_name=model_name)
        #self.input_dataframe_names = [INPUT_DF_NAME] # self._loaded_model.get('output_dataframe_names')
        #self.output_dataframe_names = [OUTPUT_DF_NAME] # self._loaded_model.get('output_dataframe_names')
        self.load_dataframes(dataframe_names={'input_df_names':[INPUT_DF_NAME], 'output_df_names':[OUTPUT_DF_NAME]})

    def fit_csv(self,input_files=None, algorithms=None, model_instance_id = None):
        if input_files is None:
            logging.getLogger('stanmo_logger').debug('Please specify the input file for ETL.')
            exit(1)
        churn_source_filename =  input_files[0]
        self.input_dataframes[INPUT_DF_NAME].from_csv(csv_file=churn_source_filename)
        self.fit_df(input_sdf = self.input_dataframes[INPUT_DF_NAME], algorithms=algorithms, model_instance_id = model_instance_id)

    def fit_df(self,input_sdf = None, algorithms=None, model_instance_id = None):
        if model_instance_id is None:
            curr_model_instance_id = self.new_model_instance_id()
            # If I still get new instance id as -1, it means no more free slots.
            if model_instance_id is None:
                raise StanmoErrorNoInstanceID('Can not get more free instance IDs')
        else:
            curr_model_instance_id = model_instance_id
        if algorithms is None:
            one_algorithm = RF
            algorithms = [RF]

        enc = ChurnInputDataEncoder(curr_sdf=self.input_dataframes[INPUT_DF_NAME], stanmoapp=self.stanmoapp)
        # Now inspect the uploaded file and check which column to exclude.

        self.input_dataframes[INPUT_DF_NAME] = input_sdf

        # fit the encoder.
        X,y = enc.fit(input_sdf.df).transform(input_sdf.df)

        ''' algorithm_list should contain only 1 algorithm for now '''
        # Construct the training and testing data.

        X_train, X_test,Y_train, Y_test  = train_test_split( X, y, test_size=0.33, random_state=42)

        # Initialize a classifier with key word arguments
        clf = one_algorithm()
        clf.fit(X_train,Y_train)

        Y_pred = clf.predict(X_test)
        new_instance_precision = np.mean(Y_test == Y_pred)

        # input_stanmodataframe['keeping_columns'] = curr_sdf.keeping_columns
        # input_stanmodataframe['to_keep_cat_columns'] = curr_sdf.to_keep_cat_columns
        # input_stanmodataframe['to_keep_num_columns'] = curr_sdf.to_keep_num_columns

        enc.slim()

        mmi = MiningModelInstance(encoder = enc,
                                  model = clf,
                                  testing_precision = new_instance_precision# ,input_stanmodataframe = input_stanmodataframe
                                  )
        self.set_model_instance(mmi,curr_model_instance_id)


    # def encode(self, input_files = None, mmi = None):
    def encode(self, input_df = None, mmi = None):
        ''' Outpudt Parameter: The metadata information used in the training and prediction .
            1. self.included_cat_attributes
            2. self.included_num_attributes
            3. self.data_instance_target_attribute
            self.data_instance_pk_attribute
            4. self.dropped_attributes
            5. self.apply_data_instance_location
        '''
        # First get the data instance (csv file) locatioin.


        # For now, one model instance can have only upto one input table instance per table
        # but each model may have many input tables.

        # new_df = input_df[source_spec_sdf.keeping_columns.keys()]
        # self.apply_df = input_df
        # self.apply_data_instance_df = self.apply_df # pd.read_csv(self.apply_data_instance_location)
        self.X_pred = mmi.encoder.transform(input_df)
        return self.X_pred, input_df[mmi.encoder.curr_sdf.pk_column]
        # X = churn_features.as_matrix().astype(np.float)


    def predict_df(self, input_df = None ):
        mmi = self.get_champion_instance()
        # apply_df = pd.read_csv(self.stanmoapp.get_dataframe_path(self.model_name, self.input_dataframes[INPUT_DF_NAME].apply_name))
        # I first find all column names and then join with drop_list,
        # X_pred,pk_df = self.encode(input_df, mmi)
        X_pred,y = mmi.encoder.transform(input_df)

        Y_pred = mmi.model.predict(X_pred)
        Y_proba = mmi.model.predict_proba(X_pred)

        result_df =  pd.DataFrame(input_df[mmi.encoder.curr_sdf.pk_column], columns=[mmi.encoder.curr_sdf.pk_column])
        y_series = pd.Series(Y_pred, index = input_df[mmi.encoder.curr_sdf.pk_column].index) #         y_series = pd.Series(Y_pred)
        result_df['churn_flag'] = y_series
        result_df['prediction_probability'] = pd.Series(Y_proba[:,1], index = input_df[mmi.encoder.curr_sdf.pk_column].index)

        result_df['record_id'] = result_df['_customer_id']
        result_df['prediction_result'] = result_df['churn_flag']
        self.add_prediction_history(model_name = self.model_name, prediction_df=result_df)

        return result_df

    def predict_csv(self, input_files = None, output_filename = None ):
        # predict and  apply are treated equally

        # Now inspect the uploaded file and check which column to exclude.
        churn_apply_filename = input_files[0]
        try:
            input_df = pd.read_csv(churn_apply_filename)
            # sdf = StanmoDataFrame(self.stanmoapp, self.model_name, 'churn_source', True)
        except :
            logging.getLogger('stanmo_logger').debug('Failed to read the data for prediction.')
            exit(1)

        result_df = self.predict_df(input_df)
        self.output_dataframes[OUTPUT_DF_NAME].df = result_df
        self.output_dataframes[OUTPUT_DF_NAME].save_dataframe(output_filename)


    def predict_json(self, input_json = None):
        # predict and  apply are treated equally

        # Now inspect the uploaded file and check which column to exclude.
        input_df = pd.DataFrame(input_json)
        result_df = self.predict_df(input_df)
        return result_df.to_json(orient='records') # simplejson.dumps(result_dict) #"duan" # simplejson.loads('["duan", "qiyang"]')


    def show(self, port = None):
        if port is None:
            port = 5010

        url = 'http://localhost:' + str(port) + '/'

        import webbrowser
        # Open URL in new window, raising the window if possible.
        webbrowser.open_new(url)

    def run(self, port = None, to_execute=True):
        if port is None:
            port = 5010
        the_flask = ChurnModelFlask(port, to_execute,self)
        return the_flask.run()

