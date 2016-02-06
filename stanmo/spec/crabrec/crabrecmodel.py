import simplejson
import json
import logging

import pandas as pd
import numpy as np
from os.path import dirname
from os.path import join
import numpy as np

from stanmo.app.basemodelspec import BaseMiningModel, MiningModelInstance, BaseInputDataEncoder
from stanmo.app import StanmoErrorNoInstanceID



from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender



#class ChurnMiningModelInstance (MiningModelInstance):
#    pass

class Bunch(dict):
    """
    Container object for datasets: dictionary-like object
    that exposes its keys and attributes. """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

class CrabRecModel(BaseMiningModel):
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

        super(CrabRecModel, self).__init__(stanmoapp=stanmoapp, model_name=model_name)
        #self.input_dataframe_names = [INPUT_DF_NAME] # self._loaded_model.get('output_dataframe_names')
        #self.output_dataframe_names = [OUTPUT_DF_NAME] # self._loaded_model.get('output_dataframe_names')

    # trasnform from csv to a bunch.
    # def fit_csv(self,input_files=None, algorithms=None, model_instance_id = None):
    def _transform(self,input_files=None):
        """ Load and return the movies dataset with
             n ratings (only the user ids, item ids and ratings).

        Return
        ------
        data: Bunch
            Dictionary-like object, the interesting attributes are:
            'data', the full data in the shape:
                {user_id: { item_id: (rating, timestamp),
                           item_id2: (rating2, timestamp2) }, ...} and
            'user_ids': the user labels with respective ids in the shape:
                {user_id: label, user_id2: label2, ...} and
            'item_ids': the item labels with respective ids in the shape:
                {item_id: label, item_id2: label2, ...} and
            DESCR, the full description of the dataset.
        """
        # base_dir = join(dirname(__file__), 'data/')
        datafiles = input_files.split( ',')
        datafile = datafiles[0]
        descfile = datafiles[1]
        #Read data
        # for now, I ignore the rest of data files. which would be user_csv & item_csv
        data_m = np.loadtxt(datafile,
                    delimiter=';', dtype=str)


        item_ids = []
        user_ids = []
        data_songs = {}
        for user_id, item_id, rating in data_m:
            if user_id not in user_ids:
                user_ids.append(user_id)
            if item_id not in item_ids:
                item_ids.append(item_id)
            u_ix = user_ids.index(user_id) + 1
            i_ix = item_ids.index(item_id) + 1
            data_songs.setdefault(u_ix, {})
            data_songs[u_ix][i_ix] = float(rating)

        data_t = []
        for no, item_id in enumerate(item_ids):
            data_t.append((no + 1, item_id))
        data_titles = dict(data_t)

        data_u = []
        for no, user_id in enumerate(user_ids):
            data_u.append((no + 1, user_id))
        data_users = dict(data_u)

        fdescr = open(descfile)
        inv_user_ids = dict((v, k) for k, v in data_users.items())
        inv_item_ids = dict((v, k) for k, v in data_titles.items())

        return Bunch(data=data_songs, item_ids=data_titles,
                     user_ids=data_users, DESCR=fdescr.read()
                     ,inv_item_ids = inv_item_ids, inv_user_ids=inv_user_ids )


    def fit_csv(self,input_file=None, model_instance_id = None):
        if model_instance_id is None:
            curr_model_instance_id = self.new_model_instance_id()
            # If I still get new instance id as -1, it means no more free slots.
            if model_instance_id is None:
                raise StanmoErrorNoInstanceID('Can not get more free instance IDs')
        else:
            curr_model_instance_id = model_instance_id

        if input_file is None:
            logging.getLogger('stanmo_logger').debug('Please specify the input file for ETL.')
            exit(1)
        data_bunch =  self._transform(input_file)
        encoder = {}


        crab_model = MatrixPreferenceDataModel(data_bunch.data)
        similarity = UserSimilarity(crab_model, pearson_correlation)
        recommender = UserBasedRecommender(crab_model, similarity, with_preference=True)

        mmi = Bunch(model=recommender, item_ids=data_bunch.item_ids,
                     user_ids=data_bunch.user_ids, desc=data_bunch.DESCR
                     ,inv_item_ids = data_bunch.inv_item_ids, inv_user_ids=data_bunch.inv_user_ids,testing_precision = -1 )

        self.set_model_instance(mmi,curr_model_instance_id)

        # Immediate testing.
        #rec1 = recommender.recommend(5)
        #print(rec1)

    def predict_csv(self, input_file = None, output_filename = None ):
        # predict and  apply are treated equally
        # Now inspect the uploaded file and check which column to exclude.
        churn_apply_filename = input_file
        try:
            input_df = pd.read_csv(churn_apply_filename)
            # sdf = StanmoDataFrame(self.stanmoapp, self.model_name, 'churn_source', True)
        except :
            logging.getLogger('stanmo_logger').debug('Failed to read the data for prediction.')
            exit(1)

        result_df = self.predict_df(input_df)
        result_df[['record_id','prediction_result']].to_csv(path_or_buf=output_filename,index=False)


    def predict_json(self, input_json = None):
        # predict and  apply are treated equally

        # Now inspect the uploaded file and check which column to exclude.
        input_df = pd.DataFrame(input_json)
        result_df = self.predict_df(input_df)
        return result_df.to_json(orient='records') # simplejson.dumps(result_dict) #"duan" # simplejson.loads('["duan", "qiyang"]')


    def predict_user(self, x, mmi=None ):
        user_id = mmi.inv_user_ids[x]
        pred = mmi.model.recommend(user_id)
        # pred_dict=[{5: 3.3477895267131013}, {1, 2.8572508984333034}, {6, 2.4473604699719846}]
        pred_dict = []
        for p in pred:
            pred_dict.append({mmi.item_ids[p[0]]:float(p[1])})
        pred_json = json.dumps(pred_dict)
        # print(pred_json)
        return pred_json

    def predict_df(self, input_df = None ):
        mmi = self.get_champion_instance()
        result_df = input_df[['record_id']].copy()
        result_df['prediction_result'] = result_df['record_id'].apply(self.predict_user,mmi=mmi)
        result_df['prediction_probability'] = 0

        self.add_prediction_history(model_name = self.model_name, prediction_df=result_df)

        return result_df


    def fit(self, input_file = None):
        return self.fit_csv(input_file = input_file)
    def predict(self, input_file = None, output_file = None):
        return self.predict_csv(input_file = input_file, output_file = output_file)

    def show(self, port = None):
        if port is None:
            port = 5011

        url = 'http://localhost:' + str(port) + '/'

        import webbrowser
        # Open URL in new window, raising the window if possible.
        webbrowser.open_new(url)

    def run(self, port = None, to_execute=True):
        if port is None:
            port = 5011
        the_flask = ChurnModelFlask(port, to_execute,self)
        print('The server will be running on: ' + 'http://localhost:' + str(port) + '/ . Press Ctrl-C to stop the server.')
        return the_flask.run()

