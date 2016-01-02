import pandas as pd
import numpy as np
import simplejson

from stanmo.app.basemodelspec import BaseMiningModel, MiningModelInstance, BaseInputDataEncoder
from stanmo.app import StanmoErrorNoInstanceID

from .models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender



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

    def load_rating_file(self,rating_filename=None):
        #Read data
        data_m = np.loadtxt(rating_filename, delimiter=';', dtype=str)
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
        return Bunch(data=data_songs, item_ids=data_titles,
                     user_ids=data_users, DESCR=None)

    def fit_csv(self,input_files=None, algorithms=None, model_instance_id = None):
        if input_files is None:
            self.stanmoapp.logger.debug('Please specify the input file for ETL.')
            exit(1)
        rating_filename =  input_files[0]
        training_data = self.load_rating_file(rating_filename)
        model = MatrixPreferenceDataModel(training_data.data)
        similarity = UserSimilarity(model, pearson_correlation)

        mmi = MiningModelInstance(model = model,
                                  similarity = similarity
                                  )
        self.set_model_instance(mmi,curr_model_instance_id)


    def predict_df(self, input_df = None ):
        mmi = self.get_champion_instance()
        # apply_df = pd.read_csv(self.stanmoapp.get_dataframe_path(self.model_name, self.input_dataframes[INPUT_DF_NAME].apply_name))
        # I first find all column names and then join with drop_list,
        # X_pred,pk_df = self.encode(input_df, mmi)
        recommender = UserBasedRecommender(mmi.model, mmi.similarity, with_preference=True)
        #Recommend items for the users one by one
        cust_list = input_df['cust_id'].values
        result_df = input_df.copy()
        result_df['item_json'] = ''
        for cust_id in cust_list:
            items = recommender.recommend(cust_id)
            result_df[result_df.cust_id == cust_id]['item_json'] = simplejson.dumps(items)

        return result_df

    def predict_cust(self, cust_id = None ):
        mmi = self.get_champion_instance()
        recommender = UserBasedRecommender(mmi.model, mmi.similarity, with_preference=True)
        items = recommender.recommend(cust_id)
        return items

    def predict_csv(self, input_files = None, output_filename = None ):
        # predict and  apply are treated equally

        # Now inspect the uploaded file and check which column to exclude.
        churn_apply_filename = input_files[0]
        try:
            input_df = pd.read_csv(churn_apply_filename)
            # sdf = StanmoDataFrame(self.stanmoapp, self.model_name, 'churn_source', True)
        except:
            logger = self.stanmoapp.logger
            logger.debug('Please specify the right target of ETL for its purpose: train or apply')
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

        from flask import Flask,jsonify
        from flask import render_template

        app = Flask(__name__,
                    template_folder = 'C:\\qduan\\Stanmo\\flask\\flask\\churn_dashboard\\templates',
                    static_folder='C:\\qduan\\Stanmo\\flask\\flask\\churn_dashboard\\static')

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/get_input_attribute_names")
        def get_input_attribute_names():
            mmi = self.get_champion_instance()
            # attr_list = mmi.input_stanmodataframe.keeping_columns.keys()
            attr_list = ["_state_code","_tenure_days","_customer_id"]
            return  simplejson.dumps(attr_list) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/get_churners")
        def get_churners():
            with open("C:\\qduan\\Stanmo\\flask\\flask\\churn_dashboard\\static\\json\\churners1.json") as f:
                json_projects = simplejson.load(f)
            return  simplejson.dumps(json_projects) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/predict_json", methods=['POST'])
        def predict_json():
            from flask import abort   # , methods=['POST']
            from flask import request
            if not request.json:
                abort(400)
            json_customers = request.json #  [] _customer_id

            #json_customers = {"cust_id": "358-1921"}
            predict_result = self.predict_cust(cust_id = json_customers['cust_id'])
            return  simplejson.dumps(predict_result) #"duan" # simplejson.loads('["duan", "qiyang"]')

        if to_execute:
            app.run(host='0.0.0.0',port=port,debug=False, threaded=True)
        else: # for testing purpose, and this app will be returned
            return app

