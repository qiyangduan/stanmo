from flask import Flask,jsonify
from flask import render_template
import os
# from flask.ext.sqlalchemy import SQLAlchemy
import simplejson
from flask import abort   # , methods=['POST']
from flask import request

class ChurnModelFlask:
    def __init__(self,port=None, to_execute=None, the_model=None):
        self.port = port
        self.to_execute = to_execute
        self.the_model = the_model

    def run(self):
        SPEC_INSTALL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'web'))
        app = Flask(__name__,
                    template_folder = os.path.join(SPEC_INSTALL_PATH,'templates'),
                    static_folder   = os.path.join(SPEC_INSTALL_PATH,'static') )

        '''app = Flask(__name__,
                    template_folder = 'C:\\qduan\\Stanmo\\flask\\flask\\churn_dashboard\\templates',
                    static_folder='C:\\qduan\\Stanmo\\flask\\flask\\churn_dashboard\\static')
       '''
        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/get_input_attribute_names")
        def get_input_attribute_names():
            # mmi = the_model.get_champion_instance()
            # attr_list = mmi.input_stanmodataframe.keeping_columns.keys()
            attr_list = ["_state_code","_tenure_days","_customer_id"]
            return  simplejson.dumps(attr_list) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/get_churners")
        def get_churners():
            with open("C:\\qduan\\Stanmo\\flask\\flask\\churn_dashboard\\static\\json\\churners1.json") as f:
                json_projects = simplejson.load(f)
            return  simplejson.dumps(json_projects) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/get_cumulative_gain_data")
        def get_cumulative_gain_data():
            # This is actaully the ROC Chart. Name might be misleading. WIll change later.
            chart_data = self.the_model.get_roc_curve_all(model_name=self.the_model.model_name)

            return  simplejson.dumps(chart_data) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/get_prediction_history")
        def get_prediction_history():
            pred_df1 = self.the_model.get_daily_prediction_count(model_name=self.the_model.model_name) # calculate_daily_precision
            pred_df = pred_df1.astype(float)
            print("prediciton begin.")
            adata = {
                'labels' : pred_df.index.tolist(), #  ["January","February","March","April","May","June","July"],
                'datasets' : [
                    {
                        'label' : "My Second dataset",
                        'fillColor' : "rgba(151,187,205,0)",
                        'strokeColor' : "rgba(151,187,205,1)",
                        'pointColor' : "rgba(151,187,205,1)",
                        'pointStrokeColor' : "#fff",
                        'pointHighlightFill' : "#fff",
                        'pointHighlightStroke' : "rgba(151,187,205,1)",
                        'data' :  pred_df.tolist() # [60, 97, 95, 12, 38, 37, 34]
                    }
                ]
            }
            return  simplejson.dumps(adata) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/get_overall_statistics")
        def get_overall_statistics():
            # prediction_count = get_number_of_predictions(model_name=self.the_model.model_name)
            print("get for model : " + self.the_model.model_name)
            overall_stat = self.the_model.get_overall_statistics(model_name=self.the_model.model_name)
            adata={
                "model_storage_path": self.the_model.stanmoapp.get_model_spec_path(self.the_model.model_name), # "c:\\temp\churn1",
                "number_of_model_instances": len(self.the_model.model_instances),
                "number_of_predictions": overall_stat['total_prediction_count'],
                "number_of_feedbacks": overall_stat['total_feedback_count'],
                "input_attributes":  ['column1', 'column2', 'column 3', 'col3', 'col5'],
                "output_attributes":  ['column11', 'column12', 'column1 3', 'col13', 'co1l5','column11', 'column12', 'column1 3', 'col13', 'co1l5'],
                "overall_precision": overall_stat['total_precision']
                }
            print(simplejson.dumps(adata))
            return  simplejson.dumps(adata) #"duan" # simplejson.loads('["duan", "qiyang"]')

        @app.route("/set_feedback_json", methods=['POST'])
        def set_feedback_json():
            if not request.json:
                abort(400)
            feedback_json = request.json #  [] _customer_id
            predict_result = self.the_model.set_feedback_json(model_name=self.the_model.model_name, feedback_json = feedback_json)
            return simplejson.dumps({"status":"ok"})


        @app.route("/predict_json", methods=['POST'])
        def predict_json():
            # from flask import abort   # , methods=['POST']
            # from flask import request
            if not request.json:
                abort(400)
            json_customers = request.json #  [] _customer_id

            #json_customers = [{"cust_id": "358-1921", "_sms_message_count_1m": "0", "_call_center_contact_count_1m": "0", "_call_count_off_peaktime_1m": "104", "_call_minute_off_peaktime_1m": "162.600000", "_call_count_peaktime_1m": "110", "_international_roaming_flag": "no", "_call_charge_off_peaktime_1m": "7.320000", "_call_charge_peaktime_1m": "10.300000", "_call_count_1m": "114", "_call_count_international_1m": "5", "_call_minute_peaktime_1m": "121.200000", "_churn_flag": "0", "_tenure_days": "137", "_pay_tv_flag": "no", "_call_minute_1m": "243.400000", "_call_minute_international_1m": "12.200000", "_call_charge_international_1m": "3.290000", "_zip_code": "415", "_total_revenue_1m": "41.380000", "_state_code": "NJ"},
            #                  {"cust_id": "358-1922", "_sms_message_count_1m": "0", "_call_center_contact_count_1m": "0", "_call_count_off_peaktime_1m": "104", "_call_minute_off_peaktime_1m": "162.600000", "_call_count_peaktime_1m": "110", "_international_roaming_flag": "no", "_call_charge_off_peaktime_1m": "7.320000", "_call_charge_peaktime_1m": "10.300000", "_call_count_1m": "114", "_call_count_international_1m": "5", "_call_minute_peaktime_1m": "121.200000", "_churn_flag": "0", "_tenure_days": "137", "_pay_tv_flag": "no", "_call_minute_1m": "243.400000", "_call_minute_international_1m": "12.200000", "_call_charge_international_1m": "3.290000", "_zip_code": "415", "_total_revenue_1m": "41.380000", "_state_code": "NJ"}]

            new_json_customers = []
            for customer in json_customers:
                new_customer = self.the_model.create_default_record()
                for customer_attr in customer.keys():
                    new_customer[customer_attr] = customer[customer_attr]
                new_json_customers.append(new_customer)
            predict_result = self.the_model.predict_json(new_json_customers)
            return  predict_result # simplejson.dumps(predict_result) #"duan" # simplejson.loads('["duan", "qiyang"]')

        #@app.route('/')
        #def hello_world_index():
        #   return 'Hello World'

        if self.to_execute:
            app.run(host='0.0.0.0',port=self.port,debug=False, threaded=True)
        else: # for testing purpose, and this app will be returned
            return app

if __name__ == "__main__":

    port = 5010
    the_flask = ChurnModelFlask(port, True)
    the_flask.run()
