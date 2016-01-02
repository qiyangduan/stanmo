__author__ = 'duan'

import unittest
import os
from unittest import TestCase
from flask.ext.testing import TestCase as FlaskTestCase
import simplejson
# from app.stanmoapp import stanmoapp
# from basemodelspec import BaseMiningModel
# from model.churnmodelspec import ChurnMiningModel

from stanmo.app import stanmoapp
from stanmo.model.churnmodelspec import ChurnMiningModel

PORT = 5011
# TESTING_FILES_DIR = 'C:\\temp\\stanmo\\model\\churn1''

class TestStanmo(TestCase):
    def test_list_model(self):
        model_list = stanmoapp.list_models()
        print(len(model_list))
        self.assertTrue(len(model_list) == 1)


class TestChurnMiningModel(FlaskTestCase):
    def create_app(self):
        self.fit_input_files = ['C:\\temp\\stanmo\\test\\churn_source.csv']
        self.predict_input_files = ['C:\\temp\\stanmo\\test\\churn_apply.csv']
        self.predict_output_file = 'C:\\temp\\stanmo\\test\\churn_apply_result.csv'
        self.test_model_name = 'churn1'

        self.fit_model_instance_id  = 1
        self.fit_model_storage_path = stanmoapp.get_model_instance_path( model_name = self.test_model_name, model_instance_id = 1)
        # 'C:\\temp\\stanmo\\model\\churn1\\instance\\1\\model_inst.pkl'
        try:
            os.remove(self.fit_model_storage_path)
            os.remove(self.predict_output_file)
        except:
            pass


        churn1 = ChurnMiningModel(stanmoapp=stanmoapp, model_name=self.test_model_name)
        app = churn1.run(port=PORT, to_execute=False)
        app.config['TESTING'] = True
        return app

    def test_predict_json(self):
        churn1 = self._fit_model()

        json_to_predict =  '[{"cust_id": "358-1921", "_sms_message_count_1m": "0", "_call_center_contact_count_1m": "0", "_call_count_off_peaktime_1m": "104", "_call_minute_off_peaktime_1m": "162.600000", "_call_count_peaktime_1m": "110", "_international_roaming_flag": "no", "_call_charge_off_peaktime_1m": "7.320000", "_call_charge_peaktime_1m": "10.300000", "_call_count_1m": "114", "_call_count_international_1m": "5", "_call_minute_peaktime_1m": "121.200000", "_churn_flag": "0", "_tenure_days": "137", "_pay_tv_flag": "no", "_call_minute_1m": "243.400000", "_call_minute_international_1m": "12.200000", "_call_charge_international_1m": "3.290000", "_zip_code": "415", "_total_revenue_1m": "41.380000", "_state_code": "NJ"}]'
        response=self.client.post('/predict_json',
                       data=json_to_predict,
                       content_type = 'application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEquals(response.json, simplejson.loads('[{"_customer_id":0,"churn_flag":0}]'))

        # negative test, wrong format
        json_to_predict =  '[{"cust_id": "358-1921"]'
        response=self.client.post('/predict_json',
                       data=json_to_predict,
                       content_type = 'application/json')
        self.assertEqual(response.status_code, 400)
        # print(response.status_code)
        # self.assertEquals(response.json, simplejson.loads('[{"_customer_id":0,"churn_flag":0}]'))


    def setUp(self):
        pass
    def tearDown(self):
        pass
    def _fit_model(self):
        churn1 = ChurnMiningModel(stanmoapp,self.test_model_name)
        churn1.fit_csv(input_files = self.fit_input_files,  algorithms=None, model_instance_id = self.fit_model_instance_id)
        return churn1

    def test_fit(self):
        self.assertTrue(os.path.exists(self.fit_model_storage_path) == 0)
        churn1 = self._fit_model()
        self.assertTrue(os.path.exists(self.fit_model_storage_path) == 1)

    def test_predict(self):
        self.assertTrue(os.path.exists(self.predict_output_file) == 0)
        churn1 = self._fit_model()
        churn1.predict_csv(self.predict_input_files, self.predict_output_file)
        self.assertTrue(os.path.exists(self.predict_output_file) == 1)

if __name__ == '__main__':
    unittest.main()