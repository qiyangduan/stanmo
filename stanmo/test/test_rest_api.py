import requests
import simplejson as json
headers = {'content-type': 'application/json'}
url = 'http://localhost:5011/predict_json'
data = ' [{"_customer_id": "358-1921", "_sms_message_count_1m": "0", "_service_center_call_count_7day": "0", "_off_peak_time_voice_call_count_7day": "104", "_off_peak_time_voice_call_minute_7day": "162.600000", "_peak_time_voice_call_count_7day": "110", "_international_roaming_flag": "no",  "_off_peak_time_voice_charge_7day": "7.320000", "_peak_time_voice_charge_7day": "10.300000", "_call_count_1m": "114",  "_international_voice_call_count_7day": "5", "_peak_time_voice_call_minute_1m": "121.200000", "_churn_flag": "0",  "_tenure_days": "137", "_pay_tv_flag": "no", "_call_minute_1m": "243.400000",  "_international_voice_call_minute_7day": "12.200000", "_international_voice_charge_7day": "3.290000", "_zip_code": "415",   "_total_revenue_1m": "41.380000", "_state_code": "NJ"}]'
myResponse = requests.post(url, data=data, headers=headers)
myResponse

# For successful API call, response code will be 200 (OK)
if(myResponse.ok):
    # Loading the response data into a dict variable
    # json.loads takes in only binary or string variables so using content to fetch binary content
    # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
    jData = json.loads(myResponse.content)
    print("The response contains {0} properties".format(len(jData)))
    print("\n")
    for cust in jData:
        print cust 
else:
    # If response code is not ok (200), print the resulting http error code with description
    myResponse.raise_for_status()
    