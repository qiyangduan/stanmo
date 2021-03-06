'''
Usage:
        stanmoctl.py -h
        stanmoctl.py pull <model_name>
        stanmoctl.py search [specs]
        stanmoctl.py list ( models | specs )
        stanmoctl.py create <model_name> --spec=<spec_name>
        stanmoctl.py fit <model_name> [--input_file=<path>]  [--instance=<id> ]
        stanmoctl.py predict <model_name> [--input_file=<path>] [--output_file=<path> ]  [--instance=<id> ]
        stanmoctl.py run <model_name>   [--port=<port> ] [--instance=<id> ]
        stanmoctl.py show <model_name>   [--port=<port> ]

C:\Users\duan>

stanmo list models
stanmo create churn5 --spec=churn.churnmodelspec.ChurnMiningModel
stanmo fit churn5 --input_file=C:\qduan\Stanmo\git\bitbucket\src\stanmo_proj\stanmo_data_to_delete\test\churn_source.csv --instance=2
stanmo predict churn5 --input=C:\qduan\Stanmo\git\bitbucket\src\stanmo_proj\stanmo_data_to_delete\test\churn_apply.csv --output=C:\qduan\Stanmo\git\bitbucket\src\stanmo_proj\stanmo_data_to_delete\test\churn_apply_result.csv

stanmo run churn5 --port=5011
stanmo show churn5  --port=5011
'''


 [{"_customer_id": "358-1921", "_sms_message_count_1m": "0", "_service_center_call_count_7day": "0", "_off_peak_time_voice_call_count_7day": "104",
 "_off_peak_time_voice_call_minute_7day": "162.600000", "_peak_time_voice_call_count_7day": "110", "_international_roaming_flag": "no",
 "_off_peak_time_voice_charge_7day": "7.320000", "_peak_time_voice_charge_7day": "10.300000", "_call_count_1m": "114", 
 "_international_voice_call_count_7day": "5", "_peak_time_voice_call_minute_1m": "121.200000", "_churn_flag": "0", 
 "_tenure_days": "137", "_pay_tv_flag": "no", "_call_minute_1m": "243.400000", 
 "_international_voice_call_minute_7day": "12.200000", "_international_voice_charge_7day": "3.290000", "_zip_code": "415", 
 "_total_revenue_1m": "41.380000", "_state_code": "NJ"}
]

