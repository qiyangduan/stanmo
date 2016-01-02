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
stanmo fit churn4 --input_file=C:\qduan\Stanmo\git\bitbucket\src\stanmo_proj\stanmo_data_to_delete\test\churn_source.csv --instance=2
stanmo predict churn4 --input=C:\qduan\Stanmo\git\bitbucket\src\stanmo_proj\stanmo_data_to_delete\test\churn_apply.csv --output=C:\qduan\Stanmo\git\bitbucket\src\stanmo_proj\stanmo_data_to_delete\test\churn_apply_result.csv

stanmo run churn4 --port=5011
stanmo show churn1  --port=5011
