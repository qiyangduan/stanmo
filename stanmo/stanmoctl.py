#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" stanmoctl.py
    
    Usage:
        stanmo -h
        stanmo list ( models | specs )
        stanmo create <model_name> --spec=<spec_name>
        stanmo fit <model_name> [--input_file=<path>]  [--instance=<id> ]
        stanmo predict <model_name> [--input_file=<path>] [--output_file=<path> ]  [--instance=<id> ]
        stanmo runserver <model_name>   [--port=<port> ] [--instance=<id> ]
        stanmo show <model_name>   [--port=<port> ]

    Options:
        -h,--help             : show this help message
        model_name            : model name given when it was created
        spec_name             : The name of model specification
        --input_file=<path>   : Path of input files to the model
        --output_file=<path>  : Path to write the model output file.
        --instance=<id>       : The model instance ID.
        --port=<port>         : The OS port where server will be listening on. It uses 5000 if omitted..
"""
# This command push will be implemented later.
#        stanmo push [specs]
#        stanmo pull [specs]  # To download a model spec from central repository
#        stanmo search [specs]

# the above is our usage string that docopt will read and use to determine
# whether or not the user has passed valid arguments.
# following: https://github.com/docopt/docopt
from docopt import docopt
from stanmo import stanmoapp
from stanmo import StanmoError, StanmoErrorNoInstanceID

import logging
# from basemodelspec import BaseMiningModel
# in the control file, it should not deal with any specific model but only use spec and model name to call the loader

def main():
    """ main-entry point for stanmo program, parse the commands and build the stanmoapp platform """
    docopt_args = docopt(__doc__)

    # Parse the User command and the required arguments
    if docopt_args["list"]:
        if docopt_args["models"] == True:
            # print(json.dumps(stanmoapp.list_models()))
            stanmoapp.list_models()
        elif docopt_args["specs"] == True:
            model_specs = stanmoapp.list_specs()
            if len(model_specs) > 0:
                print('{0:40} {1:35} '.format('Model Spec Name:','Path:  '))
                for spec in model_specs:
                    if spec["name"] is None:
                        spec_name = 'No Name'
                    else:
                        spec_name = spec["name"]
                    print('{0:40} {1:35}'.format(spec_name,   spec["path"]) )
            else:
                print('No model specs found!')


    # Parse the User command and the required arguments
    # create <model_name> --spec=<spec_name>
    if docopt_args["create"]:
        model_name = docopt_args["<model_name>"]
        spec_name = docopt_args["--spec"]
        if spec_name is None:
            print("Please specify the <model_name> and <spec_name>")
        stanmoapp.create_model(model_name=model_name, spec_name=spec_name)
        print("Model is created successfully.")


    elif docopt_args["fit"]:
        # to predict according to trained a model, given the input file.
        input_file = None
        output_file = None
        model_instance_id  = None
        if docopt_args["--input_file"] is not None:
            input_file = docopt_args["--input_file"]
        else:
            print("Please provide input training data")

        if docopt_args["--instance"] is not None:
            model_instance_id = docopt_args["--instance"]

        model_name = docopt_args["<model_name>"]
        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.fit_csv(input_file=input_file, model_instance_id = model_instance_id)
        except StanmoErrorNoInstanceID:
            logging.getLogger('stanmo_logger').error('Can not get more free instance IDs, please specify one (by --instnace=x) to override.')
            print('Can not get more free instance IDs, please specify one to override by --instance=n.')


    elif docopt_args["predict"]:
        # to predict according to a trained model, given the input file.
        input_file = None
        output_file = None
        if docopt_args["--input_file"] is not None:
            input_file = docopt_args["--input_file"]
        else:
            print("Please provide input and output information")

        if docopt_args["--output_file"] is not None:
            output_file =  docopt_args["--output_file"]
        else:
            print("Please provide input and output information")

        #not used for now
        model_instance_id  = None
        if docopt_args["--instance"] is not None:
            model_instance_id = docopt_args["--instance"]

        model_name = docopt_args["<model_name>"]

        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.predict_csv(input_file, output_file)
        except StanmoErrorNoInstanceID:
            logging.getLogger('stanmo_logger').error('Can not get more free instance IDs, please specify one (by --instnace=x) to override.')
            print('Can not get more free instance IDs, please specify one to override by --instance=n.')


    elif docopt_args["runserver"]:
        # to run a HTTP server to provide restful api services..
        port = 5011
        model_instance_id  = None
        if docopt_args["--port"] is not None:
            port =  docopt_args["--port"]
        if docopt_args["--instance"] is not None:
            model_instance_id = docopt_args["--instance"]

        model_name = docopt_args["<model_name>"]
        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.run(port=port)
        except StanmoError as e:
            print(' The server failed to start: {0}'.format(e.strerror))


    elif docopt_args["show"]:
        # to predict according to trained a model, given the input file.
        port = 5011
        if docopt_args["--port"] is not None:
            port =  docopt_args["--port"]

        model_name = docopt_args["<model_name>"]
        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.show(port=port)
        except StanmoError as e:
            print('Failed to show the model: {0}'.format(e.strerror))


# START OF SCRIPT
if __name__ == "__main__":
    main()


