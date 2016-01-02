#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" stanmoctl.py
    
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

    Options:
        -h,--help         : show this help message
        model_name        : model name given when it was created
        spec_name         : The name of model specification
        --input_file=<path>   : Path of input files to the model
        --output_file=<path>  : Path to write the model output file.
        --instance=<id> : The model instance ID.
        --port=<port>   : The OS port where server will be listening on. It uses 5000 if omitted..
"""
# This command push will be implemented later.
#        stanmoctl.py push <spec_name>

# the above is our usage string that docopt will read and use to determine
# whether or not the user has passed valid arguments.

# I removed show function, since it is not core.
#         stanmoctl.py show <model_name>  [--port=<port> ]

# following: https://github.com/docopt/docopt

# import the docopt function from the docopt module
from docopt import docopt
from stanmo import stanmoapp
from stanmo import StanmoError, StanmoErrorNoInstanceID

import logging
# from basemodelspec import BaseMiningModel
#import sys
# sys.path.append('C:\\qduan\\Stanmo\\git\\bitbucket\\src\\stanmo')
# sys.path.remove('C:\\qduan\\Stanmo\\git\\bitbucket\\src\\stanmo_proj\\stanmo')
#print sys.path

# from stanmo.model.churnmodelspec import ChurnMiningModel

def main():
    """ main-entry point for stanmo program, parse the commands and build the stanmoapp platform """
    docopt_args = docopt(__doc__)


    # Parse the User command and the required arguments
    if docopt_args["list"]:
        if docopt_args["models"] == True:
            # print "You have used the list models: "
            # print json.dumps(stanmoapp.list_models())
            stanmoapp.list_models()
        elif docopt_args["specs"] == True:
            print "Not implemented. "
        # print "Listing models."

    # Parse the User command and the required arguments
    # create <model_name> --spec=<spec_name>
    if docopt_args["create"]:
        model_name = docopt_args["<model_name>"]
        spec_name = docopt_args["--spec"]
        if spec_name is None:
            # print "You have used the list models: "
            # print json.dumps(stanmoapp.list_models())
            print("Please specify the <model_name> and <spec_name>")
        #try:
        stanmoapp.create_model(model_name=model_name, spec_name=spec_name)
        #except StanmoError as e:
        #    print('Error during model creation: {0}'.format(e.strerror) )
        print("Model is created successfully.")


    elif docopt_args["fit"]:
        # to predict according to trained a model, given the input file.
        input_file = None
        output_file = None
        model_instance_id  = None
        if docopt_args["--input_file"] is not None:
            input_file = docopt_args["--input_file"]
        else:
            print "Please provide input training data"

        if docopt_args["--instance"] is not None:
            model_instance_id = docopt_args["--instance"]
        else:
            model_instance_id = None

        model_name = docopt_args["<model_name>"]
        # print(model_name)
        # return
        the_model = stanmoapp.load_model(model_name=model_name)
        # churn1 = ChurnMiningModel(stanmoapp, 'churn1')
        # churn1.etl([input_file])
        try:
            the_model.fit_csv(input_files=[input_file], model_instance_id = model_instance_id)
        except StanmoErrorNoInstanceID:
            logging.getLogger('stanmo_logger').error('Can not get more free instance IDs, please specify one (by --instnace=x) to override.')
            print('Can not get more free instance IDs, please specify one to override by --instance=n.')


    elif docopt_args["predict"]:
        # to predict according to trained a model, given the input file.
        input_file = None
        output_file = None
        model_instance_id  = None
        if docopt_args["--input_file"] is not None:
            input_file = docopt_args["--input_file"]
        else:
            print "Please provide input and output information"

        if docopt_args["--output_file"] is not None:
            output_file =  docopt_args["--output_file"]
        else:
            print "Please provide input and output information"

        if docopt_args["--instance"] is not None:
            model_instance_id = docopt_args["--instance"]

        model_name = docopt_args["<model_name>"]
        #churn1 = ChurnMiningModel(stanmoapp, model_name)
        #churn1.predict_csv([input_file], output_file)

        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.predict_csv([input_file], output_file)
        except StanmoErrorNoInstanceID:
            logging.getLogger('stanmo_logger').error('Can not get more free instance IDs, please specify one (by --instnace=x) to override.')
            print('Can not get more free instance IDs, please specify one to override by --instance=n.')


    elif docopt_args["run"]:
        # to predict according to trained a model, given the input file.
        port = None
        model_instance_id  = None
        if docopt_args["--port"] is not None:
            port =  docopt_args["--port"]
        if docopt_args["--instance"] is not None:
            model_instance_id = docopt_args["--instance"]

        #churn1 = ChurnMiningModel(stanmoapp, 'churn1')
        #churn1.run(port=port)
        model_name = docopt_args["<model_name>"]
        #churn1 = ChurnMiningModel(stanmoapp, model_name)
        #churn1.predict_csv([input_file], output_file)
        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.run(port=port)
        except StanmoError as e:
            print('Failed to run model: {0}'.format(e.strerror))


    elif docopt_args["show"]:
        # to predict according to trained a model, given the input file.
        if docopt_args["--port"] is not None:
            port =  docopt_args["--port"]

        model_name = docopt_args["<model_name>"]
        #churn1 = ChurnMiningModel(stanmoapp, model_name)
        #churn1.predict_csv([input_file], output_file)
        the_model = stanmoapp.load_model(model_name=model_name)
        try:
            the_model.show(port=port)
        except StanmoError as e:
            print('Failed to run model: {0}'.format(e.strerror))

        #churn1 = ChurnMiningModel(stanmoapp, 'churn1')
        # churn1.show(port=port)



    # We have valid args, so run the program.
    #stanmoapp.logger.debug('often makes a very good meal of %s', 'visiting tourists')
    #stanmoapp.logger.warning('Watch out!') # will print a message to the console


# START OF SCRIPT
if __name__ == "__main__":
    # Docopt will check all arguments, and exit with the Usage string if they
    # don't pass.
    # If you simply want to pass your own modules documentation then use __doc__,
    # otherwise, you would pass another docopt-friendly usage string here.
    # You could also pass your own arguments instead of sys.argv with: docopt(__doc__, argv=[your, args])
    main()


