stanmo
==============
The goal of stanmo project is to provide standard business models for industry usage. Examples of standard business models include churn prediction in telecom, product recommendation in online ecommerce, etc. Please refer to `Stanmo <http://www.stanmo.com/>` (website still in construction) for list of models available models.

Currently stanmo support only python 2.7 only. I do not have enough time to port onto python 3. I will do that later. I hate Guido's reckless act to make python 3 incompatible to python 2. At least I did not see any benefit of using python 3.

The problem it solves
---------------------
In my previous professional experience, I have seen many customers implementing different mining models over and over again. From the earlier days, people use very simple Decision Trees and somedays later Logistic Regressions. So i thought I can create a standard model using state of arts algorithms. If it can not be used as it is, it should at least serve as a baseline template for further model creation.


Concepts
--------

This stanmo project is the execution platform of different business models. Each business model is implemented as a "model spec" under $stanmo/spec. Each model spec should be in its own folder.  

Each business model is organized in three different layers. Each business model is first abstracted as a model spec. The model spec is normally a Python class and it inherit from the BaseMiningModel class. The model spec contains all the codes to train and predict the data. The model specs can be added by putting a new folder under $stanmo/spec. 

Then using this spec, each user should create its own model. Under the same model spec, one user can create multiple models. Each model can be trained using different datasets. Each model will have its own webservices as well. Each model will be saved as one folder under $stanmo/models, with model name as the folder name.

Under one model, different model instances may be created. When running the prediction, the most accurate model instance will be used by default. It is named champion instance.


Tests
-----

I am still working on the test. Will come later.

Installation
------------
Since the business models normally require machine learning packages like scikit-learn, numpy, scipy, you have to install those packages first. I recommend installing `anaconda<https://www.continuum.io/downloads>` to install all those  packages in one go. Then you should install the stanmo package by PIP install.
You can install ``stanmo`` with ``pip``::

    # pip install stanmo

I tried to package this by conda, but did not find a proper way to do it. Since this stanmo project is a pure Python project, so far, PIP seems work ok. 


Usage
-----

Once you installed ``stanmo`` package, the first step is to create a model based on avaiable model specs. You can first look at the availables specs and models by::

    # stanmo list specs
    # stanmo list models

Then you can create your first mdoel by churn model spec::

    # stanmo create churn1 --spec=churn.ChurnMiningModel
    # stanmo list models

Before you can use the model, you should feed in some data to fit the model. Two data files are shipped with the package under $stanmo/data. You can use those files to fit your first model and test the prediction::

    # stanmo fit churn1 --input_file=~/anaconda2/envs/conda1/lib/python2.7/site-packages/stanmo/data/churn_source.csv --instance=1
    # stanmo list models

You can predict your data by two different ways. One is through the console in a batch style, as the following::
    # stanmo predict churn2 --input=~/anaconda2/envs/conda1/lib/python2.7/site-packages/stanmo/data/churn_apply.csv --output=/tmp/churn_apply_result1.csv
    # cat /tmp/churn_apply_result1.csv
    
Another way of running prediction is to start a http server and run the prediction through the REST API::
    # stanmo runserver churn2 --port=5011 &    

Then you can use any rest api caller program to execute the prediction. 
    # pip install requests  
    # python ~/anaconda2/envs/conda1/lib/python2.7/site-packages/stanmo/test_rest_api.py
    
    

Changelog
---------

0.2.0 (2016-01-04)
*******************

* First public release.
* Includes one prebuilt model -- Telecom Churn Prediction.
* Includes webservices based on Flask.
