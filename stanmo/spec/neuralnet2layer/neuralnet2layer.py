import simplejson
import logging

import pandas as pd
import numpy as np
from six.moves import cPickle


from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()
BATCH_SIZE = 128

from stanmo.app.basemodelspec import BaseMiningModel, MiningModelInstance, BaseInputDataEncoder
from stanmo.app import StanmoErrorNoInstanceID




ALLOWED_NUMERICAL_DTYPES = [np.dtype('int64'), np.dtype('int32'), np.dtype('float32'), np.dtype('float64')]
CATEGORICAL_ATTRIBUTE_CADINALITY_THRESHOLD = 8




class GeneralClassificationInputEncoder:


    def fit(self, input_df=None):
        """  The ETL logic from customer sources into the input dataframes as defined in input_dataframe_spec.
             If you use input dataframe as it is, you may simply implement it by single 'pass' command.
             The fit command must be called with self.curr_sdf.df properly setup.
        :param input_file:
        :return:
        """
        chunk_df = input_df

        ad = chunk_df.dtypes
        column_df = pd.DataFrame(ad.index.tolist(), columns=['col'])
        column_df['typ'] = ad.values.tolist()

        # str(column_df.typ[1])

        def obj_type_2_str(x):
            return str(x)

        column_df['typ_str'] = column_df['typ'].apply(obj_type_2_str)

        num_df = chunk_df.select_dtypes(include=['int64', 'float64'])
        # cat_df = chunk_df.select_dtypes(include=['object'])
        num_stat = num_df.describe().transpose()
        num_stat['mean'] = 1
        # num_stat.groupby('count').count()

        # only if at least 2/3 of the cells are populated.
        df_num_of_rows = chunk_df.shape[0]
        if df_num_of_rows < 10:
            num_of_rows_threshold = 9
        else:
            num_of_rows_threshold = df_num_of_rows * 2 / 3

        list_of_valid_num_cols = num_stat[num_stat['count'] > num_of_rows_threshold].index.tolist()
        # num_of_rows_threshold


        '''
        # To train the dict vectorizer for all categorical values.
        cat_train = input_df[all_source_categorical_names]
        x_cat_train = cat_train.T.to_dict().values()
        vectorizer = DV(sparse=False)
        self.cat_encoder = vectorizer.fit(x_cat_train)
        '''

        # To train the scaler for all numerical values.
        num_train1 = chunk_df[list_of_valid_num_cols]
        num_train = num_train1.as_matrix().astype(np.float32)
        s_scaler = StandardScaler()
        self.scaler = s_scaler.fit(num_train)
        self.list_of_valid_num_cols = list_of_valid_num_cols
        print('fitting completed.')
        return self


    def transform(self, input_df=None):
        '''
        It transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        X should be a pandas data frame.
        '''


        # Now transform all numerical values.
        num_train1 = input_df[self.list_of_valid_num_cols]
        num_train = num_train1.as_matrix().astype(np.float32)
        trans_num_train = self.scaler.transform(num_train)
        X = trans_num_train
        y = input_df['_churn_flag'].as_matrix().astype(np.float32)

        '''
        # Now transform all categorical values.
        cat_train = input_df[all_source_categorical_names]
        x_cat_train = cat_train.T.to_dict().values()
        trans_cat_train = self.cat_encoder.transform(x_cat_train)
        # Concatenate numerical and categorical into a single matrix for training.
        X = np.hstack((trans_num_train, trans_cat_train))
        '''
        return X,y




def one_hot(x,n):
	# if type(x) == list:
	x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x


class NeuralNet2Layer(BaseMiningModel):
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
        #self.input_dataframe_names = [INPUT_DF_NAME] # self._loaded_model.get('output_dataframe_names')
        #self.output_dataframe_names = [OUTPUT_DF_NAME] # self._loaded_model.get('output_dataframe_names')
        #self.load_dataframes(dataframe_names={'input_df_names':[INPUT_DF_NAME], 'output_df_names':[OUTPUT_DF_NAME]})


    def fit_df(self, input_df=None, algorithms=None, model_instance_id=None):
        if model_instance_id is None:
            curr_model_instance_id = 1
        else:
            curr_model_instance_id = model_instance_id

        if algorithms is None:
            one_algorithm = RF
            algorithms = [RF]

        enc = GeneralClassificationInputEncoder()
        # fit the encoder.
        X, y = enc.fit(input_df).transform(input_df)

        # X, y = ChurnInputDataEncoder().fit(input_df).transform(input_df)
        num_of_features = X.shape[1]

        # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        trX, teX, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        trY = one_hot(Y_train, n=2)
        teY = one_hot(Y_test, n=2)

        X = T.fmatrix()
        Y = T.fmatrix()
        w_h = init_weights((num_of_features, 400))
        w_h2 = init_weights((400, 50))
        w_o = init_weights((50, 2))

        noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
        h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
        y_x = T.argmax(py_x, axis=1)

        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
        params = [w_h, w_h2, w_o]
        updates = RMSprop(cost, params, lr=0.001)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        for i in range(3):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                cost = train(trX[start:end], trY[start:end])
            print(i, np.mean(np.argmax(teY, axis=1) == predict(teX)))

        neural_model = {'w_h': w_h, 'w_h2': w_h2, 'w_o': w_o, 'enc':enc}
        f = open('/tmp/obj.save', 'wb')
        cPickle.dump(neural_model, f)
        f.close()

    def predict_df(self, input_df = None ):

        f = open('/tmp/obj.save', 'rb')
        neural_model = cPickle.load(f)
        f.close()

        X, y = neural_model['enc'].transform(input_df)
        # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        trX, teX, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        trY = one_hot(Y_train, n=2)
        teY = one_hot(Y_test, n=2)

        X = T.fmatrix()
        Y = T.fmatrix()

        h, h2, py_x = model(X, neural_model['w_h'], neural_model['w_h2'], neural_model['w_o'], 0., 0.)
        y_pred = T.argmax(py_x, axis=1)

        cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        gradient = T.grad(cost=cost, wrt=w)
        update = [[w, w - gradient * 0.05]]

        train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
        print('Loaded precision:' , np.mean(np.argmax(teY, axis=1) == predict(teX)))


        return predict(teX)

    def fit(self, input_df=None):
        return self.fit_df(input_df=input_df)

    def predict(self, input_df=None):
        return self.predict_df(input_df=input_df)

    def fit_predict(self, X=None,y=None):
        # only for testing
        #trX, teX, trY, teY = mnist(onehot=True)

        # X, y = ChurnInputDataEncoder().fit(input_df).transform(input_df)
        num_of_features = X.shape[1]

        # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        trX, teX, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        trY = one_hot(Y_train,n=2)
        teY = one_hot(Y_test, n=2)

        X = T.fmatrix()
        Y = T.fmatrix()
        w_h = init_weights((num_of_features, 1440))
        w_h2 = init_weights((1440, 100))
        w_o = init_weights((100, 2))

        noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
        h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
        y_x = T.argmax(py_x, axis=1)

        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
        params = [w_h, w_h2, w_o]
        updates = RMSprop(cost, params, lr=0.001)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        for i in range(3):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                cost = train(trX[start:end], trY[start:end])
            print(i, np.mean(np.argmax(teY, axis=1) == predict(teX)))

        w= {'w_h':w_h, 'w_h2':w_h2, 'w_o':w_o}
        f = open('/tmp/obj.save', 'wb')
        cPickle.dump(w, f)
        f.close()





def convert_orange_data():
    import pandas as pd
    # chunk_file = '/media/wdc_d/QQDownload/orange_data/chunk1.100'
    # label_file = '/media/wdc_d/QQDownload/orange_data/label.99'

    #chunk_file = '/media/wdc_d/QQDownload/orange_data/2000.chunk'
    #label_file = '/media/wdc_d/QQDownload/orange_data/2000.label'

    chunk_file = '/media/wdc_d/QQDownload/orange_data/1000.chunk.predict'
    label_file = '/media/wdc_d/QQDownload/orange_data/1000.label.predict'

    chunk_df = pd.read_csv(chunk_file, sep='\s+')
    num_df = chunk_df # chunk_df.select_dtypes(include=['int64', 'float64'])


    label_df = pd.read_csv(label_file, sep='\s+')
    label_df[label_df['churn']==-1]=0
    label_df.rename(index=str, columns={"churn": "_churn_flag"})
    label_df['_customer_id']=label_df.index

    # label_df.index


    orange_df = pd.concat([num_df, label_df], axis=1)

    # In[43]:
    orange_df.to_csv('/media/adata/qduan/Stanmo/git/github/stanmo/stanmo/data/orange_data_1000.predict.csv',
                     sep=',', encoding='utf-8')


def load_orange_data():
    import pandas as pd
    #chunk_file = '/media/wdc_d/QQDownload/orange_data/chunk1.100'
    #label_file = '/media/wdc_d/QQDownload/orange_data/label.99'

    #chunk_file = '/media/wdc_d/QQDownload/orange_data/2000.chunk'
    #label_file = '/media/wdc_d/QQDownload/orange_data/2000.label'

    chunk_file = '/media/wdc_d/QQDownload/orange_data/10000.chunk1'
    label_file = '/media/wdc_d/QQDownload/orange_data/9999.label'

    source_df = pd.read_csv(chunk_file, sep='\s+')
    chunk_df = GeneralClassificationInputEncoder().fit(source_df).transform(source_df)

    label_df = pd.read_csv(label_file, sep='\s+')
    # orange_orig_df = pd.concat([chunk_df, label_df], axis=1)
    X = chunk_df
    y = label_df.as_matrix()

    return X, y



def main():
    X,y = load_orange_data()

    X_train, X_test,Y_train, Y_test  = train_test_split( X, y, test_size=0.33, random_state=42)

    clf = RF()
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    rf_precision = np.mean(Y_test == Y_pred)
    print('Random Forest precison: {0}'.format(rf_precision))

    #  This converts (-1,1) to (0,1) to make sure the one_hot encoder work properly.
    c=pd.DataFrame(y)
    c[c==-1]=0
    y_01=c.as_matrix()

    nlr = NeuralNet2Layer()
    nlr.fit_predict(X=X,y=y_01)
    # nlr.predict(input_df=source_df)


    # Y_pred = nlr.predict(X_test)
    # rf_precision = np.mean(Y_test == Y_pred)


if __name__ == '__main__':
    # main()
    convert_orange_data()
