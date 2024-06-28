import db2 as db

from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import Input


from keras import backend as K
from keras.layers import Activation
from keras.utils import get_custom_objects


import tensorflow as tf

import numpy as np

#************************************************

dict_db = db.dict_db
dict_dl = db.dict_dl

folder_best_train   = db.folder_best_train
folder_best_val     = db.folder_best_val
folder_best_val_all = db.folder_best_val_all

path_flag_stop = db.path_flag_stop

#************************************************

patience = 5000
learning_rate_beg = 1.0E-3
learning_rate_end = 1.0E-8
learning_rate_ratio = 1.5

#************************************************

def build_model(dbin, dbout, lays, init_weights=None):

    # def my_loss(y_true, y_pred):
    #     squared_difference = tf.square(y_true - y_pred)
    #     return tf.reduce_mean(squared_difference, axis=-1)
    
    #************************************************

    in_shape  = (dbin.shape[1],)
    out_shape = (dbout.shape[1])

    #************************************************

    input_tensor = Input(shape=in_shape)

    x = input_tensor

    for lay in lays:
        # x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dense(lay[0], activation=lay[1])(x)
    
    output_tensor = layers.Dense(out_shape, activation=dict_dl['last_activation'])(x)

    model = models.Model(input_tensor, output_tensor)

    #************************************************

    if init_weights is not None:
        model.set_weights(init_weights)

    #************************************************

    myAdam = optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )

    model.compile(optimizer=myAdam,
                  loss=dict_dl['loss'],
                  metrics=dict_dl['metrics'])

    return model

class StopLearning(callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):

        def get_flag_stop():
            f = 0

            with open(self.filepath, 'r') as file:
                try:
                    f = int(file.read())
                except:
                    pass
            
            return f

        if epoch % 100 == 99 and get_flag_stop():
            self.model.stop_training = True

class SaveBestWeights(callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

        self.logs_keys    = 0

        self.best_dict         = {}
        self.best_weights_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.logs_keys = list(logs.keys())
            for key in self.logs_keys:
                self.best_dict[key] = np.Inf
                self.best_weights_dict[key] = self.model.get_weights()

        for key in self.logs_keys:
            curr = logs.get(key)
            if np.less(curr, self.best_dict[key]):
                self.best_dict[key] = curr
                self.best_weights_dict[key] = self.model.get_weights()
    
    def on_train_end(self, logs=None):
        for key in self.logs_keys:
            self.model.set_weights(self.best_weights_dict[key])
            self.model.save(self.filepath+r'/model_'+key)

class CustomLearningRateScheduler(callbacks.Callback):

    def __init__(self, filepath, learning_rate_beg=1.E-3, learning_rate_end=1.E-8, learning_rate_ratio=1.5, patience=1000):
        super().__init__()
        self.filepath = filepath

        self.lrs = learning_rate_beg
        self.lre = learning_rate_end
        self.lrr = learning_rate_ratio

        self.patience = patience

    def on_train_begin(self, logs=None):
        self.wait         = 0
        self.best         = np.Inf
        self.best_mape    = np.Inf
        self.best_weights = self.model.get_weights()

        self.logs_keys = 0
        self.best_dict = {}

        self.logs_metrics_str = []

        tf.keras.backend.set_value(self.model.optimizer.lr, self.lrs)

    def on_epoch_end(self, epoch, logs=None):

        if epoch == 0:
            self.logs_keys = list(logs.keys())
            for key in self.logs_keys:
                self.best_dict[key] = np.Inf

            log_str = ('%13s ' + '|%12s '*len(self.logs_keys)) % tuple(['learning rate'] + self.logs_keys)
            self.logs_metrics_str.append(log_str)
            print(log_str)

        for key in self.logs_keys:
            curr = logs.get(key)
            if np.less(curr, self.best_dict[key]):
                self.best_dict[key] = curr

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if np.less(logs.get('val_loss'), self.best):
            self.best = logs.get('val_loss')
            self.best_mape = logs.get('val_mape')
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                
                if (lr/self.lrr < self.lre):
                    
                    self.model.stop_training = True

                tf.keras.backend.set_value(self.model.optimizer.lr, lr/self.lrr)
                
                log_str = ('%13.2e '+'|%12.2e '*len(self.logs_keys)) % tuple([lr/self.lrr] + [self.best_dict[key] for key in self.logs_keys])
                self.logs_metrics_str.append(log_str)
                print(log_str)
        
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        self.model.save(self.filepath+folder_best_val)

        with open(self.filepath+folder_best_val + '/logs_metrics', 'w') as file_log:
            print(self.filepath+folder_best_val + '/logs_metrics')
            for s in self.logs_metrics_str:
                file_log.write(s + '\n')


#************************************************

path_cur_model = db.get_path_model()

callbacks_list = [
    CustomLearningRateScheduler(filepath=path_cur_model, patience=patience, learning_rate_beg=learning_rate_beg, learning_rate_end=learning_rate_end, learning_rate_ratio=learning_rate_ratio),
    StopLearning(filepath=path_flag_stop),
    SaveBestWeights(filepath=path_cur_model)
]

def fit(dict_x, dict_y, debug=False, max_epoch_on_debug=10, verbose='auto'):

    def custom_tanh(x):
        return 1.7159 * K.tanh(2 / 3 * x)
    
    get_custom_objects().update({'ctanh': Activation(custom_tanh)})

    layers     = dict_dl['layers']
    epochs     = dict_dl['epochs']
    batch_size = dict_dl['batch']

    if debug:
        epochs = max_epoch_on_debug
    
    model = build_model(
                    dict_x['train'],
                    dict_y['train'], 
                    layers,
                    init_weights=None)
    
    history = model.fit(
                dict_x['train'], 
                dict_y['train'],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(dict_x['val'], dict_y['val']),
                callbacks=[callbacks_list],
                verbose=verbose          
            )
        
    return history

#************************************************

def save_history(history, path_model=path_cur_model, folder=folder_best_val):
    path = path_model + folder + r'/'
    np.save(path + 'history.npy', history.history)

def load_history(path_mod=path_cur_model, folder=folder_best_val):
    path = path_mod + folder + r'/'
    history = np.load(path + 'history.npy', allow_pickle=True).item()
    return history

#***********************************************

def get_dict_error(model, dict_x, dict_y):

    def loss1_func(x):
        return np.abs(x)

    def loss2_func(x):
        return np.square(x)

    def loss3_func(x):
        return x
    
    def get_dict_risk(y, y_pred, loss):

        relative_error         = loss(y - y_pred) / loss(y)

        overall_relative_error = np.mean(relative_error, axis=1)

        mean_overall_relative_error  = np.mean(overall_relative_error)
        std__overall_relative_error  = np.std( overall_relative_error)

        mean_separate_relative_error = np.mean(relative_error, axis=0)
        std__separate_relative_error = np.std( relative_error, axis=0)

        dict_risk = {
            'rel_err':           relative_error,
            'over_rel_err':      overall_relative_error,

            'mean_over_rel_err': mean_overall_relative_error,
            'std__over_rel_err': std__overall_relative_error,

            'mean_sep_rel_err':  mean_separate_relative_error,
            'std__sep_rel_err':  std__separate_relative_error
        }
        
        return dict_risk

    def get_orig_db(dict_db):

        mean = dict_db['mean']
        std  = dict_db['std']

        dict_orig_db = {}

        for key in ['db', 'train', 'val', 'test']:
            dict_orig_db[key] = dict_db[key] * std + mean
        
        return dict_orig_db

    dict_pred_y = {
        'db':    model.predict(dict_x['db'],    verbose=False),
        'train': model.predict(dict_x['train'], verbose=False),
        'val':   model.predict(dict_x['val'],   verbose=False),
        'test':  model.predict(dict_x['test'],  verbose=False),

        'mean':  dict_y['mean'],
        'std':   dict_y['std']
    }

    dict_opy = get_orig_db(dict_pred_y)
    dict_oy  = get_orig_db(dict_y)

    loss = loss1_func

    dict_err = {
        'train': get_dict_risk(dict_opy['train'], dict_oy['train'], loss),
        'val':   get_dict_risk(dict_opy['val'],   dict_oy['val'],   loss),
        'test':  get_dict_risk(dict_opy['test'],  dict_oy['test'],  loss),
    }

    return dict_err

def save_best_val_all(dict_x, dict_y, history):
    
    model_cur  = models.load_model(path_cur_model+folder_best_val)

    try:
        model_best = models.load_model(path_cur_model+folder_best_val_all)
    except:
        model_cur.save(path_cur_model+folder_best_val_all)
        save_history(history, folder=folder_best_val_all)
        return None
    

    loss_cur  = get_dict_error(model_cur,  dict_x, dict_y)['val']['mean_over_rel_err']
    loss_best = get_dict_error(model_best, dict_x, dict_y)['val']['mean_over_rel_err']

    if loss_cur <= loss_best:
        model_cur.save(path_cur_model+folder_best_val_all)
        save_history(history, folder=folder_best_val_all)

#************************************************

