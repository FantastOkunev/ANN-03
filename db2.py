import numpy as np
import pandas as pd
import random

from os import system

from pathlib import Path
import json as js

#************************************************

path_fort_prog = r'~/eclipse-workspace/KHN3P/Release/KHN3P'

path_db        = r'./data_bases'

path_vis       = r'./vis'

path_flag_stop = r'./flag_stop.txt'

folder_best_train   = r'/best_train'
folder_best_val     = r'/best_val'
folder_best_val_all = r'/best_val_all'

path_meta_db = r'meta_db/db.csv'
path_meta_dl = r'meta_db/dl.csv'

#************************************************

df_db = 0
df_dl = 0

dict_db = 0
dict_dl = 0

name_mod = 'mod_LH100_0520_0104_01E'

#************************************************

nlayer    = 0
nsample   = 0

full_db   = 0
crop_db   = 0

db_Enu    = 0
db_border = 0

db_E      = 0
db_nu     = 0

#************************************************ # формирование имен файлов данных

flfort = r'fort.1'  # файл параметров FORTRAN-программ

dbmt   = 0          # БД параметров материала упругой полосы
fldbmt = r'fort.31' # файл БД

dbht   = 0          # БД нормализованных границ слоев полосы
fldbht = r'fort.32' # файл БД

dbur   = 0          # БД кривых нагружения
fldbur = r'fort.41' # файл БД

dbrn   = 0          # БД усилий на микровыступах
fldbrn = r'fort.42' # файл БД

dbsm   = 0          # БД максимальных контактных напряжений
fldbsm = r'fort.43' # файл БД

dbfk   = 0          # БД относительных площадей фактического контакта
fldbfk = r'fort.44' # файл БД

#************************************************

def get_path_db(dict_db_local=None):
    dict_db_local = dict_db if dict_db_local is None else dict_db_local

    path = path_db + dict_db_local['path']
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_path_model(dict_db_local=None, dict_dl_local=None):
    dict_db_local = dict_db if dict_db_local is None else dict_db_local
    dict_dl_local = dict_dl if dict_dl_local is None else dict_dl_local

    path = path_db + dict_db_local['path'] + dict_dl_local['path']
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_path_vis(dict_db_local=None, dict_dl_local=None, in_model=True):
    dict_db_local = dict_db if dict_db_local is None else dict_db_local
    dict_dl_local = dict_dl if dict_dl_local is None else dict_dl_local

    if in_model:
        path = path_db + dict_db['path'] + dict_dl['path'] + path_vis[1:]
    else:
        path = path_vis
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

#************************************************

def save_matr(path, x):
    np.savetxt(path, x, fmt='%12e', header=str(x.shape[0])+' '+str(x.shape[1]), comments='')

def save_db():
    save_matr(fldbmt, db_Enu)
    save_matr(fldbht, db_border)

def move_db():

    fout = ['fort.1',
            'fort.31',
            'fort.32',
            'fort.41',
            'fort.42',
            'fort.43',
            'fort.44',
            'fort.70',
            'fort.71',
            'fort.72',
            'fort.3']
        
    current_path_db = get_path_db()
        
    for fn in fout:
        system('mv ' + fn + ' ' + current_path_db)
    
#************************************************
        
def write_fort_txt():

    # параметры FORTRAN-программы

    imt  = 101  # тип аппроксимации упругих параметров по толщине полосы
                # 101 - кусочно-постоянная

    #************************************************

    # количество слоев в полосе
    nht  = nlayer

    #************************************************

    kpp   = 2  # параметр управления постпроцессорной обработкой результатов (1-2)
            # БД усилий на микровыступах
            # БД максимальных контактных напряжений
            # БД относительных площадей фактического контакта
            # 1 - для конечной нагрузки
            # 2 - для всех шагов нагружения 

    kpstr = 0  # параметр управления построением графиков контактного давления (FORTRAN-программа)
            # 0 - отсутствие графиков
            # 1 - построение графиков 

    kprsf = 0  # параметр управления построением графиков по микровыступам (FORTRAN-программа)
            # 0 - отсутствие графиков
            # 1 - построение графиков 

    kpurs = 0  # параметр управления построением графиков истории нагружения (FORTRAN-программа)
            # 0 - отсутствие графиков
            # 1 - построение графиков 

    kpzzr = 0  # параметр управления построением графиков зазоров (FORTRAN-программа)
            # 0 - отсутствие графиков
            # 1 - построение графиков 

    kpemt = 0  # параметр управления построением графиков упругих постоянных (FORTRAN-программа)
            # 0 - отсутствие графиков
            # 1 - построение графиков 

    #************************************************

    ncu   = 0  # CUFFT (не изменять)
            # 0 - мой FFT-алгоритм
            # 1 - CUFFT 

    nind  = 8  # параметр точности вычислений CUFFT (не изменять)

    #************************************************

    nld   = 1       # число шагов нагружения
    n2a   = 4       # 3 показатель степени количества волн 
    n2ea  = 9       # показатель степени количества ГЭ на одной волне (7-9)
    n2v   = 1       # 5 показатель степени коэффициента для вычислительной области (3-5)

    npnch = 1       # тип основы штампа (1-2)
    nvawe = 1       # тип микровыступа  (1-2)

    bp    = 4.0     # показатель степени основы штампа 
    bv    = 4.0     # показатель степени микровыступа

    xl    = 1.0     # размер области (абсолютный)
    xc    = 0.0     # координата левой границы области (абсолютная)
    eht   = 0.12    # относительная толщина полосы

    ap    = 3.0E-05 # отношение амплитуды основы штампа к его длине 
    av    = 1.0E-04 # отношение амплитуды волны  штампа к ее  длине

    rpn1  = 5.0E-05 # отношение приложенной силы к произведению размера области на приведенный модуль упругости 
    rpn2  = 2.5E-01 # отношение эксцентриситета приложенной силы к размеру области
    
    # открытие файла параметров FORTRAN-программы
    fl = open(flfort, mode='wt')

    # запись в файл номеров каналов ввода-вывода FORTRAN-программы 
    fl.write(fldbmt.lstrip('fort.') + ' - номер канала  ввода БД параметров материала упругой полосы\n')
    fl.write(fldbht.lstrip('fort.') + ' - номер канала  ввода БД нормализованных границ слоев упругой полосы\n')
    fl.write(fldbur.lstrip('fort.') + ' - номер канала вывода БД кривых нагружения\n')
    fl.write(fldbrn.lstrip('fort.') + ' - номер канала вывода БД усилий на микровыступах\n')
    fl.write(fldbsm.lstrip('fort.') + ' - номер канала вывода БД максимальных контактных напряжений\n')
    fl.write(fldbfk.lstrip('fort.') + ' - номер канала вывода БД относительных площадей фактического контакта\n')

    # запись в файл параметров FORTRAN-программы
    
    fl.write('{0:d} - {1}\n'.format(imt, 'кусочно-постоянная аппроксимация'))

    # запись в файл параметров FORTRAN-программы
    fl.write('{0:d} - {1}\n'.format(nht,   'количество слоев в полосе'))
    fl.write('{0:d} - {1}\n'.format(nld,   'число шагов нагружения'))
    fl.write('{0:d} - {1}\n'.format(n2a,   'показатель степени количества микровыступов'))
    fl.write('{0:d} - {1}\n'.format(n2ea,  'показатель степени количества ГЭ на одном микровыступе'))
    fl.write('{0:d} - {1}\n'.format(n2v,   'показатель степени коэффициента для вычислительной области'))
    fl.write('{0:d} - {1}\n'.format(npnch, 'тип основы штампа'))
    fl.write('{0:d} - {1}\n'.format(nvawe, 'тип микровыступов'))
    fl.write('{0:d} - {1}\n'.format(kpp,   'параметр управления постпроцессорной обработкой результатов')) 
    fl.write('{0:d} - {1}\n'.format(kpstr, 'параметр управления построением графиков контактного давления'))
    fl.write('{0:d} - {1}\n'.format(kprsf, 'параметр управления построением графиков по микровыступам'))
    fl.write('{0:d} - {1}\n'.format(kpurs, 'параметр управления построением графиков истории нагружения'))
    fl.write('{0:d} - {1}\n'.format(kpzzr, 'параметр управления построением графиков зазоров'))
    fl.write('{0:d} - {1}\n'.format(kpemt, 'параметр управления построением графиков упругих постоянных'))
    fl.write('{0:d} - {1}\n'.format(ncu,   'CUFFT'))
    fl.write('{0:d} - {1}\n'.format(nind,  'параметр точности вычислений CUFFT'))

    fl.write('{0:E} - {1}\n'.format(xl,    'размер области (абсолютный)'))
    fl.write('{0:E} - {1}\n'.format(xc,    'координата левой границы области (абсолютная)'))
    fl.write('{0:E} - {1}\n'.format(eht,   'относительная толщина полосы'))
    fl.write('{0:E} - {1}\n'.format(ap,    'отношение амплитуды основы штампа к его длине'))
    fl.write('{0:E} - {1}\n'.format(av,    'отношение амплитуды микровыступа  к его длине'))
    fl.write('{0:E} - {1}\n'.format(bp,    'показатель степени основы штампа'))
    fl.write('{0:E} - {1}\n'.format(bv,    'показатель степени микровыступа'))
    fl.write('{0:E} - {1}\n'.format(rpn1,  'отношение приложенной силы к произведению размера области на приведенный модуль упругости')) 
    fl.write('{0:E} - {1}\n'.format(rpn2,  'отношение эксцентриситета приложенной силы к размеру области'))

    fl.close()

#************************************************
    
def tranform_dict_json_to_list(dict_in):
    for key in dict_in.keys():
        try:
            dict_in[key] = js.loads(dict_in[key])
        except:
            pass

def get_dict_db_dl(name_dl):
        
    tmp = df_dl.loc[df_dl['name'] == name_dl]

    if tmp.shape[0] == 0:
        return 'Нет базы данных с именем ' + name_dl, 0
    elif tmp.shape[0] > 1:
        return 'ОШИБКА количество баз данных с именем ' + name_dl + ' больше одного', 0

    dict_dl_local = df_dl.loc[df_dl['name'] == name_dl].to_dict(orient='records')[0]

    name_db = dict_dl_local['name_db']

    dict_db_local = df_db.loc[df_db['name'] == name_db].to_dict(orient='records')[0]

    return dict_db_local, dict_dl_local

def read_meta_db():

    global df_db
    global df_dl

    global dict_db
    global dict_dl

    global nlayer
    global nsample

    df_db = pd.read_csv(path_meta_db)
    df_dl = pd.read_csv(path_meta_dl)

    dict_db, dict_dl = get_dict_db_dl(name_mod)

    if type(dict_db) == str:
        return dict_db, dict_dl

    nlayer  = dict_db['nlayer']
    nsample = dict_db['ndb']

    tranform_dict_json_to_list(dict_db)
    tranform_dict_json_to_list(dict_dl)

    return dict_db, dict_dl

#************************************************

def read_db():

    global dbmt # output ANN - БД параметров материала упругой полосы
    global dbht # output ANN - БД нормализованных границ слоев полосы
    global dbur #  input ANN - БД кривых нагружения
    global dbrn #  input ANN - БД усилий на микровыступах
    global dbsm #  input ANN - БД максимальных контактных напряжений
    global dbfk #  input ANN - БД относительных площадей фактического контакта

    global db_E
    global db_nu
    global db_border

    path = get_path_db() + r'/'

    dbmt = np.loadtxt(path + fldbmt, skiprows=1)

    dbht = np.loadtxt(path + fldbht, skiprows=1)

    dbur = np.loadtxt(path + fldbur, skiprows=1)

    dbrn = np.loadtxt(path + fldbrn, skiprows=1)
    
    dbsm = np.loadtxt(path + fldbsm, skiprows=1)

    dbfk = np.loadtxt(path + fldbfk, skiprows=1)

    dbht = dbht.reshape((1, dbht.shape[0])) if len(dbht.shape) == 1 else dbht

    db_E      = np.reshape(dbmt, (dbmt.shape[0], dbmt.shape[1]//2, 2))[:, :, 0]
    db_nu     = np.reshape(dbmt, (dbmt.shape[0], dbmt.shape[1]//2, 2))[:, :, 1]
    db_border = dbht

#************************************************
    
def create_input_BEM():

    global full_db
    global crop_db

    list_a = []
    list_b = []
    list_f = []

    ab_key = ['E', 'nu', 'border']

    for key in ab_key:
        list_a += dict_db['a' + key]
        list_b += dict_db['b' + key]
        list_f += dict_db['f' + key]

    new_list_a = []
    new_list_b = []

    for i, flag in enumerate(list_f):
        if flag:
            new_list_a.append(list_a[i])
            new_list_b.append(list_b[i])

    type_db = dict_db['type']
    nnode   = dict_db['nnode']

    if   type_db == 'FF':
        crop_db = form_FF(nnode, new_list_a, new_list_b)
    elif type_db == 'LH':
        crop_db = form_LH(nnode, new_list_a, new_list_b)
    else:
        print('ОШИБКА не существующий тип базы данных: ' + type_db)

    rand_crop_db = form_RANDOM(dict_db['nrand'], new_list_a, new_list_b)

    crop_db = np.concatenate((crop_db, rand_crop_db), axis=0)

    full_db = np.empty((crop_db.shape[0], len(list_a)))

    j = 0

    for i, flag in enumerate(list_f):
        if flag:
            full_db[:, i] = crop_db[:, j]
            j += 1
        else:
            full_db[:, i] = list_a[i]

    transform_db_to_write()

def transform_db_to_write():

    global db_Enu
    global db_E
    global db_nu
    global db_border

    db_E      = full_db[:,       : nlayer]
    db_nu     = full_db[:, nlayer:-nlayer]
    db_border = full_db[:,-nlayer:]

    db_Enu    = np.empty((nsample, nlayer, 2))

    db_Enu[:,:,0] = db_E
    db_Enu[:,:,1] = db_nu

    db_Enu = np.reshape(db_Enu, (nsample, 2*nlayer), order='C')

    if dict_db['nborder'] == 0:
        db_border = db_border[:1, :]

def form_RANDOM(nrand, a, b):

    ndim = len(a)

    db = np.empty((nrand, ndim))

    for i in range(nrand):
        for j in range(ndim):
            db[i, j] = random.uniform(a[j], b[j])
    
    return db
            
def form_FF(nnode, a, b):

    ndim = len(a)

    nsample = nnode**ndim

    db = np.empty((nsample, ndim))

    for i in range(nsample):
        kk = i
        for j in range(ndim):
            k = kk % nnode
            db[i, j] = a[j] + (b[j] - a[j]) * k / (nnode - 1)
            kk //= nnode

    return db

def form_LH(nnode, a, b):

    rng = np.random.default_rng()
    
    ndim = len(a)

    nsample = nnode

    db = np.empty((nsample, ndim))

    for i in range(ndim):

        tmp = np.empty((nsample,))

        for j in range(nsample):
            tmp[j] = rng.random() * (b[i] - a[i]) / nsample + a[i] + (b[i] - a[i]) * j / nsample

        np.random.shuffle(tmp)

        db[:,i] = tmp

    return db

#************************************************

def form_xy(set_x=None, set_y=None):

    def form_dict_all_xy(dict_x, dict_y):
        
        def add_mean_std_in_dict(dict_z):
            keys = dict_z.keys()
            for key in keys:
                z = dict_z[key]
                dict_z[key] = {
                    'db':   z,
                    'mean': np.zeros((z.shape[1],)),
                    'std':  np.ones( (z.shape[1],))
                }

        def form_reduced(dict_z):

            z = dict_z['db']

            stencil = []

            for j in range(z.shape[1]):
                
                flag = 0
                z0   = z[0, j]

                for i in range(z.shape[0]):
                    if z[i, j] > z0 or z[i, j] < z0:
                        flag = 1
                        break
                
                if flag:
                    stencil.append(j)

            rz = np.empty((z.shape[0], len(stencil)))

            for i in range(z.shape[0]):
                for j in range(len(stencil)):
                    rz[i, j] = z[i, stencil[j]]

            dict_z['db'] = rz
                
            return dict_z

        def form_normed(dict_z):
            
            z = dict_z['db']

            ntrain = dict_db['ntrain']

            nz = np.copy(z)

            z_mean = np.mean(nz[:ntrain], axis=0)
            z_std  = np.std( nz[:ntrain], axis=0)

            nz -= z_mean
            nz /= z_std
            
            nz_dict = {
                'db':   nz,
                'mean': z_mean,
                'std':  z_std
            }

            return nz_dict

        def form_transf(dict_z):

            z = dict_z['db']

            ntrain = dict_db['ntrain']

            tz = np.copy(z)

            z_mean = np.mean(tz[:ntrain], axis=0)

            tz -= z_mean

            z_max = np.max(np.abs(tz[:ntrain]), axis=0)

            z_max = z_max

            tz /= z_max

            tx_dict = {
                'db':   tz,
                'mean': z_mean,
                'std':  z_max
            }

            return tx_dict

        def form_dict_num(dict_all_z):

            dict_num = {}

            for key in dict_all_z.keys():

                dict_z = dict_all_z[key]

                z    = dict_z['db']
                mean = dict_z['mean']
                std  = dict_z['std']

                for i in range(z.shape[1]):
                    zi = z[:, i].copy().reshape((z.shape[0], 1))

                    dict_num.update({
                        key + str(i + 1): {
                            'db':   zi,
                            'mean': mean[i].reshape((1,)),
                            'std':  std[i].reshape((1,))
                        }
                    })
            return dict_num

        add_mean_std_in_dict(dict_x)
        add_mean_std_in_dict(dict_y)

        dict_rx  = {'r' + key: form_reduced(dict_x[key]) for key in dict_x.keys()}
        dict_nrx = {'n' + key: form_normed(dict_rx[key]) for key in dict_rx.keys()}
        dict_trx = {'t' + key: form_transf(dict_rx[key]) for key in dict_rx.keys()}

        dict_ny  = {'n' + key: form_normed(dict_y[key])  for key in dict_y.keys()}
        dict_ty  = {'t' + key: form_transf(dict_y[key])  for key in dict_y.keys()}

        dict_y_num  = form_dict_num(dict_y)
        dict_ny_num = form_dict_num(dict_ny)
        dict_ty_num = form_dict_num(dict_ty)

        dict_all_xy = {
            **dict_x,
            **dict_y,
            **dict_rx,
            **dict_nrx,
            **dict_trx,
            **dict_ny,
            **dict_ty,
            **dict_y_num,
            **dict_ny_num,
            **dict_ty_num
        }

        return dict_all_xy

    def set_to_db(set_z, dict_all):
        tmp_z    = [dict_all[key]['db']   for key in set_z]
        tmp_mean = [dict_all[key]['mean'] for key in set_z]
        tmp_std  = [dict_all[key]['std']  for key in set_z]

        out_z = {
            'db':   np.concatenate(tmp_z,    axis=1),
            'mean': np.concatenate(tmp_mean, axis=0),
            'std':  np.concatenate(tmp_std,  axis=0),
        }

        return out_z

    dict_init_x = {
        'lc': dbur,
        'ef': dbrn,
        'vo': dbsm,
        'ar': dbfk
    }

    dict_init_y = {
        'E':  db_E,
        'nu': db_nu,
        # 'bo': db_border
    }

    dict_all_xy = form_dict_all_xy(dict_init_x, dict_init_y)

    set_x = dict_dl['x'] if set_x is None else set_x
    set_y = dict_dl['y'] if set_y is None else set_y

    out_x = [set_to_db(set_, dict_all_xy) for set_ in set_x] 
    out_y = [set_to_db(set_, dict_all_xy) for set_ in set_y]

    return out_x, out_y

def add_train_val_test(dict_x, dict_y):

    ntrain = dict_db['ntrain']
    nval   = dict_db['nval']

    x = dict_x['db']
    y = dict_y['db']

    dict_x.update({
        'train': x[:ntrain,:],
        'val':   x[ntrain:ntrain+nval,:],
        'test':  x[ntrain+nval:,:]
    })

    dict_y.update({
        'train': y[:ntrain,:],
        'val':   y[ntrain:ntrain+nval,:],
        'test':  y[ntrain+nval:,:]
    })

def sec_to_time(sec):
    return '%2d часов %2d минут %2d секунд' % (sec // 60 // 60, sec // 60 % 60, sec % 60)

def get_name_model_in_folder(path):
    pass