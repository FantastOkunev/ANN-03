import db2 as db

check = db.read_meta_db()

if check != 0:
    print(check)

import mdl2 as mdl

import vis2 as vis


db.read_db()

dict_x, dict_y = db.form_xy()

dict_x = dict_x[0]
dict_y = dict_y[0]

db.add_train_val_test(dict_x, dict_y)

mdl.fit(dict_x, dict_y, debug=False)

mdl.save_best_val_all(dict_x, dict_y)