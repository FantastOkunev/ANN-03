import os
from time import time

import db2 as db


check = db.read_meta_db()

if check != 0:
    print(check)

db.create_input_BEM()

db.save_db()


db.write_fort_txt()

start_time = time()

ierr = os.system(db.path_fort_prog)  # вызов FORTRAN-программы

if ierr != 0 : 
    print("ПРОИЗОШЛА ОШИБКА ВЫПОЛНЕНИЯ FORTRAN-ПРОГРАММЫ")

delta = int(time() - start_time)

print(db.sec_to_time(delta))


db.move_db()