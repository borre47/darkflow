import os
import numpy as np
from datetime import datetime

## # Return current time
def get_time():

    ts = str(datetime.now()).split(' ')
    date = ''.join(ts[0].split('-'))
    hour = ''.join(ts[1].split(':')).split('.')[0]
    date_s = date + '_' + hour
    return date_s