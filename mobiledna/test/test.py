

import pandas as pd
import numpy as np
import mobiledna.core.help as hlp
from os.path import *


if __name__ == '__main__':

    meta = pd.read_csv(join(hlp.CACHE_DIR, 'app_meta_custom.csv'), sep=';')
    meta.category_new.fillna(meta.category, inplace=True)

    meta.rename({'category':'genre_old','category_new':'genre','Rating':'rating',
                 'Parentalguidance':'parental_guidance','Downloads':'downloads'},axis='columns', inplace=True)

    meta_dict = {}

    for idx, app_row in meta.iterrows():
        meta_data = app_row.to_dict()
        key = meta_data.pop('appname')
        value = meta_data
        meta_dict[key] = value

    # Store app meta data cache
    np.save(file=join(hlp.CACHE_DIR, 'app_meta_custom.npy'), arr=meta_dict)