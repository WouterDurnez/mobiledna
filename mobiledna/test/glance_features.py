
from mobiledna.core.appevents import Appevents
from os.path import join,pardir
from os import listdir
from mobiledna.core import help as hlp
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


if __name__ == '__main__':

    hlp.hi()
    hlp.set_param(log_level=1)
    hlp.set_dir(join(pardir, pardir, 'data', 'glance','objects','appevents'))

    # Set dirs
    ae_objects_dir = join(pardir, pardir, 'data', 'glance','objects','appevents')
    ae_dir = join(pardir, pardir, 'data','glance','appevents')
    ae_processed_dir = join(pardir, pardir, 'data','glance','processed_appevents')

    # Get file names
    ae_data_files = sorted(listdir(ae_dir))


    big_data = pd.DataFrame()

    # Loop over files and process
    for ae_data_file in tqdm(ae_data_files[:1], desc='Processing appevents'):

        # File name
        file_name = ae_data_file.split('.')[0]

        # Set paths
        ae_path = join(ae_dir, ae_data_file)

        # Load data
        data = hlp.load(path=ae_path, index='appevents')

        # Concatenate
        ae = Appevents(data=data, add_categories=True)

        daily_events_days, daily_social_days, daily_communication_days = [], [], []

        for days in tqdm(range(1,121)):
            data_sub = ae.select_n_first_days(days)
            ae_temp = Appevents(data_sub)
            daily_events_days.append(ae_temp.select_n_first_days(days, inplace=True).get_daily_events()[0])
            daily_social_days.append(ae_temp.select_n_first_days(days, inplace=True).get_daily_events(category='social')[0])
            daily_communication_days.append(ae_temp.select_n_first_days(days, inplace=True).get_daily_events(category='communication')[0])

        df = pd.DataFrame([daily_events_days, daily_social_days, daily_communication_days], index=['daily_events_days','daily_social_days','daily_communication_days']).T

        #for col in df:
        #    df[col] = (df[col] - min(df[col]))/(max(df[col]) - min(df[col]))

        correlations = []
        for idx, row in df.iterrows():
            corr, _ = pearsonr(df.iloc[-1].values,row.values)
            correlations.append(corr)
            plt.plot(correlations)
            #plt.show()