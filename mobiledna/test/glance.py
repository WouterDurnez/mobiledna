import mobiledna.communication.elastic as es
import mobiledna.core.help as hlp
import mobiledna.core.basic as bas
import os
from datetime import datetime, timedelta
import random as rnd
import pandas as pd
import numpy as np
from dask import dataframe as dd
from mobiledna.core.help import log
from matplotlib import pyplot as plt

def remove_first_and_last(df: pd.DataFrame) -> pd.DataFrame:

    first, last = list(df.startDate.agg(['min','max']))

    df = df.loc[(df.startDate != first) & (df.startDate != last)]

    return df

def is_consecutive(df: pd.DataFrame, col='startDate', shut_up=False) -> bool:

    if df.empty:
        return False

    df_days = df[col].unique()
    #print(df_days)

    first, last = min(df_days), max(df_days)
    #print(f"first {first} - last{last}\n")

    delta = (last - first).days

    for d in range(delta+1):

        day = first + timedelta(days=d)
        #print(f"Checking day {day}")

        if day not in df_days:
            if not shut_up:
                print(f"Nothing logged on {day}!")
            return False

    if not shut_up:
        print("We got a good one!")
    return True


    # This variable will store the end of the window
    window_end = first

    counts = []
    #for days in range(delta.days + 1):

    return df


if __name__ == '__main__':

    # .astype("datetime64[s]")

    hlp.hi()
    hlp.set_param(log_level=1)

    # Load the __data__
    appevents = hlp.load(path=os.path.join(hlp.DATA_DIR,"glance_small_appevents.parquet"),
                          index='appevents')

    # Add dates from timestamp
    appevents['startDate'] = appevents.startTime.astype("datetime64[s]").dt.date
    appevents['endDate'] = appevents.endTime.astype("datetime64[s]").dt.date

    # Check out 1 app
    appevents = appevents.loc[appevents.application == 'com.whatsapp']

    # We'll do it for each id separately (using a loop), for now.
    # This should be rewritten as a function, and applied with groupby.
    '''for n in range(len(appevents.id.unique())):
        id = appevents.id.unique()[n]
        ae = appevents.loc[appevents.id == id].reset_index(drop=True)

        test = is_consecutive(ae)'''



    def new_baseline(df: pd.DataFrame, col='startDate') -> pd.DataFrame:
        first = df[col].min()

        df[col] = df[col].apply(lambda x: (x - first).days)

        return df

    test = appevents.groupby(['id']).apply(new_baseline)


    #for days in range(delta.days + 1):


    '''for n in range(len(appevents.id.unique())):

        # Get an id and get his/her __data__
        id = appevents.id.unique()[n]
        ae = appevents.loc[appevents.id == id].reset_index(drop=True)
        
        
        
        ae = appevents

        # Get rid of the first and last day
        ae = remove_first_and_last(ae)
        ae.id = ae.id.astype(str)

        # Count total events
        total_event_count = bas.count_events(ae)

        # Get the first and the last day
        first, last = list(ae.startDate.agg(['min','max']))
        delta = last-first

        # This variable will store the end of the window
        window_end = first

        counts = []
        for days in range(delta.days+1):
            print(f"Start {first} -- stop {window_end}")
            ae_windowed = ae.loc[(ae.startDate >= first) & (ae.startDate <= window_end)]
            counts.append(bas.active_screen_time(ae_windowed)[0]/(days+1))
            window_end = first + timedelta(days=days+1)

        plt.plot(counts, 'ro')
        plt.hlines(bas.active_screen_time(ae)[0]/len(counts), xmin=0, xmax=len(counts),colors='b')
        plt.title(id)
        plt.show()

        if n==0:
            break'''





