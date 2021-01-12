# -*- coding: utf-8 -*-

"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

ANNOTATION FUNCTIONS

-- Coded by Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

import datetime as dt
import random as rnd
from collections import Counter
from os import listdir
from os.path import join, pardir

import holidays
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm

from mobiledna.core import help as hlp
from mobiledna.core.help import log


##################
# App categories #
##################

def scrape_play_store(app_names: list, cache: dict, overwrite=False) -> (dict, list):
    """
    Scrape app meta data from Google play store.

    :param app_name: the official app name (e.g., com.facebook.katana)
    :return: dict with meta data for apps that got a hit, list with remaining apps
    """

    '''try:
        cache = np.load(file=join(hlp.CACHE_DIR, 'app_meta_custom.npy'), allow_pickle=True).item()
    except:
        log('No cache was found for app meta data.', lvl=3)'''

    # Play store URL prefix
    play_store_url = 'https://play.google.com/store/apps/details?id='

    # Initialize dict of knowns and list of unknowns
    known_apps = {}
    unknown_apps = []
    cached_apps = 0

    # Loop over app names
    t_app_names = app_names if hlp.LOG_LEVEL > 1 else tqdm(app_names, position=0, leave=True)
    if hlp.LOG_LEVEL == 1:
        t_app_names.set_description('Scraping')
    for app_name in t_app_names:

        # Check with local cache, which must be a dict
        if isinstance(cache, dict):

            # Is the app name in the cache's keys? Is the genre attached to it a NaN?
            if app_name in cache.keys() and not pd.isna(cache[app_name]['genre']):

                log(f"Info for f{app_name} is in cache.", lvl=3)
                cached_apps += 1

                # If we don't want to overwrite, skip this one
                if not overwrite:
                    continue

        # Combined into full URLs per app
        url = f'{play_store_url}{app_name}'

        # Get HTML from URL
        response = get(url)

        # Create BeautifulSoup object
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get attributes
        try:

            # Store all meta data for this app here
            meta = {'source': 'play_store'}

            # Find the name
            name = soup.find('h1', {'class': 'AHFaub'})

            # If we can't even find that, get out of here
            if not name:
                raise Exception(f'Could not find anything on {app_name}.')
            else:
                meta['name'] = name.text

            # Find info
            info = soup.find_all(attrs={'class': 'R8zArc'})

            # ... extract text where possible
            info_text = [info_bit.text for info_bit in info]

            # ... and fill in the blanks
            while len(info_text) < 3:
                info_text.append(None)

            meta['company'] = info_text[0]
            meta['genre'] = info_text[1]
            meta['genre2'] = info_text[2]

            # Find purchase info
            purchases = soup.find('div', {'class': 'bSIuKf'})
            if purchases:
                meta['purchases'] = purchases.text

            # Find rating info
            rating = soup.find('div', {'class': 'BHMmbe'})
            if rating:
                meta['rating'] = rating.text

            # Add it to the big dict (lol)
            log(f'Got it! <{app_name}> meta data was scraped.', lvl=3)
            known_apps[app_name] = meta

        except Exception as e:
            log(f'Problem for <{app_name}> - {e}', lvl=3)
            unknown_apps.append(app_name)

        zzz = rnd.uniform(1, 3)
        # print(f'Sleeping for {round(zzz, 2)} seconds.')
        # print()
        # time.sleep(zzz)

    log(f"Obtained info for {len(known_apps)} apps.", lvl=2)
    log(f"Failed to get info on {len(unknown_apps)} apps.", lvl=2)
    log(f"{cached_apps} apps were already cached.", lvl=2)

    # Merge new info with cache
    if isinstance(cache, dict):

        # If we specified overwrite, store scraped info in cache over old info
        if overwrite:
            # known_apps |= cache # Python3.9
            known_apps = {**known_apps, **cache}
        # ... else retain app info
        else:
            # known_apps = cache|known_apps
            known_apps = {**cache, **known_apps}

    # Store app meta data cache
    np.save(file=join(pardir, pardir, 'cache', 'app_meta.npy'), arr=known_apps)

    return known_apps, unknown_apps


def add_category(df: pd.DataFrame, scrape=False, overwrite=False) -> pd.DataFrame:
    """
    Take a data frame and annotate rows with category field, based on application name.

    :param df:data frame (appevents or notifications)
    :param scrape: scrape Play Store for new info (set to True if no meta data is found)
    :return: Annotated data frame
    """

    # Load app meta data
    try:
        meta = dict(np.load(join(hlp.CACHE_DIR, 'app_meta.npy'), allow_pickle=True).item())
    except Exception as e:
        log('No app meta data found. Scraping Play store.', lvl=1)
        scrape = True
        meta = {}

    # Check if data frame has an application field
    if 'application' not in df:
        raise Exception('Cannot find <application> column in data frame!')

    # Scape the Play store if requested
    if scrape:
        applications = list(df.application.unique())

        meta, _ = scrape_play_store(app_names=applications, cache=meta, overwrite=overwrite)

    # Add category field to row
    def adding_category_row(app: str):

        if app in meta.keys() and meta[app]['genre']:
            try:
                return meta[app]['genre'].lower()
            except:
                print(f"Exception for {app}")
                return meta[app]['genre']
        else:
            return 'unknown'

    tqdm.pandas(desc="Adding category", position=0, leave=True)
    df['category'] = df.application.progress_apply(adding_category_row)

    return df


##################################################
# Weekends, holidays, working hours, time of day #
##################################################

# Holidays --> complete with non-standard days
be_holidays = holidays.BE()

# Schedule
morning = (dt.time(8, 30), dt.time(12))
afternoon = (dt.time(13, 30), dt.time(16))

schedule = {
    0: [morning, afternoon],
    1: [morning, afternoon],
    2: [morning],
    3: [morning, afternoon],
    4: [morning, afternoon]
}


def add_date_annotation(df: pd.DataFrame, date_cols: list, holidays_separate=False) -> pd.DataFrame:
    """
    Annotate dates in dataframe (holiday, week or weekend)
    :param df: data frame
    :param date_cols: datetime columns to process
    :return: annotated data frame
    """

    # Type check
    date_cols = date_cols if isinstance(date_cols, list) else [date_cols]

    # Mapping dates to types of days
    def label_date(date: dt.datetime, holidays_separate=holidays_separate) -> str:
        # Get weekday, hour, minute and second
        dt = date.date()
        wd = date.weekday()

        # Weekend?
        if wd >= 5:
            return "weekend"

        # Holiday?
        if dt in be_holidays and holidays_separate:
            return "holiday"

        # Else: regular weekday
        return "week"

    # Loop over date columns
    for date_col in date_cols:
        # Make sure they're in the correct format
        df[date_col] = pd.to_datetime(df[date_col])

        # Get new name (subtract date, add day of the week)
        new_col = date_col[:-4] + 'DOTW'

        # Process each row
        tqdm.pandas(desc=f"Adding dotw <{date_col}>", position=0, leave=True)
        df[new_col] = df[date_col].progress_apply(lambda row: label_date(row))

    return df


def add_time_of_day_annotation(df: pd.DataFrame, time_cols: list = ['startTime']):
    """
    Add time of day annotation depending on datetime field
    :param df: appevents dataframe
    :return: annotated dataframe
    """

    # Type check
    time_cols = time_cols if isinstance(time_cols, list) else [time_cols]

    # Mapping hours to time zones
    def label_hour(x):
        if x <= 4:
            return 'late_night'
        elif x <= 8:
            return 'early_morning'
        elif x <= 12:
            return 'morning'
        elif x <= 16:
            return 'noon'
        elif x <= 20:
            return 'eve'
        else:
            return 'night'

    # Looping over time columns
    for time_col in time_cols:
        # Make sure they're in the correct format
        df[time_col] = pd.to_datetime(df[time_col])

        # Get hour of day information
        hours = df[time_col].dt.hour

        # Get new name (subtract date, add day of the week)
        new_col = time_col[:-4] + 'TOD'

        # Process each row
        tqdm.pandas(desc=f"Adding tod <{time_col}>", position=0, leave=True)
        df[new_col] = hours.progress_apply(label_hour)

    return df


if __name__ == '__main__':
    # Let's go
    hlp.hi()
    hlp.set_dir(join(pardir, 'cache'))
    hlp.set_param(log_level=1,
                  data_dir=join(pardir, pardir, 'data', 'glance', 'processed_appevents'),
                  cache_dir=join(pardir, 'cache'))

    # Load the data and gather apps
    log('Collecting app names.', lvl=1)
    appevents_files = listdir(hlp.DATA_DIR)
    apps = {}

    # Load data
    data = hlp.load(path=join(hlp.DATA_DIR, appevents_files[0]), index='appevents')

    # Add apps to the set (no duplicates)
    app_counts = Counter(list(data.application))
    apps = {**apps, **app_counts}

    data = add_date_annotation(data, ['startDate', 'endDate'])

# Sort apps by number of times they occurred in data
'''apps = {k: v for k, v in sorted(apps.items(), key=lambda item: item[1], reverse=True)}

data2 = add_category(df=data, scrape=True, overwrite=False)'''

# Go through bing
'''bing_url_prefix = 'https://www.bing.com/search?q=site%3Ahttps%3A%2F%2Fapkpure.com+'

for app_name in unknowns_play:

    bing_url = bing_url_prefix + app_name

    # Get HTML from URL
    response = get(bing_url)

    # Create BeautifulSoup object
    soup = BeautifulSoup(response.text, 'html.parser')

    a_s = soup.find_all('a', href=True)

    links = set()

    for a in a_s:
        if (a['href'].startswith('https://apkpure.com') and
            a['href'].__contains__(app_name) and
            not (a['href'].__contains__('/fr/') or
                 a['href'].__contains__('/id/') or
                 a['href'].__contains__('/in/') or
                 a['href'].__contains__('/es/') or
                 a['href'].__contains__('/versions') or
                 a['href'].__contains__('/download') or
                 a['href'].__contains__('/nl/'))):
            links.add(a['href'])

    if links and len(links) > 1:
        print(app_name, len(links), links)
    '''
