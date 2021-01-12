# -*- coding: utf-8 -*-

"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

APPEVENTS CLASS

-- Coded by Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

import pickle

import pandas as pd
from tqdm import tqdm

import mobiledna.core.help as hlp
from mobiledna.core.annotate import add_category, add_date_annotation, add_time_of_day_annotation
from mobiledna.core.help import log, remove_first_and_last, longest_uninterrupted

tqdm.pandas()


# TODO
# * Calculate session duration (based on first and last appevent)

class Appevents:

    def __init__(self, data: pd.DataFrame = None, add_categories=False, add_date_annotation=False,
                 get_session_sequences=False, strip=False):

        # Drop 'Unnamed' columns
        for col in data.columns:

            if col.startswith('Unnamed'):
                data.drop(labels=[col], axis=1, inplace=True)

        # Set dtypes #
        ##############

        # Set datetimes
        try:
            data.startTime = data.startTime.astype('datetime64[ns]')
        except Exception as e:
            log('Could not convert startTime column to datetime format: ', e)
        try:
            data.endTime = data.endTime.astype('datetime64[ns]')
        except Exception as e:
            log('Could not convert endTime column to datetime format: ', e)

        # Downcast battery column
        try:
            data.battery = data.battery.astype('uint8')
        except Exception as e:
            log('Could not convert battery column to uint8 format: ', e)

        # Factorize ids
        # data.id = data.id.astype('category')

        # Factorize apps
        # data.application = data.application.astype('category')

        # Factorize sessions
        # data.session = data.session.astype('category')

        # Sort data frame
        data.sort_values(by=['id', 'startTime'], inplace=True)

        # Set data attribute
        self.__data__ = data

        # Keep track of stripping
        self.__stripped__ = False

        # Add date columns
        self.__data__ = hlp.add_dates(df=self.__data__, index='appevents')
        data.startDate = data.startDate.astype('datetime64[D]')
        data.endDate = data.endDate.astype('datetime64[D]')

        # Add duration columns
        self.__data__ = hlp.add_duration(df=self.__data__)

        # Add categories on request
        if add_categories:
            self.add_category()

        # Add date annotations on request
        if add_date_annotation:
            self.add_date_type()

        # Strip on request
        if strip:
            self.strip(number_of_days=14)

        # Initialize attributes
        self.__session_sequences__ = self.get_session_sequences() if get_session_sequences else None

    @classmethod
    def load_data(cls, path: str, file_type='infer', sep=',', decimal='.'):
        """
        Construct Appevents object from path to data

        :param path: path to the file
        :param file_type: file extension (csv, parquet, or pickle)
        :param sep: separator for csv files
        :param decimal: decimal for csv files
        :return: Appevents object
        """

        data = hlp.load(path=path, index='appevents', file_type=file_type, sep=sep, dec=decimal)

        return cls(data=data)

    @classmethod
    def from_pickle(cls, path: str):
        """
        Construct an Appevents object from pickle
        :param path: path to file
        :return: Appevents object
        """

        with open(file=path, mode='rb') as file:
            object = pickle.load(file)
        file.close()

        return object

    def save_data(self, dir: str, name: str, csv=False, pickle=False, parquet=True):
        """
        Save data from Appevents object to data frame
        :param dir: directory to save
        :param name: file name
        :param csv: csv format
        :param pickle: pickle format
        :param parquet: parquet format
        :return: None
        """

        hlp.save(df=self.__data__, dir=dir, name=name, csv_file=csv, pickle=pickle, parquet=parquet)

    def to_pickle(self, path: str):
        """
        Store an Appevents object to pickle
        :param path: path to file
        :return: None
        """

        # Setting directory
        dir = '/'.join(path.split('/')[:-1])
        hlp.set_dir(dir)

        # Storing pickle
        with open(file=path, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def filter(self, users=None, category=None, application=None, from_push=None, day_types=None, time_of_day=None,
               inplace=False):

        # If we want category-specific info, make sure we have category column
        if category:
            categories = [category] if not isinstance(category, list) else category

            if 'category' not in self.__data__.columns:
                self.add_category()

            # ... and filter
            data = self.__data__.loc[self.__data__.category.isin(categories)]

        # If we want application-level info
        elif application:
            applications = [application] if not isinstance(application, list) else application

            # ... filter
            data = self.__data__.loc[self.__data__.application.isin(applications)]

        else:
            data = self.__data__

        # If we only want appevents started from push notifications
        if from_push:
            data = data.loc[data.notification == from_push]

        # If we want specific users
        if users:
            users = [users] if not isinstance(users, list) else users

            data = data.loc[data.id.isin(users)]

        # If we want specific day types (week, weekend)
        if day_types:
            day_types = [day_types] if not isinstance(day_types, list) else day_types

            if 'startDOTW' not in self.__data__.columns:
                self.add_date_type()

            # ... and filter
            data = data.loc[data.startDOTW.isin(day_types)]

        # If we want specific times fo day (morning, noon, etc.)
        if time_of_day:
            time_of_day = [time_of_day] if not isinstance(time_of_day, list) else time_of_day

            if 'startTOD' not in self.__data__.columns:
                self.add_time_of_day()

            # ... and filter
            data = data.loc[data.startTOD.isin(time_of_day)]

        if inplace:
            self.__data__ = data
            return self
        else:
            return data

    def strip(self, number_of_days=None, min_log_days=None):

        if self.__stripped__:
            log('Already stripped this Appevents object!', lvl=1)
            return self

        log('Stripping object to longest uninterrupted sequence.')
        # Get longest uninterrupted sequence
        tqdm.pandas(desc="Finding longest uninterrupted sequence.", position=0, leave=True)
        self.__data__ = self.__data__.groupby('id').progress_apply(lambda df: longest_uninterrupted(df=df)).reset_index(
            drop=True)

        # Cut off head and tail
        tqdm.pandas(desc="Cutting off head and tail.", position=0, leave=True)
        self.__data__ = self.__data__.groupby('id').progress_apply(lambda df: remove_first_and_last(df=df)).reset_index(
            drop=True)

        # If a number of days is set
        if number_of_days:
            self.select_n_first_days(n=number_of_days, inplace=True)

        # If a minimum number of log days is set
        if min_log_days:
            self.impose_min_days(n=min_log_days, inplace=True)

        # Remember that we did this
        self.__stripped__ = True

        return self

    def select_n_first_days(self, n: int, inplace=False):
        """
        Select the first n days in the data frame, either inplace or on a copy that is returned.

        :param n: number of days
        :param inplace: modify object or return copy of data
        :return: modified Appevents object or modified copy of data frame
        """

        def select_helper(data: pd.DataFrame, n: int):

            start = data.startDate.min()
            end = start + pd.Timedelta(n - 1, 'D')

            return data.loc[(data.startDate >= start) & (data.startDate <= end)]

        selection = self.__data__.groupby('id'). \
            apply(lambda data: select_helper(data=data, n=n)).reset_index(drop=True)

        if inplace:
            self.__data__ = selection
            return self

        else:
            return selection

    def impose_min_days(self, n: int, inplace=False):
        """
        Filter out users that don't have a minimum of n log days
        :param n: number of log days
        :param inplace: either manipulate object, or return copy
        :return: object or copy, depending on inplace parameter
        """

        if inplace:
            self.filter(users=list(self.get_days()[(self.get_days() >= n)].index), inplace=True)
            return self
        else:
            return self.filter(users=list(self.get_days()[(self.get_days() >= n)].index), inplace=False)

    def merge(self, *appevents: pd.DataFrame):
        """
        Merge new data into existing Appevents object.

        :param appevents: data frame with appevents
        :return: new Appevents object
        """

        new_data = pd.concat([self.__data__, *appevents], sort=False)

        return Appevents(data=new_data)

    def add_category(self, scrape=False, overwrite=False):

        self.__data__ = add_category(df=self.__data__, scrape=scrape, overwrite=overwrite)

        return self

    def add_date_type(self, date_cols='startDate', holidays_separate=False):

        self.__data__ = add_date_annotation(df=self.__data__, date_cols=date_cols, holidays_separate=holidays_separate)

        return self

    def add_time_of_day(self, time_col='startTime'):

        self.__data__ = add_time_of_day_annotation(df=self.__data__, time_cols=time_col)

        return self

    # Getters #
    ###########

    def get_data(self) -> pd.DataFrame:
        """
        Return appevents data frame
        """
        return self.__data__

    def get_users(self) -> list:
        """
        Returns a list of unique users
        """
        return list(self.__data__.id.unique())

    def get_applications(self, by: str = 'events') -> dict:
        """
        Returns applications and their frequency
        """
        if by == 'events':
            return self.__data__.application.value_counts()
        elif by == 'duration':
            return self.__data__.groupby('application').duration.sum().sort_values(ascending=False)
        else:
            log("Cannot get applications according to that metric. Choose 'events' or 'duration'.", lvl=1)
            return {}

    def get_categories(self, by: str = 'events') -> dict:
        """
        Returns categories and their frequency
        """

        # Add categories if not present
        if 'category' not in self.__data__.columns:
            log('Data not annotated with categories yet. Fixing...', lvl=1)
            self.add_category()

        if by == 'events':
            return self.__data__.category.value_counts()
        elif by == 'duration':
            return self.__data__.groupby('category').duration.sum().sort_values(ascending=False)
        else:
            log("Cannot get categories according to that metric. Choose 'events' or 'duration'.", lvl=1)
            return {}

    def get_dates(self, relative=False) -> list:
        """
        Returns a list of unique dates
        """
        unique_dates = self.__data__.groupby('id').startDate.unique()

        if relative:
            unique_dates = unique_dates - self.__data__.groupby('id').startDate.min()

            for idx in range(len(unique_dates)):
                unique_dates.iloc[idx] = [delta.days for delta in unique_dates.iloc[idx]]

        return unique_dates

    def get_days(self) -> pd.Series:
        """
        Returns the number of unique days
        """
        return self.__data__.groupby('id').startDate.nunique().rename('days')

    def get_events(self) -> pd.Series:
        """
        Returns the number of appevents
        """

        return self.__data__.groupby('id').application.count().rename('events')

    def get_durations(self) -> pd.Series:
        """
        Returns the total duration
        """
        return self.__data__.groupby('id').duration.sum().rename('durations')

    def get_session_sequences(self) -> list:
        """
        Returns a list of all session sequences
        """

        sessions = []

        t_sessions = tqdm(self.__data__.session.unique())
        t_sessions.set_description('Extracting sessions')

        for session in t_sessions:
            sessions.append(tuple(self.__data__.loc[self.__data__.session == session].application))

        return sessions

    # Compound getters #
    ####################

    def get_daily_events(self, category=None, application=None, from_push=None, day_types=None,
                         time_of_day=None) -> pd.Series:
        """
        Returns number of appevents per day
        """

        # Field name
        name = ('daily_events' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '') +
                (f'_{day_types}' if day_types else '') +
                (f'_{time_of_day}' if time_of_day else '')).lower()

        # Filter data on request
        data = self.filter(category=category, application=application, from_push=from_push, day_types=day_types,
                           time_of_day=time_of_day)

        return data.groupby(['id', 'startDate']).application.count().reset_index(). \
            groupby('id').application.mean().rename(name)

    def get_daily_duration(self, category=None, application=None, from_push=None, day_types=None,
                           time_of_day=None) -> pd.Series:
        """
        Returns duration per day
        """

        # Field name
        name = ('daily_durations' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '') +
                (f'_{day_types}' if day_types else '') +
                (f'_{time_of_day}' if time_of_day else '')).lower()

        # Filter data on request
        data = self.filter(category=category, application=application, from_push=from_push, day_types=day_types,
                           time_of_day=time_of_day)

        return data.groupby(['id', 'startDate']).duration.sum().reset_index(). \
            groupby('id').duration.mean().rename(name)

    def get_daily_active_sessions(self) -> pd.Series:
        """
        Returns daily number of sessions based on appevent activity
        """

        name = 'daily_active_sessions'

        data = self.__data__

        return data.groupby(['id', 'startDate']).session.count().reset_index(). \
            groupby('id').session.mean().rename(name)

    def get_daily_number_of_apps(self) -> pd.Series:

        name = 'daily_number_of_apps'

        data = self.__data__

        return data.groupby(['id', 'startDate']).application.count().reset_index(). \
            groupby('id').application.mean().rename(name)

    def get_daily_events_sd(self, category=None, application=None, from_push=None, day_types=None,
                            time_of_day=None) -> pd.Series:
        """
        Returns standard deviation on number of events per day
        """

        # Field name
        name = ('daily_events_sd' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '') +
                (f'_{day_types}' if day_types else '') +
                (f'_{time_of_day}' if time_of_day else '')).lower()

        # Filter __data__ on request
        data = self.filter(category=category, application=application, from_push=from_push, day_types=day_types,
                           time_of_day=time_of_day)

        return data.groupby(['id', 'startDate']).application.count().reset_index(). \
            groupby('id').application.std().rename(name)

    def get_daily_duration_sd(self, category=None, application=None, from_push=None, day_types=None,
                              time_of_day=None) -> pd.Series:
        """
        Returns standard deviation on duration per days
        """

        # Field name
        name = ('daily_durations_sd' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '') +
                (f'_{day_types}' if day_types else '') +
                (f'_{time_of_day}' if time_of_day else '')).lower()

        # Filter __data__ on request
        data = self.filter(category=category, application=application, from_push=from_push, day_types=day_types,
                           time_of_day=time_of_day)

        return data.groupby(['id', 'startDate']).duration.sum().reset_index(). \
            groupby('id').duration.std().rename(name)

    def get_daily_active_sessions_sd(self) -> pd.Series:
        """
        Returns standard deviation on daily number of sessions based on appevent activity
        """

        name = 'daily_active_sessions_sd'

        data = self.__data__

        return data.groupby(['id', 'startDate']).session.count().reset_index(). \
            groupby('id').session.std().rename(name)

    def get_daily_number_of_apps_sd(self) -> pd.Series:

        name = 'daily_number_of_apps_sd'

        data = self.__data__

        return data.groupby(['id', 'startDate']).application.count().reset_index(). \
            groupby('id').application.std().rename(name)

    def get_sessions_starting_with(self, category=None, application=None, normalize=False):

        # Field name
        name = ('sessions_starting_with' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '')).lower()

        if category:
            categories = [category] if not isinstance(category, list) else category

            return (self.__data__.groupby(['id', 'session']).category.first().isin(categories)). \
                groupby('id').value_counts(normalize=normalize).rename(name)

        if application:
            applications = [application] if not isinstance(application, list) else application

            return (self.__data__.groupby(['id', 'session']).application.first().isin(applications)). \
                groupby('id').value_counts(normalize=normalize).rename(name)


if __name__ == "__main__":
    ###########
    # EXAMPLE #
    ###########

    hlp.hi()
    hlp.set_param(log_level=3)

    # Read sample data
    data = hlp.add_dates(
        pd.read_parquet(
            path='../../data/glance/processed_appevents/0a0fe3ed-d788-4427-8820-8b7b696a6033_appevents.parquet'),
        'appevents')

    # Data path
    data_path = '../../data/glance/appevents/0a0fe3ed-d788-4427-8820-8b7b696a6033_appevents.parquet'

    # More sample data
    data2 = pd.read_parquet(path='../../data/glance/appevents/0a9edba1-14e3-466a-8d0c-f8a8170cefc8_appevents.parquet')
    data3 = pd.read_parquet(path='../../data/glance/appevents/0a48d1e8-ead2-404a-a5a2-6b05371200b1_appevents.parquet')
    data4 = hlp.add_dates(pd.concat([data, data2, data3], sort=True), 'appevents')

    # Initialize object by loading from path
    print(1)
    ae = Appevents.load_data(path=data_path)

    # Initialize object and add categories
    print(2)
    ae2 = Appevents(data2, add_categories=False, strip=True)

    # Initialize object by adding more data (in this case data2 and data3)
    print(3)
    ae3 = ae.merge(data2, data3)

    ae2.to_pickle('../../data/glance/meta/ae2.ae')

    ae.save_data(dir='../../data/glance/processed_appevents', name='test')

    ae5 = Appevents.from_pickle('../../data/glance/meta/ae2.ae')
