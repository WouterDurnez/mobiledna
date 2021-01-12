# -*- coding: utf-8 -*-

"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

NOTIFICATIONS CLASS

-- Coded by Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

from collections import Counter

import pandas as pd

import mobiledna.core.help as hlp
from mobiledna.core.annotate import add_category
from mobiledna.core.help import log

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Notifications:

    def __init__(self, data: pd.DataFrame = None, add_categories=False):

        # Set dtypes #
        ##############

        # Set datetimes
        try:
            data.time = data.time.astype('datetime64[ns]')
        except Exception as e:
            print('Could not convert startTime column to datetime format: ', e)

        # Sort data frame
        data.sort_values(by=['id', 'time'], inplace=True)

        # Set data attribute
        self.__data__ = data

        # Add date columns
        self.__data__ = hlp.add_dates(df=self.__data__, index='notifications')

        # Add categories
        if add_categories:
            self.add_category()

    @classmethod
    def load(cls, path: str, file_type='infer', sep=',', decimal='.'):
        """
        Construct Notifications object from path

        :param path: path to the file
        :param file_type: file extension (csv, parquet, or pickle)
        :param sep: separator for csv files
        :param decimal: decimal for csv files
        :return: Notifications object
        """

        # Load data frame, depending on file type
        if file_type == 'infer':

            # Get extension
            file_type = path.split('.')[-1]

            # Only allow the following extensions
            if file_type not in ['csv', 'pickle', 'pkl', 'parquet']:
                raise Exception("ERROR: Could not infer file type!")

            log("Recognized file type as <{type}>.".format(type=file_type), lvl=3)

        # CSV
        if file_type == 'csv':
            data = pd.read_csv(filepath_or_buffer=path,
                               # usecols=,
                               sep=sep, decimal=decimal,
                               error_bad_lines=False)

        # Pickle
        elif file_type == 'pickle' or file_type == 'pkl':
            data = pd.read_pickle(path=path)

        # Parquet
        elif file_type == 'parquet':
            data = pd.read_parquet(path=path, engine='auto')

        # Unknown
        else:
            raise Exception("ERROR: You want me to read what now? Invalid file type! ")

        return cls(data=data)

    def filter(self, category=None, application=None, priority=None, posted=True):

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

        # Filter on whether notification was posted
        if posted:
            data = data.loc[data.posted == posted]

        # Filter on priority
        if priority:
            data = data.loc[data.priority >= priority]

        return data

    def merge(self, *notifications: pd.DataFrame):
        """
        Merge new data into existing Notifications object.

        :param notifications: data frame with notifications
        :return: new Notifications object
        """

        new_data = pd.concat([self.__data__, *notifications], sort=False)

        return Notifications(data=new_data)

    def add_category(self, scrape=False, overwrite=False):

        self.__data__ = add_category(df=self.__data__, scrape=scrape, overwrite=overwrite)

    # Getters #
    ###########

    def get_data(self) -> pd.DataFrame:
        """
        Return notifications data frame
        """
        return self.__data__

    def get_users(self) -> list:
        """
        Returns a list of unique users
        """
        return list(self.__data__.id.unique())

    def get_applications(self) -> dict:
        """
        Returns an {app: app count} dictionary
        """

        return Counter(list(self.__data__.application))

    def get_days(self) -> pd.Series:
        """
        Returns the number of unique days
        """
        return self.__data__.groupby('id').date.nunique().rename('days')

    def get_notifications(self) -> pd.Series:
        """
        Returns the number of notifications
        """

        return self.__data__.groupby('id').application.count().rename('notifications')

    # Compound getters #
    ####################

    def get_daily_notifications(self, category=None, application=None, priority=0, posted=True) -> pd.Series:
        """
        Returns number of notifications per day
        """

        # Field name
        name = ('daily_notifications' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '')).lower()

        # Filter data on request
        data = self.filter(category=category, application=application, priority=priority, posted=posted)

        return data.groupby(['id', 'date']).application.count().reset_index(). \
            groupby('id').application.mean().rename(name)

    def get_daily_notifications_sd(self, category=None, application=None, priority=0, posted=True) -> pd.Series:
        """
        Returns standard deviation on number of events per day
        """

        # Field name
        name = ('daily_notifications_sd' +
                (f'_{category}' if category else '') +
                (f'_{application}' if application else '')).lower()

        # Filter __data__ on request
        data = self.filter(category=category, application=application, priority=priority, posted=posted)

        return data.groupby(['id', 'date']).application.count().reset_index(). \
            groupby('id').application.std().rename(name)


if __name__ == "__main__":
    ###########
    # EXAMPLE #
    ###########

    hlp.hi()
    hlp.set_param(log_level=1)

    # Read sample data
    data = pd.read_csv('../../data/glance/not.csv', sep=';')

    n = Notifications(data)
