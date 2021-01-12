# -*- coding: utf-8 -*-

"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

CONNECTIVITY CLASS

-- Coded by Kyle Van Gaeveren & Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

import pickle

import pandas as pd

import mobiledna.core.help as hlp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class Connectivity:

    def __init__(self, data: pd.DataFrame = None):

        # Set dtypes #
        ##############

        # Set datetimes
        try:
            data.timestamp = data.timestamp.astype('datetime64[ns]')
        except Exception as e:
            print('Could not convert timestamp column to datetime format: ', e)

        # Set data attribute
        self.__data__ = data

        # Add date columns
        self.__data__ = hlp.add_dates(df=self.__data__, index='connectivity')

    @classmethod
    def load_data(cls, path: str, file_type='infer', sep=',', decimal='.'):
        """
        Construct Connectivity object from path to data

        :param path: path to the file
        :param file_type: file extension (csv, parquet, or pickle)
        :param sep: separator for csv files
        :param decimal: decimal for csv files
        :return: Connectivity object
        """

        data = hlp.load(path=path, index='connectivity', file_type=file_type, sep=sep, dec=decimal)

        return cls(data=data)

    @classmethod
    def from_pickle(cls, path: str):
        """
        Construct an Sessions object from pickle
        :param path: path to file
        :return: Connectivity object
        """

        with open(file=path, mode='rb') as file:
            object = pickle.load(file)
        file.close()

        return object

    def save_data(self, dir: str, name: str, csv=False, pickle=False, parquet=True):
        """
        Save data from Connectivity object to data frame
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
        Store an Connectivity object to pickle
        :param path: path to file
        :return: None
        """

        with open(file=path, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def merge(self, *connectivity: pd.DataFrame):
        """
        Merge new data into existing Session object.

        :param connectivity: data frame with connectivity
        :return: new Connectivity object
        """

        new_data = pd.concat([self.__data__ , *connectivity], sort=False)

        return Connectivity(data=new_data)

    # Getters #
    ###########

    def get_data(self) -> pd.DataFrame:
        """
        Return Connectivity data frame
        """
        return self.__data__

    def get_users(self) -> list:
        """
        Returns a list of unique users
        """
        return list(self.__data__.id.unique())

    def get_days(self) -> pd.Series:
        """
        Returns the number of unique days
        """
        return self.__data__.groupby('id').date.nunique().rename('days')

    # Compound getters #
    ####################

    def get_average_signal_strength(self, signal_type: str = "dbm") -> pd.Series:
        """
        Returns average signal strength

        :param signal_type: signal type variable selection (default: "dbm", alternative: "asu")
        """
        if signal_type.lower() == "dbm":
            name = "average_signal_dbm"
            return self.__data__.groupby("id").signalStrengthDbm.mean().rename(name)

        elif signal_type.lower() == "asu":
            name = "average_signal_asu"
            return self.__data__.groupby("id").signalStrengthAsu.mean().rename(name)

        else:
            raise Exception("ERROR: Incorrect signal type. Please use 'asu' or 'dbm'.")


if __name__ == "__main__":
    ###########
    # EXAMPLE #
    ###########

    hlp.hi()
    hlp.set_param(log_level=1)

    data = hlp.load(path='../../data/connectivity/connectivity_test_connectivity.csv',
                    index='connectivity')

    se = Connectivity(data=data)
    se2 = Connectivity.load_data(path="../../data/connectivity/connectivity_test_connectivity.csv",
                        sep=";")

    se3 = se2.merge(data)

    print(se3.get_days())

    print(se3.get_average_signal_strength(signal_type="asu"))
    print(se3.get_average_signal_strength(signal_type="dbm"))
    print(se3.get_average_signal_strength(signal_type="error"))