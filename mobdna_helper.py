# -*- coding: utf-8 -*-

"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

HELPER FUNCTIONS

-- Coded by Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

import os
import time
from pprint import PrettyPrinter

import numpy as np
import pandas as pd

pp = PrettyPrinter(indent=4)


def timeit(f):
    """Timer decorator: shows how long execution of function took."""

    def timed(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()

        print("\'", f.__name__, "\' took ", round(t2 - t1, 3), " seconds to complete.", sep="")

        return res

    return timed


def set_dir(subfolder: str) -> str:
    """Set subdirectory, starting from current folder as root. Create it if need be."""

    # Create path
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + subfolder + "/"

    # If it doesn't exist on drive yet, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Return path
    return dir_path


def hi():
    """Say hello. (It's a stupid function, I know.)"""
    print("\n")
    print("    __  ___      __    _ __     ____  _   _____ ")
    print("   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |")
    print("  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |")
    print(" / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |")
    print("/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|")
    print("\n")

    pd.set_option('chained_assignment', None)


def initialize(project="", cache=False, pickle=True, csv=False) -> dict:
    """Set paths and load data. Specify name of data set (must be in folder of the same name).
    Returns dict containing location and app category caches, data (taken from csv or pickles files),
    and directories to data and cache."""

    # Set warning
    pd.set_option('chained_assignment', None)

    # Present working directory
    file_path = os.getcwd()

    # Create paths
    data_path = set_dir("data" + ("" if project == "" else ("/" + project)))
    cache_path = set_dir("cache")
    print("Using data in folder: " + data_path + "\n")
    print("Using caches in folder: " + cache_path + "\n")


def data_frame_cleanup(df) -> pd.DataFrame:
    """Drop 'Unnamed columns' (redundant indices)"""

    for n in df.columns:

        # Does the column name start with 'Unnamed'?
        if type(n) != str:
            t = True
        else:
            t = n.startswith("Unnamed")

        # If yes, boot it
        if t:
            df = df.drop(columns=[n])

    # Reset index
    df = df.reset_index(drop=True)

    # Return cleaned df
    return df


def df_get_unique(column: str, df: pd.DataFrame) -> np.ndarray:
    """Get list of unique column values in given data frame."""

    # Checking if df has necessary column
    if column not in df.columns:
        print("Could not find variable {column} in dataframe - terminating script\n", column=column)
        return np.nan

    else:
        unique_values = df[column].unique()
        return unique_values


if __name__ == "__main__":

    hi()
