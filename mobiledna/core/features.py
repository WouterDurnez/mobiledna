"""
    __  ___      __    _ __     ____  _   _____
   /  |/  /___  / /_  (_) /__  / __ \/ | / /   |
  / /|_/ / __ \/ __ \/ / / _ \/ / / /  |/ / /| |
 / /  / / /_/ / /_/ / / /  __/ /_/ / /|  / ___ |
/_/  /_/\____/_.___/_/_/\___/_____/_/ |_/_/  |_|

FEATURES
Preliminary non-OOP way of calculating features

-- Coded by Kyle Van Gaeveren & Wouter Durnez
-- mailto:Wouter.Durnez@UGent.be
"""

from mobiledna.core.appevents import Appevents
from mobiledna.core.sessions import Sessions
from mobiledna.core.notifications import Notifications
from mobiledna.core.annotate import add_category
import mobiledna.core.help as hlp

import pandas as pd
import datetime as dt
import numpy as np
from os.path import join

# TODO:
# - use Appevents functions for daily duration and appevents
# - add function to add a date column into the annotate.py class

### ### ### #
# Anhedonia #
### ### ### #
def features_calc_anhedonia(df: pd.DataFrame) -> pd.DataFrame:
    """ Takes an appevents dataframe and calculates all Anhedonia variables:
    """
    logdays = df.groupby("id")["date"].nunique()
    # logdays = ae.get_days()

    ## less smartphone use
    # sum of appevents per person
    sum_appevents = df.groupby("id")["application"].count()

    # average daily appevents per person
    daily_appevents = (sum_appevents / logdays).rename("daily_appevents")

    # sum of duration
    sum_duration = df.groupby("id")["duration"].sum()

    # average daily duration
    daily_duration = (sum_duration / logdays).rename("daily_duration")

    ## Losing interest in social media
    # filter on social media apps
    mask = df["category"].isin(["social"])

    # average daily tapped notifications
    socmed_daily_notification_taps = (
            df[mask & (df["notification"] == True)].groupby(["id"])["application"].count() / logdays).rename(
        "socmed_daily_notification_taps")

    # average daily appevents
    socmed_daily_appevents = (df[mask].groupby("id")["application"].count() / logdays).rename("socmed_daily_appevents")

    # average daily duration
    socmed_daily_duration = (df[mask].groupby("id")["duration"].sum() / logdays).rename("socmed_daily_duration")

    ## less incoming and outgoing calls
    mask = df["category"].isin(["calling"])
    calls_daily_appevents = (df[mask].groupby("id")["application"].count() / logdays).rename("calls_daily_appevents")
    calls_daily_duration = (df[mask].groupby("id")["duration"].sum() / logdays).rename("calls_daily_duration")

    ## create result dataframe
    result = (pd.merge(
        daily_appevents,
        daily_duration,
        left_index=True,
        right_index=True,
        how="left"
    ).merge(
        socmed_daily_appevents, on="id", how="left"
    ).merge(
        socmed_daily_notification_taps, on="id", how="left"
    ).merge(
        socmed_daily_duration, on="id", how="left"
    ).merge(
        calls_daily_appevents, on="id", how="left"
    ).merge(
        calls_daily_duration, on="id", how="left"
    ))

    return result


### ### ### ### ### ###
# Executive functions #
### ### ### ### ### ###
def calc_app_cat_use(df: pd.DataFrame, appdict: dict) -> pd.DataFrame:
    """ Takes a dataframe and a dictionary consisting of {category_name: list_of_apps} and calculates
        average daily appevents and duration for provided category.
        Returns a new dataframe with these aggregated measures.
    """

    # Loops over the dictionary, k(ey) is the name of the category,
    # v(alue) the list of apps belonging to the category
    for k, v in appdict.items():
        cat_name = k
        category = v

    # filter for the dataframe, only relevant apps
    mask = df["application"].isin(category)

    logdays = df.groupby("id")["date"].nunique()

    # average daily appevents and duration for the category
    daily_appevents = (df[mask].groupby("id")["application"].count() / logdays).rename(f"{cat_name}_daily_appevents")
    daily_duration = (df[mask].groupby("id")["duration"].sum() / logdays).rename(f"{cat_name}_daily_duration")

    result = pd.merge(
        daily_appevents,
        daily_duration,
        left_index=True,
        right_index=True,
        how="left"
    )

    return result

def calc_category_measures(df: pd.DataFrame) -> pd.DataFrame:
    """ Takes a dataframe, and a predefined list of app categories, loops over these categories
        and merges the aggregated measures into a new dataframe.
    """
    # DF with app names and app categories
    # appcat = pd.read_excel("../data/app_categories.xlsx")
    meta = np.load(join(hlp.CACHE_DIR, 'app_meta.npy'), allow_pickle=True).item()
    appcat = pd.DataFrame.from_dict(meta, orient="index")[["fancyname", "genre", "genre_old"]]
    appcat.reset_index(inplace=True)
    appcat.rename({"index": "app"}, axis=1, inplace=True)

    # Creating the different category dictionaries
    entertainment = list(appcat[appcat["genre"] == "entertainment"]["app"])
    entertainment_dict = {"entertainment": entertainment}

    productivity = list(appcat[appcat["genre"] == "productivity"]["app"])
    productivity_dict = {"productivity": productivity}

    email = list(appcat[appcat["app"].str.contains("mail").fillna(False)]["app"])
    email_dict = {"email": email}
    sms = [
        "com.google.android.apps.messaging",  # google berichten
        "com.android.mms",  # android mms
        "com.textra",  # textra sms
        "xyz.klinker.messenger",  # pulse sms
        "com.p1.chompsms",  # chomp sms
        "com.samsung.android.messaging",  # samsung sms
        "com.handcent.app.nextsms",
        "com.mysms.android.sms",
        "com.supertext.phone",
        "com.MSMS",
        "com.texty.sms",
        "com.htc.sense.mms",
        "com.icq.mobile.client",
        "com.disa",
    ]
    sms_dict = {"sms": sms}

    whatsapp = ["com.whatsapp"]
    whatsapp_dict = {"whatsapp": whatsapp}

    messenger = ["com.facebook.orca"]
    messenger_dict = {"messenger": messenger}

    communication = [
        "com.facebook.orca",  # messenger,
        "com.facebook.mlite",  # messenger lite
        "com.snapchat.android",  # snapchat,
        "com.whatsapp",  # whatsapp,
        "com.skype.raider",  # skype,
        "com.skype.m2",  # skype lite
        "com.google.android.talk",  # hangouts,
        "com.discord",  # discord
        "org.telegram.messenger",  # telegram
        "com.viber.voip",
    ]
    communication_dict = {"communication": communication}

    # creates a list of dictionaries to loop over
    dict_list = [entertainment_dict, productivity_dict, email_dict, sms_dict, whatsapp_dict, messenger_dict,
                 communication_dict]

    # takes a "starting" dataframe, with one ID per row to add the variables to
    app_category_use = df[["id"]].drop_duplicates()

    # loop over the list of dictionaries, calculates the variables and add them to the starting dataframe
    for appcat in dict_list:
        app_category_use = pd.merge(
            app_category_use,
            calc_app_cat_use(df, appcat),
            on="id"
        )

    return app_category_use

def calc_scatter(df: pd.DataFrame) -> pd.DataFrame:
    """ Takes a dataframe and returns a new one with average amount of daily session duration binned in four categories
    """

    # Bin session duration in 4 categories, count the daily session per bin and takes the daily average
    scatter = (
        pd.cut(
            df.groupby(["id", "date", "session"])["duration"].sum(),
            bins=[0, 30, 60, 300, float("inf")],
        )
            .groupby(["id", "date"])
            .value_counts()
            .groupby(["id", "duration"])
            .mean()
    )
    scatter = scatter.reset_index(name="count")
    scatter_pivot = scatter.pivot(index="id", columns="duration", values="count")
    scatter_pivot.columns = ["0s-30s", "30s-1m", "1m-5m", "+5m"]

    return scatter_pivot

def calc_session_lapse(df: pd.DataFrame) -> pd.DataFrame:
    """ Takes a dataframe and calculates the average time between sessions.
    """

    # Groups the dataframe and takes the first row [head(1)] of each groupby(["id", "session"])
    # important to have a SORTED dataframe!
    session_start_stop = df.groupby(["id", "session"]).head(1)[["id", "session", "date", "startTime", "endTime"]]

    # add the start moment of session+1 as a new variable for the existing session
    session_start_stop = session_start_stop.assign(start_next=session_start_stop.groupby(["id"])["startTime"].shift(-1))

    # adds the duration variable, calculates the time between the end of session and start of session+1
    session_start_stop = session_start_stop.assign(
        session_lapse=(session_start_stop["start_next"] - session_start_stop["endTime"]).dt.total_seconds())

    # aggregates the lapse duration and calculates mean per person
    avg_session_lapse = (
        session_start_stop.groupby(["id", pd.Grouper(key="startTime", freq="D")])["session_lapse"].mean().groupby(
            "id").mean()).rename("avg_lapse_duration")

    return avg_session_lapse

# TODO: filter notifications
def calc_average_notif_between(df: pd.DataFrame, df_n: pd.DataFrame):
    """ Takes an appevents and notifications dataframe and calculates the average amount of received notifications between two app sessions, per person.
    """

    # Groups the dataframe and takes the first row [head(1)] of each groupby(["id", "session"])
    # important to have a SORTED dataframe!
    session_start_stop = df.groupby(["id", "session"]).head(1)[
        ["id", "session", "date", "startTime", "endTime"]
    ]

    # add the start moment of session+1 as a new variable for the existing session
    session_start_stop = session_start_stop.assign(
        start_next=session_start_stop.groupby(["id"])["startTime"].shift(-1)
    )

    # add filter to keep only "relevant" notifications
    mask = df_n["ongoing"] == False
    mask &= df_n["priority"] >= 0
    df_n = df_n[mask]

    # merge notifications dataframe with sessions overview, find "last" session per notification
    notif_session = pd.merge_asof(
        df_n.sort_values("time"),
        session_start_stop[
            ["startTime", "endTime", "start_next", "session", "id"]
        ].sort_values("startTime"),
        by=["id"],
        right_on="startTime",
        left_on="time",
        direction="backward",
        allow_exact_matches=False,
    )

    # switch sessions to str?
    # notif_session["session"] = notif_session["session"].astype(str)

    # groupby ID and backfill session so each notification row has a "next session"
    notif_session = notif_session.assign(
        session_bfill=notif_session.groupby("id")["session"].bfill()
    )

    # groupby id and session_bfill and count amount of incoming notifications before the next session starts,
    # calculate mean per person
    mean_notifications_between = (
        notif_session.groupby(["id", "session_bfill"])["application"]
            .count()
            .groupby("id")
            .mean()
    )

    return mean_notifications_between.rename("mean_notif_between")

# TODO: use notifications function
def calc_avg_daily_notifications(df_n: pd.DataFrame):
    total_days = df_n.groupby("id")["date"].nunique()
    notifs_pd = df_n.groupby("id")["application"].count() / total_days

    return notifs_pd.rename("avg_daily_notifications")

def calc_reaction_time(df: pd.DataFrame, df_n: pd.DataFrame):
    """ Takes an appevents and notifications dataframe and
        calculates the average reaction time (in s) between receiving a notification and opening the application.
    """

    # Filter DF's to merge
    df = df.sort_values("startTime", ascending=True)
    df_n = df_n.sort_values("time", ascending=True)

    # merge_asof to find closest match
    notif_merge = pd.merge_asof(
        df[df["notification"] == True],
        df_n[["time", "id", "application"]],
        left_on="startTime",
        right_on="time",
        by=["id", "application"],
        allow_exact_matches=False,
        direction="backward",
        tolerance=pd.Timedelta("1h"),
    )

    # calculate reaction speed for each notifcation
    reaction_s = (notif_merge["startTime"] - notif_merge["time"]).dt.total_seconds()
    notif_merge = notif_merge.assign(reaction_s=reaction_s)

    # calculate average reaction speed
    mean_reaction_s = notif_merge.groupby(["id"])["reaction_s"].mean()

    return mean_reaction_s.rename("avg_reaction_time")

def features_calc_executive_function(df: pd.DataFrame, df_n: pd.DataFrame) -> pd.DataFrame:
    """ Takes an Appevents and Notifications DataFrame and calculates all executive function variables.
    """

    # Sort dataframe, important for session lapses
    df = df.sort_values(by=["id", "startTime"])

    # Unique days someone used their smartphone
    logdays = df.groupby("id")["date"].nunique()

    # Average amount of apps per session
    apps_per_session = df.groupby(["id", "session"])["application"].count().groupby("id").mean().rename(
        "apps_per_session")

    # Average unique apps per session
    unique_apps_per_session = df.groupby(["id", "session"])["application"].nunique().groupby("id").mean().rename(
        "unique_apps_per_session")

    # Average amount of daily _active_ sessions (so sessions with apps)
    daily_sessions = (df.groupby(["id"])["session"].nunique() / logdays).rename("daily_active_sessions")

    # Duration and frequency per app category
    category_measures = calc_category_measures(df)

    # Distribution of app sessions based on duration
    scatter_sessions = calc_scatter(df)

    # Average amount of time spent on one appevent
    avg_app_duration = df.groupby(["id", "date"])["duration"].mean().groupby("id").mean().rename("avg_app_duration")

    # Average amount of time between sessions
    avg_lapse_duration = calc_session_lapse(df)

    # Average daily notifications
    avg_daily_notifications = calc_avg_daily_notifications(df_n)

    # Average amount of notifications between two sessions
    avg_notifications_between = calc_average_notif_between(df=df, df_n=df_n)

    # Average notification reaction speed
    avg_notification_reaction = calc_reaction_time(df=df, df_n=df_n)

    # Merge variables to new dataframe
    result = pd.merge(
        apps_per_session,
        unique_apps_per_session,
        on="id",
        how="left").merge(
        daily_sessions,
        on="id",
        how="left"
    ).merge(
        category_measures,
        on="id",
        how="left"
    ).merge(
        scatter_sessions,
        on="id",
        how="left",
    ).merge(
        avg_app_duration,
        on="id",
        how="left",
    ).merge(
        avg_lapse_duration,
        on="id",
        how="left"
    ).merge(
        avg_daily_notifications,
        on="id",
        how="left",
    ).merge(
        avg_notifications_between,
        on="id",
        how="left",
    ).merge(
        avg_notification_reaction,
        on="id",
        how="left"
    )

    return result


### ### ###
# Memory  #
### ### ###

def calc_daily_appevents(df: pd.DataFrame, apps: list) -> pd.DataFrame:
    """

    :param df: appevents DataFrame
    :param apps: list of apps to filter on
    :return: results DataFrame with daily appevents for the selected app(s)
    """
    logdays = df.groupby("id")["date"].nunique()
    mask = df["application"].isin(apps)

    # Need to manually rename series afterwards
    avg_daily_appevents = (df[mask].groupby(["id"])["application"].count() / logdays).rename("avg_daily_appevents")

    return avg_daily_appevents

def calc_daily_app_duration(df: pd.DataFrame, apps: list) -> pd.DataFrame:
    """

    :param df: appevents DataFrame
    :param apps: list of apps to filter on
    :return: results DataFrame with daily duration for the selected app(s)
    """
    logdays = df.groupby("id")["date"].nunique()
    mask = df["application"].isin(apps)

    # Need to manually rename series afterwards
    avg_daily_app_duration = (df[mask].groupby(["id"])["duration"].sum() / logdays).rename("avg_daily_app_duration")

    return avg_daily_app_duration

def calc_empty_sessions(df: pd.DataFrame, df_s: pd.DataFrame):
    """ Takes an appevents and sessions dataframe and calculates the percentage of _empty_ sessions.
    """

    # Percentage of "active" sessions divided by total amount of sessions
    pct_empty = (
            df.groupby("id")["session"].nunique() / df_s.groupby("id")["session on"].count()
    )

    return pct_empty.rename("sessions_empty_pct")

def calc_app_reopen_same_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the time (in seconds) until a same application is opened again, in the same session.
    :param df: appevents DataFrame
    :return: results DataFrame
    """
    data = df[["id", "session", "application", "startTime", "endTime", "duration"]]
    data = data.sort_values(by=["id", "startTime"])
    data.reset_index(drop=True, inplace=True)

    ae_session_overview = (
            data.groupby(["id", "session"])["application"].value_counts() > 1
    ).reset_index(name="multi")

    ae_session_overview = ae_session_overview[ae_session_overview["multi"] == True][
        ["id", "session", "application"]
    ]

    ae_session_merge = pd.merge(
        ae_session_overview, data, on=["id", "session", "application"]
    )

    ae_session_merge = ae_session_merge.assign(
        start_shift=ae_session_merge.groupby(["id", "session", "application"])[
            "startTime"
        ].shift(-1)
    )

    ae_session_merge = ae_session_merge.assign(
        app_revisit_s=(
                ae_session_merge["start_shift"] - ae_session_merge["endTime"]
        ).dt.total_seconds()
    )

    mean_same_app_s = ae_session_merge.groupby("id")["app_revisit_s"].mean()
    std_same_app_s = ae_session_merge.groupby("id")["app_revisit_s"].std()
    median_same_app_s = ae_session_merge.groupby("id")["app_revisit_s"].median()

    res = (
        pd.merge(
            mean_same_app_s.rename("mean_same_app_s"),
            std_same_app_s.rename("std_same_app_s"),
            left_index=True,
            right_index=True
        ).merge(
            median_same_app_s.rename("median_same_app_s"),
            left_index=True,
            right_index=True
        )
    )

    return res

def features_calc_memory(df: pd.DataFrame, df_s: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all memory features and returns a results DataFrame.
    :param df: appevents DataFrame
    :param df_s: sessions DataFrame
    :return: results DataFrame with all memory features calculated
    """

    search = [
        "com.google.android.googlequicksearchbox"
    ]

    avg_daily_search = calc_daily_appevents(df=df, apps=search).rename("avg_daily_search")
    avg_daily_search_duration = calc_daily_app_duration(df=df, apps=search).rename("avg_daily_search_duration")

    notes = [
        "com.microsoft.office.onenote",
        "com.zoho.notebook",
        "com.splendapps.adler",
        "com.evernote",
        "com.yocto.wenote",
        "com.google.android.keep",
        "org.whiteglow.keepmynotes",
        "com.dencreak.esmemo"
    ]

    avg_daily_notes = calc_daily_appevents(df=df, apps=notes).rename("avg_daily_notes")
    avg_daily_notes_duration = calc_daily_app_duration(df=df, apps=notes).rename("avg_daily_notes_duration")

    empty_sessions = calc_empty_sessions(df=df, df_s=df_s)

    app_reopen_s = calc_app_reopen_same_session(df=df)

    res = pd.merge(
        avg_daily_search,
        avg_daily_search_duration,
        on="id",
        how="left"
    ).merge(
        avg_daily_notes,
        on="id",
        how="left"
    ).merge(
        avg_daily_notes_duration,
        on="id",
        how="left"
    ).merge(
        empty_sessions,
        on="id",
        how="left"
    ).merge(
        app_reopen_s,
        on="id",
        how="left"
    )

    return res


### ### ##
# Sleep  #
### ### ##

def calc_use_during_sleep(
        df: pd.DataFrame, start=dt.time(0, 0, 0), end=dt.time(7, 0, 0)
) -> pd.DataFrame:
    """ Takes an appevents dataframe and _roughly_ calculates the average amount and duration (in seconds) of app-use during a period
        standard hours: between 00:00:00 and 7:00:00
    """

    # resample data per hour
    df_per_hour = (
        df.set_index("startTime")
            .groupby("id")
            .resample("1H")["duration"]
            .agg(["count", "sum"])
            .rename({"count": "apps", "sum": "duration"}, axis=1)
            .reset_index()
    )

    # filter based on time
    mask = df_per_hour["startTime"].dt.time >= start
    mask &= df_per_hour["startTime"].dt.time < end

    # calculate averages per person based on filtered dataframe
    avg_apps_per_hour = df_per_hour[mask].groupby(["id"])["apps"].mean()
    avg_duration_per_hour = df_per_hour[mask].groupby(["id"])["duration"].mean()

    # merge results
    res = pd.merge(
        avg_apps_per_hour.rename("sleep_avg_apps_per_hour"),
        avg_duration_per_hour.rename("sleep_avg_duration_per_hour_s"),
        on="id",
    )

    return res

def calc_sleep_pattern(df: pd.DataFrame):
    """
    Takes an appevents dataframe and calculates descriptives on "stop time" and "start time".
    In order to check for usage after 24h, we correct the timestamps with four hours.
    :param df: an appevents DataFrame
    :return: a results DataFrame with average, median and std of both start & stop time.
    """

    # Correct for usage after 24h
    df = df.assign(start_correct=df["startTime"] - dt.timedelta(hours=4))
    df = df.assign(end_correct=df["endTime"] - dt.timedelta(hours=4))
    df = df.assign(start_date_correct=df["start_correct"].dt.date)
    df = df.assign(end_date_correct=df["end_correct"].dt.date)

    # Groupby per id, into new dataframe
    start_stop = (
        df.groupby(["id", "start_date_correct"])
            .agg({"start_correct": "min", "end_correct": "max"})
            .reset_index()
    )

    start_stop = start_stop.assign(start_h=start_stop["start_correct"].dt.hour + 4)
    start_stop = start_stop.assign(end_h=start_stop["end_correct"].dt.hour + 4)

    # Calculate the descriptives for start and stop, give variables unique names
    start_pattern = start_stop.groupby("id")["start_h"].agg(["mean", "median", "std"]).rename({
        "mean": "sleep_start_mean",
        "median": "sleep_start_median",
        "std": "sleep_start_std"
    }, axis=1)
    stop_pattern = start_stop.groupby("id")["end_h"].agg(["mean", "median", "std"]).rename({
        "mean": "sleep_stop_mean",
        "median": "sleep_stop_median",
        "std": "sleep_stop_std"
    }, axis=1)

    return pd.merge(start_pattern, stop_pattern, on="id", how="left")

def features_calc_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all sleep features: sleep pattern and use during "sleep" hours.
    :param df: an Appevents DataFrame
    :return: a results DataFrame with all sleep features
    """

    """
    df = df.assign(hour=df["startTime"].dt.hour)
    start_stop_hour_pd = df.groupby(["id", "date"])["hour"].agg(["min", "max"])

    start_stop_hour_pd_descriptives = start_stop_hour_pd.groupby("id")["max"].describe()
    avg_end_hour = start_stop_hour_pd_descriptives["mean"].rename("avg_sleep_hour")
    std_end_hour = start_stop_hour_pd_descriptives["std"].rename("std_sleep_hour")

    start_stop_hour_pd = start_stop_hour_pd.reset_index()
    start_stop_hour_pd = start_stop_hour_pd.assign(start_next=start_stop_hour_pd.groupby(["id"])["min"].shift(-1))
    start_stop_hour_pd = start_stop_hour_pd.assign(difference=start_stop_hour_pd["max"] - start_stop_hour_pd["start_next"])
    start_stop_difference_descriptives = start_stop_hour_pd.groupby(["id"])["difference"].describe()
    avg_start_stop_difference = start_stop_difference_descriptives["mean"].rename("avg_start_stop_difference")
    std_start_stop_difference = start_stop_difference_descriptives["std"].rename("std_start_stop_difference")
    """

    sleep_use = calc_use_during_sleep(df=df)
    sleep_pattern = calc_sleep_pattern(df=df)

    return pd.merge(sleep_pattern, sleep_use, on="id")

### ### ###
# BUNDLED #
### ### ###

def features_calc_all(df: pd.DataFrame, df_s: pd.DataFrame, df_n: pd.DataFrame) -> pd.DataFrame:
    """
    Bundles all seperate feature calculations into one function. Calculates features from:
    Anhedonia, Executive functions, Memory, Sleep.
    :param df: an Appevents DataFrame
    :param df_s: a Sessions DataFrame
    :param df_n: a Notifications DataFrame
    :return: a results DataFrame
    """
    anhedonia_features = features_calc_anhedonia(df=df)
    executive_features = features_calc_executive_function(df=df, df_n=df_n)
    memory_features = features_calc_memory(df=df, df_s=df_s)
    sleep_features = features_calc_sleep(df=df)

    all_features = pd.merge(
        anhedonia_features,
        executive_features,
        on="id",
        how="left"
    ).merge(
        memory_features,
        on="id",
        how="left"
    ).merge(
        sleep_features,
        on="id",
        how="left"
    )

    return all_features

if __name__ == "__main__":
    ###########
    # EXAMPLE #
    ###########
    ae = Appevents.load_data(hlp.DATA_DIR + "/appevents.csv", sep=";")
    se = Sessions.load_data(hlp.DATA_DIR + "/sessions.csv", sep=";")
    nc = Notifications.load(hlp.DATA_DIR + "/notifications.csv", sep=";")

    df_appevents = ae.__data__
    df_sessions = se.__data__
    df_notifications = nc.__data__

    df_appevents = add_category(df=df_appevents)
    df_appevents = df_appevents.assign(date=df_appevents["startTime"].dt.date)

    anhedonia_features = features_calc_anhedonia(df=df_appevents)
    executive_features = features_calc_executive_function(df=df_appevents, df_n=df_notifications)
    memory_features = features_calc_memory(df=df_appevents, df_s=df_sessions)
    sleep_features = features_calc_sleep(df=df_appevents)

    all_features = pd.merge(
        anhedonia_features,
        executive_features,
        on="id",
        how="left"
    ).merge(
        memory_features,
        on="id",
        how="left"
    ).merge(
        sleep_features,
        on="id",
        how="left"
    )

    print(all_features.head())
    print(all_features.describe())