import numpy as np
import mobiledna.core.help as hlp
import pandas as pd
from os.path import join
from mobiledna.core.appevents import Appevents
from mobiledna.core.notifications import Notifications
from tqdm import tqdm
import mobiledna.test.mdecline_features as mf

if __name__ == '__main__':
    # Set some parameters
    hlp.hi('Data merge')
    orig_data_dir = hlp.DATA_DIR
    hlp.set_param(log_level=3)

    ae = Appevents.load_data(join(hlp.DATA_DIR, 'mdecline_newest/m-decline_newest_appevents.csv'), sep=';')
    nf = Notifications.load(join(hlp.DATA_DIR, 'mdecline_newest/m-decline_newest_notifications.csv'),sep=';')
    # Annotate (already scraped so set to False)
    ae.add_category(scrape=False)
    ae.add_time_of_day()
    ae.add_date_type()
    ae.strip(number_of_days=28,min_log_days=5)

    # BUILD FEATURES
    feature_list = []

    apps = ae.get_applications()[:30].index.tolist()
    categories = ae.get_categories()[:30].index.to_list()
    times_of_day = ['late_night', 'early_morning', 'morning', 'noon', 'eve', 'night']

    '''find_apps = lambda term: [app for app in apps if app.__contains__(term)]
    apps = list(ae.get_applications().index)
    facebook = find_apps('facebook')
    whatsapp = find_apps('whatsapp')'''

    # Build features
    categories.append(None)
    #apps.append(None) --> Already covered
    times_of_day.append(None)

    # Get features for apps & categories per time of day
    for tod in tqdm(times_of_day, desc='Building features per time of day'):
        for cat in tqdm(categories, desc='Building features per category'):
            feature_list.append(ae.get_daily_duration(category=cat,time_of_day=tod))
            feature_list.append(ae.get_daily_events(category=cat,time_of_day=tod))
            feature_list.append(ae.get_daily_duration_sd(category=cat,time_of_day=tod))
            feature_list.append(ae.get_daily_events_sd(category=cat,time_of_day=tod))
        for app in tqdm(apps, desc='Building features per app'):
            feature_list.append(ae.get_daily_duration(application=app,time_of_day=tod))
            feature_list.append(ae.get_daily_events(application=app,time_of_day=tod))
            feature_list.append(ae.get_daily_duration_sd(application=app,time_of_day=tod))
            feature_list.append(ae.get_daily_events_sd(application=app,time_of_day=tod))

    # Extra features
    feature_list.append(ae.get_daily_active_sessions())
    feature_list.append(ae.get_daily_active_sessions_sd())
    feature_list.append(ae.get_daily_number_of_apps())
    feature_list.append(ae.get_daily_number_of_apps_sd())

    # Create feature matrix
    features = pd.DataFrame(feature_list).transpose().sort_index()

    # Replace Nan with 0 (if category wasn't present)
    features.fillna(value=0, inplace=True)

    # Create m-decline features
    executive = mf.calc_executive_function(df=ae.get_data(),df_n=nf.get_data()).set_index('id')
    executive.fillna(value=0, inplace=True)

    full = pd.concat([features, executive], axis=1, sort=False)
    full.to_csv(join(hlp.DATA_DIR, 'mdecline_newest/m-full.csv'))

    # Get data
    ## m-decline
    '''mdecline_data = pd.read_csv(join(hlp.DATA_DIR, 'm-decline', 'mdecline_labels.csv'), sep=';')
    mdecline_age_vector = mdecline_data.set_index('id')['leeftijd'].rename('age')

    ## digimeter
    ae_digi = Appevents.from_pickle(join(hlp.DATA_DIR, 'm-decline_pred', 'ae_digi.npy'))
    ae_digi.get_data().groupby(['surveyId', 'id']).size()
    digi_data = pd.read_excel(join(hlp.DATA_DIR, 'digimeter2020', 'DM20_Loggers_Wouter.xlsx')).\
        rename(columns={'SurveyID':'surveyId'})

    ids = ae_digi.get_data().groupby(['surveyId', 'id']).size().reset_index().iloc[:,:2]
    mdecline_data = pd.read_csv(join(hlp.DATA_DIR, 'm-decline', 'mdecline_labels.csv'), sep=';')
    digi_age_vector = pd.merge(digi_data, ids).set_index('id')['Leeftijd'].rename('age')

    ## implicit
    implicit_data = pd.read_csv(join(hlp.DATA_DIR, 'implicit', 'data2.csv'), sep='\t')
    implicit_age_vector = implicit_data.set_index('code')['age']

    ## Concatenate
    full_age_vector = pd.concat([mdecline_age_vector, digi_age_vector, implicit_age_vector])
    full_age_vector = full_age_vector.groupby(full_age_vector.index).first()
    np.save(join(hlp.DATA_DIR, 'm-decline_pred', 'full_age_vector.npy'), full_age_vector)

    # Get overlapping ids
    age_ids = dict(full_age_vector)
    ae = Appevents.from_pickle(path=join(hlp.DATA_DIR, 'm-decline_pred', 'ae_full.npy'))
    ae_ids = set(ae.get_users())

    final_ids = list(set(age_ids.keys()).intersection(ae_ids))
    ae.filter(users=final_ids, inplace=True)
    ae.to_pickle(path=join(hlp.DATA_DIR, 'm-decline_pred', 'ae_full_intersection.npy'))
    #np.save(join(hlp.DATA_DIR, 'm-decline_pred', 'full_age_vector_intersection.npy'), full_age_vector[final_ids])
    full_age_vector[final_ids].to_pickle(path=join(hlp.DATA_DIR, 'm-decline_pred', 'full_age_vector_intersection.npy'))
    log_data = pd.read_csv(join(hlp.DATA_DIR, 'implicit', 'log_data.csv'), sep=';')

    ae_mdecline = Appevents(data=log_data)
    #ae_mdecline.load_data(path=join(hlp.DATA_DIR, 'm-decline', 'mdecline_appevents.csv'), sep=';')
    ae_mdecline.strip(min_log_days=14,number_of_days=28)
    ae_mdecline.to_pickle(path=join(hlp.DATA_DIR, 'm-decline_pred', 'ae_implicit.npy'))'''

    '''ae_digi = Appevents.from_pickle(join(hlp.DATA_DIR, 'm-decline_pred', 'ae_digi.npy'))
    ae_mdec = Appevents.from_pickle(join(hlp.DATA_DIR, 'm-decline_pred', 'ae_mdecline.npy'))
    ae_impl = Appevents.from_pickle(join(hlp.DATA_DIR, 'm-decline_pred', 'ae_implicit.npy'))

    ae_new = ae_digi.merge(ae_mdec.get_data(), ae_impl.get_data())
    del ae_mdec, ae_impl, ae_digi

    ae_new.to_pickle(path=join(hlp.DATA_DIR, 'm-decline_pred', 'ae_full.npy'))'''
