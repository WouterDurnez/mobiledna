# Implicit Attitude Test - data

from mobiledna.core.appevents import Appevents
import mobiledna.core.help as hlp
import numpy as np
import pandas as pd
from os.path import join
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xg

if __name__ == '__main__':

    # Set some parameters
    hlp.hi('Implicit attitude & mobileDNA')
    hlp.set_param(data_dir=join(hlp.DATA_DIR, 'implicit'), log_level=1)

    # Get data
    survey_data = pd.read_csv(join(hlp.DATA_DIR, 'data.csv'), sep='\t')
    log_data = pd.read_csv(join(hlp.DATA_DIR, 'log_data.csv'), sep=';').iloc[:, 1:]

    # Build object
    ae = Appevents(log_data, add_categories=False, add_date_annotation=False, strip=True)
    #ae = Appevents.from_pickle(path=join(hlp.DATA_DIR, 'implicit.ae'))
    del log_data

    # Filter object (only users with over two weeks of logging)
    ae.filter(users=list(ae.get_days()[(ae.get_days() >= 14)].index), inplace=True)

    # Only users for which we have all data (survey & log)
    users = ae.get_users()
    survey_data = survey_data.loc[survey_data.code.isin(users)]
    ae.filter(users=list(survey_data.code.unique()), inplace=True)

    # Annotate (already scraped so set to False)
    ae.add_category(scrape=False)
    ae.add_time_of_day()
    ae.add_date_type()
    #ae.to_pickle(path=join(hlp.DATA_DIR, 'implicit.ae'))
    # Get some info on specific categories (20 most prevalent ones)
    categories = ae.get_categories()[:30].index.to_list()

    '''find_apps = lambda term: [app for app in apps if app.__contains__(term)]
    apps = list(ae.get_applications().index)
    facebook = find_apps('facebook')
    whatsapp = find_apps('whatsapp')'''

    # Build features
    feature_list = []
    categories.append(None)

    for cat in tqdm(categories, desc='Building features per category'):
        feature_list.append(ae.get_daily_duration(category=cat))
        feature_list.append(ae.get_daily_events(category=cat))
        feature_list.append(ae.get_daily_duration_sd(category=cat))
        feature_list.append(ae.get_daily_events_sd(category=cat))

    feature_list.append(ae.get_daily_active_sessions())
    feature_list.append(ae.get_daily_active_sessions_sd())

    # Create feature matrix
    features = pd.DataFrame(feature_list).transpose().sort_index()

    # Replace Nan with 0 (if category wasn't present)
    features.fillna(value=0, inplace=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Build score vector
    sassv_cols = [f'sassv{idx}' for idx in range(1, 11)]
    survey_data['sassv'] = survey_data[sassv_cols].T.sum()
    survey_data = survey_data.set_index('code').sort_index()

    # Build regression
    X = scaled_features
    #y = scaler.fit_transform(survey_data.sassv.values.reshape(-1, 1)).flatten()
    y = survey_data.age.values.reshape(-1, 1).flatten()

    # Initialize models
    lr = LinearRegression()
    mlp = MLPRegressor(max_iter=500)
    rfr = RandomForestRegressor()
    ada = AdaBoostRegressor()

    data_dmatrix = xg.DMatrix(data=X, label=y)
    #xgr = xg.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
    #            max_depth = 5, alpha = 10, n_estimators = 10)

    # Evaluate using crossvalidation (5-fold)
    lr_results = cross_validate(lr, X, y, cv=5,
                                scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                         'neg_root_mean_squared_error'])
    mlp_results = cross_validate(mlp, X, y, cv=5,
                                 scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                          'neg_root_mean_squared_error'])
    rf_results = cross_validate(rfr, X, y, cv=5,
                                scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                         'neg_root_mean_squared_error'])
    ada_results = cross_validate(ada, X, y, cv=5,
                                scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                         'neg_root_mean_squared_error'])

    params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3, 'learning_rate': 0.2,
              'max_depth': 10, 'alpha': 10}
    xg_results = xg.cv(dtrain=data_dmatrix, params=params, nfold=5,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

    metric = 'test_neg_root_mean_squared_error'
    plt.boxplot([#lr_results[metric],
                mlp_results[metric],
                rf_results[metric],
                 ada_results[metric],
                 -xg_results['test-rmse-mean'][-1:]],
                labels=['mlp','rfr','ada','xgb'])
    plt.show()

    #plt.plot(y)
    rfr.fit(X,y)
    plt.boxplot(y - rfr.predict(X),'r')
    plt.show()