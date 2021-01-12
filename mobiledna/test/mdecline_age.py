# Implicit Attitude Test - data

from mobiledna.core.appevents import Appevents
from mobiledna.core.help import log
import mobiledna.core.help as hlp
import numpy as np
import pandas as pd
from os.path import join
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xg
from xgboost import XGBRegressor, plot_importance
import mobiledna.test.mdecline_features as mf

if __name__ == '__main__':

    # Set some parameters
    hlp.hi('M-decline age prediction')
    hlp.set_param(data_dir=join(hlp.DATA_DIR, 'm-decline_pred'), log_level=1)

    # Get data
    ae = Appevents.from_pickle(join(hlp.DATA_DIR, 'ae_full_intersection.npy'))
    age = pd.read_pickle(path=join(hlp.DATA_DIR, 'full_age_vector_intersection.npy'))
    age = age.sort_index()
    sns.set_palette('Accent')
    sns.set_style('white')
    sns.distplot(age)
    plt.show()

    #age = age[age>35]
    #selection_ids = list(age.index)
    #ae.filter(users=selection_ids,inplace=True)

    # Annotate (already scraped so set to False)
    ae.add_category(scrape=False)
    ae.add_time_of_day()
    ae.add_date_type()

    # BUILD FEATURES
    log('Getting agnostic features.')
    feature_list = []

    apps = ae.get_applications()[:30].index.tolist()
    categories = ae.get_categories()[:30].index.to_list()
    times_of_day = ['late_night', 'early_morning', 'morning', 'noon', 'eve', 'night']
    from_push = [False, True, None]
    '''find_apps = lambda term: [app for app in apps if app.__contains__(term)]
    apps = list(ae.get_applications().index)
    facebook = find_apps('facebook')
    whatsapp = find_apps('whatsapp')'''

    # Build features
    categories.append(None)
    #apps.append(None) --> Already covered
    times_of_day.append(None)

    # Get features for apps & categories per time of day
    for push in tqdm(from_push, desc='Building features per push response'):
        for tod in tqdm(times_of_day, desc='Building features per time of day'):
            for cat in tqdm(categories, desc='Building features per category'):
                feature_list.append(ae.get_daily_duration(category=cat, time_of_day=tod, from_push=push))
                feature_list.append(ae.get_daily_events(category=cat, time_of_day=tod, from_push=push))
                feature_list.append(ae.get_daily_duration_sd(category=cat, time_of_day=tod, from_push=push))
                feature_list.append(ae.get_daily_events_sd(category=cat, time_of_day=tod, from_push=push))
            for app in tqdm(apps, desc='Building features per app'):
                feature_list.append(ae.get_daily_duration(application=app, time_of_day=tod, from_push=push))
                feature_list.append(ae.get_daily_events(application=app, time_of_day=tod, from_push=push))
                feature_list.append(ae.get_daily_duration_sd(application=app, time_of_day=tod, from_push=push))
                feature_list.append(ae.get_daily_events_sd(application=app, time_of_day=tod, from_push=push))

    # Extra features
    feature_list.append(ae.get_daily_active_sessions())
    feature_list.append(ae.get_daily_active_sessions_sd())
    feature_list.append(ae.get_daily_number_of_apps())
    feature_list.append(ae.get_daily_number_of_apps_sd())

    # Create feature matrix
    agnostic = pd.DataFrame(feature_list).transpose().sort_index()

    # Add m-decline features
    log('Getting theoretical features.')
    executive = mf.calc_executive_function_without_notifications(df=ae.get_data()).set_index('id')

    # Combine
    features = pd.concat([agnostic, executive], axis=1)

    # Replace Nan with 0 (if category wasn't present)
    features.fillna(value=0, inplace=True)
    features.to_csv(join(hlp.DATA_DIR, 'feature_matrix.csv'))

    #features = pd.read_csv(join(hlp.DATA_DIR, 'feature_matrix.csv')).set_index('id')

    # Scale
    log('Scaling.')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Build regression
    X = scaled_features
    # y = scaler.fit_transform(survey_data.sassv.values.reshape(-1, 1)).flatten()
    y = age.values.reshape(-1, 1).flatten()

    # Initialize models
    log('Building models.')
    dum = DummyRegressor(strategy='mean')
    en = ElasticNet()
    rfr = RandomForestRegressor(n_estimators=100)
    ada = AdaBoostRegressor()

    data_dmatrix = xg.DMatrix(data=X, label=y)
    # xgr = xg.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
    #            max_depth = 5, alpha = 10, n_estimators = 10)

    # Evaluate using crossvalidation (5-fold)
    log('Crossvalidation.')
    dum_results = cross_validate(dum, X, y, cv=5,
                                 scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                          'neg_root_mean_squared_error'])
    lr_results = cross_validate(lr, X, y, cv=5,
                                scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                         'neg_root_mean_squared_error'])
    svm_results = cross_validate(svm, X, y, cv=5,
                                 scoring=['explained_variance', 'r2', 'neg_mean_absolute_error',
                                          'neg_root_mean_squared_error'])
    en_results = cross_validate(en, X, y, cv=5,
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

    #########
    # Plots #
    #########

    colors = sns.color_palette('Accent')
    sns.set_style('white')

    log('Plotting results.')
    metric = 'test_neg_root_mean_squared_error'
    plt.boxplot([dum_results[metric],
                 lr_results[metric],
                 svm_results[metric],
                 en_results[metric],
                 rf_results[metric],
                 ada_results[metric],
                 -xg_results['test-rmse-mean'][-1:]],
                labels=['dummy', 'lr', 'svm', 'en', 'rfr', 'ada', 'xgb'])
    plt.title('RMSE')
    plt.show()
    metric = 'test_r2'
    plt.boxplot([dum_results[metric],
                 lr_results[metric],
                 svm_results[metric],
                 en_results[metric],
                 rf_results[metric],
                 ada_results[metric]],
                labels=['dummy', 'lr', 'svm', 'en', 'rfr', 'ada'])
    plt.title('R2')
    plt.show()

    # RANDOM FOREST PLOTS #
    #######################

    # 1. Error on full set
    rfr.fit(X, y)
    sns.distplot(y - rfr.predict(X),color=colors[0])
    plt.title('Error distribution on full set (RF)')
    plt.show()

    # 2. Scatterplot of predictions on full set
    sns.scatterplot(y, rfr.predict(X), alpha=.3, color=colors[0])
    plt.plot(np.arange(0, max(y)), np.arange(0, max(y)), c='grey')
    plt.xlim(min(y), max(y))
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.title('Predictions on full set (RF)')
    plt.show()

    # 3. Crossvalidation error
    rfr.fit(X, y)
    y_predicted = cross_val_predict(rfr, X, y, cv=5)
    sns.distplot(y-y_predicted,color=colors[1])
    plt.title('Crossvalidated prediction error (RF)')
    plt.show()

    # 4. Scatterplot of crossvalidated predictions
    ax = sns.scatterplot(y, y_predicted, alpha=.3, color=colors[1])
    ax.plot(np.arange(0, max(y)), np.arange(0, max(y)), c='grey')
    ax.set_xlim(min(y), max(y))
    ax.set_xlabel('True age')
    ax.set_ylabel('Predicted age')
    plt.title('Crossvalidated predictions (RF)')
    plt.show()


    # XGB REGRESSION PLOTS #
    ########################

    # Build general model
    xgb = XGBRegressor(learning_rate=0.01, subsample=.7,
                       gamma=1.5, colsample_bytree=.6, objective='reg:squarederror', nthread=4,
                       max_depth=5, min_child_weight=3, n_estimators=3500)
    xgb_scores = cross_validate(xgb, X, y, scoring=['r2', 'explained_variance', 'neg_root_mean_squared_error', 'max_error', 'neg_mean_absolute_error'], cv=5)
    dum_scores = cross_validate(dum, X, y, scoring=['r2', 'explained_variance', 'neg_root_mean_squared_error',  'max_error', 'neg_mean_absolute_error'], cv=5)

    xgb.fit(X, y)
    y_predicted_xgb = xgb.predict(X)
    y_predicted_xgb_cv = cross_val_predict(xgb, X, y, cv=5)

    # 1. Error on full set
    sns.distplot(y - y_predicted_xgb, color=colors[2])
    plt.title('Error distribution on full set (XGB)')
    plt.show()

    # 2. Scatterplot of predictions on full set
    ax = sns.scatterplot(y, y_predicted_xgb, alpha=.3, color=colors[2])
    plt.plot(np.arange(0, max(y)), np.arange(0, max(y)), c='grey')
    plt.xlim(min(y), max(y))
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.title('Predictions on full set (XGB)')
    plt.show()

    # 3. Crossvalidation error
    sns.distplot(y - y_predicted_xgb_cv, color=colors[4])
    plt.title('Crossvalidated prediction error (XGB)')
    plt.show()

    # 4. Scatterplot of crossvalidated predictions
    ax = sns.scatterplot(y, y_predicted_xgb_cv, alpha=.3, color=colors[4])
    ax.plot(np.arange(0, max(y)), np.arange(0, max(y)), c='grey')
    ax.set_xlim(min(y), max(y))
    ax.set_xlabel('True age')
    ax.set_ylabel('Predicted age')
    plt.title('Crossvalidated predictions (XGB)')
    plt.show()

    # FEATURE IMPORTANCES #
    #######################

    feature_importances = pd.Series(data=xgb.feature_importances_, index=features.columns)
    feature_importances.to_csv(join(hlp.DATA_DIR, 'feature_importances.csv'))
    fig, ax = plt.subplots(figsize=(14, 10))
    xgb.get_booster().feature_names = list(features.columns)
    plot_importance(xgb, max_num_features=30, ax=ax,importance_type='gain')
    plt.show()

    feature_importances = feature_importances.sort_values(ascending=False)
    proportions = []
    for i in range(1,len(feature_importances)):
        temp = feature_importances[:i]
        total = len(temp)
        md = len([f for f in temp.iteritems() if f[0].startswith('md')])
        proportions.append(md/i)
    plt.plot(proportions, label='Proportion in top')
    plt.title('Importance of theory-driven features')
    plt.xlabel('Top n features')
    plt.axhline(y=23/features.shape[1], color='grey', label='Overall',linestyle='dashed')
    plt.legend()
    plt.show()

    md_ordered = [f for f in feature_importances.iteritems() if f[0].startswith('md')]