# m-decline grid search
import mobiledna.core.help as hlp
from os.path import join
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix, plot_importance,train, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # Set some parameters
    hlp.hi('M-decline grid search')
    hlp.set_param(data_dir=join(hlp.DATA_DIR, 'm-decline_pred'), log_level=1)

    # Get data
    age = pd.read_pickle(path=join(hlp.DATA_DIR, 'full_age_vector_intersection.npy'))
    age = age.sort_index()
    features = pd.read_csv(join(hlp.DATA_DIR, 'feature_matrix.csv')).set_index('id').sort_index()
    X = features.values
    y = age

    bins = np.array([0,28,40,100])
    inds = np.digitize(age,bins)

    label_encoder = LabelEncoder()
    y_new = label_encoder.fit_transform(inds)

    xgb = XGBClassifier()
    xgb.fit(X,y_new)

    results = cross_validate(xgb,X,y_new,cv=5,scoring=['accuracy','f1_macro'])
    xgb.get_booster().feature_names = list(features.columns)
    plot_importance(xgb,max_num_features=20)
    plt.show()
    # A parameter grid for XGBoost
    '''params = {
        'subsample': [0.6, 0.7, .8, 1],
        'n_estimators': [600, 700, 1000],
        'min_child_weight': [3, 4, 5, 6],
        'max_depth': [3,4, 5, 7, 10]
    }

    # Create model
    xgb = XGBRegressor(learning_rate=0.01, gamma=1.5, colsample_bytree=.6, objective='reg:squarederror', nthread=4)
    data_dmatrix = DMatrix(data=X, label=y, feature_names=features.columns)

    # Search grid
    param_comb = 50

    # Conduct random search
    random_search = RandomizedSearchCV(xgb, param_distributions=params,
                                       n_iter=param_comb,
                                       scoring='neg_root_mean_squared_error',
                                       n_jobs=4,
                                       cv=5,
                                       verbose=3,
                                       random_state=616) 
    # BEST: {'subsample': 0.7, 'n_estimators': 1000, 'min_child_weight': 3, 'max_depth': 5}

    params = {
        'n_estimators': range(500,4000,250)
    }
    xgb = XGBRegressor(learning_rate=0.01, subsample=.7,
                       gamma=1.5, colsample_bytree=.6, objective='reg:squarederror', nthread=4,
                       max_depth=5, min_child_weight=3)

    grid_search = GridSearchCV(xgb, param_grid=params,
                               scoring='neg_root_mean_squared_error',
                               cv=5,
                               verbose=3,
                               n_jobs=-1,
                               refit=True)

    # Here we go
    #random_search.fit(X, y)
    grid_search.fit(X, y)
    best_params = {'max_depth': 6, 'min_child_weight': 7, 'n_estimators': 900, 'subsample': 0.6}
    xgb = XGBRegressor(objective='reg:squarederror', learning_rate=0.01, colsample_bytree=.6,
                       max_depth= 6, min_child_weight= 7, n_estimators= 900, subsample= 0.6)
    xgb.get_booster().feature_names = list(features.columns)
    xgb.fit(X,y)
    sns.set_style('white')
    sns.set_palette('pastel')
    plot_importance(xgb,max_num_features=30)
    plt.show()

    # Other approach
    xgb_params = {'max_depth': 6, 'min_child_weight': 7, 'n_estimators': 900, 'subsample': 0.6,
                  'learning_rate': 0.01, 'colsample_bytree':.6}
    xgb2 = train(xgb_params, dtrain=data_dmatrix)
    plot_importance(xgb2)
    plt.show()'''
