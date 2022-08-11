import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor

def model_scoring(model, X, y, argument_dict=None, cv=5):
    # for sklearn API compatible gradient boosting models
    
    if not argument_dict:
        # train vanilla model
        cv_results = cross_validate(model, X, y, cv=cv,
                                    scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error'],
                                    return_estimator=True,
                                    return_train_score=True)
        return cv_results
    else:
        # do a grid search
        gs_results = GridSearchCV(model,
                                  param_grid = [argument_dict], cv=cv,
                                  scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error'],
                                  refit='neg_root_mean_squared_error',
                                  return_train_score=True)
        gs_results.fit(X, y)
        return gs_results


def get_time_series_splits(X, cv=3, min_train_length=2000, era_col='era', test_split_pct=0.3):
    """
    This function is meant to split our training data into chronologically stable
    partitions to train. I added min_train_length as some eras are extremely short,
    so partitioning equally by era can create instability in train/test size.
    """
    remaining_eras = list(X[era_col].unique())
    total_split = divmod(len(remaining_eras), cv)
    test_length = round(total_split[0] * test_split_pct)
    if test_length == 0:
        test_length = 1
    train_eras = []
    test_eras = []
    start = 0
    while remaining_eras:
        next_split = remaining_eras[:total_split[0]]
        remaining_eras = remaining_eras[total_split[0]:]
        if len(remaining_eras) == total_split[1]:
            next_split += remaining_eras
            remaining_eras = None
        next_test_split = [next_split.pop() for i in range(test_length)]
        next_test_split.reverse()
        train_eras.append(next_split)
        test_eras.append(next_test_split)
    return zip(train_eras, test_eras)


def get_cross_val_indeces(data, splits):
    tr_indeces = list()
    te_indeces = list()
    for tr, te in splits:
        tr_indeces.append(list(data[data.era.isin(tr)].index))
        te_indeces.append(list(data[data.era.isin(te)].index))
    return tr_indeces, te_indeces



print('Reading minimal training data')
# read the feature metadata amd get the "medium" feature set
with open("features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["medium"]
# read in just those features along with era and target columns
read_columns = features + ['era', 'data_type', 'target']
train = pq.ParquetFile("numerai_training_data.parquet")
train_read = train.read(columns=read_columns)
print(f'Completed Read, data has shape {train_read.shape}')

# set number of rows used for training and read it
x = 1000000
last_xk = [i for i in range(train_read.shape[0] - x, train_read.shape[0])]
df = train_read.take(last_xk).to_pandas()

del(train)
del(train_read)
gc.collect()

# extract features, era data (for time series splitting), and targets
feature_bool = list(map(lambda x: True if x.count('target') == 0 else False, list(df.columns)))
feature_names = [list(df.columns)[i] for i in range(len(feature_bool)) if feature_bool[i]]
era_data = df[['era', 'data_type']]
features = df[feature_names].drop(['era', 'data_type'], axis=1)
targets = df.filter(regex='target')

# grab train/test indeces needed for sklearn api
era_data = era_data.reset_index()
splits = get_time_series_splits(era_data)
tr, te = get_cross_val_indeces(era_data, splits)

# vanilla training / testing
print('Training vanilla models...')
xgb_vanilla = model_scoring(xgb.XGBRegressor(tree_method='gpu_hist'), features, targets['target'], cv=zip(tr, te))
lgbm_vanilla = model_scoring(lgb.LGBMRegressor(), features, targets['target'], cv=zip(tr, te))
ada_vanilla = model_scoring(AdaBoostRegressor(), features, targets['target'], cv=zip(tr, te))
cat_vanilla = model_scoring(cb.CatBoostRegressor(verbose=0), features, targets['target'], cv=zip(tr, te))
print('Vanilla training comlete')

# grid search params
xgb_params = {
    'learning_rate': [0.001, 0.01],
    'max_depth': [4, 5, 6, 7],
    'colsample_bytree': [0.05, 0.1, 0.25],
    'max_leaves': [2**5, 2**6]
}
ada_params = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [25, 50, 100]
}
lgbm_params = {
    'learning_rate': [0.001, 0.01],
    'max_depth': [4, 5, 6, 7],
    'colsample_bytree': [0.05, 0.1, 0.25],
    'num_leaves': [2**5, 2**6]
}
cat_params = {
    'learning_rate': [0.001, 0.01],
    'depth': [4, 5, 6, 7],
    'rsm': [0.05, 0.1, 0.25],
    'max_leaves': [2**5, 2**6]
}

# grid search initialization
print('Performing grid search...')
xgb_param_testing = model_scoring(xgb.XGBRegressor(tree_method='gpu_hist'),
                                  features,
                                  targets['target'],
                                  cv=zip(tr, te),
                                  argument_dict=xgb_params)
print('XGB Complete')
lgbm_param_testing = model_scoring(lgb.LGBMRegressor(),
                                  features,
                                  targets['target'],
                                  cv=zip(tr, te),
                                  argument_dict=lgbm_params)
print('LGBM Complete')
ada_param_testing = model_scoring(AdaBoostRegressor(),
                                  features,
                                  targets['target'],
                                  cv=zip(tr, te),
                                  argument_dict=ada_params)
print('Ada Complete')
cat_model = cb.CatBoostRegressor(verbose=0)
cat_param_testing = cat_model.grid_search(X=features,
                                          y=targets['target'],
                                          cv=zip(tr, te),
                                          param_grid=cat_params)
print('Catboost Complete')
print('Grid Search complete')

# write json to ouput
with open('rmse_scoring_w_params.json', 'w') as f:
    json.dump(test_score_dict, f)

# test voting regressor models using untrained vanilla regressors
xgb_reg = xgb.XGBRegressor(tree_method='gpu_hist')
lgbm_reg = lgb.LGBMRegressor()
ada_reg = AdaBoostRegressor()
cat_reg = cb.CatBoostRegressor(verbose=0)

reg_1 = VotingRegressor([('xgb', xgb_reg), ('lgbm', lgbm_reg)])
reg_1_vanilla = model_scoring(reg_1, features, targets['target'], cv=zip(tr, te))

reg_2 = VotingRegressor([('lgbm', lgbm_reg), ('cat', cat_reg)])
reg_2_vanilla = model_scoring(reg_2, features, targets['target'], cv=zip(tr, te))

reg_3 = VotingRegressor([('xgb', xgb_reg), ('cat', cat_reg)])
reg_3_vanilla = model_scoring(reg_3, features, targets['target'], cv=zip(tr, te))

reg_4 = VotingRegressor([('xgb', xgb_reg), ('lgbm', lgbm_reg), ('cat', cat_reg)])
reg_4_vanilla = model_scoring(reg_4, features, targets['target'], cv=zip(tr, te))

reg_5 = VotingRegressor([('ada', ada_reg), ('lgbm', lgbm_reg)])
reg_5_vanilla = model_scoring(reg_5, features, targets['target'], cv=zip(tr, te))

reg_6 = VotingRegressor([('ada', ada_reg), ('cat', cat_reg)])
reg_6_vanilla = model_scoring(reg_6, features, targets['target'], cv=zip(tr, te))

reg_7 = VotingRegressor([('ada', ada_reg), ('xgb', xgb_reg)])
reg_7_vanilla = model_scoring(reg_7, features, targets['target'], cv=zip(tr, te))

reg_8 = VotingRegressor([('xgb', xgb_reg), ('lgbm', lgbm_reg), ('cat', cat_reg), ('ada', ada_reg)])
reg_8_vanilla = model_scoring(reg_8, features, targets['target'], cv=zip(tr, te))

reg_9 = VotingRegressor([('lgbm', lgbm_reg), ('cat', cat_reg), ('ada', ada_reg)])
reg_9_vanilla = model_scoring(reg_9, features, targets['target'], cv=zip(tr, te))

test_score_dict = {
    'vanilla':
        {
            
            'xgboost': [-np.mean(xgb_vanilla['test_neg_root_mean_squared_error']), np.mean(xgb_vanilla['test_r2']),-np.mean(xgb_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(xgb_vanilla['test_neg_mean_absolute_error'])],
            'lgbm': [-np.mean(lgbm_vanilla['test_neg_root_mean_squared_error']), np.mean(lgbm_vanilla['test_r2']),-np.mean(lgbm_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(lgbm_vanilla['test_neg_mean_absolute_error'])],
            'adaboost': [-np.mean(ada_vanilla['test_neg_root_mean_squared_error']), np.mean(ada_vanilla['test_r2']),-np.mean(ada_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(ada_vanilla['test_neg_mean_absolute_error'])],
            'catboost': [-np.mean(cat_vanilla['test_neg_root_mean_squared_error']), np.mean(cat_vanilla['test_r2']),-np.mean(cat_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(cat_vanilla['test_neg_mean_absolute_error'])],
            'reg_1:xgb+lgbm': [-np.mean(reg_1_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_1_vanilla['test_r2']),-np.mean(reg_1_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_1_vanilla['test_neg_mean_absolute_error'])],
            'reg_2:lgbm+cat': [-np.mean(reg_2_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_2_vanilla['test_r2']),-np.mean(reg_2_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_2_vanilla['test_neg_mean_absolute_error'])],
            'reg_3:xgb+cat': [-np.mean(reg_3_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_3_vanilla['test_r2']),-np.mean(reg_3_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_3_vanilla['test_neg_mean_absolute_error'])],
            'reg_4:xgb+lgbm+cat': [-np.mean(reg_4_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_4_vanilla['test_r2']),-np.mean(reg_4_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_4_vanilla['test_neg_mean_absolute_error'])],
            'reg_5:ada+lgbm': [-np.mean(reg_5_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_5_vanilla['test_r2']),-np.mean(reg_5_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_5_vanilla['test_neg_mean_absolute_error'])],
            'reg_6:ada+cat': [-np.mean(reg_6_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_6_vanilla['test_r2']),-np.mean(reg_6_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_6_vanilla['test_neg_mean_absolute_error'])],
            'reg_7:ada+xgb': [-np.mean(reg_7_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_7_vanilla['test_r2']),-np.mean(reg_7_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_7_vanilla['test_neg_mean_absolute_error'])],
            'reg_8:ada+xgb+lgb+cat': [-np.mean(reg_8_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_8_vanilla['test_r2']),-np.mean(reg_8_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_8_vanilla['test_neg_mean_absolute_error'])],
            'reg_9:ada+lgb+cat': [-np.mean(reg_9_vanilla['test_neg_root_mean_squared_error']), np.mean(reg_9_vanilla['test_r2']),-np.mean(reg_8_vanilla['test_neg_mean_absolute_percentage_error']), -np.mean(reg_9_vanilla['test_neg_mean_absolute_error'])]
            
        }
}
test_score_dict = {
    'vanilla_results':
        {
            'xgboost': {val: list(xgb_vanilla[val]) for val in xgb_vanilla.keys() if val != 'estimator'},
            'lgbm': {val: list(lgbm_vanilla[val]) for val in lgbm_vanilla.keys() if val != 'estimator'},
            'adaboost': {val: list(ada_vanilla[val]) for val in ada_vanilla.keys() if val != 'estimator'},
            'catboost': {val: list(cat_vanilla[val]) for val in cat_vanilla.keys() if val != 'estimator'}
        },
    'vanilla_voting_regressors':
        {
            'reg_1:xgb+lgbm': {val: list(xgb_vanilla[val]) for val in reg_1_vanilla.keys() if val != 'estimator'},
            'reg_2:lgbm+cat': {val: list(xgb_vanilla[val]) for val in reg_2_vanilla.keys() if val != 'estimator'},
            'reg_3:xgb+cat': {val: list(xgb_vanilla[val]) for val in reg_3_vanilla.keys() if val != 'estimator'},
            'reg_4:xgb+lgbm+cat': {val: list(xgb_vanilla[val]) for val in reg_4_vanilla.keys() if val != 'estimator'},
            'reg_5:ada+lgbm': {val: list(xgb_vanilla[val]) for val in reg_5_vanilla.keys() if val != 'estimator'},
            'reg_6:ada+cat': {val: list(xgb_vanilla[val]) for val in reg_6_vanilla.keys() if val != 'estimator'},
            'reg_7:ada+xgb': {val: list(xgb_vanilla[val]) for val in reg_7_vanilla.keys() if val != 'estimator'},
            'reg_8:ada+xgb+lgb+cat': {val: list(xgb_vanilla[val]) for val in reg_8_vanilla.keys() if val != 'estimator'},
            'reg_9:ada+lgb+cat': {val: list(xgb_vanilla[val]) for val in reg_9_vanilla.keys() if val != 'estimator'}
        },
    'gs_results':
        {
            'xgboost': {
                'best_params': xgb_param_testing.best_params_,
                'params': xgb_param_testing.cv_results_['params'],
                'mean_test_scores': list(xgb_param_testing.cv_results_['mean_test_score']),
                'std_test_scores': list(xgb_param_testing.cv_results_['std_test_score']),
                'mean_fit_time': list(xgb_param_testing.cv_results_['mean_fit_time'])
            },
            'lgbm': {
                'best_params': lgbm_param_testing.best_params_,
                'params': lgbm_param_testing.cv_results_['params'],
                'mean_test_scores': list(lgbm_param_testing.cv_results_['mean_test_score']),
                'std_test_scores': list(lgbm_param_testing.cv_results_['std_test_score']),
                'mean_fit_time': list(lgbm_param_testing.cv_results_['mean_fit_time'])
            },
            'adaboost': {
                'best_params': ada_param_testing.best_params_,
                'params': ada_param_testing.cv_results_['params'],
                'mean_test_scores': list(ada_param_testing.cv_results_['mean_test_score']),
                'std_test_scores': list(ada_param_testing.cv_results_['std_test_score']),
                'mean_fit_time': list(ada_param_testing.cv_results_['mean_fit_time'])
            },
            'catboost': {
                'best_params': cat_param_testing['params'],
                'cv_results': dict(cat_param_testing['cv_results'])
            }
        }
}