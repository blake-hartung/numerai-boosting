import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
import gc
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor
import pickle

## environment stuff
TRAIN_DATA_PATH = 'numerai_training_data.parquet'
MODEL_PATH = 'models/'

### Train the models using our full dataset

# Read in entire dataset
train = pq.ParquetFile(TRAIN_DATA_PATH)
train_read = train.read()
print(f'Completed Training Set Read, data has shape {train_read.shape}')

df = train_read.to_pandas()

# garbage collection for memory
del(train)
del(train_read)
gc.collect()

feature_bool = list(map(lambda x: True if x.count('target') == 0 else False, list(df.columns)))
feature_names = list(set([list(df.columns)[i] for i in range(len(feature_bool)) if feature_bool[i]]) - set(['era', 'data_type']))

# train the vanilla models
print('Training vanilla models...')

xgb_vanilla = xgb.XGBRegressor(tree_method='hist')
xgb_vanilla.fit(df[feature_names], df['target'])
pickle.dump(xgb_vanilla, open(MODEL_PATH + 'xgb_vanilla.pkl', "wb"))


lgbm_vanilla = lgb.LGBMRegressor()
lgbm_vanilla.fit(df[feature_names], df['target'])
pickle.dump(lgbm_vanilla, open(MODEL_PATH + 'lgbm_vanilla.pkl', "wb"))


ada_vanilla = AdaBoostRegressor()
ada_vanilla.fit(df[feature_names], df['target'])
pickle.dump(ada_vanilla, open(MODEL_PATH + 'ada_vanilla.pkl', "wb"))


cat_vanilla = cb.CatBoostRegressor(verbose=0)
cat_vanilla.fit(df[feature_names], df['target'])
pickle.dump(cat_vanilla, open(MODEL_PATH + 'cat_vanilla.pkl', "wb"))

print('Vanilla models trained.')

del(xgb_vanilla)
del(lgbm_vanilla)
del(ada_vanilla)
del(cat_vanilla)
gc.collect()

# train vanilla voting models
xgb_reg = xgb.XGBRegressor(tree_method='hist')
lgbm_reg = lgb.LGBMRegressor()
ada_reg = AdaBoostRegressor()
cat_reg = cb.CatBoostRegressor(verbose=0)

print('Training vanilla voting models...')

reg_1 = VotingRegressor([('xgb', xgb_reg), ('lgbm', lgbm_reg), ('cat', cat_reg)])
reg_1.fit(df[feature_names], df['target'])
pickle.dump(reg_1, open(MODEL_PATH + 'xlc_vanilla.pkl', "wb"))


reg_2 = VotingRegressor([('xgb', xgb_reg), ('lgbm', lgbm_reg), ('cat', cat_reg), ('ada', ada_reg)])
reg_2.fit(df[feature_names], df['target'])
pickle.dump(reg_2, open(MODEL_PATH + 'xlca_vanilla.pkl', "wb"))


reg_3 = VotingRegressor([('lgbm', lgbm_reg), ('cat', cat_reg)])
reg_3.fit(df[feature_names], df['target'])
pickle.dump(reg_3, open(MODEL_PATH + 'lc_vanilla.pkl', "wb"))


reg_4 = VotingRegressor([('ada', ada_reg), ('cat', cat_reg)])
reg_4.fit(df[feature_names], df['target'])
pickle.dump(reg_4, open(MODEL_PATH + 'ac_vanilla.pkl', "wb"))

print('Vanilla voting models trained.')

del(reg_1)
del(reg_2)
del(reg_3)
del(reg_4)
gc.collect()

# optimal model params
params = {
    'xgb': {"tree_method": "hist",
            "colsample_bytree": 0.25,
            "learning_rate": 0.01,
            "max_depth": 7,
            "max_leaves": 64},
    'lgbm': {"colsample_bytree": 0.25,
             "learning_rate": 0.01,
             "max_depth": 7,
             "num_leaves": 64},
    'ada': {"learning_rate": 0.1,
            "n_estimators": 50},
    'cat': {"max_leaves": 64,
            "depth": 6,
            "rsm": 0.25,
            "learning_rate": 0.01,
            "verbose": 0}
}

# train tuned models

print('Training tuned models...')

xgb_tuned = xgb.XGBRegressor(**params['xgb'])
xgb_tuned.fit(df[feature_names], df['target'])
pickle.dump(xgb_tuned, open(MODEL_PATH + 'xgb_tuned.pkl', "wb"))


lgbm_tuned = lgb.LGBMRegressor(**params['lgbm'])
lgbm_tuned.fit(df[feature_names], df['target'])
pickle.dump(xgb_tuned, open(MODEL_PATH + 'lgbm_tuned.pkl', "wb"))


ada_tuned = AdaBoostRegressor(**params['ada'])
ada_tuned.fit(df[feature_names], df['target'])
pickle.dump(xgb_tuned, open(MODEL_PATH + 'ada_tuned.pkl', "wb"))


cat_tuned = cb.CatBoostRegressor(**params['cat'])
cat_tuned.fit(df[feature_names], df['target'])
pickle.dump(cat_tuned, open(MODEL_PATH + 'cat_tuned.pkl', "wb"))

print('Tuning models trained.')

del(xgb_tuned)
del(lgbm_tuned)
del(ada_tuned)
del(cat_tuned)
gc.collect()

# train tuned voting regressors

xgb_regt = xgb.XGBRegressor(**params['xgb'])
lgbm_regt = lgb.LGBMRegressor(**params['xgb'])
ada_regt = AdaBoostRegressor(**params['xgb'])
cat_regt = cb.CatBoostRegressor(**params['xgb'])

print('Training tuned voting models...')

reg_1t = VotingRegressor([('xgb', xgb_regt), ('lgbm', lgbm_regt), ('cat', cat_regt)])
reg_1t.fit(df[feature_names], df['target'])
pickle.dump(reg_1t, open(MODEL_PATH + 'xlc_tuned.pkl', "wb"))


reg_2t = VotingRegressor([('xgb', xgb_regt), ('lgbm', lgbm_regt), ('cat', cat_regt), ('ada', ada_regt)])
reg_2t.fit(df[feature_names], df['target'])
pickle.dump(reg_2t, open(MODEL_PATH + 'xlca_tuned.pkl', "wb"))


reg_3t = VotingRegressor([('lgbm', lgbm_regt), ('cat', cat_regt)])
reg_3t.fit(df[feature_names], df['target'])
pickle.dump(reg_3t, open(MODEL_PATH + 'lc_tuned.pkl', "wb"))


reg_4t = VotingRegressor([('ada', ada_regt), ('cat', cat_regt)])
reg_4t.fit(df[feature_names], df['target'])
pickle.dump(reg_4t, open(MODEL_PATH + 'ac_tuned.pkl', "wb"))

print('Tuned voting models trained...')
print('Training complete')
