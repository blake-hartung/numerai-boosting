import pickle
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import gc
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor

VAL_DATA_PATH = 'numerai_validation_data.parquet'
PREDICTIONS_PATH = 'boosting_predictions_vanilla.csv'
MODEL_PATH = 'models/'

# import validation set
val = pq.ParquetFile(VAL_DATA_PATH)
val_read = val.read(f'Completed Training Set Read, data has shape {train_read.shape}')

df_val = val_read.to_pandas()
feature_bool = list(map(lambda x: True if x.count('target') == 0 else False, list(df_val.columns)))
feature_names = list(set([list(df_val.columns)[i] for i in range(len(feature_bool)) if feature_bool[i]]) - set(['era', 'data_type']))

predictions = pd.DataFrame(df_val['target'])

# make predictions (vanilla)
xgb_vanilla = pickle.load(open(MODEL_PATH + 'xgb_vanilla.pkl', "rb"))
lgbm_vanilla = pickle.load(open(MODEL_PATH + 'lgbm_vanilla.pkl', "rb"))
ada_vanilla = pickle.load(open(MODEL_PATH + 'ada_vanilla.pkl', "rb"))
cat_vanilla = pickle.load(open(MODEL_PATH + 'cat_vanilla.pkl', "rb"))
predictions['xgb_vanilla_preds'] = xgb_vanilla.predict(df_val[feature_names])
predictions['lgbm_vanilla_preds'] = lgbm_vanilla.predict(df_val[feature_names])
predictions['ada_vanilla_preds'] = ada_vanilla.predict(df_val[feature_names])
predictions['cat_vanilla_preds'] = cat_vanilla.predict(df_val[feature_names])

del(xgb_vanilla)
del(lgbm_vanilla)
del(ada_vanilla)
del(cat_vanilla)
gc.collect()

# voting regressor (vanilla)
reg_1 = pickle.load(open(MODEL_PATH + 'xlc_vanilla.pkl', "rb"))
reg_2 = pickle.load(open(MODEL_PATH + 'xlca_vanilla.pkl', "rb"))
reg_3 = pickle.load(open(MODEL_PATH + 'lc_vanilla.pkl', "rb"))
reg_4 = pickle.load(open(MODEL_PATH + 'ac_vanilla.pkl', "rb"))
predictions['xlv_vanilla_preds'] = reg_1.predict(df_val[feature_names])
predictions['xlca_vanilla_preds'] = reg_2.predict(df_val[feature_names])
predictions['lc_vanilla_preds'] = reg_3.predict(df_val[feature_names])
predictions['ac_vanilla_preds'] = reg_4.predict(df_val[feature_names])

del(reg_1)
del(reg_2)
del(reg_3)
del(reg_4)
gc.collect()

# tuned
xgb_tuned = pickle.load(open(MODEL_PATH + 'xgb_tuned.pkl', "rb"))
lgbm_tuned = pickle.load(open(MODEL_PATH + 'lgbm_tuned.pkl', "rb"))
ada_tuned = pickle.load(open(MODEL_PATH + 'ada_tuned.pkl', "rb"))
cat_tuned = pickle.load(open(MODEL_PATH + 'cat_tuned.pkl', "rb"))
predictions['xgb_tuned_preds'] = xgb_tuned.predict(df_val[feature_names])
predictions['lgbm_tuned_preds'] = lgbm_tuned.predict(df_val[feature_names])
predictions['ada_tuned_preds'] = ada_tuned.predict(df_val[feature_names])
predictions['cat_tuned_preds'] = cat_tuned.predict(df_val[feature_names])

del(xgb_tuned)
del(lgbm_tuned)
del(ada_tuned)
del(cat_tuned)
gc.collect()

# tuned voting regressor
reg_1t = pickle.load(open(MODEL_PATH + 'xlc_tuned.pkl', "rb"))
reg_2t = pickle.load(open(MODEL_PATH + 'xlca_tuned.pkl', "rb"))
reg_3t = pickle.load(open(MODEL_PATH + 'lc_tuned.pkl', "rb"))
reg_4t = pickle.load(open(MODEL_PATH + 'ac_tuned.pkl', "rb"))
predictions['xlv_tuned_preds'] = reg_1.predict(df_val[feature_names])
predictions['xlca_tuned_preds'] = reg_2.predict(df_val[feature_names])
predictions['lc_tuned_preds'] = reg_3.predict(df_val[feature_names])
predictions['ac_tuned_preds'] = reg_4.predict(df_val[feature_names])


