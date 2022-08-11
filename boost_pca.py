import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import gc
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import AdaBoostRegressor
import pickle
from sklearn.decomposition import PCA

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

pca = PCA(n_components=100)
pca_feats = pca.fit_transform(df[feature_names].to_numpy())

targets = df['target']

del(df)
gc.collect()

# train the pca models
print('Training pca models...')
xgb_pca = xgb.XGBRegressor(tree_method='hist')
xgb_pca.fit(pca_feats, targets)
pickle.dump(xgb_pca, open(MODEL_PATH + 'xgb_pca.pkl', "wb"))


lgbm_pca = lgb.LGBMRegressor()
lgbm_pca.fit(pca_feats, targets)
pickle.dump(lgbm_pca, open(MODEL_PATH + 'lgbm_pca.pkl', "wb"))


ada_pca = AdaBoostRegressor()
ada_pca.fit(pca_feats, targets)
pickle.dump(ada_pca, open(MODEL_PATH + 'ada_pca.pkl', "wb"))


cat_pca = cb.CatBoostRegressor(verbose=0)
cat_pca.fit(pca_feats, targets)
pickle.dump(cat_pca, open(MODEL_PATH + 'cat_pca.pkl', "wb"))

print('Vanilla models trained.')
