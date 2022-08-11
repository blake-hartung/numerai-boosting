### Testing various models on numer.ai data
Data source: https://numer.ai/

boost_training.py -- utilizes Adaboost, XGBoost, LightGBM, and Catboost to train various
Voting Regressor models on the numerai training data. Models used are both vanilla and
coarsely grid searched through some important hyperparameters within the bounds of the
virtual machine we used on the GCP.

boost_testing.py -- performs testing on the boosting models mentioned above, utilizing
grid search to find optimal parameters on smaller sets of numerai data. This became somewhat
of a chalenge because numerai data is roughly chronological, so creating a random enough
dataset to grid search on was far from trivial.

boost_predictions.py -- uses the numerai validation data to output predictions of the
tuned and vanilla models for performance testing.

boost_pca.py -- loose testing on pca data made from the numerai dataset

boosting.ipynb -- initial testing notebook

rmse_scoring_w_params.json -- output file of boost_testing.py giving optimal
parameters of each model used and some performance metrics on the smaller
data cuts used to grid search.