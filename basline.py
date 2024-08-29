import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter, CoxPHFitter 
from model import model as EfnBoost
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
       
from lifelines.statistics import logrank_test, multivariate_logrank_test 
from sklearn.model_selection import train_test_split, GridSearchCV

from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines.utils import k_fold_cross_validation
from lifelines.utils import concordance_index


import json
import hyperopt as hpt
from sklearn.model_selection import RepeatedKFold
from base import survival_dmat
global Logval, eval_cnt, time_start
global train_data
global max_iters, k_fold, n_repeats 
global T_col, E_col


pd.options.mode.use_inf_as_na = True
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

data = pd.read_csv('data.csv')
features = [f for f in data.columns if f not in ['Unnamed: 0','stkcd','time','T','E']]
train_data = data.drop(['Unnamed: 0','stkcd','time'],axis=1)
x_train = data[features]
y_train = data['E']

# #

# #分类器
def cv_model(clf,x_train,y_train,clf_name):
	"""
	    @param clf: 分类器包名(lgb&xgb)、分类器实例名(CatBoostClassifier)
        @param x_train: 训练集X
        @param y_train: 训练集y
        @param clf_name: 分类器名：'xgb' 'lr' xgboost-Cox'

        @return: model 最后一次交叉训练后的模型
        @return: train 分类器在训练集上的预测结果
	"""
	folds = 5
	seed = 2020
	kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state = seed)

	train = np.zeros(x_train.shape[0])
	train_score = np.zeros(x_train.shape[0])
	# cv_accuracy_scores = []
	# cv_f1_scores = []
	cv_auc_scores = []

	for i,(train_index,valid_index) in enumerate(kf.split(x_train,y_train)):
		print('************************************ {0}/{1} ************************************'.format(str(i+1), folds))
		trn_x, trn_y, val_x, val_y = x_train.iloc[train_index], y_train[train_index], x_train.iloc[valid_index], y_train[valid_index]
		if clf_name == 'xgb':
			train_matrix = clf.DMatrix(trn_x,label=trn_y)
			valid_matrix = clf.DMatrix(val_x,label=val_y)
			params = {'booster': 'gbtree',
	                  'objective': 'binary:logistic',
	                  'eval_metric': 'auc',
	                  'gamma': 1,
	                  'min_child_weight': 1.5,
	                  'max_depth': 5,
	                  'lambda': 10,
	                  'subsample': 0.7,
	                  'colsample_bytree': 0.7,
	                  'colsample_bylevel': 0.7,
	                  'eta': 0.04,
	                  'tree_method': 'exact',
	                  'seed': 2020,
	                  'nthread': 36,
	        }
			watchlist = [(train_matrix,'train'),(valid_matrix,'test')]
			model = clf.train(params,train_matrix,num_boost_round=1000,evals=watchlist,verbose_eval=200,early_stopping_rounds=200)
			val_score  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)

		if clf_name == 'lr':
			model = clf()
			model = model.fit(trn_x, trn_y)
			val_score = model.predict_proba(val_x)[:, 1]

	    # 交叉验证每次得到一部分结果
		val_pred = np.int8(np.round(val_score))
		train[valid_index] = val_pred
		train_score[valid_index] = val_score

		cv_auc_scores.append(roc_auc_score(val_y, val_score))
		print(cv_auc_scores)

	if clf_name == 'xgb':
		train_matrix=clf.DMatrix(x_train , label=y_train)
		watchlist=[(train_matrix,'train')]
		model=clf.train(params, train_matrix, num_boost_round=1000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
	
	if clf_name == 'lr':
		model = clf()
		model = model.fit(x_train, y_train)

	print("%s_auc_score_list:" % clf_name, cv_auc_scores)
	print("%s_auc_score_mean:" % clf_name, np.mean(cv_auc_scores))
	print("%s_auc_score_std:" % clf_name, np.std(cv_auc_scores))

	return model,train,train_score

def xgb_model(x_train,y_train):
	xgb_trained_model, xgb_train, xgb_train_score = cv_model(xgb, x_train, y_train, 'xgb')
	return xgb_trained_model,xgb_train, xgb_train_score
xgb_trained_model, xgb_train, xgb_train_score = xgb_model(x_train, y_train)
print('XGBoost AUC Score: {0:.2f}'.format(roc_auc_score(y_train, xgb_train_score)))

def lr_model(x_train, y_train):
	lr_trained_model, lr_train, lr_train_score = cv_model(LogisticRegression, x_train, y_train, 'lr')
	return lr_trained_model, lr_train, lr_train_score
lr_trained_model, lr_train, lr_train_score = lr_model(x_train, y_train)
print('Logistic Regression AUC Score: {0:.2f}'.format(roc_auc_score(y_train, lr_train_score)))

# #XGBoost-Cox
# params = {
#           'eta':0.08,
#           'max_depth':6, 
#           'min_child_weight': 0.17, 
#           'subsample': 0.6,
#           'colsample_bytree': 0.5,
#           #'gamma': 0.20,
#           #'lambda': 0,
# }
#     # Build and train model

# rskf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=64)

# for i,(train_index, test_index ) in enumerate(rskf.split(train_data)):
# 	data_train = train_data.iloc[train_index, :]
# 	data_validate = train_data.iloc[test_index,  :]
# 	dtrain = survival_dmat(data_train, t_col='T', e_col='E', label_col="Y")
# 	dtest = survival_dmat(data_validate, t_col='T', e_col='E', label_col="Y")
# 	model = EfnBoost(params)
# 	eval_result = model.train(
#                              dtrain,
#                              num_rounds=200,
#                              skip_rounds=20,
#                              evals=[(dtrain, 'train'), (dtest, 'test')],
#                              plot=False
#     )
# 	print("CI on training set:", model.evals(dtrain))
# 	print("CI on training set:", model.evals(dtest))




