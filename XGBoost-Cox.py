import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
'''
数据预处理
'''
data = pd.read_csv('DATA.csv')
print('stkcd:',len(data['stkcd'].unique()))
data.info()
data.shape
data.describe()
print(f'There are {len(data.stkcd.unique())} companies.')
print(f'There are {data.isnull().any().sum()} columns in  dataset with missing values.')
missing = data.isnull().sum()/len(data)
missing = missing[missing > 0]
missing.sort_values(inplace = True)
missing.plot.bar()
plt.show()
one_value_fea = [col for col in data.columns if data[col].nunique() <= 1]
print(f'There are {len(one_value_fea)} columns in test dataset with one unique value.')
corr_mat = data.corr()
f, ax = plt.subplots(figsize=(12, 8))
mask = np.zeros_like(corr_mat)
for i in range(1,len(mask)):
  for j in range(0,i):
    mask[i][j] = True
sns.heatmap(corr_mat, annot=True,mask=mask,linewidths=.05,square=False,annot_kws={'size':8})
print(corr_mat)
plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
plt.show()
# --------------------------------------------------------------------------------
'''
特征集构建
利用与T，E相关性高的风险变量生成新的因子，筛选出其中相关性更高的新特征，标准为与T，E相关性在0.1以上,变量间的相关性在0.7以下
时间序列因子；波动率因子
'''
window = 8;data['f1'] = data.groupby('stkcd',as_index=False).apply(lambda x : x['z1']- x['z1'].shift(window)).values;data['f1'].values[:window] =0 
window = 4;data['f2'] = data.groupby('stkcd',as_index=False).apply(lambda x : x['z1'].rolling(window).std()/x['z1'].rolling(window).mean()).values;data['f2'].values[:window] = 0
window = 4;data['f3'] = data.groupby('stkcd',as_index=False).apply(lambda x : x['z3'].rolling(window).mean()).values;data['f3'].values[:window] = 0
window = 4;data['f4'] = data.groupby('stkcd',as_index=False).apply(lambda x : x['z5'].rolling(window).std()/x['z5'].rolling(window).mean()).values;data['f4'].values[:window] = 0
window = 4;data['f5'] = data.groupby('stkcd',as_index=False).apply(lambda x : x['z17'].rolling(window).mean()).values;data['f5'].values[:window] = 0
for column in list(data.columns[data.isnull().sum() > 0]):
    mean_val = data[column].mean()
    data[column].fillna(mean_val, inplace=True)

data['z15'] = np.log(data['z15'])
data.to_csv('data.csv')
train = data[data['time'] <= '2019/10/1']
test = data[data['time'] > '2019/10/1']
# ---------------------------------------------------------------------
'''
XGBoost-Cox 模型         
'''
# -------------------------------------------------------------------
import collections
from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test 
from functools import cmp_to_key
from lifelines.utils import concordance_index
from _efn_core import _abs_sort,_label_abs_sort,efn_loss,_efn_grads
from vision import plot_train_curve, plot_surv_curve
from utils import concordance_index, baseline_survival_function, _check_params
from xgboost import DMatrix
from xgboost import plot_importance
from model import model as EfnBoost
from base import survival_stats,survival_dmat,survival_df

fig,ax = plt.subplots(figsize=(10,8))
kmf = KaplanMeierFitter()
kmf.fit(data['T'],event_observed=data['E'])
kmf.plot_survival_function(at_risk_counts=True,ax=ax)
plt.show()
#longrank检验
# data.drop(['stkcd','time','T','E'],axis=1,inplace=True)
# for column in data.columns:
#     data[column] = data[column].apply(lambda x : '1' if x>= data[column].median() else '0')
# data.to_csv('data_1.csv')
# data_1 = pd.read_csv('data_1.csv')
# data_1.drop('Unnamed: 0',axis=1,inplace = True)
# for feature in data_1.columns:
#     p_value = multivariate_logrank_test(event_durations = data['T'],  
#                                     groups=data_1[feature],  
#                                     event_observed=data['E']
#                                     ).p_value
#     p_value_text = ['p_value = %.4F'%p_value][0]
#     print(f'X = {feature} logrank test {p_value_text}.')

#
survival_stats(train, t_col="T", e_col="E", plot=True)
plt.show()
survival_stats(test, t_col="T", e_col="E", plot=True)
# via survival_dmat function
train = train.drop(['stkcd','time'],axis=1)
train.to_csv('train.csv')
# X = train.drop('T', axis=1)
# y = train['T']
# base_class = sklearn_adapter(CoxPHFitter, event_col='E')
# cph = base_class()
# params = {
#     "penalizer": 10.0 ** np.arange(-2, 3),
#     "l1_ratio": np.linspace(0,1,6)
# }
# clf = GridSearchCV(cph, param_grid=params, cv=10)
# clf.fit(X, y)
# cph = CoxPHFitter(**clf.best_params_)
# print(clf.best_params_)


cph = CoxPHFitter(penalizer=0.005)
cph.fit(train,duration_col='T',event_col='E')
cph.print_summary()
fig,ax = plt.subplots(figsize=(12,9))
cph.plot(ax=ax)
plt.show()
cph.check_assumptions(train,show_plots=True)
print(train['z1'].describe())
print(train['z3'].describe())
fig,ax = plt.subplots(figsize=(12,9))
cph.plot_partial_effects_on_outcome(['z1','z3'],
                                    values=[[0,-2],
                                            [0,4],
                                            [4,0],
                                            [4,-2],
                                            [4,4]],
                                    cmap='coolwarm',ax=ax)
plt.show()

# train_describe = train.describe()
# train_describe = pd.DataFrame(train_describe)
# train_describe.to_csv('train.dec.csv')
data_train = survival_df(train, t_col="T", e_col="E", label_col="Y")
test = test.drop(['stkcd','time'],axis=1)
# test_describe = test.describe()
# test_describe = pd.DataFrame(test_describe)
# test_describe.to_csv('test.dec.csv')
data_test = survival_df(test, t_col="T", e_col="E", label_col="Y")
print(data_test)

x_cols = list(data_train.columns)[:-1]
print(x_cols)
surv_train = DMatrix(data_train[x_cols], label=data_train['Y'].values)
print(surv_train)
surv_test = DMatrix(data_test[x_cols], label=data_test['Y'].values)

params = {
    'eta':0.08,
    'max_depth':6, 
    'min_child_weight': 0.17, 
    'subsample': 0.6,
    #'colsample_bytree': 0.5,
    #'gamma': 0.20,
    #'lambda': 0,
}

model = EfnBoost(params)

eval_result = model.train(
    surv_train,
    num_rounds=90,
    skip_rounds=50,
    evals=[(surv_train, 'train'), (surv_test, 'test')],
    plot=True
)


'''
model.predict: Hazard Ratio of coxph
model.predict_survival_function: survival function transformed from hazard ratio
'''
# select the first 5 samples
surv_data_now = surv_test.slice([i for i in range(1)])

# Predict Hazard Ratio
print("Prediction of hazard ratio:", model.predict(surv_data_now, output_margin=False))
surv_data_now = surv_test.slice([i])
result_survf = model.predict_survival_function(surv_data_now, plot=True)
print(result_survf)
# Predict Hazard Ratio
print("Prediction of hazard ratio:", model.predict(surv_test, output_margin=False))
hazard_ratio = model.predict(surv_test, output_margin=False)
print(hazard_ratio)
hazard_ratio = pd.DataFrame(hazard_ratio)
hazard_ratio.to_csv('hazard_ratio.csv')
# Predict Survival Function
#result_survf = model.predict_survival_function(surv_test, plot=True)
print(result_survf)

# evaluate model performance 
print("CI on training set:", model.evals(surv_train))
print("CI on test set:", model.evals(surv_test))
fscore = model.get_factor_score(importance_type='weight')

# normalize the scores
sum_score = sum(fscore.values())
fscore = {k: v / sum_score for k, v in fscore.items()}

list= sorted(fscore.items(),key=lambda x:x[1],reverse=True)
print(list)
score = pd.DataFrame(list)
score.to_csv('score.csv')
model.save_model("XGBoost-Cox.model")

fig,ax = plt.subplots(figsize=(12,9))
cph.plot(ax=ax)
plt.show()
