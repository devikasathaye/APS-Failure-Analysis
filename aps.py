#!/usr/bin/env python
# coding: utf-8

# ## 1. The LASSO and Boosting for Regression

# ### Importing Libraries

import sklearn
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, mean_squared_error
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
import weka.core.converters as converters
import os.path
from os import path
import weka.core.serialization as serialization
from weka.classifiers import Evaluation
from weka.core.classes import Random
import weka.core.packages as packages
from weka.filters import Filter
import weka.plot.classifiers as plcls  # NB: matplotlib is required
from weka.classifiers import FilteredClassifier

# To suppress warnings
warnings.filterwarnings('ignore')


# ### (a) Download the Communities and Crime data and APS Failure data from GIT local repository

get_ipython().system(' git clone https://github.com/devikasathaye/APS-Failure-Analysis')


# ### Data Management. Train and test split.

column_names = []
file = open("APS-Failure-Analysis/communities.names", "r")
for line in file:
    if line.startswith("@attribute"):
        column_names.append(line.split()[1])

print(column_names)

df_all = pd.read_csv('APS-Failure-Analysis/communities.csv', sep=',', header=None, skiprows=0, na_values='?', names=column_names)
df_train = df_all.iloc[:1495]
df_test = df_all.iloc[1495:]

print("Top few rows of entire data")
print(df_all.head())
print("Size of entire data")
print(df_all.shape)
print("")

print("Top few rows of train data")
print(df_train.head())
print("Size of train data")
print(df_train.shape)
print("")

print("Top few rows of test data")
print(df_test.head())
print("Size of test data")
print(df_test.shape)


# ### (b) Use a data imputation technique to deal with the missing values in the data set.

X_train = df_train.drop(columns=['ViolentCrimesPerPop'])
y_train = df_train['ViolentCrimesPerPop']
X_test = df_test.drop(columns=['ViolentCrimesPerPop'])
y_test = df_test['ViolentCrimesPerPop']

simple_imputer = SimpleImputer()
X_train_full = pd.DataFrame(simple_imputer.fit_transform(X_train.iloc[:,5:]), columns=column_names[5:-1])
X_train = X_train_full.copy()
# now transform test
X_test_full = pd.DataFrame(simple_imputer.transform(X_test.iloc[:,5:]), columns=column_names[5:-1], index=X_test.index)
X_test = X_test_full.copy()
print("After data imputation")
print("Train dataset")
print(X_train)
print("Test dataset")
print(X_test)


# ### (c) Plot a correlation matrix for the features in the data set.

corr = X_train.corr()
f, ax = plt.subplots(figsize=(20, 17))
corr_mat = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, linewidths=.1)


# ### (d) Calculate the Coefficient of Variation CV for each feature

CoV=X_train.std()/X_train.mean()
print("Coefficient of Variation CV for each feature is")
CoV


# ### (e) Pick &lfloor;&radic;128&rfloor; features with highest CV

# Pick floor(sqrt(128)) features with highest CV
n_fea = math.floor(math.sqrt(128))
n_fea

CoV_highest = CoV.sort_values(ascending=False)[:n_fea].axes[0].to_list()
CoV_highest


# ### Make scatter plots

a=0

def diagfunc(x, **kws):
    global a
    ax = plt.gca()
    ax.annotate(CoV_highest[a], xy=(0.3, 0.5), xycoords=ax.transAxes)
    a=a+1

sns.set(context="paper")

g = sns.PairGrid(X_train,vars=CoV_highest).map_diag(diagfunc)
g = g.map_offdiag(plt.scatter)

for ax in g.axes.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')


# ### Make box plots
plt.figure(figsize=(20,20))
sns.boxplot(data=X_train[CoV_highest],orient='h')


# ### Can you draw conclusions about signicance of those features, just by the scatter plots?

# No.

# ### (f) Fit a linear model using least squares to the training set and report the test error.

lrmodel = LinearRegression().fit(X_train, y_train)
lrmodel_predict = lrmodel.predict(X_test)
test_error_lrmodel=mean_squared_error(y_test,lrmodel_predict)
print("The test error of the Linear Regression model is", test_error_lrmodel)


# ### (g) Fit a ridge regression model on the training set, with &lambda; chosen by cross-validation. Report the test error obtained.

g_alpha=[]
g_acc=[]
alpha=0.00001
while(alpha<10000):
    g1_ridge_model = Ridge(alpha=alpha)
    crossval_scores = cross_val_score(g1_ridge_model,X_train,y_train,cv=5)
    g_alpha.append(alpha)
    g_acc.append(crossval_scores.mean())
    alpha = alpha*10

best_alpha_ridge = g_alpha[g_acc.index(max(g_acc))]
print("Best value of alpha for Ridge Regression model is", best_alpha_ridge)
print("The cross-validation accuracy is {}%".format(max(g_acc)*100))

g2_ridge_model = Ridge(alpha=best_alpha_ridge)
g2_ridge_model.fit(X_train, y_train)
g_test_err = 1-(g2_ridge_model.score(X_test, y_test))
print("The test error of Ridge Regression model is", g_test_err)


# ### (h) Fit a LASSO model on the training set, with &lambda; chosen by cross-validation.

h_alpha=[]
h_acc=[]
alpha=0.00001
while(alpha<10000):
    h1_lasso_model = linear_model.Lasso(alpha=alpha)
    crossval_scores = cross_val_score(h1_lasso_model,X_train,y_train,cv=5)
    h_alpha.append(alpha)
    h_acc.append(crossval_scores.mean())
    alpha = alpha*10

best_alpha_lasso = h_alpha[h_acc.index(max(h_acc))]
print("Best value of alpha for Lasso is", best_alpha_lasso)
print("The cross-validation accuracy is {}%".format(max(h_acc)*100))


# ### Report the test error obtained, along with a list of the variables selected by the model.

h2_lasso_model = linear_model.Lasso(alpha=best_alpha_lasso)
h2_lasso_model.fit(X_train,y_train)
h_test_err = 1-(h2_lasso_model.score(X_test, y_test))
print("The test error of Lasso Regression model is", h_test_err)

sel_fea_lasso = []
lasso_coeffs = h2_lasso_model.coef_
for i in range(len(lasso_coeffs)):
    if(lasso_coeffs[i]!=0):
        sel_fea_lasso.append(column_names[i+5])
print("List of variables selected by the model is")
print(sel_fea_lasso)
print("Total number of features selected is", len(sel_fea_lasso))


# ### Repeat with standardized features.

scaler = StandardScaler()
X_train_std = pd.DataFrame(scaler.fit_transform(X_train), columns=column_names[5:-1])
X_test_std = pd.DataFrame(scaler.fit_transform(X_test), columns=column_names[5:-1])

h_alpha_std=[]
h_acc_std=[]
alpha=0.00001
while(alpha<10000):
    h1_lasso_model_std = linear_model.Lasso(alpha=alpha)
    crossval_scores = cross_val_score(h1_lasso_model_std,X_train_std,y_train,cv=5)
    h_alpha_std.append(alpha)
    h_acc_std.append(crossval_scores.mean())
    alpha = alpha*10

best_alpha_lasso_std = h_alpha_std[h_acc_std.index(max(h_acc_std))]
print("Best value of alpha for Lasso is", best_alpha_lasso_std)
print("The cross-validation accuracy is {}%".format(max(h_acc_std)*100))

h2_lasso_model_std = linear_model.Lasso(alpha=best_alpha_lasso_std)
h2_lasso_model_std.fit(X_train_std,y_train)
h_test_err_std = 1-(h2_lasso_model_std.score(X_test_std, y_test))
print("The test error of Lasso Regression model with standardized features is", h_test_err_std)

sel_fea_lasso_std = []
lasso_coeffs_std = h2_lasso_model_std.coef_
for i in range(len(lasso_coeffs_std)):
    if(lasso_coeffs_std[i]!=0):
        sel_fea_lasso_std.append(column_names[i+5])
print("List of variables selected by the model is")
print(sel_fea_lasso_std)
print("Total number of features selected is", len(sel_fea_lasso_std))


# ### Report the test error for both cases and compare them.

print("Model\t\t\t\t\tTest Error")
print("Lasso(Normalized features)\t\t",h_test_err)
print("Lasso(Standardized features)\t\t",h_test_err_std)


# ### (i) Fit a PCR model on the training set, with M (the number of principal components) chosen by cross-validation. Report the test error obtained.

i_acc = []
for m in range(1, len(column_names[5:-1])+1):
    pcr = PCA(n_components=m)
    X_train_pca = pcr.fit_transform(X_train)
    i_lrmodel = LinearRegression()
    crossval_scores = cross_val_score(i_lrmodel,X_train_pca, y_train,cv=5)
    i_acc.append(crossval_scores.mean())

best_m_pcr = 1+(i_acc.index(max(i_acc)))
print("Best M(number of principal components) is", best_m_pcr)
print("Cross-validation accuracy is {}%".format(max(i_acc)*100))

pcr2 = PCA(n_components=best_m_pcr)
X_train_pca2 = pcr2.fit_transform(X_train)
X_test_pca2 = pcr2.transform(X_test)
i2_lrmodel = LinearRegression().fit(X_train_pca2, y_train)
i_test_err = 1-(i2_lrmodel.score(X_test_pca2, y_test))
print("The test error of PCR is", i_test_err)


# ### (j) Fit a boosting tree to the data. Determine &alpha;(the regularization term) using cross-validation.

j_acc = []
j_reg_alpha = []
reg_alpha=0.00001
while(reg_alpha<10000):
    xr = xgboost.XGBRegressor(reg_alpha=reg_alpha, reg_lambda=0, objective='reg:squarederror')
    crossval_scores = cross_val_score(xr,X_train,y_train,cv=5)
    j_reg_alpha.append(reg_alpha)
    j_acc.append(crossval_scores.mean())
    reg_alpha = reg_alpha*10

best_reg_alpha_xgboost = j_reg_alpha[j_acc.index(max(j_acc))]
print("Best value of alpha for L1-penalized gradient boosting tree is", best_reg_alpha_xgboost)
print("The cross-validation accuracy is {}%".format(max(j_acc)*100))

xr2 = xgboost.XGBRegressor(reg_alpha=best_reg_alpha_xgboost, reg_lambda=0, objective='reg:squarederror')
xr2.fit(X_train, y_train)
j_test_err = 1-(xr2.score(X_test, y_test))
print("The test error of L1-penalized gradient boosting tree is", j_test_err)


# ## 2. Tree-Based Methods

# ### (a) Train and test split
df_train_tree = pd.read_csv('APS-Failure-Analysis/aps_failure_training_set.csv', sep=',', header=0, skiprows=20, na_values='na')
df_test_tree = pd.read_csv('APS-Failure-Analysis/aps_failure_test_set.csv', sep=',', header=0, skiprows=20, na_values='na')

print("Top few rows of train data")
print(df_train_tree.head())
print("Size of train data")
print(df_train_tree.shape)
print("")

print("Top few rows of test data")
print(df_test_tree.head())
print("Size of test data")
print(df_test_tree.shape)

y_train_tree = df_train_tree['class']
y_test_tree = df_test_tree['class']


# ### (b) Data Preparation

# ### i. Research what types of techniques are usually used for dealing with data with missing values.

# The techniques used for data imputation are-<br>
#     Mean imputation- Simply calculate the mean of the observed values for that variable for all individuals who are non-missing.
#     Substitution- Impute the value from a new individual who was not selected to be in the sample.
#     Hot deck imputation- A randomly chosen value from an individual in the sample who has similar values on other variables.
#     Cold deck imputation- A systematically chosen value from an individual who has similar values on other variables.
#     Regression imputation- The predicted value obtained by regressing the missing variable on other variables.
#     Stochastic regression imputation- The predicted value from a regression plus a random residual value.
#     Interpolation and extrapolation- An estimated value from other observations from the same individual.
#     Iterative Imputation- A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.

# ### Data imputation

# impute train data if the imputed data file does not exist
if(path.exists('APS-Failure-Analysis/train_tree_weka.csv') and path.exists('APS-Failure-Analysis/test_tree_weka.csv')):
    df_train_tree = pd.read_csv("APS-Failure-Analysis/train_tree_weka.csv")
    X_train_tree = df_train_tree.drop(columns=['class'])
    y_train_tree = df_train_tree['class']

    df_test_tree = pd.read_csv("APS-Failure-Analysis/test_tree_weka.csv")
    X_test_tree = df_test_tree.drop(columns=['class'])
    y_test_tree = df_test_tree['class']
else:
    X_train_tree = df_train_tree.drop(columns=['class'])
    y_train_tree = df_train_tree['class']
    iterative_imputer = IterativeImputer()
    X_train_tree = pd.DataFrame(iterative_imputer.fit_transform(X_train_tree), columns=X_train_tree.columns)
    print("After data imputation")
    print("Train dataset")
    print(X_train_tree)
    df_train_tree = X_train_tree.copy()
    df_train_tree['class'] = y_train_tree
    df_train_tree.to_csv(path_or_buf="APS-Failure-Analysis/train_tree_weka.csv", index=False)

# impute test data if the imputed data file does not exist
    X_test_tree = df_test_tree.drop(columns=['class'])
    y_test_tree = df_test_tree['class']
    X_test_tree = pd.DataFrame(iterative_imputer.transform(X_test_tree), columns=X_test_tree.columns, index=X_test_tree.index)
    print("After data imputation")
    print("Test dataset")
    print(X_test_tree)
    df_test_tree = X_test_tree.copy()
    df_test_tree['class'] = y_test_tree
    df_test_tree.to_csv(path_or_buf="APS-Failure-Analysis/test_tree_weka.csv", index=False)

print(X_train_tree.head())
print(X_train_tree.shape)
print("")
print(y_train_tree.head())
print(y_train_tree.shape)
print("")
print(X_test_tree.head())
print(X_test_tree.shape)
print("")
print(y_test_tree.head())
print(y_test_tree.shape)


# ### ii. For each of the 170 features, calculate the coefficient of variation CV

CoV_tree=X_train_tree.std()/X_train_tree.mean()
print("Coefficient of Variation CV for each feature is")
CoV_tree


# ### iii. Plot a correlation matrix for your features using pandas or any other tool.

corr = X_train_tree.corr()
f, ax = plt.subplots(figsize=(20, 17))
corr_mat = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, linewidths=.1)


# ### iv. Pick &lfloor;&radic;170&rfloor; features with highest CV

# Pick floor(sqrt(170)) features with highest CV
n_fea_tree = math.floor(math.sqrt(170))
n_fea_tree

CoV_highest_tree = CoV_tree.sort_values(ascending=False)[:n_fea_tree].axes[0].to_list()
CoV_highest_tree


# ### Make scatter plots

sp_df_train_tree=pd.DataFrame()
for col in CoV_highest_tree:
    sp_df_train_tree[col] = X_train_tree[col]
sp_df_train_tree['class'] = y_train_tree
sp_df_train_tree

sp_df_train_tree.columns

# function to draw scatter plots
b=0

colname = []
for i in range(len(sp_df_train_tree.columns)-1):
    colname.extend(["",sp_df_train_tree.columns[i]])

def diagfunc(x, **kws):
    global b
    ax = plt.gca()
    ax.annotate(colname[b], xy=(0.3, 0.5), xycoords=ax.transAxes)
    b=b+1

sns.set(context="paper")

g = sns.PairGrid(sp_df_train_tree,hue="class",hue_kws={"marker":["o","+"]}, height=3,palette=["#FF0000","#0C41CE"],vars=CoV_highest_tree).map_diag(diagfunc)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()

for ax in g.axes.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')


# ### Make box plots

plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=1)
for i in range(len(sp_df_train_tree.columns)-1):
    plt.subplot(2,7,i+1)
    sns.boxplot(x='class',y=sp_df_train_tree.columns[i],data=sp_df_train_tree)


# ### Can you draw conclusions about significance of those features, just by the scatter plots?

# No, significant conclusions cannot be drawn only from the scatter plots.

# ### v. Determine the number of positive and negative data. Is this data set imbalanced?

pos=0
neg=0
for x in y_train_tree:
    if(x=='pos'):
        pos=pos+1
    else:
        neg=neg+1

print("The number of positive data points in train data is",pos)
print("The number of negative data points in train data is",neg)

pos=0
neg=0
for x in y_test_tree:
    if(x=='pos'):
        pos=pos+1
    else:
        neg=neg+1

print("The number of positive data points in test data is",pos)
print("The number of negative data points in test data is",neg)


# Yes, the data is imbalanced

# ### (c) Train a random forest to classify the data set.

c_rfc_tree = RandomForestClassifier(oob_score=True, n_estimators=100)
c_rfc_tree.fit(X_train_tree, y_train_tree)
c_predictions_train_tree = c_rfc_tree.predict(X_train_tree)
c_predictions_tree = c_rfc_tree.predict(X_test_tree)
c_rfc_train_score = c_rfc_tree.score(X_train_tree, y_train_tree)
c_rfc_test_score = c_rfc_tree.score(X_test_tree, y_test_tree)


# ### Calculate the confusion matrix, ROC, AUC, and misclassification for training and test sets and report them

c_cm_train_tree = confusion_matrix(y_train_tree,c_predictions_train_tree)
c_cm_tree = confusion_matrix(y_test_tree,c_predictions_tree)
c_df_cm_tree = pd.DataFrame(c_cm_tree, index = ["Negative","Positive"],
                  columns = ["Negative","Positive"])
plt.figure(figsize = (7,7))
x = sns.heatmap(c_df_cm_tree, annot=True)

def convert_to_binary_labels(test_target):
    new_test_target=[]
    for z in test_target:
        if(z=='pos'):
            new_test_target.append(1)
        else:
            new_test_target.append(0)
    return new_test_target

# ROC and AUC

c_fpr_tree = dict()
c_tpr_tree = dict()
c_roc_auc_tree = dict()

c_fpr_tree, c_tpr_tree, d4_threshold = roc_curve(convert_to_binary_labels(y_test_tree), convert_to_binary_labels(c_predictions_tree))
c_roc_auc_tree = auc(c_fpr_tree, c_tpr_tree)

plt.plot(c_fpr_tree, c_tpr_tree,label='ROC curve (area = %0.2f)' % c_roc_auc_tree)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right")
plt.show()

miscl_tr = c_cm_train_tree[0][1]+c_cm_train_tree[1][0]
miscl_te = c_cm_tree[0][1]+c_cm_tree[1][0]
print("Misclassifications for training set are", miscl_tr)
print("Misclassifications for testing set are", miscl_te)


# ### Calculate Out of Bag error estimate for your random forset and compare it to the test error.

oob_score = c_rfc_tree.oob_score_
oob_error = 1-oob_score
print("Out of Bag error estimate for random forest classifier is", oob_error)
test_error_rfc = 1-c_rfc_tree.score(X_test_tree,y_test_tree)
print("Test error is", test_error_rfc)


# ### (d) Compensate for class imbalance in your random forest and repeat 2c.

d_rfc_tree = RandomForestClassifier(oob_score=True, class_weight="balanced",n_estimators=100)
d_rfc_tree.fit(X_train_tree, y_train_tree)
d_predictions_train_tree=d_rfc_tree.predict(X_train_tree)
d_predictions_tree=d_rfc_tree.predict(X_test_tree)
d_rfc_train_score = d_rfc_tree.score(X_train_tree, y_train_tree)
d_rfc_test_score = d_rfc_tree.score(X_test_tree, y_test_tree)


# ### Confusion Matrix

d_cm_train_tree = confusion_matrix(y_train_tree,d_predictions_train_tree)
d_cm_tree = confusion_matrix(y_test_tree,d_predictions_tree)
d_df_cm_tree = pd.DataFrame(d_cm_tree, index = ["Negative","Positive"],
                  columns = ["Negative","Positive"])
plt.figure(figsize = (9,7))
x = sns.heatmap(d_df_cm_tree, annot=True)


# ### ROC and AUC

# ROC and AUC

d_fpr_tree = dict()
d_tpr_tree = dict()
d_roc_auc_tree = dict()

d_fpr_tree, d_tpr_tree, d4_threshold = roc_curve(convert_to_binary_labels(y_test_tree), convert_to_binary_labels(d_predictions_tree))
d_roc_auc_tree = auc(d_fpr_tree, d_tpr_tree)

plt.plot(d_fpr_tree, d_tpr_tree,label='ROC curve (area = %0.2f)' % d_roc_auc_tree)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right")
plt.show()

d_miscl_tr = d_cm_train_tree[0][1]+d_cm_train_tree[1][0]
d_miscl_te = d_cm_tree[0][1]+d_cm_tree[1][0]
print("Misclassifications for training set are", d_miscl_tr)
print("Misclassifications for testing set are", d_miscl_te)

d_oob_score = d_rfc_tree.oob_score_
d_oob_error = 1-d_oob_score
print("Out of Bag error estimate for random forest classifier is", d_oob_error)
d_test_error_rfc = 1-d_rfc_tree.score(X_test_tree,y_test_tree)
print("Test error is", d_test_error_rfc)


# ### Compare the results with those of 2c.

tbl = []
tr_score = []
te_score = []
oobscore = []

method = ['Random Forest Classifier','RFC with class imbalance compensation']
tbl.append(method)

tr_score.append(c_rfc_train_score)
tr_score.append(d_rfc_train_score)
tbl.append(tr_score)

te_score.append(c_rfc_test_score)
te_score.append(d_rfc_test_score)
tbl.append(te_score)

oobscore.append(oob_score)
oobscore.append(d_oob_score)
tbl.append(oobscore)
tbl
tbl=list(map(list,zip(*tbl)))
tbl=pd.DataFrame(tbl, columns=['Method', 'Train Score', 'Test Score', 'OOB Score'])
tbl


# ### (e) Model Trees

jvm.start(packages=True, max_heap_size="512m")

train_weka = converters.load_any_file("APS-Failure-Analysis/train_tree_weka.csv")
train_weka.class_is_last()

test_weka = converters.load_any_file("APS-Failure-Analysis/test_tree_weka.csv")
test_weka.class_is_last()

if(path.exists("APS-Failure-Analysis/lmt.model")):
    cls = Classifier(jobject=serialization.read("APS-Failure-Analysis/lmt.model"))
else:
    # train classifier
    cls = Classifier(classname="weka.classifiers.trees.LMT")
    cls.build_classifier(train_weka)
    # save model
    serialization.write("APS-Failure-Analysis/lmt.model", cls)

eval = Evaluation(train_weka)
eval.crossvalidate_model(cls, train_weka, 5, Random(1))

# cross validation error, summary

e_cv_train_error_lmt = eval.error_rate
e_summary_lmt = eval.summary()
e_cm_lmt = eval.confusion_matrix

print("Train error is",e_cv_train_error_lmt)
print("")
print("Summary-")
print(e_summary_lmt)
print("")
print("Confusion matrix")
print(e_cm_lmt)
print("")
print("Class details")
print(eval.class_details())

print("ROC")
plcls.plot_roc(eval, class_index=[1], wait=True)

auc_pos = eval.area_under_roc(1)
print("AUC pos=", auc_pos)

e_cm_lmt = eval.confusion_matrix
e_df_cm_test_lmt = pd.DataFrame(e_cm_lmt, index = ["Negative","Positive"], columns = ["Negative","Positive"])
plt.figure(figsize = (9,7))
x = sns.heatmap(e_df_cm_test_lmt, annot=True, square=False)

# to calculate test error, summary
eval_test_lmt = Evaluation(train_weka)
eval_test_lmt.test_model(cls, test_weka)

e_test_error_lmt = eval_test_lmt.error_rate
e_summary_test_lmt = eval_test_lmt.summary()
e_cm_test_lmt = eval_test_lmt.confusion_matrix

print("Test error is", e_test_error_lmt)
print("")
print("Summary-")
print(e_summary_test_lmt)
print("")
print("Confusion matrix")
print(e_cm_test_lmt)
print("")
print("Class details")
print(eval_test_lmt.class_details())

print("ROC")
plcls.plot_roc(eval_test_lmt, class_index=[1], wait=True)

auc_test_pos = eval_test_lmt.area_under_roc(1)
print("AUC pos=", auc_test_pos)

def convert_to_binary_labels(test_target):
    new_test_target=[]
    for z in test_target:
        if(z=='bending'):
            new_test_target.append(1)
        else:
            new_test_target.append(0)
    return new_test_target

e_cm_test_lmt = eval_test_lmt.confusion_matrix
e_df_cm_test_lmt = pd.DataFrame(e_cm_test_lmt, index = ["Negative","Positive"],
                  columns = ["Negative","Positive"])
plt.figure(figsize = (9,7))
x = sns.heatmap(e_df_cm_test_lmt, annot=True)


# ### (f) SMOTE

packages.install_package('SMOTE')

smote = Filter(classname="weka.filters.supervised.instance.SMOTE", options=["-P", "4000"])
f_smote_lmt = Classifier(classname="weka.classifiers.trees.LMT")

fc = FilteredClassifier()
fc.filter = smote
fc.classifier = f_smote_lmt

eval_smote = Evaluation(train_weka)
eval_smote.crossvalidate_model(fc, train_weka, 5, Random(1))

f_cv_train_error_smote = eval_smote.error_rate
f_summary_smote = eval_smote.summary()
f_cm_smote = eval_smote.confusion_matrix

print("Train error is",f_cv_train_error_smote)
print("")
print("Summary-")
print(f_summary_smote)
print("")
print("Confusion matrix")
print(f_cm_smote)
print("")
print("Class details")
print(eval_smote.class_details())

print("ROC")
plcls.plot_roc(eval_smote, class_index=[1], wait=True)

auc_pos_smote = eval_smote.area_under_roc(1)
print("AUC pos=", auc_pos_smote)

fc.build_classifier(train_weka)

# to calculate test error, summary
eval_test_smote = Evaluation(train_weka)
eval_test_smote.test_model(fc, test_weka)
# serialization.write("e_eval_test_smote.model", eval_test_smote)

e_test_error_smote = eval_test_smote.error_rate
e_summary_test_smote = eval_test_smote.summary()
e_cm_test_smote = eval_test_smote.confusion_matrix

print("Test error is", e_test_error_smote)
print("")
print("Summary-")
print(e_summary_test_smote)
print("")
print("Confusion matrix")
print(e_cm_test_smote)
print("")
print("Class details")
print(eval_test_smote.class_details())

print("ROC")
plcls.plot_roc(eval_test_smote, class_index=[1], wait=True)

auc_test_pos_smote = eval_test_smote.area_under_roc(1)
print("AUC pos=", auc_test_pos_smote)


# ### Compare the uncompensated case with SMOTE.

tbl2 = []
tr_score = []
te_score = []

method = ['LMT(imbalanced dataset)','LMT(SMOTE)']
tbl2.append(method)

tr_score.append(e_cv_train_error_lmt)
tr_score.append(f_cv_train_error_smote)
tbl2.append(tr_score)

te_score.append(e_test_error_lmt)
te_score.append(e_test_error_smote)
tbl2.append(te_score)

tbl2=list(map(list,zip(*tbl2)))
tbl2=pd.DataFrame(tbl2, columns=['Method', 'Train Score', 'Test Score'])
tbl2

jvm.stop()
