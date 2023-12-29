import numpy as np
import pandas as pd
import sklearn.svm as svm
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
df_1=pd.read_csv('input/AMP-BERT(prob).txt',header=None)
df_2=pd.read_csv('input/Bert-Protein(prob).txt',header=None)
df_3=pd.read_csv('input/cAMPs_pred(prob).txt',header=None)
df_4=pd.read_csv('input/LMPred(prob).txt',header=None)
label=pd.read_csv('True lable/label.txt',header=None)
df=pd.concat([df_1,df_2,df_3,df_4,label],axis=1,ignore_index=False)
# df.columns=['AMP-BERT','AMP-BERT','cAMPs_pred','LMPred','true lable']
df.columns=['0','1','2','3','4']
# print(df.columns)
# --------------fold1-----------------
x_train=df.iloc[768:,:4]
x_test=df.iloc[:768,:4]
y_train=df.iloc[768:,4]
y_test=df.iloc[:768,4]
print(x_train)
print(x_test)
print(y_train)
print(y_test)
# --------------fold2-----------------
# x_train=df.iloc[np.r_[:768,1534:3832],:4]
# x_test=df.iloc[768:1534,:4]
# y_train=df.iloc[np.r_[:768,1534:3832],4]
# y_test=df.iloc[768:1534,4]
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# --------------fold3-----------------
# x_train=df.iloc[np.r_[:1534,2300:3832],:4]
# x_test=df.iloc[1534:2300,:4]
# y_train=df.iloc[np.r_[:1534,2300:3832],4]
# y_test=df.iloc[1534:2300,4]
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# --------------fold4-----------------
# x_train=df.iloc[np.r_[:2300,3066:3832],:4]
# x_test=df.iloc[2300:3066,:4]
# y_train=df.iloc[np.r_[:2300,3066:3832],4]
# y_test=df.iloc[2300:3066,4]
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# --------------fold5-----------------
# x_train=df.iloc[:3066,:4]
# x_test=df.iloc[3066:,:4]
# y_train=df.iloc[:3066,4]
# y_test=df.iloc[3066:,4]
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# ------------模型构建SVM----------------
model = svm.SVC(C=0.1,kernel='sigmoid',gamma=0.1,probability=True)
model.fit(x_train,y_train)
pred = model.predict(x_test)
prob = model.predict_proba(x_test)
prob=prob[:,1]
# print(prob)
np.savetxt("SVM_prob(fold1).txt",prob, fmt='%f', delimiter=',')
# ------------模型构建Xgboost----------------
# model=XGBClassifier()
# model.fit(x_train,y_train)
# pred = model.predict(x_test)
# prob = model.predict_proba(x_test)
# prob=prob[:,1]
# np.savetxt("XGBoost_prob(fold5).txt",prob, fmt='%f', delimiter=',')
# ------------评价指标----------------
accuracy = accuracy_score(y_test, pred)
f1_score= f1_score(y_test,pred)
precision=precision_score(y_test,pred)
MCC=matthews_corrcoef(y_test,pred)
confusion_matrix=confusion_matrix(y_test,pred)
auc=roc_auc_score(y_test,prob)
TP=confusion_matrix[0][0]
FN=confusion_matrix[0][1]
TN=confusion_matrix[1][1]
FP=confusion_matrix[1][0]
Sn=TP/(TP+FN)
Sp=TN/(TN+FP)
print(f'Test Sensitivity: {Sn:.4f}')
print(f'Test Specificity: {Sp:.4f}')
print(f'Test precision: {precision:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test f1_score: {f1_score:.4f}')
print(f'Test matthews_corrcoef: {MCC:.4f}')
print(f'Test AUC: {auc:.4f}')




