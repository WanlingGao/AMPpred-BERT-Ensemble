import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(8,6), dpi=100)
file=['predicte_prob/AMP-BERT(prob).txt',
      'predicte_prob/Bert-Protein(prob).txt',
      'predicte_prob/cAMPs_pred(prob).txt',
      'predicte_prob/LMPred(prob).txt',
      'predicte_prob/Ensemble-SVM(prob).txt',
      'predicte_prob/Ensemble-XGBoost(prob).txt'
      ]
label=pd.read_csv('true lable/label.txt',header=None)
for i in range(len(file)):
    data=pd.read_csv(file[i],header=None)
    fpr, tpr, thresholds = roc_curve(label,data)
    # precision, recall, _  = precision_recall_curve(label, data)
    roc_auc = auc(fpr, tpr)
    # aupr=auc(recall, precision)
    plt.plot(fpr, tpr, label='{} (AUC= %0.3f)'.format(file[i][14:-10])% roc_auc)
    # plt.plot(recall, precision, label='{} (AUPR= %0.3f)'.format(file[i][14:-10])% aupr)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.rc('font',family='Times New Roman')
plt.xlabel('False Positive Rate',size=16)
plt.ylabel('True Positive Rate',size=16)
# plt.xlabel('Recall',size=16)
# plt.ylabel('Precision',size=16)
plt.xticks(size=16)
plt.yticks(size=16)
# plt.title('Receiver Operating Characteristic Curve\nModels Tested on Independent Dataset', fontweight='bold')
plt.legend(loc="lower right",fontsize=16)
plt.savefig('ROC_Curves.png', bbox_inches='tight')
# plt.savefig('PR_Curves.png', bbox_inches='tight')
plt.show()