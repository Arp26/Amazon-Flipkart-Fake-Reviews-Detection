import FunctionsSVM
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import pandas as pd


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()


data = FunctionsSVM.clean_data("Combined.csv")
vp_pred_arrays,rating_pred_arrays,xtr_r,xte_r,xtr_vp,xte_vp,xtr,xte,ytr,yte,text_pred_arrays = FunctionsSVM.getEmbeddings_text(data)
xtr_t,xte_t,title_pred_arrays = FunctionsSVM.getEmbeddings_title(data)

# rating_pred_arrays=np.array(data.loc[len(xtr):,"RATING"].values)/5
# vp_pred_arrays=np.array(data.loc[:len(xtr)-1,"VERIFIED_PURCHASE"].values).astype(dtype="int32")

# xtr_r=np.array(data.loc[:len(xtr)-1,"RATING"].values)/5
# xte_r=np.array(data.loc[len(xtr):,"RATING"].values)/5
# xtr_vp=np.array(data.loc[:len(xtr)-1,"VERIFIED_PURCHASE"].values).astype(dtype="int32")
# xte_vp=np.array(data.loc[len(xtr):,"VERIFIED_PURCHASE"].values).astype(dtype="int32")


xtr=np.c_[xtr,xtr_t,xtr_r,xtr_vp]
xte=np.c_[xte,xte_t,xte_r,xte_vp]


xpred=np.c_[text_pred_arrays,title_pred_arrays,rating_pred_arrays,vp_pred_arrays]

clf = SVC()
clf.fit(xtr, ytr)
y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")

plot_cmat(yte, y_pred)




y_final = clf.predict(xpred)

data_pred = pd.read_csv("Data_Pred.csv")

import csv
rowi=0
with open("Data_Pred.csv", "r" ) as cfile:
  reader = csv.reader(cfile)
  with open("Data_Pred_final_SVM.csv", "w") as cwrite:
    writer = csv.writer(cwrite)
    for i in reader:
      try:
        writer.writerow([data_pred.loc[rowi, 'REVIEW_TEXT'],data_pred.loc[rowi, 'REVIEW_TITLE'],data_pred.loc[rowi, 'RATING'],data_pred.loc[rowi, 'VERIFIED_PURCHASE'],y_final[rowi]])
        rowi=rowi+1
        print(rowi)
      except:
        print("Done")

