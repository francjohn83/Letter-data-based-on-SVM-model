
# coding: utf-8

# In[127]:


#libraries to import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[128]:


#read the data

letter = pd.read_csv(r'C:\Users\francis\Downloads\letterdata.csv')


# In[99]:


letter.info(50)


# In[100]:


letter.describe()


# In[12]:


# density 
letter.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1) 
plt.show()


# In[131]:


#validating the dataset

letter.info()


# In[122]:


validation_size = 0.20 
seed = 7
X_train, X_validation, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=seed)


# In[113]:


# Test options and evaluation metric 
num_folds = 10 
seed = 7
scoring = 'accuracy'


# In[114]:


# Spot-Check Algorithms 
models = []
models.append(('LR', LogisticRegression(solver='liblinear'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier())) 
models.append(('SVM', SVC(gamma='auto')))


# In[132]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
results = [] 
names = [] 
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)


# In[134]:


# Compare Algorithms
fig = plt.figure() 
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names) 
plt.show()


# In[57]:


# We can see that CART and LR is still doing well, even better than before. 
#We can also see that the standardization of the data has lifted the skill of SVM to be the most accurate algorithm tested so far.


# In[58]:


#algorithm Tuning
#We can tune two key parameters of the SVM algorithm, the value of C (how much to relax the margin) and the type of kernel. The default for SVM (the SVC class) is to use the Radial Basis Function (RBF) kernel with a C value set to 1.0. Like with KNN, we will perform a grid search using 10-fold cross-validation with a standardized copy of the training dataset. 
#We will try a number of simpler kernel types and C values with less bias and more bias (less than and more than 1.0 respectively).


# In[139]:


#tune scaled SVM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train) 
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0] 
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid'] 
param_grid = dict(C=c_values, kernel=kernel_values) 
model = SVC(gamma='auto')
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, iid=True) 
grid_result = grid.fit(rescaledX, y_train) 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#finalize the model

#The SVM showed the most promise as a low complexity and stable model for this problem. 
#In this section we will ﬁnalize the model by training it on the entire training dataset and make predictions for the hold-out validation dataset to conﬁrm our ﬁndings. A part of the ﬁndings was that SVM performs better when the dataset is standardized so that all attributes have a mean value of zero and a standard deviation of on


# In[142]:


#prepare the model
from sklearn.metrics import accuracy_score
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
model = SVC(C=1.5) 
model.fit(rescaledX, y_train) 
# estimate accuracy on validation dataset 
rescaledValidationX = scaler.transform(X_validation) 
predictions = model.predict(rescaledValidationX) 
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions)) 
print(classification_report(y_test, predictions))

