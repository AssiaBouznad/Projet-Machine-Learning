#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

data_smoking = pd.read_csv("smoking.csv")
X = data_smoking[['dental caries','tartar', 'oral', 'Gtp']].astype(str)
y = data_smoking['smoking']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Il y a %i exemples dans le 'training set' et %i dans le 'test set'" % (X_train.shape[0], X_test.shape[0]))
V = DictVectorizer(sparse=False)
X_train = V.fit_transform(X_train.to_dict(orient='records'))
X_test = V.transform(X_test.to_dict(orient='records'))


# In[10]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=9)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
accuracy_perceptron = accuracy_score(y_test, y_pred)
print('Exactitude pour Perceptron: %.2f' % accuracy_perceptron)

print('Bonne classification: %d' % (y_test == y_pred).sum())
print('Erreurs: %d' % (y_test != y_pred).sum())


# In[11]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print('Exactitude pour Naive Bayes: %.2f' % accuracy_nb)

print('Bonne classification pour Naive Bayes: %d' % (y_test == y_pred_nb).sum())
print('Erreurs pour Naive Bayes: %d' % (y_test != y_pred_nb).sum())


# In[27]:


from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('Exactitude pour SVM: %.2f' % accuracy_svm)

print('Bonne classification pour SVM: %d' % (y_test == y_pred_svm).sum())
print('Erreurs pour SVM: %d' % (y_test != y_pred_svm).sum()) 


# In[16]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print('Exactitude pour knn: %.2f' % accuracy_knn)

print('Bonne classification pour KNeighborsClassifier: %d' % (y_test == y_pred_knn).sum())
print('Erreurs pour KNeighborsClassifier: %d' % (y_test != y_pred_knn).sum()) 


# In[25]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
accuracy_clf = accuracy_score(y_test, y_pred_clf)
print('Exactitude pour clf: %.2f' % accuracy_clf)


print('Bonne classification pour DecisionTreeClassifier: %d' % (y_test == y_pred_clf).sum())
print('Erreurs pour DecisionTreeClassifier: %d' % (y_test != y_pred_clf).sum()) 


# In[141]:


from sklearn.metrics import accuracy_score

accuracy_perceptron = accuracy_score(y_test, y_pred)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_clf = accuracy_score(y_test, y_pred_clf)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print('Exactitude pour Perceptron: %.2f' % accuracy_perceptron)
print('Exactitude pour Naive Bayes: %.2f' % accuracy_nb)
print('Exactitude pour SVM: %.2f' % accuracy_svm)
print('Exactitude pour clf: %.2f' % accuracy_clf)
print('Exactitude pour knn: %.2f' % accuracy_knn)


# In[90]:


from sklearn.metrics import classification_report
report = classification_report (y_test , y_pred)
print(report)


# In[91]:


nom_classes = ["1","0"]
report = classification_report (y_test, y_pred, target_names = nom_classes )
print(report)


# In[92]:


from sklearn.metrics import precision_recall_fscore_support
stats = precision_recall_fscore_support(y_test, y_pred)
print(stats)


# In[93]:


from sklearn.metrics import confusion_matrix
matrice_confusion = confusion_matrix (y_test, y_pred)
print(matrice_confusion)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(figsize=(5,5))
classes = ["1", "0"]
sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=classes, yticklabels=classes, 
            annot=True, fmt ="d")


# In[94]:


from sklearn import tree
DT = tree. DecisionTreeClassifier()
DT = DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
matrice_confusion = confusion_matrix (y_test, y_pred)
print(matrice_confusion)


# In[95]:


print("Avec la valeur par defaut de random state")
for i in range(3):
    DT = tree.DecisionTreeClassifier()
    DT = DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_pred)
    print(matrice_confusion)
    stats = precision_recall_fscore_support(y_test, y_pred)
    print(stats)


# In[96]:


print("En fixant random state")

for i in range(3):
    DT = tree. DecisionTreeClassifier(random_state=0)
    DT = DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_pred)
    print(matrice_confusion)
    stats = precision_recall_fscore_support(y_test, y_pred)
    print(stats)
    print("−−"*10)


# In[97]:


y_0 = [1 for x in y_test]
report = classification_report(y_negatif, y_pred, target_names=nom_classes)
matrice_confusion = confusion_matrix (y_negatif, y_pred)
print(matrice_confusion)
print(report)
y_1 = [0 for x in y_test]
report = classification_report(y_positif, y_pred, target_names=nom_classes)
matrice_confusion = confusion_matrix(y_positif, y_pred)
print(matrice_confusion)
print(report)


# In[ ]:





# In[ ]:





# In[ ]:




