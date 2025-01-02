#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_music = pd.read_csv("music.csv")

data_music = data_music[data_music['genre'].isin(['hiphop', 'jazz', 'classical'])]

X = data_music[['tempo','acousticness', 'danceability','liveness', 'speechiness', 'instrumentalness', 'mode', 'key', 'energy']]
y = data_music['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Il y a %i exemples dans le 'training set' et %i dans le 'test set'"%( X_train.shape[0], X_test.shape[0]))


# In[102]:


from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=6)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
accuracy_perceptron = accuracy_score(y_test, y_pred)
print('Exactitude pour Perceptron: %.2f' % accuracy_perceptron)

print('Bonne classification: %d' % (y_test == y_pred).sum())
print('Erreurs: %d' % (y_test != y_pred).sum())


# In[103]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print('Exactitude pour Naive Bayes: %.2f' % accuracy_nb)

print('Bonne classification pour Naive Bayes: %d' % (y_test == y_pred_nb).sum())
print('Erreurs pour Naive Bayes: %d' % (y_test != y_pred_nb).sum())


# In[120]:


from sklearn.svm import SVC

svm = SVC(kernel='linear', random_state=0, gamma=.10, C=20.0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('Exactitude pour SVM: %.2f' % accuracy_svm)

print('Bonne classification pour SVM: %d' % (y_test == y_pred_svm).sum())
print('Erreurs pour SVM: %d' % (y_test != y_pred_svm).sum())


# In[108]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=17, metric='manhattan', weights='distance')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print('Exactitude pour knn: %.2f' % accuracy_knn)

print('Bonne classification pour knn: %d' % (y_test == y_pred_knn).sum())
print('Erreurs pour KNeighborsClassifier: %d' % (y_test != y_pred_knn).sum())


# In[109]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
accuracy_clf = accuracy_score(y_test, y_pred_clf)
print('Exactitude pour clf: %.2f' % accuracy_clf)

print('Bonne classification pour DecisionTreeClassifier: %d' % (y_test == y_pred_clf).sum())
print('Erreurs pour DecisionTreeClassifier: %d' % (y_test != y_pred_clf).sum()) 


# In[80]:


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
print('Exactitude pour SVM: %.2f' % accuracy_knn)


# In[110]:


from sklearn.metrics import classification_report
report = classification_report (y_test , y_pred_svm)
print(report)


# In[111]:


nom_classes = ["classical", "hiphop", "jazz"]
report = classification_report (y_test, y_pred_svm, target_names = nom_classes )
print(report)


# In[112]:


from sklearn.metrics import precision_recall_fscore_support
stats = precision_recall_fscore_support(y_test, y_pred_svm)
print(stats)


# In[113]:


from sklearn.metrics import confusion_matrix
matrice_confusion = confusion_matrix (y_test, y_pred_svm)
print(matrice_confusion)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(figsize=(5,5))
classes = nom_classes
sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=classes, yticklabels=classes, 
            annot=True, fmt ="d")


# In[117]:


from sklearn import svm
from sklearn.metrics import confusion_matrix

SVM = svm.SVC()
SVM.fit(X_train, y_train)
y_pred_svm = SVM.predict(X_test)
matrice_confusion = confusion_matrix(y_test, y_pred_svm)
print(matrice_confusion)


# In[118]:


from sklearn.svm import SVC
print("Avec la valeur par defaut de random state")
for i in range(3):
    svm = SVC()
    svm = svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_pred_svm)
    print(matrice_confusion)
    stats = precision_recall_fscore_support(y_test, y_pred_svm)
    print(stats)
    


# In[119]:


from sklearn.svm import SVC

print("En fixant random state")

for i in range(3):
    SVM = SVC(random_state=0)
    SVM = SVM.fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_pred_svm)
    print(matrice_confusion)
    stats = precision_recall_fscore_support(y_test, y_pred_svm)
    print(stats)
    print("−−"*10)


# In[ ]:




