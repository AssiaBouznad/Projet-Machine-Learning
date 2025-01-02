#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_alcohol = pd.read_csv("Maths.csv")
X = data_alcohol[['Dalc', 'Walc']]
y = data_alcohol['G3']

def classify_notes(note):
    if note >= 10:
        return 'positif'
    else:
        return 'negatif'

data_alcohol['G3_category'] = data_alcohol['G3'].apply(classify_notes)
y = data_alcohol['G3_category']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
print("Il y a %i exemples dans le 'training set' et %i dans le 'test set'"%( X_train.shape[0], X_test.shape[0]))


# In[30]:


from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
accuracy_perceptron = accuracy_score(y_test, y_pred)
print('Exactitude pour Perceptron: %.2f' % accuracy_perceptron)

print('Bonne classification: %d' % (y_test == y_pred).sum())
print('Erreurs: %d' % (y_test != y_pred).sum())


# In[31]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print('Exactitude pour Naive Bayes: %.2f' % accuracy_nb)

print('Bonne classification pour Naive Bayes: %d' % (y_test == y_pred_nb).sum())
print('Erreurs pour Naive Bayes: %d' % (y_test != y_pred_nb).sum())


# In[38]:


from sklearn.svm import SVC

svm = SVC(kernel='poly', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('Exactitude pour SVM: %.2f' % accuracy_svm)

print('Bonne classification pour SVM: %d' % (y_test == y_pred_svm).sum())
print('Erreurs pour SVM: %d' % (y_test != y_pred_svm).sum())


# In[32]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print('Exactitude pour knn: %.2f' % accuracy_knn)

print('Bonne classification pour knn: %d' % (y_test == y_pred_knn).sum())
print('Erreurs pour KNeighborsClassifier: %d' % (y_test != y_pred_knn).sum()) 


# In[34]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
accuracy_clf = accuracy_score(y_test, y_pred_clf)
print('Exactitude pour clf: %.2f' % accuracy_clf)

print('Bonne classification pour DecisionTreeClassifier: %d' % (y_test == y_pred_clf).sum())
print('Erreurs pour DecisionTreeClassifier: %d' % (y_test != y_pred_clf).sum()) 


# # Exactitude : 

# In[39]:


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


# In[13]:


from sklearn.metrics import classification_report
report = classification_report (y_test, y_pred_knn)
print(report)


# In[16]:


nom_classes = ["negatif", "positif"]
report = classification_report (y_test, y_pred_knn, target_names = nom_classes )
print(report)


# In[17]:


from sklearn.metrics import precision_recall_fscore_support
stats = precision_recall_fscore_support(y_test, y_pred_knn)
print(stats)


# In[18]:


from sklearn.metrics import confusion_matrix
matrice_confusion = confusion_matrix (y_test, y_pred_knn)
print(matrice_confusion)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(figsize=(5,5))
classes = ["negatif", "positif"]
sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=classes, yticklabels=classes, 
            annot=True, fmt ="d")


# In[19]:


from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
y_pred_knn = KNN.predict(X_test)
matrice_confusion = confusion_matrix(y_test, y_pred_knn)
print(matrice_confusion)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
print("Avec la valeur par defaut de random state")
for i in range(3):
    KNN = KNeighborsClassifier()
    KNN = KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_pred_knn)
    print(matrice_confusion)
    stats = precision_recall_fscore_support(y_test, y_pred_knn)
    print(stats)
    


# In[36]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
print("En fixant random state")

for i in range(3):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_pred_knn)
    print(matrice_confusion)
    stats = precision_recall_fscore_support(y_test, y_pred_knn)
    print(stats)
    print("−−"*10)


# In[ ]:




