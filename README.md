# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2.Analyse the data. 
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R.Bharathi Shankar
RegisterNumber: 212225230032 
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
<img width="983" height="625" alt="Screenshot 2026-05-27 134937" src="https://github.com/user-attachments/assets/eb61514c-9084-4f7e-bf89-da68b841bd1c" />
<img width="578" height="299" alt="Screenshot 2026-05-27 134956" src="https://github.com/user-attachments/assets/7f90488f-961f-4077-b019-00d4d14c0925" />
<img width="864" height="330" alt="Screenshot 2026-05-27 135011" src="https://github.com/user-attachments/assets/b836ccc6-e37e-45b4-9373-50bbbec2cd19" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
