# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

    1.Import the necessary packages.
    2.Read the given csv file and display the few contents of the data.
    3.Assign the features for x and y respectively.
    4.Split the x and y sets into train and test sets.
    5.Convert the Alphabetical data to numeric using CountVectorizer.
    6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
    7.Find the accuracy of the model.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Praveen S
RegisterNumber:  212222240077
*/
import pandas as pd
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
data.info()
data.isnull()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Result  
![Screenshot 2023-06-02 141321](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/e57d5bb2-c420-462d-926f-8b9f7adfd6ee)

### Data Head
![Screenshot 2023-06-02 141327](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/a372fb17-427d-403c-bb54-91862bdc81c9)

### Data Info



![Screenshot 2023-06-02 141332](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/8320c603-d5d8-4186-b28b-a008e18a7347)

### Data Isnull


![Screenshot 2023-06-02 141337](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/1d98f101-7308-453c-8d0e-91e5fcb99536)

### Data Isnull Sum
![Screenshot 2023-06-02 141342](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/82b4c0dd-6392-4487-9eae-7fb604a72b20)


### Y_pred


![Screenshot 2023-06-02 141349](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/2097250d-8eac-4405-88f0-2d88f97d979c)

### accuracy
![Screenshot 2023-06-02 141353](https://github.com/praveenst13/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118787793/7e5e2a5b-e53b-4765-a993-f27bf48eeeed)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
