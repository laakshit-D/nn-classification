# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/Harishspice/nn-classification/assets/117935868/f6400e0b-e39f-4d8d-9570-e79a9510982f)

## DESIGN STEPS

### STEP 1:
We start by reading the dataset using pandas.


### STEP 2:
The dataset is then preprocessed, i.e, we remove the features that don't contribute towards the result.


### STEP 3:
The null values are removed aswell

### STEP 4:
The resulting data values are then encoded. We, ensure that all the features are of the type int, or float, for the model to better process the dataset.

### STEP 5:
The Sequential model is then build using 4 dense layers(hidden) and, 1 input and output layer.

### STEP 6:
Once the model is done training, we validate and use the model to predict values.



## PROGRAM

### Name: LAAKSHIT D
### Register Number:212222230071

```python
import pandas as pd
data = pd.read_csv("customers.csv")
data.head()
```

```python
data_cleaned=data.drop(columns=["ID","Var_1"])
data_col=list(data_cleaned.columns)
print("The shape of the data before removing null values is\nRow:"+str(data_cleaned.shape[0])+"\nColumns:"+str(data_cleaned.shape[1]))
```

```python
data_col_obj=list()
for c in data_col:
  if data_cleaned[c].dtype=='O':
      data_col_obj.append(c)
data_col_obj.remove("Segmentation")
print("The Columns/Features that have Objects(dataType) before encoding are:\n")
print(data_col_obj)

from sklearn.preprocessing import OrdinalEncoder
data_cleaned[data_col_obj]=OrdinalEncoder().fit_transform(data_cleaned[data_col_obj])
from sklearn.preprocessing import MinMaxScaler
data_cleaned[["Age"]]=MinMaxScaler().fit_transform(data_cleaned[["Age"]])
data_cleaned.head()

from sklearn.preprocessing import OneHotEncoder
y=data_cleaned[["Segmentation"]].values
y=OneHotEncoder().fit_transform(y).toarray()
pd.DataFrame(y)
```
```python
X=data_cleaned.iloc[:,:-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential([
    Dense(64,input_shape=X_train.iloc[0].shape,activation="relu"),
    Dense(32,activation='tanh'),
    Dense(16,activation='relu'),
    Dense(8,activation='tanh'),
    Dense(4,activation='softmax'),
])

model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
model.fit(x=X_train,y=y_train,
          epochs=400,
          validation_data=(X_test,y_test),
          verbose=0, 
          callbacks=[early_stop]
          )
```
```python
metrics = pd.DataFrame(model.history.history)
metrics.iloc[metrics.shape[0]-1,:]4
```
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(metrics[['accuracy','val_accuracy']])
plt.legend(["Training Accuracy","Validation Accuracy"])
plt.title("Accuracy vs Test Accuracy")
plt.subplot(1,2,2)
plt.plot(metrics[['loss','val_loss']])
plt.legend(["Training Loss","Validation Loss"])
plt.title("Loss vs Test Loss")
plt.show()
```
```python
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
predictions=np.argmax(model.predict(X_test),axis=1)
y_test=np.argmax(y_test, axis=1)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
import seaborn as sn
sn.heatmap(confusion_matrix(y_test,predictions))
plt.show()
```

## Dataset Information

![image](https://github.com/Harishspice/nn-classification/assets/117935868/831a0fb2-ec16-4a35-8d26-848ee5c0b326)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Harishspice/nn-classification/assets/117935868/05e078dd-5b6b-4992-8ddb-234df6c50de0)
![image](https://github.com/Harishspice/nn-classification/assets/117935868/b43f36f3-59a7-455b-9f30-9285f7144325)


### Classification Report

![image](https://github.com/Harishspice/nn-classification/assets/117935868/bbcd250b-241b-4405-81b1-b332224badb7)

### Confusion Matrix

![image](https://github.com/Harishspice/nn-classification/assets/117935868/c48d3ffa-9c27-4ff8-a989-2c4e077019cf)


### New Sample Data Prediction

![image](https://github.com/Harishspice/nn-classification/assets/117935868/02d7ae97-54f5-47ee-9232-8d6ae5ee9475)

## RESULT
Hence we have constructed a Neural Network model for Multiclass Classification.

