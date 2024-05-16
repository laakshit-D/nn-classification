# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/Yuvan291205/nn-classification/assets/138849170/84792821-faa1-4050-9bfb-6da7dab1784c)

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
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')
customer_df.columns

customer_df.dtypes
customer_df.shape

customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis=0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
customers_1.dtypes

# Calculate the correlation matrix
corr = customers_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

# Plot scatterplot
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)

customers_1.describe()
customers_1['Segmentation'].unique()

one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape

y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

# Creating the model
model = Sequential([
    Dense(units=8,activation='relu',input_shape=[8]),
    Dense(units=16,activation='relu'),
    Dense(units=4,activation='softmax')
])

model.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(x=X_train_scaled,y=y_train,
             epochs= 2000,
             batch_size= 256,
             validation_data=(X_test_scaled,y_test),
             )

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
predi = model.predict(X_test_scaled)
predi
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

# Saving the Model
model.save('customer_classification_model.h5')

# Saving the data ,PICKLE:  Stores as binary file and then converts
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

# Loading the Model
model = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))

```

## Dataset Information:

![image](https://github.com/Rithigasri/DL_Exp02/assets/93427256/59ce1619-686c-4ae9-835f-9c77bdb146ed)

## OUTPUT:
### Training Loss, Validation Loss Vs Iteration Plot:

![image](https://github.com/Rithigasri/DL_Exp02/assets/93427256/714e5816-8daf-49df-8ee1-bff332a8b6b6)

### Classification Report:

![image](https://github.com/Rithigasri/DL_Exp02/assets/93427256/6bfd3d31-f93b-42d9-83e4-a8ec21f6a983)

### Confusion Matrix:

![image](https://github.com/Rithigasri/DL_Exp02/assets/93427256/419517ce-f97d-4f3e-af4a-977ccb0d24d4)

### New Sample Data Prediction:

![image](https://github.com/Rithigasri/DL_Exp02/assets/93427256/11617352-b34c-4762-b686-39798d90c82d)

## RESULT
Hence we have constructed a Neural Network model for Multiclass Classification.

