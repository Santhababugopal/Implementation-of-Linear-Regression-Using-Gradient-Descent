# Implementation-of-Linear-Regression-Using-Gradient-Descent
NAME :   SANTHABABU   G


REGISTER NUMBER:   212224040292
## AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:


1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


Load data from CSV into a DataFrame.

2.Extract features (x) and target values (y) from the DataFrame.

3.Convert all values to float for computation compatibility.

4.Initialize two StandardScalers: one for x, one for y.

5.Standardize (normalize) both x and y to have mean 0 and std 1.

6.Define linear_regression function that:

*Adds bias term (intercept) to x

*Initializes theta (weight vector)

7.Runs num_iters iterations of gradient descent to minimize MSE loss

8.Train the model by calling linear_regression(x_scaled, y_scaled)

9.Prepare a new input sample, scale it using the same x_scaler.

10.Make prediction using theta, then inverse transform the result using y_scaler.

11.Print predicted output in original scale.

## Program:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors =(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")



```

## Output:
DATA INFORMATION:

<img width="701" height="157" alt="image" src="https://github.com/user-attachments/assets/9f53af87-d42d-4d7c-86d9-7cc3248db75c" />

THE VALUE OF X:

<img width="395" height="272" alt="image" src="https://github.com/user-attachments/assets/04df17d8-88d1-4903-ac98-79c8badd1e64" />


THE VALUE OF Y:

<img width="266" height="268" alt="image" src="https://github.com/user-attachments/assets/44a03580-2890-4e0e-b41e-7be5f20c5186" />



THE VALUE OF X_SCALED:

<img width="515" height="247" alt="image" src="https://github.com/user-attachments/assets/8d8296b9-9a6a-4a13-a56a-e100e4860a65" />


THE VALUE OF Y_SCALED:

<img width="233" height="280" alt="image" src="https://github.com/user-attachments/assets/a3a95f20-cac3-4e10-ae36-a51b26eba54c" />


PREDICTED VALUE:


<img width="374" height="46" alt="image" src="https://github.com/user-attachments/assets/288edbea-2bdc-45aa-bc75-8ff14b0c02bf" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
