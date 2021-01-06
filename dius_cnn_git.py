#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:01:17 2021

@author: ross
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


############################################
# EDA
############################################
print('Loading data and performing EDA')
df = pd.read_csv('./test_1/train.csv')
df_sorted = df.sort_values(['Y'], ascending=False)
df_sorted = df_sorted.reset_index(drop=True)
fig, ax = plt.subplots()
ax.imshow(df, extent=[-10,10,-10,10])

fig, ax = plt.subplots()
ax.imshow(df_sorted, extent=[-10,10,-10,10])

fig, ax = plt.subplots()
t = df_sorted.index.values
ax.scatter(t,df_sorted['Y'], s=1, color='b')


############################################
# baseline model
############################################
y = df_sorted['Y']
X = df_sorted[df_sorted.columns.values[:-1]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training baseline Random Forrest model')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_score = rf.score(X_test,y_test)

print(rf.feature_importances_)
print("MAE: %.4f" % mean_absolute_error(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict


def plot_regression_results(y_true, y_pred,  scores):
    """Scatter plot of the predicted vs true targets."""
    fig, ax = plt.subplots()
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')


score = cross_validate(rf, X_test, y_test,
                       scoring=['r2', 'neg_mean_absolute_error'],
                       n_jobs=-1, verbose=0)

y_pred = cross_val_predict(rf, X_test, y_test, n_jobs=-1, verbose=0)

plot_regression_results(
    y_test, y_pred,
    (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
    .format(np.mean(score['test_r2']),
            np.std(score['test_r2']),
            -np.mean(score['test_neg_mean_absolute_error']),
            np.std(score['test_neg_mean_absolute_error'])))
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


print('Loading tensorflow')
############################################
# initial cnn model
############################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print('Deleting old variables')
del X, y, X_train, X_test, y_train, y_test, y_pred

print('Loading and scaling data')
df = pd.read_csv('./test_1/train.csv')
data = np.asarray(df[df.columns.values])
trans = MinMaxScaler()
data = trans.fit_transform(data)

y = data[:,-1]
x = data[:,:-1]
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15, random_state=42) 

model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(10, 1)))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
 
model.summary()

print('Training cnn')
model.fit(xtrain, ytrain, batch_size=32,epochs=50, verbose=0)
ypred = model.predict(xtest)
print(model.evaluate(xtrain, ytrain))
print("MSE: %.4f" % mean_squared_error(ytest, ypred))
print("MAE: %.4f" % mean_absolute_error(ytest, ypred))

ytest = np.array([ytest]).T
r = np.hstack((ytest,ypred))
result = r[np.argsort(-r[:, 0])]

fig, ax = plt.subplots()
x_ax = range(result.shape[0])
ax.scatter(x_ax, result[:,0], s=3, color="blue", label="original")
ax.scatter(x_ax, result[:,1], s=1, color="red", label="predicted")
ax.legend()



############################################
# ground truth data test
############################################
df_test_gt = pd.read_csv('./test_1/test_gt.csv')

data2 = trans.fit_transform(df_test_gt)
ygt = data2[:,-1]
xgt = data2[:,:-1]
xgt = xgt.reshape(xgt.shape[0], xgt.shape[1], 1)
print(xgt.shape)
ypred_gt = model.predict(xgt)

print('Ground truth data evaluation')
print(model.evaluate(xgt, ygt))
print("MSE: %.4f" % mean_squared_error(ygt, ypred_gt))
print("MAE: %.4f" % mean_absolute_error(ygt, ypred_gt))

ygt = np.array([ygt]).T
np.hstack((ygt,ypred_gt))
df2 = pd.DataFrame(np.hstack((ygt,ypred_gt)), columns=['ygt', 'ypred_gt'])

# Do the sorting with pandas, just for demonstration
df_sorted = df2.sort_values(['ygt'], ascending=False)
df_sorted = df_sorted.reset_index(drop=True)

fig, ax = plt.subplots()
t = df_sorted.index.values
ax.scatter(t,df_sorted['ygt'], s=1, color='b', label="original")
ax.scatter( t, df_sorted['ypred_gt'], s=1, color='r', label="predicted")
ax.legend()


############################################
# model applied to test data set
############################################
df_test = pd.read_csv('./test_1/test.csv')

data3 = trans.fit_transform(df_test)
xt = data3
xt = xt.reshape(xt.shape[0], xt.shape[1], 1)
print(xt.shape)
ypred_t = model.predict(xt)

print('Writing predictions for test.csv')
np.savetxt("./test_1/test_pred.csv", ypred_t, delimiter=",")

