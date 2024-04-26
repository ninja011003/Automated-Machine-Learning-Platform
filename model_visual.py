import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer,StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
import streamlit as st
import joblib


def ConfusionMatrixDisp(y_test,y_predict):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test,y_predict,ax=ax,cmap="Blues")
    #plt.savefig("./images/confusion_mat.png",bbox_inches='tight')
    st.pyplot(fig)

def ROC(y_train,y_test,y_predict_prob):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(0,len(np.unique(y_train))):
        CoI=np.unique(y_train)[i]
        RocCurveDisplay.from_predictions(y_onehot_test[:,i-1],y_predict_prob[:,i-1],name=f"{CoI} vs the rest",ax=ax)
    ax.plot([0,1],[0,1],"k--",label="Chance Level (AUC = 0.50)",figure=fig)
    ax.legend()
    #plt.savefig("./images/roc.png",bbox_inches='tight')
    st.pyplot(fig)

def ErrorPlot(x_train,y_train,x_test,y_test):
    alphas = np.logspace(-5, 1, 100)
    model=joblib.load("./model/best_model.sav")
    train_errors = list()
    test_errors = list()
    for alpha in alphas:
        train_errors.append(model.score(x_train, y_train))
        test_errors.append(model.score(x_test, y_test))

    q_grid = np.linspace(start=0.0, stop=1.0, num=100)
    train_error_quantiles = np.quantile(a=train_errors, q=q_grid)
    test_error_quantiles = np.quantile(a=test_errors, q=q_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=q_grid, y=train_error_quantiles, label='train', ax=ax)
    sns.lineplot(x=q_grid, y=test_error_quantiles, label='test', ax=ax)
    ax.legend(loc='upper left')
    ax.set(title='Quantile Plot (Errors)', xlabel='q', ylabel='quantile')
    st.pyplot(fig)

def Reg_Mod_Eval(x_test,y_test,y_pred):
    y_test=y_test.values
    ss=StandardScaler()
    y_test=ss.fit_transform(y_test.reshape(-1, 1))
    y_pred=ss.fit_transform(y_pred.reshape(-1, 1))
    residuals = np.array(y_test) - np.array(y_pred)

    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    standardized_residuals = (residuals - residuals_mean) / residuals_std
    
    # Plotting Residuals vs. Predicted Values
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(y_pred, standardized_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')  # Add a horizontal line at y=0
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    #plt.savefig("./images/resid.png",bbox_inches='tight')
    st.pyplot(fig)