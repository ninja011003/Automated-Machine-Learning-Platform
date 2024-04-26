from info_gain_gen import genetic
from PreProcessor import PreProcessor
import numpy as np
import pandas as pd
import copy
from model_selection import model_seletion
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import joblib


import streamlit as st
PreProcessor = PreProcessor()

def train(data,target):
	with st.status("Training Models",expanded=True) as status:
		st.write("Fetching Models")
		model_sel = model_seletion(data,target)
		models=model_sel.run_process()
		acc_m=list(models.values())
		st.write("Selecting Best Models")
		model_df=pd.DataFrame({"models":models.keys(),"Accuracy": [acc[1] for acc in acc_m]})
		#print(model_df)
		best_model=max(models.values(),key=lambda x:x[1])[0]
		#print(best_model)
		x=data.drop(columns=target)
		y=data[target]
		X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
		st.write("Splitting data into train and test set")
		best_model.fit(X_train,Y_train)
		y_pred=best_model.predict(X_test)

		st.write("Validating Model with Data")
		#if model_sel.is_classification(data,target):
		#	accuracy=accuracy_score(Y_test,y_pred)
		#else:
		#	accuracy=r2_score(Y_test,y_pred)
		accuracy=max(model_df["Accuracy"])
		#Save Model
		st.write("Saving best model")
		joblib.dump(best_model,"./model/best_model.sav")
		status.update(label="Done!",state="complete",expanded=False)
	return accuracy,X_train,X_test,Y_train,Y_test,y_pred,model_df

def D_prepare(data,target,file_type):
	with st.status("Preparing Data",expanded=True) as status:
		org=copy.deepcopy(data)
		st.write("Preprocessing Data")
		data,y_labels= PreProcessor.clean_data(data,target)
		for i in data.columns:
			if data[i].dtype=="object":
				if (is_float(data[i][0])==True):
					data[i]=data[i].astype(float)
				elif (is_int(data[i][0])==True):
					data[i]=data[i].astype(int)
				else:
					continue
		
		st.write("Selecting Best Features")
		gen_algo = genetic(Data=data,targer_var=target)
		
		status.update(label="Performing Genetic Algorithm",state="running",expanded=True)
		population, generations = gen_algo.run_evolution(
				popluation_size=10,
				genome_length=len(data.columns)-1,
				generation_limit=20
			)
		status.update(label="Performing Final Checks",state="running",expanded=True)
		st.write("Optimizing Features")
		data = gen_algo.best_feature(population,data.columns.tolist())
		total_cols=len(org.drop(columns=target).columns)
		remain_cols=len(data.drop(columns=target).columns)
		col_diff=total_cols-remain_cols
		if file_type=="csv":
			data.to_csv("./data/Preprocessed.csv",index=False)
		else:
			data.to_parquet("./data/Preprocessed.parquet",index=False)
		status.update(label="Data Preparation Complete!",state="complete",expanded=False)
	return data,data.drop(columns=target).columns,y_labels,[total_cols,remain_cols,col_diff]

def is_float(value):
    try:
        float_value = float(value)
        if isinstance(float_value, float):
            return True
        else:
            return False
    except ValueError:
        return False
    
def is_int(value):
    try:
        int_value = int(value)
        if isinstance(int_value, int):
            return True
        else:
            return False
    except ValueError:
        return False
#train("C:/Users/andre/Downloads/mldata/Steel_industry.csv","Load_Type")

# import sklearn 
# print(sklearn.__version__)



