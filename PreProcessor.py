from timeit import default_timer as timer
import numpy as np
import pandas as pd
from math import isnan
from sklearn.utils.multiclass import type_of_target
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from streamlit import progress
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dateutil.parser import parse
from loguru import logger
import copy
import warnings
warnings.filterwarnings('ignore')
class PreProcessor:
    def __init__(self):
        self.unique_count = 1
        
    def RemoveIrrelevantColumn(self,df):
        #regex expressions for various formats of dates
        regex_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
        r'\b\d{4}/\d{2}/\d{2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b',
        r'\b\d{10,13}\b',
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        r'\d+\s+([a-zA-Z]+|[a-zA-Z]+\s[a-zA-Z]+)\s+\w+,\s[A-Za-z\s]+,\s[A-Za-z]{2}\s\d{5}',  # Full Address
        # Phone Number Patterns
        # r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US Phone Number
        #r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International Phone Number
        #r'\+\d{2}\s\d{5}\s\d{5}'  # Indian Phone Number
        ]
        # count=0
        # for column in df.columns:
        #     if df[column].nunique(dropna=True)==self.unique_count:
        #         print(f'Column :{column} is removed')
        #         count=count+1
        #         df=df.drop(column,axis=1)
        # print(f'{count} irrelavant columns found.')
        columns_to_remove = []    
        for column in df.columns:
            # Check if any value in the column matches any regex pattern
            if any(df[column].astype(str).str.match(pattern).any() for pattern in regex_patterns):
                columns_to_remove.append(column)
        print("Columns Removed :",columns_to_remove)
        df = df.drop(columns=columns_to_remove)
        
        return df

        

    def HandlingMissingData(self,df,num_strategy='most_frequent',cat_strategy='knn',n_neighbors=3,null_threshold=0.1):
        
        
        for column in df.columns:
            null_val=df[column].isnull().mean()
            if(null_val!=0 and null_val<=null_threshold):
                print(f'{null_val}% NaN values found on column: {column}')
                df=df.dropna(subset=[column])
                df= df.reset_index(drop=True)
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = [column for column in df.columns if column not in num_cols]
        imputer=None
        if(num_strategy=='knn'):
            imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            if num_strategy in ['mean','median','mode','most_frequent']:
                imputer = SimpleImputer(strategy=num_strategy)
            else:
                print('Invalid imputer strategy specified :{}\nDefault strategy Mean is applied',num_strategy)
                imputer = SimpleImputer(strategy='mean')
        # print('imputation process started...')
        for feature in num_cols:
            if df[feature].isna().sum().sum() != 0:
                try:
                    df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
                    if (df[feature].fillna(-9999) % 1  == 0).all():
                        df[feature] = df_imputed
                        # round back to INTs, if original df were INTs
                        df[feature] = df[feature].round()
                        df[feature] = df[feature].astype('Int64')                                        
                    else:
                        df[feature] = df_imputed
                except:
                    print('imputation failed for feature "{}"',feature)
        if(cat_strategy=='knn'):
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif(cat_strategy=='logreq'):
            df = PreProcessor.LogisticRegressionImputer(
                df=df
            )
            return df
        else:
            imputer = SimpleImputer(strategy='most_frequent')
        
        for feature in cat_cols:
            if df[feature].isna().sum()!= 0:
                try:
                    mapping = dict()
                    mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                    mapping[feature] = mappings
                    df[feature] = df[feature].map(mapping[feature])

                    df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])    

                    # round to integers before mapping back to original values
                    df[feature] = df_imputed
                    df[feature] = df[feature].round()
                    df[feature] = df[feature].astype('Int64')  

                    # map values back to original
                    mappings_inv = {v: k for k, v in mapping[feature].items()}
                    df[feature] = df[feature].map(mappings_inv)
                except:
                    print('Imputation failed for feature "{}"',  feature)
        return df
    
    
    def normalization(self,df):
        sc = StandardScaler()
        normalize_columns = []
        for column in df.columns:
            if (df[column].dtype == 'int64' or df[column].dtype == 'float64') and df[column].nunique() > 10:
                normalize_columns.append(column)
        df[normalize_columns] = sc.fit_transform(df[normalize_columns])
        # Is normalization Done well
        normalized_feature = df[normalize_columns]
        mean_normalized = np.mean(normalized_feature)
        std_dev_normalized = np.std(normalized_feature)
        print("Mean of normalized feature:", mean_normalized)
        print("Standard deviation of normalized feature:", std_dev_normalized)
        return df
    

    def encoding(self,df,target):
        org=copy.deepcopy(df)
        lable_encoder = preprocessing.LabelEncoder()
        print(df.columns)
        for column in df.columns:
                if df[column].dtype == 'object' and df[column].nunique()<3: #binary 
                    df[column]=lable_encoder.fit_transform(df[column])
                elif df[column].dtype == 'object' : #Multi-class 
                    df[column]=lable_encoder.fit_transform(df[column])
                elif df[column].dtype == 'bool':
                    df[column]=lable_encoder.fit_transform(df[column])
                elif df[column].dtype == 'str':
                    df[column]=lable_encoder.fit_transform(df[column])
                else:
                    pass
        
        if org[target].dtype=="object":
            y_labels=lable_encoder.inverse_transform(df[target])
        else:
            y_labels=None
        return df,y_labels
    
    
    
    def LogisticRegressionImputer(self,df):
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])} #.dropna().unique(), 0)}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])

        target_cols = [x for x in df.columns if x not in cols_num]
            
        for feature in df.columns: 
            if feature in target_cols:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), LogisticRegression())

                        y = train_df[feature]
                        train_df.drop(feature, axis=1, inplace=True)
                        test_df.drop(feature, axis=1, inplace=True)

                        model = pipe.fit(train_df, y)
                        
                        pred = model.predict(test_df) # predict values
                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original df were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])                             
                        logger.debug('LOGREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LOGREG imputation failed for feature "{}"', feature)
        for feature in df.columns: 
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass     
        return df
    def clean_data(self,df,target):
        bar=progress(0,text="Starting Data Cleansing")
        print("Data cleansing has started...")
        bar.progress(20,text="Removing irrelevant columns")
        df = self.RemoveIrrelevantColumn(df)
        bar.progress(40,text="Handling Missing Data")
        df = self.HandlingMissingData(df)
        bar.progress(70,text="Encoding Categorical values")
        df,y_labels = self.encoding(df,target)
        print("Data cleansing completed.")
        bar.progress(100,text="Data Cleansing Done!")
        return df,y_labels
        
   
    