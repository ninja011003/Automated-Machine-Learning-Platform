import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from prediction import predict,predict_prob
from pathlib import Path
from main import train,D_prepare
from math import ceil
from sklearn.preprocessing import LabelEncoder
from visualize import visualization
from model_visual import ConfusionMatrixDisp,ROC,ErrorPlot,Reg_Mod_Eval
from model_selection import model_seletion
import time

#st.set_page_config(page_title="Automated ML", layout="wide", initial_sidebar_state="expanded")

def downloadbutton(button,model, save_file):
    with open(model, 'rb') as f:
        mlmodel = f.read()
    st.download_button(button, data=mlmodel, file_name=save_file)

def text_hl(text, color='blue', font_weight='bold', font_size='16px', padding='5px'):
    return f'<span style="color:{color}; font-weight:{font_weight}; font-size:{font_size}; padding:{padding}">{text}</span>'

def U_Manual():
    img_path="./images/"
    st.caption("Team Higgs Bosons'")
    st.header("User Manual: Guide to Automated ML Platform",divider='red')
    st.markdown("Welcome to our Data Analysis Platform! Whether you're a tech-savvy individual or completely new to data analysis, this user manual will guide you through every step of the process in a simple and straightforward manner.")
                
    st.subheader("Step 1",divider="blue")
    st.markdown(''':orange[**Data Upload:**] Begin by uploading your dataset onto the platform.  
- Click on the :green[Data Upload] option.  
- Choose the :violet[file] you want to upload.  
- After uploading, you'll see a success message.  
- If you want to view the uploaded data, click on :green[View Data].  
- Otherwise, proceed to the next step.  ''')
    with st.expander("Demo for Data Upload"):
        st.video("./videos/step1.mp4")
    
    st.subheader("Step 2",divider="blue")
    st.markdown('''**:orange[Visualization:]** :rainbow[V I S U A L I Z E] your data to gain insights.  
 - Select the :violet[target] variable.  
 - Click on :green[Perform Data Visualisation] to generate visualizations based on your selection.  ''')
    with st.expander("Demo for Data Visualisation"):
        st.video("./videos/step2.mp4")
    
    st.subheader("Step 3",divider="blue")
    st.markdown('''**:orange[Data Preparation:]** Prepare your data for analysis.  
 - Select the :violet[target] variable.  
 - Click on :green[Prepare Data] to perform preprocessing and feature selection.  
 - The platform will display the selected best features for model selection.  ''')
    with st.expander("Demo for Data Preparation"):
        st.video("./videos/step3.mp4")
    
    st.subheader("Step 4",divider="blue")
    st.markdown('''**:orange[Model Training:]** Train your model to make predictions.  
 - Select the :violet[target] variable.  
 - Click on :green[Train] to initiate the training process.  
 - Once completed, the platform will show the :violet[best model] along with a list of other models to choose from.  ''')
    with st.expander("Demo for Model Training"):
        st.video("./videos/step4.mp4")

    st.subheader("Step 5",divider="blue")
    st.markdown('''**:orange[Best Model:]** Download the :violet[best model] and :violet[preprocessed data].  
 - In this section, you can download the model and preprocessed data for future use.  ''')
    
    st.subheader("Step 6",divider="blue")
    st.markdown('''**:orange[Model Evaluation:]** Evaluate the :violet[performance] of your model.  
 - You can visualize the model's performance metrics in this section.  ''')
    
    tab1, tab2= st.tabs(["Classification Graphs", "Regression Graphs"])
    with tab1:
        st.markdown(''' - For Classification models, :violet[Confusion Matrix] and :violet[ROC Curve Graph] will be displayed.''')
        col1,col2=st.columns(2)
        with col1:
            st.image(img_path+"confusion_mat.png",
                    caption="Ideal Confusion Matrix should have deep colored diagonal elements",
                    width=355)
        with col2:
            st.image(img_path+"roc.png",
                    caption="AUC of classes(colored lines) should be above 0.5, that is, above the red dotted line",
                    width=397)
    with tab2:
        st.markdown(''' - For Regression models, :violet[Residue vs Predicted Graph] will be shown''')
        st.image(img_path+"resid.png",
                caption="The points should be concentrated around the red line.",
                width=370)
        
    st.subheader("Step 7",divider="blue")
    st.markdown('''**:orange[Prediction:]** Make :violet[predictions] using your trained model.  
 - :green[Input] the :violet[values] you want to predict within the platform.  
 - Check the :violet[output] to see the predicted results.  ''')
    with st.expander("Demo for Prediction"):
        st.video("./videos/step7.mp4")
    st.divider()
    st.markdown("**Congratulations!** You've successfully navigated through our Automaled ML Platform. If you have any questions or need further assistance, feel free to reach out to our support team.")
    st.markdown("[Go to top](#automated-ml)")
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)


def Upload_data():
    st.session_state.target=None
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv','parquet'])
    if uploaded_file:
        save_folder = "./data"
        save_path = Path(save_folder, uploaded_file.name)
        st.session_state.filename=uploaded_file.name
        with open(save_path, 'wb') as w:
            w.write(uploaded_file.getvalue())
        if save_path.exists():
            st.success(f'File {uploaded_file.name} is successfully saved!')
            #text_to_speech("The file is uploaded successfully, click next option in sidebar to proceed further")
        if str(save_path).endswith(".csv"):
            data = pd.read_csv(save_path)
        else:
            data = pd.read_parquet(save_path)
        return data

def View_data(data):
    if st.button("View Data"):
        c1,c2=st.columns(2)
        c2.caption(f"{data.shape[0]} Rows")
        c1.caption(f"{data.shape[1]} Columns")
        st.write(data.head())

def Visualize():
    st.header("Data Visualization")
    if st.session_state.target==None:
        st.session_state.target = st.selectbox("Select the target variable:", list(st.session_state.data.columns),index=None,placeholder="-Select target to visualize")
    else:
        st.session_state.target = st.selectbox("Select the target variable:", list(st.session_state.data.columns),index=list(st.session_state.data.columns).index(st.session_state.target),placeholder="-Select target to visualize")
    st.write("Target Variable selected:", st.session_state.target)
    if st.button("Perform Data Visualisation"):
        visualization(st.session_state.data, st.session_state.target)
        st.markdown("[Go to top](#automated-ml)")

def Data_Prepare():
    st.header("Data Preparation")
    if st.session_state.target==None:
        st.session_state.target = st.selectbox("Select the target variable:", list(st.session_state.data.columns),index=None,placeholder="-Select value as target-")
    else:
        st.session_state.target = st.selectbox("Select the target variable:", list(st.session_state.data.columns),index=list(st.session_state.data.columns).index(st.session_state.target),placeholder="-Select value as target-")

    if st.session_state.target!=None:
        if st.button("Prepare Data"):
            data_res=D_prepare(st.session_state.data,st.session_state.target,st.session_state.filename.split(".")[1])
            st.session_state.data=data_res[0]
            st.session_state.features=data_res[1]
            st.session_state.y_labels=data_res[2]
            total_cols,remain_cols,cols_diff=data_res[3]
        
        if type(st.session_state.features)!=list:
            st.session_state.features=st.session_state.features.tolist()
        
        try:
            if len(st.session_state.features)!=0:
                c1,c2,c3=st.columns(3)
                c1.text(f'Total Columns: {total_cols}')
                c2.text(f'Columns Removed: {cols_diff}')
                c3.text(f'Remaining Columns: {remain_cols}')
                #st.text(f"Features Available:{st.session_state.pre_feat}")
                #st.caption(f"Model Accuracy:{round(st.session_state.pre_acc*100,2)}%")
                st.text(f"Features selected using Feature Selection:")
                feats=pd.DataFrame({"Columns Selected":st.session_state.features})
                st.write(feats)
        
        except:
            st.info(f"Select the target variable and click Prepare Data button to preprocess and select best features.")
    return st.session_state.features

def Training():
    st.header("Model Training")
    if st.session_state.target==None:
        st.session_state.target = st.selectbox("Select the target variable:", list(st.session_state.data.columns),index=None,placeholder="-Select value to predict-")
    else:
        st.session_state.target = st.selectbox("Select the target variable:", list(st.session_state.data.columns),index=list(st.session_state.data.columns).index(st.session_state.target),placeholder="-Select value to predict-")
    st.write("Target Variable selected:", st.session_state.target)
    if st.session_state.target!=None:
        if st.button("Train"):
            mod=model_seletion(st.session_state.data,st.session_state.target)
            model_res=train(st.session_state.data,st.session_state.target)
            
            st.session_state.model_acc=model_res[0]
            st.session_state.X_train=model_res[1]
            st.session_state.X_test=model_res[2]
            st.session_state.Y_train=model_res[3]
            st.session_state.Y_test=model_res[4]
            st.session_state.y_pred=model_res[5]
            st.session_state.models_table=model_res[6]
            st.session_state.mod_type=mod.is_classification(st.session_state.data,st.session_state.target)
            st.success(f'Model trained successfully')
            st.write(st.session_state.models_table.head())
            #st.info(f"Best Model Accuracy :{round(st.session_state.model_acc,2)}%")
            st.info(f"Best Model Accuracy :{round(st.session_state.model_acc*100,2)}%")
    #X_feat = list(st.session_state.data.drop(columns=[target]).columns)
    #st.session_state.X_feat=X_feat
    return st.session_state.mod_type

def download_model():
    st.header("Downloads")
    try:
        st.subheader("Download the model here!")
        downloadbutton("Download Model","./model/best_model.sav", "best_model.sav")
        st.subheader("Download processed data here!")
        downloadbutton("Download Dataset","./data/Preprocessed."+st.session_state.filename.split(".")[1], st.session_state.filename)
        
    except:
        st.text("You have now uploaded the data. Click Train to train the model and download it.")

def Model_Visualize():
    st.header("Model Analysis")
    #prediction=predict(st.session_state.X_test)
    prediction=st.session_state.y_pred
    # print(st.session_state.Y_test)
    if st.session_state.mod_type:
        y_prob=predict_prob(st.session_state.X_test)
        ConfusionMatrixDisp(st.session_state.Y_test,prediction)
        ROC(st.session_state.Y_train,st.session_state.Y_test,y_prob)
    else:
        Reg_Mod_Eval(st.session_state.X_test,st.session_state.Y_test,st.session_state.y_pred)
        #ErrorPlot(st.session_state.X_train,st.session_state.Y_train,st.session_state.X_test,st.session_state.Y_test)
    st.markdown("[Go to top](#automated-ml)")
    

def predictions():
    st.header("Features")
    co1=[]
    col1, col2 = st.columns(2)
    if len(st.session_state.features)%2!=0:
        with col1:
            for i in range(0, ceil(len(st.session_state.features) / 2)):
                co1.append(st.number_input(f"Enter value for {st.session_state.features[i]}",key=i))
        with col2:
            for i in range((len(st.session_state.features) // 2)+1, len(st.session_state.features)):
                co1.append(st.number_input(f"Enter value for {st.session_state.features[i]}",key=i+1))
    else:
        with col1:
            for i in range(0, ceil(len(st.session_state.features) / 2)):
                co1.append(st.number_input(f"Enter value for {st.session_state.features[i]}",key=i))
        with col2:
            for i in range((len(st.session_state.features) // 2), len(st.session_state.features)):
                co1.append(st.number_input(f"Enter value for {st.session_state.features[i]}",key=i+1))
    if st.button("Predict"):
        # print(st.session_state.y_labels)
        if st.session_state.y_labels is not None:
            le=LabelEncoder()
            le.classes_=st.session_state.y_labels
            res = le.inverse_transform(predict(np.array([co1])))
        else:
            res = predict(np.array([co1]))
        styled_pred = text_hl(res[0], color="yellow", font_weight='700', font_size='20px', padding='10px')
        st.markdown(f"Prediction Result:{styled_pred}", unsafe_allow_html=True)

# Build App
st.title("Automated ML")
st.header("The service to automate the ml training.")
st.divider()

st.sidebar.title("Menu")
#app = st.sidebar.radio("Choose", options=["Load Data", "Data Visualisation", "Model Training", "Best Model","Model Visualisation", "Prediction"])
with st.sidebar:        
    app = option_menu(
        menu_title='',
        options=["User Manual","Load Data", "Data Visualisation","Data Preparation","Model Training", "Best Model","Model Visualisation", "Prediction"],
        icons=['book-fill','cloud-upload-fill','bar-chart-fill','wrench','gear-fill','award-fill','magic','chat-square-text-fill'],
        menu_icon='',
        default_index=1,
        styles={
            "container": {"padding": "5!important","background-color":'#080A29',"font-family":'"Trebuchet MS",Helvatica,sans-serif'},
"icon": {"color": "white", "font-size": "20px"}, 
"nav-link": {"color":"white","font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
"nav-link-selected": {"background-color": "#630490"},}
        )



if 'data' not in st.session_state:
    st.session_state.data = None

if 'filename' not in st.session_state:
    st.session_state.filename=None

if "model_acc" not in st.session_state:
   st.session_state.model_acc=None

if 'features' not in st.session_state:
    st.session_state.features=pd.Series([])

if 'pre_feat' not in st.session_state:
    st.session_state.pre_feat=None

if 'pre_acc' not in st.session_state:
    st.session_state.pre_acc=None

if 'X_train' not in st.session_state:
    st.session_state.X_train=None

if 'X_test' not in st.session_state:
    st.session_state.X_test=None

if 'Y_train' not in st.session_state:
    st.session_state.Y_train=None

if 'Y_test' not in st.session_state:
    st.session_state.Y_test=None

if 'y_pred' not in st.session_state:
    st.session_state.y_pred=None

if 'y_labels' not in st.session_state:
    st.session_state.y_labels=None

if 'mod_type' not in st.session_state:
    st.session_state.mod_type=None

if 'target' not in st.session_state:
    st.session_state.target=None

if app == "User Manual":
    U_Manual()
if app == "Load Data":
    st.session_state.data = Upload_data()
    if st.session_state.data is not None:
        View_data(st.session_state.data)

if app == "Data Visualisation":
    if st.session_state.data is not None:
        Visualize()
    else:
        st.warning("Load the data first")

if app == "Data Preparation":
    if st.session_state.data is not None:
        st.session_state.features=Data_Prepare()
    else:
        st.warning("Load the data first")

if app == "Model Training":
    if st.session_state.data is not None:
        with st.spinner("Training...."):
            Training()
    else:
        st.warning("Load the data first")

if app == "Best Model":
    if st.session_state.data is not None:
        download_model()
    else:
        st.warning("Could you atleast load the data and run the model?")

if app == "Model Visualisation":
    if st.session_state.data is not None:
        Model_Visualize()
    else:
        st.warning("Load the data first")

if app == "Prediction":
    if st.session_state.data is not None:
        predictions()
    else:
        st.warning("Load the data first")