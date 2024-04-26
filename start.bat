@echo off
echo Installing Python packages listed in requirements.txt...
pip install -r requirements.txt
echo Installation completed.


streamlit run streamlit_app.py