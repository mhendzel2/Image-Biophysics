@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Starting GUI...
streamlit run app.py

pause
