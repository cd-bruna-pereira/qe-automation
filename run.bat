@echo off
IF NOT EXIST ".venv" (
    echo Criando ambiente virtual Python...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Instalando dependencias...
pip install -r requirements.txt

echo Iniciando aplicacao Streamlit...
streamlit run app.py
pause