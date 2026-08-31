#!/bin/bash
if [ ! -d ".venv" ]; then
    echo "Criando ambiente virtual Python..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Instalando dependências..."
pip install -r requirements.txt

echo "Iniciando aplicação Streamlit..."
streamlit run app.py