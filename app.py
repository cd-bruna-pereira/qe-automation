import os
import time
import pandas as pd
import plotly.express as px
import streamlit as st

from qe_engine import check_convergence, parse_total_energy, update_input_param
from ssh_manager import SSHConnector, discover_local_ssh_keys

st.set_page_config(page_title="QE Convergence Automation", layout="wide")

st.title("Automação de Convergência — Quantum ESPRESSO")

# Inicialização de variáveis de sessão
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False

# Sidebar: Configuração SSH
st.sidebar.header("Conexão com o Supercomputador")
host = st.sidebar.text_input("Host / IP", value="cluster.hpc.br")
port = st.sidebar.number_input("Porta", value=22, step=1)
username = st.sidebar.text_input("Usuário", value="pesquisador")

auth_type = st.sidebar.radio(
    "Método de Autenticação",
    ["Chave Local Detectada", "Upload de Chave Privada"],
)

key_path = None
key_buffer = None

if auth_type == "Chave Local Detectada":
    detected_keys = discover_local_ssh_keys()
    if detected_keys:
        key_path = st.sidebar.selectbox("Selecione a chave", detected_keys)
    else:
        st.sidebar.warning("Nenhuma chave encontrada na pasta ~/.ssh ou %USERPROFILE%/.ssh")
        key_path = st.sidebar.text_input("Caminho absoluto da chave local")
else:
    uploaded_file = st.sidebar.file_uploader("Upload da chave (.pem / .id_rsa)", type=None)
    if uploaded_file:
        key_buffer = uploaded_file.getvalue().decode("utf-8")

passphrase = st.sidebar.text_input("Passphrase da chave (opcional)", type="password")
remote_work_dir = st.sidebar.text_input("Diretório Remoto de Trabalho", value="~/qe_runs")

# Botão de Teste de Conexão
if st.sidebar.button("Testar Conexão"):
    connector = SSHConnector(host=host, port=port, username=username)
    success, msg = connector.test_connection(
        key_path=key_path,
        key_buffer=key_buffer,
        passphrase=passphrase if passphrase else None,
        remote_dir=remote_work_dir,
    )
    if success:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

# Aba Principal
tab_config, tab_run, tab_results = st.tabs(
    ["Configuração do Input", "Execução & Monitoramento", "Resultados"]
)

with tab_config:
    st.subheader("Parâmetros do Quantum ESPRESSO")
    uploaded_in = st.file_uploader("Arquivo de entrada base (.in)", type=["in", "txt"])

    if uploaded_in:
        input_base_content = uploaded_in.getvalue().decode("utf-8")
        st.text_area("Pré-visualização do Input Base", input_base_content, height=180)
    else:
        input_base_content = "&CONTROL\n  calculation = 'scf'\n/\n&SYSTEM\n  ibrav = 1, celldm(1) = 7.0, nat = 1, ntyp = 1\n  ecutwfc = 30.0\n/\n&ELECTRONS\n/\nATOMIC_SPECIES\n  Si 28.085 Si.pbe-rrkjus.UPF\nATOMIC_POSITIONS (alat)\n  Si 0.0 0.0 0.0\nK_POINTS (automatic)\n  4 4 4 0 0 0"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        param_to_test = st.selectbox("Parâmetro para Teste", ["ecutwfc", "ecutrho"])
    with col2:
        val_start = st.number_input("Valor Inicial", value=30.0, step=5.0)
    with col3:
        val_stop = st.number_input("Valor Final", value=80.0, step=5.0)
    with col4:
        val_step = st.number_input("Passo", value=10.0, step=5.0)

    tolerance = st.number_input(
        "Tolerância de Convergência ΔE (Ry)",
        value=0.001,
        format="%.4f",
    )

    sbatch_script = st.text_area(
        "Script de Submissão Slurm (Exemplo)",
        value=f"#!/bin/bash\n#SBATCH --job-name=qe_conv\n#SBATCH --nodes=1\n#SBATCH --ntasks=32\n\ncd {remote_work_dir}\nmpirun -np 32 pw.x -in run.in > run.out",
        height=120,
    )

with tab_run:
    st.subheader("Controle de Execução")

    if st.button("Iniciar Loop de Convergência", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.history = []

        connector = SSHConnector(host=host, port=port, username=username)

        try:
            connector.connect(
                key_path=key_path,
                key_buffer=key_buffer,
                passphrase=passphrase if passphrase else None,
            )
            connector.execute_command(f"mkdir -p {remote_work_dir}")

            current_val = val_start
            energies = []
            values_tested = []

            plot_spot = st.empty()
            status_spot = st.empty()

            while current_val <= val_stop and st.session_state.running:
                status_spot.info(f"Executando cálculo para {param_to_test} = {current_val}...")

                # Gerar e enviar input atualizado
                current_input = update_input_param(input_base_content, param_to_test, current_val)
                connector.upload_string_as_file(
                    current_input,
                    f"{remote_work_dir}/run.in",
                )
                connector.upload_string_as_file(
                    sbatch_script,
                    f"{remote_work_dir}/job.sh",
                )

                # Submeter Job
                exit_code, out, err = connector.execute_command(
                    f"cd {remote_work_dir} && sbatch job.sh"
                )
                job_id = out.strip().split()[-1] if exit_code == 0 else "LOCAL_EXEC"

                # Monitoramento da execução
                complete = False
                while not complete:
                    time.sleep(3)
                    # Verifica se o arquivo run.out foi gerado e finalizou
                    exit_code, out_content, _ = connector.execute_command(
                        f"cat {remote_work_dir}/run.out"
                    )
                    if "JOB DONE." in out_content or "End of calculation" in out_content:
                        complete = True

                # Leitura e parsing dos resultados
                energy = parse_total_energy(out_content)

                if energy is not None:
                    energies.append(energy)
                    values_tested.append(current_val)

                    is_conv, delta_e = check_convergence(energies, tolerance)

                    st.session_state.history.append({
                        param_to_test: current_val,
                        "Total Energy (Ry)": energy,
                        "Delta E (Ry)": delta_e if delta_e else 0.0,
                        "Status": "Convergido" if is_conv else "Em Progresso",
                    })

                    # Atualização do Gráfico em Tempo Real
                    df_chart = pd.DataFrame(st.session_state.history)
                    fig = px.line(
                        df_chart,
                        x=param_to_test,
                        y="Total Energy (Ry)",
                        markers=True,
                        title=f"Convergência de Energia vs {param_to_test}",
                    )
                    plot_spot.plotly_chart(fig, use_container_width=True)

                    if is_conv:
                        status_spot.success(
                            f"Convergência atingida em {param_to_test} = {current_val}! "
                            f"(ΔE = {delta_e:.6f} Ry <= {tolerance} Ry)"
                        )
                        st.session_state.running = False
                        break

                current_val += val_step

            connector.close()
            st.session_state.running = False

        except Exception as e:
            st.error(f"Erro na execução do workflow: {str(e)}")
            st.session_state.running = False

with tab_results:
    st.subheader("Relatório Final de Convergência")
    if st.session_state.history:
        df_results = pd.DataFrame(st.session_state.history)
        st.dataframe(df_results, use_container_width=True)

        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Exportar Tabela de Resultados (CSV)",
            data=csv_data,
            file_name="qe_convergence_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Nenhum dado de execução disponível no momento.")