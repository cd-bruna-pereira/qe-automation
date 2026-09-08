import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qe_engine import calculate_delta_energies, parse_final_energy
from ssh_manager import SSHConnector, discover_local_ssh_keys


st.set_page_config(page_title="Análise de Final Energy", layout="wide")
st.title("Análise de energia no supercomputador")
st.caption("Leia os arquivos .out de uma pasta remota, compare as energias e visualize o DeltaE.")


def connect_to_cluster(host, port, username, key_path, key_buffer, passphrase):
    connector = SSHConnector(host=host, port=port, username=username)
    connector.connect(
        key_path=key_path,
        key_buffer=key_buffer,
        passphrase=passphrase or None,
    )
    return connector


with st.sidebar:
    st.header("Conexão SSH")
    host = st.text_input("Host / IP")
    port = st.number_input("Porta", min_value=1, max_value=65535, value=22, step=1)
    username = st.text_input("Usuário")

    auth_type = st.radio("Autenticação", ["Chave local", "Upload de chave privada"])
    key_path = None
    key_buffer = None
    if auth_type == "Chave local":
        detected_keys = discover_local_ssh_keys()
        key_path = st.selectbox("Chave privada", detected_keys) if detected_keys else None
        if not key_path:
            st.warning("Nenhuma chave privada foi encontrada em ~/.ssh.")
            key_path = st.text_input("Caminho da chave")
    else:
        uploaded_key = st.file_uploader("Arquivo da chave privada")
        if uploaded_key:
            key_buffer = uploaded_key.getvalue().decode("utf-8")

    passphrase = st.text_input("Passphrase (opcional)", type="password")
    remote_dir = st.text_input("Pasta remota com os arquivos .out", value=".")
    analyze = st.button("Ler arquivos e analisar", type="primary")


if analyze:
    missing_fields = [
        label
        for label, value in [
            ("Host / IP", host),
            ("Usuário", username),
            ("Pasta remota", remote_dir),
        ]
        if not value
    ]
    out_files = []
    rows = []
    if missing_fields:
        st.error(f"Preencha: {', '.join(missing_fields)}.")
    elif auth_type == "Chave local" and not key_path:
        st.error("Selecione ou informe uma chave privada.")
    elif auth_type == "Upload de chave privada" and not key_buffer:
        st.error("Envie uma chave privada.")
    else:
        connector = None
        try:
            with st.spinner("Conectando e lendo os arquivos .out..."):
                connector = connect_to_cluster(
                    host, port, username, key_path, key_buffer, passphrase
                )
                out_files = connector.list_remote_out_files(remote_dir)
                for filename in out_files:
                    remote_path = f"{remote_dir.rstrip('/')}/{filename}"
                    content = connector.read_remote_file(remote_path)
                    energy = parse_final_energy(content)
                    if energy is not None:
                        rows.append({"Arquivo": filename, "Final energy": energy})
        except Exception as error:
            st.error(f"Não foi possível analisar a pasta remota: {error}")
        finally:
            if connector:
                connector.close()

        if not out_files:
            st.warning("Nenhum arquivo .out foi encontrado nessa pasta.")
        elif not rows:
            st.warning('Os arquivos .out encontrados não possuem uma linha "Final energy = x".')
        else:
            energies = [row["Final energy"] for row in rows]
            for row, delta_e in zip(rows, calculate_delta_energies(energies)):
                row["DeltaE"] = delta_e

            results = pd.DataFrame(rows)
            st.subheader("Resultados")
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar tabela CSV",
                results.to_csv(index=False).encode("utf-8"),
                "final_energy_results.csv",
                "text/csv",
            )

            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=results["Arquivo"],
                    y=results["DeltaE"],
                    mode="markers",
                    name="DeltaE",
                )
            )
            if len(results) >= 2:
                figure.add_trace(
                    go.Scatter(
                        x=results["Arquivo"],
                        y=results["DeltaE"],
                        mode="lines",
                        name="Interpolação",
                        line_shape="spline",
                    )
                )
            figure.update_layout(
                title="DeltaE por arquivo",
                xaxis_title="Arquivo",
                yaxis_title="DeltaE",
                xaxis={"type": "category"},
            )
            st.plotly_chart(figure, use_container_width=True)