# Automação de Convergência - Quantum ESPRESSO

Esta solução é uma plataforma interativa desenvolvida em Python e Streamlit para automatizar testes de convergência de parâmetros de entrada (como ecutwfc e ecutrho) no Quantum ESPRESSO, executando cálculos em supercomputadores remotos via SSH e SFTP.

---

## 1. Estrutura do Projeto

* app.py: Interface gráfica interativa construída em Streamlit.
* ssh_manager.py: Módulo responsável pela autenticação SSH, transferência de arquivos via SFTP e execução de comandos no cluster.
* qe_engine.py: Módulo responsável por atualizar parâmetros nos arquivos de input (.in) e extrair a energia total dos arquivos de saída (.out).
* requirements.txt: Dependências do projeto Python.
* run.sh: Script de inicialização para sistemas Linux e macOS.
* run.bat: Script de inicialização para sistemas Windows.

---

## 2. Pré-requisitos

* Python 3.8 ou superior instalado localmente.
* Acesso SSH a um cluster/supercomputador com Quantum ESPRESSO e gerenciador de filas (ex: Slurm).
* Chave SSH cadastrada no cluster (chave privada local em ~/.ssh ou arquivo .pem/.id_rsa para upload).

---

## 3. Como Rodar a Solução

### No Linux / macOS

1. Abra o terminal no diretório do projeto.
2. Torne o script de inicialização executável:
   chmod +x run.sh
3. Execute o script:
   ./run.sh

O script criará o ambiente virtual (.venv), instalará as dependências do arquivo requirements.txt e iniciará o dashboard no navegador.

### No Windows

1. Dê um duplo clique no arquivo run.bat ou abra o Prompt de Comando (CMD) / PowerShell no diretório do projeto.
2. Execute o comando:
   run.bat

O script criará o ambiente virtual, instalará as bibliotecas necessárias e abrirá a interface no seu navegador padrão.

---

## 4. Como Usar a Aplicação

### Passo 1: Configurar a Conexão SSH
Na barra lateral esquerda:
1. Informe o Host/IP, Porta (padrão 22) e Usuário do cluster.
2. Selecione o Método de Autenticação:
   * Chave Local Detectada: O sistema busca automaticamente chaves no diretório ~/.ssh (Linux/macOS) ou %USERPROFILE%\.ssh (Windows).
   * Upload de Chave Privada: Permite enviar um arquivo de chave privada (.pem, .id_rsa) diretamente pela interface.
3. Se a sua chave possuir senha, preencha o campo Passphrase da chave.
4. Informe o Diretório Remoto de Trabalho (ex: ~/qe_runs).
5. Clique no botão Testar Conexão para validar o acesso ao cluster e as permissões de escrita.

### Passo 2: Configurar o Input e os Parâmetros de Teste
Na aba Configuração do Input:
1. Faça o upload do arquivo de entrada base do Quantum ESPRESSO (.in). Caso nenhum arquivo seja enviado, um modelo padrão de Silício será utilizado.
2. Selecione o Parâmetro para Teste (ex: ecutwfc ou ecutrho).
3. Defina os valores do teste:
   * Valor Inicial (ex: 30.0)
   * Valor Final (ex: 80.0)
   * Passo (ex: 10.0)
4. Defina a Tolerância de Convergência ΔE em Ry (ex: 0.001 Ry).
5. Ajuste o Script de Submissão Slurm conforme as especificações do seu cluster (número de nós, tarefas, módulos, etc.).

### Passo 3: Executar e Monitorar
Na aba Execução & Monitoramento:
1. Clique no botão Iniciar Loop de Convergência.
2. A aplicação enviará sequencialmente os arquivos .in e os scripts de submissão para o cluster, executará os jobs e monitorará o término de cada cálculo.
3. O gráfico de linha em tempo real atualizará a cada ponto calculado (Energia Total vs. Parâmetro).
4. O loop interromperá automaticamente quando a diferença de energia entre dois passos consecutivos for menor ou igual à tolerância estabelecida, ou ao atingir o valor final do intervalo.

### Passo 4: Visualizar e Exportar Resultados
Na aba Resultados:
1. Visualize a tabela consolidada contendo o valor do parâmetro, energia total acumulada, variação de energia (ΔE) e status de convergência.
2. Clique no botão Exportar Tabela de Resultados (CSV) para baixar os dados brutos para análise posterior.
