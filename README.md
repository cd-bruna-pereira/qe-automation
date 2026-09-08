# Análise de energia do Quantum ESPRESSO

Aplicação Streamlit que acessa um supercomputador por SSH, lê arquivos `.out` de uma pasta remota e apresenta os valores de `Final energy` e `DeltaE` em uma tabela e em um gráfico.

## Pré-requisitos

- Python 3.8 ou superior.
- Acesso SSH ao supercomputador.
- Uma chave privada SSH válida, disponível em `~/.ssh` ou em um arquivo que possa ser enviado pela aplicação.
- Arquivos de saída do Quantum ESPRESSO com extensão `.out`.

## Instalação e inicialização

### Linux ou macOS

No terminal, dentro da pasta do projeto:

```bash
chmod +x run.sh
./run.sh
```

O script cria o ambiente virtual, instala as dependências e inicia o Streamlit.

### Windows

Execute `run.bat` pelo Explorador de Arquivos ou pelo Prompt de Comando, dentro da pasta do projeto.

Depois da inicialização, abra no navegador o endereço exibido pelo Streamlit, normalmente:

```text
http://localhost:8501
```

## Como usar

1. Na barra lateral, informe o **Host / IP** do supercomputador.
2. Informe a **Porta SSH**, normalmente `22`.
3. Informe o **Usuário** usado no acesso remoto.
4. Escolha uma forma de autenticação:
	- **Chave local**: selecione uma chave detectada em `~/.ssh` ou informe o caminho completo dela.
	- **Upload de chave privada**: envie o arquivo da chave pela interface.
5. Informe a passphrase da chave, caso ela possua uma.
6. No campo **Pasta remota com os arquivos .out**, informe o caminho da pasta no supercomputador. Use `.` para a pasta inicial da sessão SSH ou informe um caminho absoluto, como `/home/usuario/calculos`.
7. Clique em **Ler arquivos e analisar**.

A aplicação conecta ao supercomputador, localiza todos os arquivos `.out` diretamente na pasta informada e lê cada arquivo.

## Formato dos arquivos

Cada arquivo precisa conter uma linha no seguinte formato:

```text
Final energy = -114.123456
```

Também são aceitos valores em notação científica, por exemplo:

```text
Final energy = -1.14123456e+02
```

Arquivos `.out` que não contêm uma linha válida de `Final energy` são ignorados. A busca não diferencia letras maiúsculas de minúsculas.

## Resultados

A tabela apresenta:

- **Arquivo**: nome do arquivo `.out`.
- **Final energy**: valor extraído do arquivo.
- **DeltaE**: diferença entre a energia do arquivo e a menor energia encontrada.

O cálculo é:

```text
DeltaE = Final energy - menor Final energy
```

Consequentemente, o menor valor de energia sempre terá `DeltaE = 0`.

O gráfico exibe os arquivos em ordem alfabética no eixo X e os valores de `DeltaE` no eixo Y. Os pontos representam os valores calculados e a linha suavizada representa a interpolação visual entre eles.

Use **Baixar tabela CSV** para exportar os resultados e analisá-los em outra ferramenta.

## Problemas comuns

### Nenhuma chave privada foi encontrada

Verifique se a chave está em `~/.ssh`, informe seu caminho completo no campo **Caminho da chave** ou use a opção de upload.

### Falha na conexão SSH

Confira o host, a porta, o usuário, a chave e a passphrase. Também confirme que a máquina onde o Streamlit está rodando tem acesso à rede do supercomputador.

### Nenhum arquivo `.out` foi encontrado

Confira o caminho da pasta remota e se os arquivos estão diretamente nela. A aplicação não pesquisa subpastas.

### Os arquivos foram encontrados, mas não há resultados

Abra um dos arquivos e confirme se existe uma linha `Final energy = valor`. Arquivos sem essa linha são desconsiderados.
