import io
import os
from pathlib import Path
import paramiko


def discover_local_ssh_keys():
    """Detecta automaticamente chaves SSH públicas/privadas no diretório padrão do sistema."""
    home_ssh = Path.home() / ".ssh"
    if not home_ssh.exists():
        return []

    valid_extensions = ["", ".pem", ".id_rsa", ".id_ed25519", ".id_ecdsa"]
    keys_found = []

    for file in home_ssh.iterdir():
        if file.is_file() and not file.name.endswith(".pub") and file.name != "known_hosts":
            if file.suffix in valid_extensions or "id_" in file.name:
                keys_found.append(str(file))

    return keys_found


class SSHConnector:
    """Gerencia autenticação, execução remota e transferência SFTP."""

    def __init__(self, host, port=22, username=""):
        self.host = host
        self.port = int(port)
        self.username = username
        self.client = None

    def _get_pkey(self, key_path=None, key_buffer=None, passphrase=None):
        """Carrega a chave privada a partir de arquivo local ou buffer em memória."""
        key_classes = [paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey]

        if key_buffer:
            for k_cls in key_classes:
                try:
                    key_file = io.StringIO(key_buffer)
                    return k_cls.from_private_key(key_file, password=passphrase)
                except Exception:
                    continue

        if key_path and os.path.exists(key_path):
            for k_cls in key_classes:
                try:
                    return k_cls.from_private_key_file(key_path, password=passphrase)
                except Exception:
                    continue

        raise ValueError("Não foi possível carregar a chave SSH fornecida.")

    def connect(self, key_path=None, key_buffer=None, passphrase=None):
        """Abre a conexão SSH."""
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        pkey = self._get_pkey(key_path, key_buffer, passphrase)
        self.client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            pkey=pkey,
            timeout=10,
        )

    def test_connection(self, key_path=None, key_buffer=None, passphrase=None, remote_dir="~/"):
        """Testa autenticação, execução de comandos e escrita em disco remoto."""
        try:
            self.connect(key_path, key_buffer, passphrase)

            # Teste de comando basico
            stdin, stdout, stderr = self.client.exec_command("whoami && hostname")
            user_info = stdout.read().decode().strip().replace("\n", " @ ")

            # Teste de escrita SFTP
            sftp = self.client.open_sftp()
            test_file = f"{remote_dir.rstrip('/')}/.ssh_test_write.tmp"
            with sftp.file(test_file, "w") as f:
                f.write("test")
            sftp.remove(test_file)
            sftp.close()

            self.close()
            return True, f"Conectado com sucesso! ({user_info})"
        except Exception as e:
            self.close()
            return False, f"Falha na conexão: {str(e)}"

    def execute_command(self, command):
        """Executa um comando no cluster e retorna o código de saída, stdout e stderr."""
        if not self.client:
            raise ConnectionError("Cliente SSH não está conectado.")
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode(), stderr.read().decode()

    def upload_string_as_file(self, content, remote_path):
        """Envia uma string direta para um arquivo remoto via SFTP."""
        if not self.client:
            raise ConnectionError("Cliente SSH não está conectado.")
        sftp = self.client.open_sftp()
        with sftp.file(remote_path, "w") as f:
            f.write(content)
        sftp.close()

    def read_remote_file(self, remote_path):
        """Lê o conteúdo de um arquivo remoto."""
        if not self.client:
            raise ConnectionError("Cliente SSH não está conectado.")
        sftp = self.client.open_sftp()
        with sftp.file(remote_path, "r") as f:
            content = f.read().decode()
        sftp.close()
        return content

    def close(self):
        """Fecha a conexão SSH atenta."""
        if self.client:
            self.client.close()
            self.client = None