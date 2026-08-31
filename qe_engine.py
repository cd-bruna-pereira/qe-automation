import re


def parse_total_energy(out_content):
    """Extrai o valor da Energia Total (em Ry) do output do Quantum ESPRESSO (pw.x)."""
    # Procura por linhas do tipo: ! total energy = -114.12345678 Ry ou total energy = ...
    matches = re.findall(r"total energy\s+=\s+([-\d\.]+)\s+Ry", out_content)
    if matches:
        return float(matches[-1])
    return None


def update_input_param(input_content, param_name, new_value):
    """Atualiza ou insere um parâmetro numérico no arquivo de input .in do Quantum ESPRESSO."""
    pattern = rf"({param_name}\s*=\s*)([\d\.\w\-]+)"

    if re.search(pattern, input_content, flags=re.IGNORECASE):
        updated_content = re.sub(
            pattern,
            rf"\g<1>{new_value}",
            input_content,
            flags=re.IGNORECASE,
        )
    else:
        # Insere o parâmetro dentro do bloco &SYSTEM caso não exista
        system_match = re.search(r"(&SYSTEM\b)", input_content, flags=re.IGNORECASE)
        if system_match:
            updated_content = input_content.replace(
                system_match.group(1),
                f"{system_match.group(1)}\n  {param_name} = {new_value},",
            )
        else:
            updated_content = input_content

    return updated_content


def check_convergence(energies, tolerance):
    """Verifica se a diferença de energia entre o passo atual e o anterior atinge a tolerância."""
    if len(energies) < 2:
        return False, None

    delta_e = abs(energies[-1] - energies[-2])
    is_converged = delta_e <= tolerance
    return is_converged, delta_e