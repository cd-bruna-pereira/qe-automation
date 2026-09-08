import re


FINAL_ENERGY_PATTERN = re.compile(
    r"^\s*Final\s+energy\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_final_energy(out_content):
    """Retorna o valor da linha ``Final energy = x`` de um arquivo .out."""
    matches = FINAL_ENERGY_PATTERN.findall(out_content)
    return float(matches[-1]) if matches else None


def calculate_delta_energies(energies):
    """Calcula DeltaE como a energia de cada arquivo menos a menor energia."""
    if not energies:
        return []

    minimum_energy = min(energies)
    return [energy - minimum_energy for energy in energies]