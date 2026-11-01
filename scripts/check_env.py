"""Comprueba que .env tiene las variables obligatorias de .env.example, sin mostrar sus valores.

No depende de paquetes externos, así que funciona antes de instalar requirements.txt.
Secciones de .env.example: '# === REQUIRED ===' y '# === OPTIONAL ==='.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(path: Path, with_sections: bool = False):
    section = "OPTIONAL"
    out: dict[str, tuple[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("# ===") and "REQUIRED" in line.upper():
            section = "REQUIRED"
            continue
        if line.startswith("# ===") and "OPTIONAL" in line.upper():
            section = "OPTIONAL"
            continue
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        out[key.strip()] = (section, val)
    return out


def main() -> int:
    example = ROOT / ".env.example"
    env = ROOT / ".env"
    if not env.exists():
        env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print("Creado .env a partir de .env.example. Rellena los valores y vuelve a ejecutar.")
    spec = parse(example)
    vals = {k: v for k, (_, v) in parse(env).items()}
    missing, present_opt, missing_opt = [], [], []
    for key, (section, _) in spec.items():
        filled = bool(vals.get(key))
        if section == "REQUIRED" and not filled:
            missing.append(key)
        elif section == "OPTIONAL":
            (present_opt if filled else missing_opt).append(key)

    # Comprobaciones especiales
    warnings = []
    key_file = vals.get("EE_SERVICE_ACCOUNT_KEY")
    if spec.get("EE_PROJECT", ("OPTIONAL",))[0] == "REQUIRED":
        if key_file:
            p = Path(key_file).expanduser()
            if not p.exists():
                missing.append("EE_SERVICE_ACCOUNT_KEY (el fichero no existe)")
            else:
                try:
                    json.loads(p.read_text(encoding="utf-8"))["client_email"]
                except Exception:
                    missing.append("EE_SERVICE_ACCOUNT_KEY (JSON sin client_email)")
        else:
            warnings.append(
                "EE_SERVICE_ACCOUNT_KEY vacío: se usarán credenciales locales de "
                "`earthengine authenticate` (deben existir en esta máquina)."
            )

    print("== Variables obligatorias ==")
    for key, (section, _) in spec.items():
        if section == "REQUIRED":
            print(f"  [{'OK' if key not in missing else 'FALTA'}] {key}")
    print("== Opcionales configuradas ==", ", ".join(present_opt) or "ninguna")
    print("== Opcionales vacías ==", ", ".join(missing_opt) or "ninguna")
    for w in warnings:
        print("AVISO:", w)
    if missing:
        print("\nFALTAN variables obligatorias:", ", ".join(missing))
        print("Rellénalas en .env (instrucciones en .env.example) y vuelve a ejecutar.")
        return 1
    print("\nOK: credenciales obligatorias presentes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
