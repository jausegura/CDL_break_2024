"""Fase 1c: precisión oficial NASS (metadatos CDL).

Estados del piloto 2021-2023 (metadatos por estado) y nacional 2024-2025 (los metadatos por
estado ya no se publican a partir de 2024). Salidas: results/tables/official_accuracy.csv (cultivos
principales y exactitud global) y official_accuracy_classes.csv (todas las clases del informe: píxeles
correctos, exactitud del productor y del usuario).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agrors.cdl import FOCUS  # noqa: E402
from agrors.io import ROOT, download, load_config  # noqa: E402

BASE = "https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/"
ROW = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z/&.,\- ]+?)\s+(?P<code>\d+)\s+(?P<px>[\d,]+)\s+"
    r"(?P<pa>[\d.]+)%\s+[\d.]+%\s+[\d.]+\s+(?:(?P<ua>[\d.]+)%|n/a)"
)
OVERALL = re.compile(r"(?:OVERALL ACCURACY\*+|FSA Crops)\s+([\d,]+)\s+([\d.]+)%\s+[\d.]+%\s+([\d.]+)")


def parse(text: str) -> tuple[dict, list[dict]]:
    plain = re.sub(r"<[^>]+>", "", text)
    i = plain.find("ACCURACY REPORT")
    block = plain[i : i + 40000]
    m = OVERALL.search(block)
    overall = (
        {
            "overall_accuracy": float(m.group(2)),
            "kappa": float(m.group(3)),
            "n_correct": int(m.group(1).replace(",", "")),
        }
        if m
        else {}
    )
    rows = []
    for line in block.splitlines():
        r = ROW.match(line.strip())
        if r:
            rows.append(
                {
                    "code": int(r.group("code")),
                    "name": r.group("name").strip(),
                    "correct_px": int(r.group("px").replace(",", "")),
                    "producer_acc": float(r.group("pa")),
                    "user_acc": float(r.group("ua")) if r.group("ua") else float("nan"),
                }
            )
        if line.strip().startswith("**The Overall"):
            break
    return overall, rows


def main() -> int:
    cfg = load_config()
    out, classes = [], []
    targets = [(st.lower(), y) for st in cfg["pilot"]["states"] for y in (2021, 2022, 2023)]
    targets += [("national", 2024), ("national", 2025)]
    for st, y in targets:
        name = (
            f"metadata_CDL{y % 100}_FGDC-STD-001-1998.htm"
            if st == "national"
            else f"metadata_{st}{y % 100}.htm"
        )
        dest = ROOT / "data" / "raw" / "metadata" / name
        download(BASE + name, dest, "Public domain (USDA NASS)", f"Metadatos CDL {st} {y}")
        overall, rows = parse(dest.read_text(encoding="utf-8", errors="replace"))
        inv = {v: k for k, v in FOCUS.items()}
        classes += [{"scope": st.upper(), "year": y, **r} for r in rows]
        focus = [r for r in rows if r["code"] in inv]
        for r in focus:
            keep = {k: r[k] for k in ("code", "producer_acc", "user_acc")}
            out.append({"scope": st.upper(), "year": y, "crop": inv[r["code"]], **keep, **overall})
        if not focus:
            out.append({"scope": st.upper(), "year": y, "crop": None, **overall})
    pd.DataFrame(classes).to_csv(ROOT / "results" / "tables" / "official_accuracy_classes.csv", index=False)
    df = pd.DataFrame(out)
    path = ROOT / "results" / "tables" / "official_accuracy.csv"
    df.to_csv(path, index=False)
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
