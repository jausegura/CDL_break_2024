# ruff: noqa: E501  (el informe de prueba reproduce el ancho fijo de los informes de NASS)
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acc01c", ROOT / "scripts" / "01c_official_accuracy.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)

REPORT = """
STATEWIDE AGRICULTURAL ACCURACY REPORT

Crop-specific covers only             *Correct  Accuracy       Error      Kappa
-------------------------              -------   --------     ------      -----
FSA Crops                              598,114      90.5%       9.5%      0.828

Cover                     Attribute   *Correct Producer's   Omission                User's Commission     Cond'l
Type                           Code     Pixels   Accuracy      Error      Kappa   Accuracy      Error      Kappa
----                           ----     ------   --------      -----      -----   --------      -----      -----
Corn                              1    340,138      94.3%       5.7%      0.912      97.5%       2.5%      0.960
Sunflower                         6          0       0.0%     100.0%      0.000       n/a        n/a        n/a
Dbl Crop WinWht/Soybeans         26         19      16.2%      83.8%      0.162     100.0%       0.0%      1.000
Other Hay/Non Alfalfa            37        656       7.2%      92.8%      0.070      35.7%      64.3%      0.351
**The Overall Accuracy represents only the FSA row crops
"""


def test_parse_all_classes_and_overall():
    overall, rows = acc.parse(REPORT)
    assert overall == {"overall_accuracy": 90.5, "kappa": 0.828, "n_correct": 598114}
    by_code = {r["code"]: r for r in rows}
    assert set(by_code) == {1, 6, 26, 37}
    assert by_code[1]["correct_px"] == 340138 and by_code[1]["producer_acc"] == 94.3
    assert by_code[37]["user_acc"] == 35.7 and by_code[37]["name"] == "Other Hay/Non Alfalfa"
    assert by_code[6]["user_acc"] != by_code[6]["user_acc"]  # n/a → NaN
