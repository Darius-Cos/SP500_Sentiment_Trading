"""
run_all.py
==========
Ruleaza toate etapele pipeline-ului in ordine.
Foloseste: python run_all.py
"""

import subprocess
import sys
import time

# Lista de scripturi in ordinea in care trebuie rulate
SCRIPTURI = [
    ("Etapa 1 - Colectare date",          "src/1_collect_data.py"),
    ("Etapa 2 - Analiza sentiment",        "src/2_sentiment_analysis.py"),
    ("Etapa 3 - Feature engineering",      "src/3_feature_engineering.py"),
    ("Etapa 4 - Antrenare model",          "src/4_train_model.py"),
    ("Etapa 5 - Evaluare si vizualizare",  "src/5_evaluate.py"),
]

print("=" * 50)
print("  PIPELINE COMPLET - S&P 500 SENTIMENT TRADING")
print("=" * 50)

for nume, cale in SCRIPTURI:
    print(f"\n>>> {nume}")
    print("-" * 50)

    start = time.time()

    # subprocess.run ruleaza scriptul ca si cum l-ai scrie in terminal
    # sys.executable = calea catre Python-ul curent (acelasi cu cel activ)
    rezultat = subprocess.run([sys.executable, cale])

    durata = round(time.time() - start, 1)

    # Daca scriptul a esuat (returncode != 0), oprim tot
    if rezultat.returncode != 0:
        print(f"\nERORE in {cale}! Pipeline oprit.")
        print("Rezolva eroarea de mai sus si reporni run_all.py")
        sys.exit(1)

    print(f"\n[{nume} completat in {durata}s]")

print("\n" + "=" * 50)
print("  PIPELINE COMPLET! Graficele sunt in reports/")
print("=" * 50)
