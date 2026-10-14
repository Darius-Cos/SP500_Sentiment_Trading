"""
5_evaluate.py
=============
Ce face acest fisier:
  - Incarca modelul si rezultatele din etapele anterioare
  - Genereaza 4 grafice vizuale in folderul reports/
  - Ruleaza un backtest simplu (simulare de tranzactionare)

Grafice generate:
  1. Confusion matrix - cate predictii corecte vs gresite
  2. Feature importance - ce variabile conteaza cel mai mult
  3. Performanta per fold - cum se comporta modelul in timp
  4. Backtest - ce randament ar fi produs strategia noastra

Ce este backtestul?
  Simulam ca in ultimii 2 ani am folosit modelul pentru a tranzactiona:
    - Daca modelul zicea "creste maine" → cumparam azi
    - Daca modelul zicea "scade maine"  → stam in cash
  Comparam randamentul nostru cu buy-and-hold (cumpara si tine fara sa faci nimic).
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

os.makedirs("reports", exist_ok=True)

# Stilul graficelor
plt.style.use("dark_background")

# Culori folosite in grafice
VERDE  = "#00d4aa"
ROSU   = "#ff4b6e"
GALBEN = "#f5c518"
GRI    = "#7b8fa1"
FUNDAL = "#1a1a2e"


# ─────────────────────────────────────────────
# PASUL 1: INCARCAM DATELE
# ─────────────────────────────────────────────

def incarca_datele():
    """
    Incarca toate fisierele generate in etapele anterioare.
    """
    print("Incarcam datele pentru evaluare...")

    # Verificam ca fisierele exista
    fisiere_necesare = [
        "data/processed/predictions.csv",
        "data/processed/walk_forward_results.csv",
        "data/processed/feature_importance.csv",
        "data/processed/features.csv",
        "models/xgb_classifier.pkl",
        "models/feature_cols.pkl",
    ]
    for fisier in fisiere_necesare:
        if not os.path.exists(fisier):
            print("EROARE:", fisier, "nu exista!")
            print("Ruleaza mai intai etapele anterioare.")
            exit()

    # Citim fisierele CSV
    predictii    = pd.read_csv("data/processed/predictions.csv")
    folduri      = pd.read_csv("data/processed/walk_forward_results.csv")
    importanta   = pd.read_csv("data/processed/feature_importance.csv")
    features     = pd.read_csv("data/processed/features.csv")

    # Incarcam modelul din fisierul pickle
    with open("models/xgb_classifier.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/feature_cols.pkl", "rb") as f:
        lista_features = pickle.load(f)

    features["Date"] = pd.to_datetime(features["Date"])

    print("Date incarcate cu succes!")
    return predictii, folduri, importanta, features, model, lista_features


# ─────────────────────────────────────────────
# GRAFIC 1: CONFUSION MATRIX
# ─────────────────────────────────────────────

def grafic_confusion_matrix(predictii):
    """
    Genereaza confusion matrix - un tabel 2x2 care arata:
      - Cate zile de crestere am prezis corect (True Positive)
      - Cate zile de scadere am prezis corect  (True Negative)
      - Cate zile de crestere am prezis gresit (False Negative)
      - Cate zile de scadere am prezis gresit  (False Positive)

    Celulele de pe diagonala = predictii corecte (vrem sa fie mari)
    Celulele in afara diagonalei = greseli (vrem sa fie mici)
    """
    print("\nGeneram confusion matrix...")

    y_real   = predictii["y_real"].values
    y_prezis = predictii["y_prezis"].values

    # Calculam matricea de confuzie
    matrice = confusion_matrix(y_real, y_prezis)

    # Cream figura
    figura, ax = plt.subplots(figsize=(7, 6))
    figura.patch.set_facecolor(FUNDAL)
    ax.set_facecolor(FUNDAL)

    # Desenam heatmap-ul cu seaborn
    sns.heatmap(
        matrice,
        annot=True,           # afisam numerele in celule
        fmt="d",              # format intreg (fara zecimale)
        cmap="YlOrRd",        # paleta de culori
        xticklabels=["Prezis: Scade (0)", "Prezis: Creste (1)"],
        yticklabels=["Real: Scade (0)", "Real: Creste (1)"],
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"}
    )

    ax.set_title("Confusion Matrix - Walk-Forward Validation",
                 color="white", fontsize=14, pad=15)
    ax.set_xlabel("Ce a prezis modelul", color="white", fontsize=12)
    ax.set_ylabel("Ce s-a intamplat in realitate", color="white", fontsize=12)
    ax.tick_params(colors="white")

    # Afisam accuracy si F1 sub grafic
    acc = accuracy_score(y_real, y_prezis)
    f1  = f1_score(y_real, y_prezis, zero_division=0)
    figura.text(0.5, 0.01,
                f"Accuracy: {round(acc, 3)}   |   F1 Score: {round(f1, 3)}",
                ha="center", color=GALBEN, fontsize=12)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("reports/confusion_matrix.png", dpi=150,
                bbox_inches="tight", facecolor=FUNDAL)
    plt.close()

    print("Salvat in: reports/confusion_matrix.png")


# ─────────────────────────────────────────────
# GRAFIC 2: FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def grafic_feature_importance(importanta):
    """
    Genereaza un grafic cu importanta fiecarei features.

    Ce masoara importanta?
    De cate ori XGBoost a folosit acea feature pentru a lua o decizie,
    ponderat de cati itemi trec prin acel split.
    O feature importanta nu inseamna neaparat ca e pozitiva sau negativa -
    inseamna ca modelul se bazeaza mult pe ea.

    Daca sentiment_score apare primul, inseamna ca stirile chiar conteaza!
    Acesta e argumentul principal al proiectului nostru.
    """
    print("\nGeneram grafic feature importance...")

    # Luam primele 12 features (cele mai importante)
    top_features = importanta.sort_values("importanta", ascending=True).tail(12)

    # Coloram diferit features de sentiment vs tehnice
    culori = []
    for feature in top_features["feature"]:
        if "sentiment" in feature:
            culori.append(VERDE)    # verde = features NLP/sentiment
        elif feature in ["rsi_14", "macd", "atr_14"]:
            culori.append(GALBEN)   # galben = indicatori tehnici clasici
        else:
            culori.append(GRI)      # gri = price/volume features

    figura, ax = plt.subplots(figsize=(10, 7))
    figura.patch.set_facecolor(FUNDAL)
    ax.set_facecolor(FUNDAL)

    # Grafic bara orizontal
    bare = ax.barh(top_features["feature"], top_features["importanta"],
                   color=culori, height=0.6)

    # Adaugam valorile la capatul fiecarei bare
    for bara, valoare in zip(bare, top_features["importanta"]):
        ax.text(bara.get_width() + 0.001,
                bara.get_y() + bara.get_height() / 2,
                str(round(valoare, 4)),
                va="center", ha="left", color="white", fontsize=9)

    ax.set_title("Feature Importance - XGBoost", color="white", fontsize=14)
    ax.set_xlabel("Importanta", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="white", alpha=0.1)

    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor=FUNDAL)
    plt.close()

    print("Salvat in: reports/feature_importance.png")


# ─────────────────────────────────────────────
# GRAFIC 3: PERFORMANTA PER FOLD
# ─────────────────────────────────────────────

def grafic_performanta_folduri(folduri):
    """
    Grafic liniar cu accuracy si F1 per fold.

    Util pentru a vedea daca modelul e consistent in timp sau
    daca performanta variaza mult (semn de instabilitate).
    O linie plata = model stabil. O linie cu varfuri mari = instabil.
    """
    print("\nGeneram grafic performanta per fold...")

    figura, ax = plt.subplots(figsize=(10, 5))
    figura.patch.set_facecolor(FUNDAL)
    ax.set_facecolor(FUNDAL)

    folduri_nr = folduri["fold"]

    # Linia de accuracy
    ax.plot(folduri_nr, folduri["accuracy"],
            color=VERDE, marker="o", linewidth=2,
            markersize=7, label="Accuracy")

    # Linia de F1
    ax.plot(folduri_nr, folduri["f1"],
            color=GALBEN, marker="s", linewidth=2,
            markersize=7, label="F1 Score", linestyle="--")

    # Linia de referinta la 0.5 (un model complet aleator ar obtine 0.5)
    ax.axhline(0.5, color=ROSU, linestyle=":",
               alpha=0.7, label="Baseline aleator (0.50)")

    # Banda pentru media accuracy
    acc_medie = folduri["accuracy"].mean()
    acc_std   = folduri["accuracy"].std()
    ax.axhline(acc_medie, color=VERDE, linestyle="-", alpha=0.3)
    ax.fill_between(folduri_nr,
                    acc_medie - acc_std,
                    acc_medie + acc_std,
                    alpha=0.1, color=VERDE,
                    label=f"Accuracy medie ({round(acc_medie, 3)} ± {round(acc_std, 3)})")

    ax.set_title("Performanta per Fold - Walk-Forward Validation",
                 color="white", fontsize=14)
    ax.set_xlabel("Fold (avansare in timp →)", color="white", fontsize=12)
    ax.set_ylabel("Scor", color="white", fontsize=12)
    ax.set_ylim(0.3, 1.0)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="white", alpha=0.1)
    ax.legend(facecolor=FUNDAL, edgecolor="white",
              labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig("reports/fold_performance.png", dpi=150,
                bbox_inches="tight", facecolor=FUNDAL)
    plt.close()

    print("Salvat in: reports/fold_performance.png")


# ─────────────────────────────────────────────
# GRAFIC 4: BACKTEST
# ─────────────────────────────────────────────

def grafic_backtest(features, model, lista_features):
    """
    Simuleaza o strategie de tranzactionare bazata pe semnalele modelului.

    Strategia noastra (Long-Only):
      - Daca modelul prezice 1 (creste maine): cumparam azi
      - Daca modelul prezice 0 (scade maine): stam in cash

    Buy & Hold (referinta):
      - Cumparam la prima zi si tinem pana la final, fara sa facem nimic

    Limitari importante de mentionat la prezentare:
      - Nu include comisioane de tranzactionare
      - Nu include slippage (diferenta dintre pretul dorit si cel executat)
      - Nu este sfat financiar!
    """
    print("\nRulam backtestul...")

    tabel = features.copy()
    tabel = tabel.sort_values("Date").reset_index(drop=True)

    # Facem predictii pe tot setul cu modelul final
    X_tot = tabel[lista_features].values
    tabel["semnal"] = model.predict(X_tot)

    # Randamentul zilnic al S&P 500
    tabel["randament_piata"] = tabel["Close"].pct_change()

    # Randamentul strategiei noastre:
    # Daca semnalul de ieri era 1 → azi suntem pe piata
    # shift(1): semnalul de azi se aplica la randamentul de maine
    tabel["randament_strategie"] = tabel["randament_piata"] * tabel["semnal"].shift(1)

    # Eliminam randurile cu NaN (primele randuri nu au randament)
    tabel = tabel.dropna(subset=["randament_piata", "randament_strategie"])

    # Randament cumulativ: (1 + r1) * (1 + r2) * ... - 1
    # Pornind de la 1, fiecare zi multiplicam cu (1 + randamentul zilei)
    tabel["cumulat_piata"]     = (1 + tabel["randament_piata"]).cumprod()
    tabel["cumulat_strategie"] = (1 + tabel["randament_strategie"]).cumprod()

    # Calculam metricile de performanta
    total_piata     = tabel["cumulat_piata"].iloc[-1] - 1
    total_strategie = tabel["cumulat_strategie"].iloc[-1] - 1

    # Sharpe Ratio: randament mediu / volatilitate, anualizat
    # sqrt(252) anualizaza: 252 zile de tranzactionare pe an
    # Sharpe > 1 = bun, > 2 = excelent
    randamente_zilnice = tabel["randament_strategie"]
    sharpe = (randamente_zilnice.mean() / randamente_zilnice.std()) * np.sqrt(252)

    # Maximum Drawdown: cea mai mare scadere de la un varf
    # Masoara cat de mult pierde strategia in cel mai rau moment
    varf_rulant  = tabel["cumulat_strategie"].cummax()
    drawdown     = (tabel["cumulat_strategie"] - varf_rulant) / varf_rulant
    max_drawdown = drawdown.min()

    # Cat timp am stat pe piata (vs cash)
    zile_pe_piata = tabel["semnal"].shift(1).sum()
    procent_piata = round(zile_pe_piata / len(tabel) * 100, 1)

    print(f"  Randament total piata:     {round(total_piata * 100, 1)}%")
    print(f"  Randament total strategie: {round(total_strategie * 100, 1)}%")
    print(f"  Sharpe Ratio:              {round(sharpe, 2)}")
    print(f"  Maximum Drawdown:          {round(max_drawdown * 100, 1)}%")
    print(f"  Timp pe piata:             {procent_piata}%")

    # Cream graficul
    figura, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={"height_ratios": [3, 1]},
                                       sharex=True)
    figura.patch.set_facecolor(FUNDAL)

    # Subplot 1: Randament cumulativ
    ax1.set_facecolor(FUNDAL)
    ax1.plot(tabel["Date"], tabel["cumulat_piata"],
             color=GRI, linewidth=1.5,
             label=f"Buy & Hold S&P 500 ({round(total_piata * 100, 1)}%)")
    ax1.plot(tabel["Date"], tabel["cumulat_strategie"],
             color=VERDE, linewidth=2,
             label=f"Strategia Modelului ({round(total_strategie * 100, 1)}%)")

    ax1.set_title("Backtest: Strategia Modelului vs Buy & Hold",
                  color="white", fontsize=14)
    ax1.set_ylabel("Randament Cumulativ", color="white", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_visible(False)
    ax1.yaxis.grid(True, color="white", alpha=0.1)
    ax1.legend(facecolor=FUNDAL, edgecolor="white",
               labelcolor="white", fontsize=11)

    # Text cu metrici
    text_metrici = (f"Sharpe: {round(sharpe, 2)}  |  "
                    f"Max Drawdown: {round(max_drawdown * 100, 1)}%  |  "
                    f"Timp pe piata: {procent_piata}%")
    ax1.text(0.5, 0.03, text_metrici,
             transform=ax1.transAxes, ha="center",
             color=GALBEN, fontsize=10)

    # Subplot 2: Drawdown
    ax2.set_facecolor(FUNDAL)
    ax2.fill_between(tabel["Date"], drawdown * 100, 0,
                     color=ROSU, alpha=0.6)
    ax2.set_ylabel("Drawdown %", color="white", fontsize=10)
    ax2.set_xlabel("Data", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_visible(False)
    ax2.yaxis.grid(True, color="white", alpha=0.1)

    # Disclaimer
    figura.text(0.5, 0.001,
                "Backtestul nu include comisioane sau slippage. Nu este sfat financiar.",
                ha="center", color=GRI, fontsize=8, style="italic")

    plt.tight_layout(rect=[0, 0.015, 1, 1])
    plt.savefig("reports/backtest_returns.png", dpi=150,
                bbox_inches="tight", facecolor=FUNDAL)
    plt.close()

    print("Salvat in: reports/backtest_returns.png")

    # Salvam si datele backtestului
    tabel[["Date", "semnal", "randament_piata", "randament_strategie",
           "cumulat_piata", "cumulat_strategie"]].to_csv(
        "data/processed/backtest_data.csv", index=False
    )


# ─────────────────────────────────────────────
# RULAM TOT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("ETAPA 5: EVALUARE SI VIZUALIZARE")
    print("=" * 50)

    # Incarcam datele
    predictii, folduri, importanta, features, model, lista_features = incarca_datele()

    # Generam graficele
    grafic_confusion_matrix(predictii)
    grafic_feature_importance(importanta)
    grafic_performanta_folduri(folduri)
    grafic_backtest(features, model, lista_features)

    print("\n" + "=" * 50)
    print("PROIECT COMPLET!")
    print("\nGrafice generate in reports/:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - fold_performance.png")
    print("  - backtest_returns.png")
    print("\nSucces la prezentare!")
    print("=" * 50)
