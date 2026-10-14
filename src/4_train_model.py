"""
4_train_model.py
================
Ce face acest fisier:
  - Incarca tabelul de features din etapa anterioara
  - Gaseste cei mai buni parametri pentru model (grid search)
  - Testeaza modelul corect in timp (walk-forward validation)
  - Antreneaza modelul final pe toate datele disponibile
  - Salveaza modelul si rezultatele

Ce este XGBoost?
  Un algoritm care construieste sute de arbori de decizie mici,
  unul dupa altul. Fiecare arbore invata din greselile celui anterior.
  La final, decizia colectiva a tuturor arborilor e mult mai buna
  decat a oricarui arbore individual.

De ce walk-forward validation si nu train/test clasic?
  Pe date financiare NU putem amesteca datele aleator.
  Daca antrenezi pe 2022 si testezi pe 2021, modelul "stie viitorul".
  Walk-forward merge mereu inainte in timp: antrenam pe trecut,
  testam pe viitorul imediat, exact cum am face in realitate.
"""

import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Seed pentru reproducibilitate
# Cu acelasi seed, rezultatele sunt identice la fiecare rulare
SEED = 42

# Fereastra initiala de antrenament = ~1 an de tranzactionare
ZILE_ANTRENAMENT_INITIAL = 252

# Avansam cu ~3 luni la fiecare pas
ZILE_PAS = 63

# Testam pe ~3 luni la fiecare pas
ZILE_TEST = 63


# ─────────────────────────────────────────────
# PASUL 1: CITIM DATELE
# ─────────────────────────────────────────────

def citeste_features():
    """
    Citeste tabelul de features si il pregateste pentru modelare.
    Returneaza tabelul, lista de features si numele coloanei target.
    """
    print("Citim features.csv...")

    if not os.path.exists("data/processed/features.csv"):
        print("EROARE: features.csv nu exista!")
        print("Ruleaza mai intai: python src/3_feature_engineering.py")
        exit()

    tabel = pd.read_csv("data/processed/features.csv")

    # Sortam cronologic - esential pentru walk-forward
    tabel = tabel.sort_values("Date")
    tabel = tabel.reset_index(drop=True)

    # Coloanele care NU sunt features pentru model
    coloane_excluse = {"Date", "Close", "target"}

    # Toate celelalte coloane sunt features
    lista_features = [col for col in tabel.columns if col not in coloane_excluse]

    print("Numar de zile:", len(tabel))
    print("Numar de features:", len(lista_features))
    print("Perioada:", tabel["Date"].iloc[0], "pana la", tabel["Date"].iloc[-1])

    return tabel, lista_features, "target"


# ─────────────────────────────────────────────
# PASUL 2: GASIM CEI MAI BUNI PARAMETRI
# ─────────────────────────────────────────────

def gaseste_parametri_optimi(tabel, lista_features, coloana_target):
    """
    Testam diferite combinatii de parametri si o alegem pe cea mai buna.
    Aceasta se numeste grid search.

    Ce sunt parametrii XGBoost?
      - n_estimators:  cati arbori construim (mai multi = mai precis, mai lent)
      - max_depth:     cat de adanc poate creste un arbore
                       (adanc = invata mai mult, dar risca sa memoreze datele)
      - learning_rate: cat de mult contribuie fiecare arbore nou
                       (mic = invatare prudenta, mai stabila)
      - subsample:     ce procent din date folosim pentru fiecare arbore
                       (mai mic = mai multa diversitate intre arbori)

    De ce testam parametrii pe primul fold?
    Ca sa nu "contaminam" evaluarea finala. Parametrii sunt alesi
    exclusiv pe date din trecut, niciodata pe datele de test.
    """
    print("\nCautam cei mai buni parametri (grid search)...")

    # Folosim primele 80% din fereastra initiala pentru antrenament in grid search
    # si restul de 20% pentru validare
    limita = int(ZILE_ANTRENAMENT_INITIAL * 0.8)

    X_antrenament = tabel[lista_features].values[:limita]
    y_antrenament = tabel[coloana_target].values[:limita]

    X_validare = tabel[lista_features].values[limita:ZILE_ANTRENAMENT_INITIAL]
    y_validare = tabel[coloana_target].values[limita:ZILE_ANTRENAMENT_INITIAL]

    # Calculam dezechilibrul de clase pentru scale_pos_weight
    # Piata creste cam 55% din zile si scade 45%
    # scale_pos_weight corecteaza acest dezechilibru
    # Formula: numar_zile_scadere / numar_zile_crestere
    nr_scaderi  = (y_antrenament == 0).sum()
    nr_cresteri = (y_antrenament == 1).sum()
    scale_pw = nr_scaderi / nr_cresteri

    # Toate combinatiile de parametri pe care le testam
    combinatii = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 1.0},
    ]

    print("Testam", len(combinatii), "combinatii de parametri...")

    cel_mai_bun_f1     = -1
    cei_mai_buni_param = {}

    for parametri in combinatii:
        # Cream si antrenam modelul cu parametrii curenti
        model = XGBClassifier(
            **parametri,
            scale_pos_weight=scale_pw,
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_antrenament, y_antrenament)

        # Testam pe datele de validare
        predictii = model.predict(X_validare)
        f1 = f1_score(y_validare, predictii, zero_division=0)

        # Pastram parametrii cu cel mai bun F1
        if f1 > cel_mai_bun_f1:
            cel_mai_bun_f1     = f1
            cei_mai_buni_param = parametri

    print("Cei mai buni parametri gasiti:", cei_mai_buni_param)
    print("F1 pe validare:", round(cel_mai_bun_f1, 3))

    return cei_mai_buni_param


# ─────────────────────────────────────────────
# PASUL 3: WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def walk_forward_validation(tabel, lista_features, coloana_target, parametri):
    """
    Evalueaza modelul in mod corect pe date de timp.

    Cum functioneaza pas cu pas:
      Fold 1: Antrenam pe zilele 1-252   → testam pe zilele 253-315
      Fold 2: Antrenam pe zilele 1-315   → testam pe zilele 316-378
      Fold 3: Antrenam pe zilele 1-378   → testam pe zilele 379-441
      ... si tot asa pana la final

    La fiecare pas:
      - Fereastra de antrenament creste (include si datele precedente de test)
      - Testam mereu pe date pe care modelul nu le-a vazut niciodata
      - Exact cum ai folosi modelul in realitate: antrenezi pe trecut,
        prezici viitorul

    De ce e important sa nu amestecam datele?
      Daca dam modelului date din 2024 la antrenament si ii cerem
      sa prezica date din 2023, el "stie" deja ce s-a intamplat.
      Rezultatele ar arata foarte bine, dar in realitate modelul nu stie nimic.
    """
    print("\nIncepem walk-forward validation...")

    n = len(tabel)

    # Calculam scale_pos_weight pe tot setul
    toate_targeturi = tabel[coloana_target].values
    nr_scaderi  = (toate_targeturi == 0).sum()
    nr_cresteri = (toate_targeturi == 1).sum()
    scale_pw = nr_scaderi / nr_cresteri

    print("Dezechilibru clase - scale_pos_weight:", round(scale_pw, 2))

    # Listele in care colectam rezultatele
    rezultate_folduri = []
    toate_y_real      = []
    toate_y_prezis    = []

    numar_fold = 1
    sfarsit_antrenament = ZILE_ANTRENAMENT_INITIAL

    # Continuam cat timp mai avem zile de test
    while sfarsit_antrenament + ZILE_TEST <= n:

        # Datele de antrenament: de la inceput pana la sfarsit_antrenament
        X_antrenament = tabel[lista_features].values[:sfarsit_antrenament]
        y_antrenament = tabel[coloana_target].values[:sfarsit_antrenament]

        # Datele de test: urmatoarele ZILE_TEST zile
        inceput_test = sfarsit_antrenament
        sfarsit_test = min(sfarsit_antrenament + ZILE_TEST, n)

        X_test = tabel[lista_features].values[inceput_test:sfarsit_test]
        y_test = tabel[coloana_target].values[inceput_test:sfarsit_test]

        # Cream si antrenam modelul pe fereastra curenta
        model = XGBClassifier(
            **parametri,
            scale_pos_weight=scale_pw,
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_antrenament, y_antrenament)

        # Facem predictii pe fereastra de test
        y_prezis = model.predict(X_test)

        # Calculam metricile pentru acest fold
        accuracy = accuracy_score(y_test, y_prezis)
        f1       = f1_score(y_test, y_prezis, zero_division=0)

        # Data de inceput si sfarsit a perioadei de test
        data_inceput_test = tabel["Date"].iloc[inceput_test]
        data_sfarsit_test = tabel["Date"].iloc[sfarsit_test - 1]

        print(f"  Fold {numar_fold}: antrenament={sfarsit_antrenament} zile | "
              f"test={data_inceput_test} → {data_sfarsit_test} | "
              f"accuracy={round(accuracy, 3)} | f1={round(f1, 3)}")

        # Salvam rezultatele
        rezultate_folduri.append({
            "fold":       numar_fold,
            "zile_antrenament": sfarsit_antrenament,
            "data_start": data_inceput_test,
            "data_final": data_sfarsit_test,
            "accuracy":   round(accuracy, 4),
            "f1":         round(f1, 4),
        })

        # Adaugam predictiile la listele totale (pentru raportul final)
        toate_y_real.extend(y_test.tolist())
        toate_y_prezis.extend(y_prezis.tolist())

        # Avansam fereastra de antrenament
        sfarsit_antrenament = sfarsit_antrenament + ZILE_PAS
        numar_fold = numar_fold + 1

    return rezultate_folduri, toate_y_real, toate_y_prezis


# ─────────────────────────────────────────────
# PASUL 4: MODEL FINAL PE TOATE DATELE
# ─────────────────────────────────────────────

def antreneaza_model_final(tabel, lista_features, coloana_target, parametri):
    """
    Antrenam modelul final pe TOATE datele disponibile.

    De ce antrenam din nou pe toate datele?
    Walk-forward ne-a spus CAT DE BINE functioneaza modelul.
    Acum ca stim ca performanta e acceptabila, antrenam pe tot setul
    ca sa avem cel mai bun model posibil pentru predictii viitoare.
    Mai multe date = model mai bun.

    Modelul salvat la final este cel care ar fi folosit "in productie".
    """
    print("\nAntrenam modelul final pe toate datele...")

    X_tot = tabel[lista_features].values
    y_tot = tabel[coloana_target].values

    nr_scaderi  = (y_tot == 0).sum()
    nr_cresteri = (y_tot == 1).sum()
    scale_pw = nr_scaderi / nr_cresteri

    model_final = XGBClassifier(
        **parametri,
        scale_pos_weight=scale_pw,
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0
    )
    model_final.fit(X_tot, y_tot)

    # Salvam modelul cu pickle
    # pickle transforma obiectul Python intr-un fisier binar
    # La etapa 5, incarcam fisierul si modelul e gata de folosit
    with open("models/xgb_classifier.pkl", "wb") as fisier:
        pickle.dump(model_final, fisier)

    # Salvam si lista de features in aceeasi ordine
    # Importanta: la predictii, datele trebuie sa fie in aceeasi ordine
    with open("models/feature_cols.pkl", "wb") as fisier:
        pickle.dump(lista_features, fisier)

    print("Model salvat in: models/xgb_classifier.pkl")
    return model_final


# ─────────────────────────────────────────────
# PASUL 5: RAPORT FINAL
# ─────────────────────────────────────────────

def afiseaza_raport(rezultate_folduri, y_real, y_prezis, model, lista_features):
    """
    Afiseaza un raport complet si salveaza rezultatele pentru etapa 5.
    """
    print("\n" + "=" * 50)
    print("RAPORT WALK-FORWARD VALIDATION")
    print("=" * 50)

    tabel_folduri = pd.DataFrame(rezultate_folduri)

    print("\nFolduri completate:", len(tabel_folduri))
    print("Accuracy medie:   ", round(tabel_folduri["accuracy"].mean(), 3),
          "(±", round(tabel_folduri["accuracy"].std(), 3), ")")
    print("F1 mediu:         ", round(tabel_folduri["f1"].mean(), 3),
          "(±", round(tabel_folduri["f1"].std(), 3), ")")

    print("\nRaport detaliat pe toate foldurile:")
    print(classification_report(y_real, y_prezis,
                                target_names=["Scade (0)", "Creste (1)"]))

    # Importanta features
    importanta = model.feature_importances_
    tabel_importanta = pd.DataFrame({
        "feature":    lista_features,
        "importanta": importanta
    })
    tabel_importanta = tabel_importanta.sort_values("importanta", ascending=False)

    print("Top 10 features dupa importanta:")
    for i, rand in tabel_importanta.head(10).iterrows():
        bara = "█" * int(rand["importanta"] * 200)
        print(f"  {rand['feature']:<35} {bara} {round(rand['importanta'], 4)}")

    # Salvam fisierele pentru etapa 5
    tabel_folduri.to_csv("data/processed/walk_forward_results.csv", index=False)
    tabel_importanta.to_csv("data/processed/feature_importance.csv", index=False)

    tabel_predictii = pd.DataFrame({"y_real": y_real, "y_prezis": y_prezis})
    tabel_predictii.to_csv("data/processed/predictions.csv", index=False)

    print("\nFisiere salvate pentru etapa 5:")
    print("  - data/processed/walk_forward_results.csv")
    print("  - data/processed/feature_importance.csv")
    print("  - data/processed/predictions.csv")


# ─────────────────────────────────────────────
# RULAM TOT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("ETAPA 4: ANTRENARE MODEL (XGBoost)")
    print("=" * 50)

    # 1. Citim datele
    tabel, lista_features, coloana_target = citeste_features()

    # 2. Gasim parametrii optimi
    parametri = gaseste_parametri_optimi(tabel, lista_features, coloana_target)

    # 3. Walk-forward validation
    rezultate, y_real, y_prezis = walk_forward_validation(
        tabel, lista_features, coloana_target, parametri
    )

    # 4. Model final pe toate datele
    model_final = antreneaza_model_final(tabel, lista_features, coloana_target, parametri)

    # 5. Raport
    afiseaza_raport(rezultate, y_real, y_prezis, model_final, lista_features)

    print("\n" + "=" * 50)
    print("ETAPA 4 COMPLETA!")
    print("Urmatorul pas: python src/5_evaluate.py")
    print("=" * 50)
