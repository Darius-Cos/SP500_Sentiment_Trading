"""
3_feature_engineering.py
========================
Ce face acest fisier:
  - Combina preturile cu sentimentul zilnic
  - Calculeaza indicatori tehnici din preturi (RSI, ATR, MACD etc.)
  - Adauga informatii de sentiment din zilele anterioare
  - Defineste ce vrem sa prezica modelul (target-ul)
  - Salveaza tabelul final gata pentru antrenament

De ce feature engineering?
  Modelul XGBoost nu poate citi direct preturile brute sau titlurile stirilor.
  El primeste un tabel de numere si cauta tipare in el.
  Cu cat ii dam mai multe perspective diferite asupra datelor,
  cu atat are mai multe sanse sa gaseasca ceva util.

Regula de aur: tot ce calculam trebuie sa fie disponibil INAINTE
de ziua pe care vrem sa o prezica. Altfel, dam modelului informatii
din viitor, si rezultatele sunt false.
"""

import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# Fereastra scurta = o saptamana de tranzactionare
ZILE_SCURT = 5

# Fereastra lunga = o luna de tranzactionare
ZILE_LUNG = 20


# ─────────────────────────────────────────────
# PASUL 1: CITIM SI COMBINAM DATELE
# ─────────────────────────────────────────────

def citeste_si_combina():
    """
    Citeste preturile si sentimentul si le uneste pe coloana Date.

    Folosim LEFT JOIN pe preturi:
    Pastram toate zilele de tranzactionare, chiar daca nu avem stiri
    pentru acea zi. In zilele fara stiri, sentimentul va fi 0 (neutru).
    """
    print("Citim datele din etapele anterioare...")

    # Verificam ca fisierele exista
    if not os.path.exists("data/raw/prices_raw.csv"):
        print("EROARE: prices_raw.csv nu exista!")
        print("Ruleaza mai intai: python src/1_collect_data.py")
        exit()

    if not os.path.exists("data/processed/sentiment_daily.csv"):
        print("EROARE: sentiment_daily.csv nu exista!")
        print("Ruleaza mai intai: python src/2_sentiment_analysis.py")
        exit()

    # Citim preturile
    preturi = pd.read_csv("data/raw/prices_raw.csv")

    # Citim sentimentul zilnic
    sentiment = pd.read_csv("data/processed/sentiment_daily.csv")

    # Unim cele doua tabele pe coloana Date
    # how="left" inseamna: pastram toate randurile din preturi
    # chiar daca nu au corespondent in sentiment
    tabel = pd.merge(preturi, sentiment, on="Date", how="left")

    # Sortam cronologic (de la trecut la prezent)
    # Esential pentru calculul indicatorilor tehnici
    tabel = tabel.sort_values("Date")
    tabel = tabel.reset_index(drop=True)

    # Ne asiguram ca coloanele de preturi sunt numere (nu text)
    # yfinance uneori le salveaza ca string in anumite versiuni de Python
    for coloana in ["Open", "High", "Low", "Close", "Volume"]:
        if coloana in tabel.columns:
            tabel[coloana] = pd.to_numeric(tabel[coloana], errors="coerce")

    print("Zile de tranzactionare:", len(tabel))
    print("Zile cu date de sentiment:", tabel["sentiment_score"].notna().sum())
    print("Zile fara stiri:", tabel["sentiment_score"].isna().sum())

    return tabel


# ─────────────────────────────────────────────
# PASUL 2: INDICATORI TEHNICI DIN PRETURI
# ─────────────────────────────────────────────

def calculeaza_rsi(coloana_close, numar_zile=14):
    """
    Calculeaza RSI - Relative Strength Index.

    Ce este RSI?
    Masoara daca piata a crescut prea mult sau prea putin recent.
    Valori intre 0 si 100:
      - Peste 70: piata e "supraincalzita", probabil urmeaza o scadere
      - Sub 30:   piata e "supravanduta",  probabil urmeaza o revenire

    Cum se calculeaza?
    1. Calculam diferenta de pret fata de ziua anterioara
    2. Separam zilele pozitive (cresteri) de cele negative (scaderi)
    3. Calculam media cresterilor si media scaderilor pe ultimele 14 zile
    4. RSI = 100 - (100 / (1 + media_cresteri / media_scaderi))
    """

    # Diferenta de pret fata de ziua precedenta
    # diff() calculeaza: valoarea_curenta - valoarea_precedenta
    diferente = coloana_close.diff()

    # Zilele cu crestere: diferenta pozitiva
    # clip(lower=0) inseamna: valorile negative devin 0
    cresteri = diferente.clip(lower=0)

    # Zilele cu scadere: diferenta negativa (o transformam in pozitiva)
    # clip(upper=0) inseamna: valorile pozitive devin 0
    # Negam rezultatul ca sa fie pozitiv
    scaderi = -diferente.clip(upper=0)

    # Media mobila exponentiala a cresterilor si scaderilor
    # ewm = exponential weighted mean (media ponderata exponential)
    # Zilele recente conteaza mai mult decat cele vechi
    medie_cresteri = cresteri.ewm(com=numar_zile - 1, min_periods=numar_zile).mean()
    medie_scaderi  = scaderi.ewm(com=numar_zile - 1, min_periods=numar_zile).mean()

    # Calculam RSI
    # Adaugam 0.0001 ca sa evitam impartirea la zero
    rs  = medie_cresteri / (medie_scaderi + 0.0001)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculeaza_atr(coloana_high, coloana_low, coloana_close, numar_zile=14):
    """
    Calculeaza ATR - Average True Range.

    Ce este ATR?
    Masoara cat de volatila este piata - cat de mult oscileaza pretul zilnic.
    ATR mare = piata instabila, ATR mic = piata calma.

    True Range = cel mai mare dintre:
      - High - Low (diferenta dintre max si min ale zilei)
      - |High - Close_ieri| (gap fata de inchiderea de ieri, sus)
      - |Low  - Close_ieri| (gap fata de inchiderea de ieri, jos)

    ATR = media True Range pe ultimele 14 zile
    """

    # Pretul de inchidere de ieri
    # shift(1) muta coloana "in jos" cu o pozitie
    close_ieri = coloana_close.shift(1)

    # Calculam cele 3 componente ale True Range
    varianta1 = coloana_high - coloana_low
    varianta2 = (coloana_high - close_ieri).abs()
    varianta3 = (coloana_low  - close_ieri).abs()

    # Cream un tabel cu cele 3 variante si luam maximul pe fiecare rand
    tabel_variante = pd.concat([varianta1, varianta2, varianta3], axis=1)
    true_range = tabel_variante.max(axis=1)

    # ATR = media mobila simpla a True Range
    atr = true_range.rolling(numar_zile).mean()

    return atr


def adauga_indicatori_tehnici(tabel):
    """
    Adauga toti indicatorii tehnici la tabel.
    """
    print("\nCalculam indicatori tehnici...")

    close  = tabel["Close"]
    high   = tabel["High"]
    low    = tabel["Low"]
    volume = tabel["Volume"]

    # Randamentul zilnic: cat % a crescut/scazut pretul fata de ieri
    # pct_change() = (valoare_curenta - valoare_anterioara) / valoare_anterioara
    tabel["randament_zilnic"] = close.pct_change()

    # Randamentul pe ultimele 5 si 20 de zile
    # pct_change(5) = cat % s-a schimbat pretul in ultimele 5 zile
    tabel["randament_5zile"]  = close.pct_change(periods=5)
    tabel["randament_20zile"] = close.pct_change(periods=20)

    # RSI pe 14 zile
    tabel["rsi_14"] = calculeaza_rsi(close, numar_zile=14)

    # Volatilitate = cat de mult variaza randamentul zilnic
    # rolling(5).std() = deviatia standard pe ultimele 5 zile
    # Deviatia standard masoara cat de imprastiate sunt valorile
    tabel["volatilitate_5zile"]  = tabel["randament_zilnic"].rolling(ZILE_SCURT).std()
    tabel["volatilitate_20zile"] = tabel["randament_zilnic"].rolling(ZILE_LUNG).std()

    # ATR pe 14 zile
    tabel["atr_14"] = calculeaza_atr(high, low, close, numar_zile=14)

    # MACD = diferenta dintre doua medii mobile exponentiale
    # Media pe 12 zile (termen scurt) vs media pe 26 zile (termen lung)
    # Cand MACD > 0: trendul scurt e mai puternic decat cel lung = semnal pozitiv
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    tabel["macd"] = ema_12 - ema_26

    # Raportul volumului fata de media sa pe 20 de zile
    # Daca volumul e dublu fata de medie, ceva important s-a intamplat
    tabel["raport_volum"] = volume / volume.rolling(ZILE_LUNG).mean()

    # Pretul fata de media mobila pe 20 de zile
    # Daca > 1: pretul e deasupra mediei (trend pozitiv)
    # Daca < 1: pretul e sub medie (trend negativ)
    medie_20 = close.rolling(ZILE_LUNG).mean()
    tabel["pret_vs_medie20"] = close / medie_20

    print("Indicatori tehnici calculati!")
    return tabel


# ─────────────────────────────────────────────
# PASUL 3: FEATURES DE SENTIMENT
# ─────────────────────────────────────────────

def adauga_features_sentiment(tabel):
    """
    Adaugam versiuni multiple ale sentimentului pentru a capta trendul.

    De ce lag-uri?
    Sentimentul de azi influenteaza pretul de maine.
    Dar sentimentul de ieri, de alaltaieri etc. conteaza si el.
    Lag-urile ii permit modelului sa vada trendul pe mai multe zile.

    De ce medii mobile?
    O singura zi de stiri negative poate fi zgomot intamplator.
    Media pe 5 zile e un semnal mai stabil si mai de incredere.
    """
    print("\nAdaugam features de sentiment...")

    # Inlocuim zilele fara stiri cu 0 (neutru)
    # Lipsa stirilor nu inseamna sentiment negativ sau pozitiv
    tabel["sentiment_score"] = tabel["sentiment_score"].fillna(0)
    tabel["article_count"]   = tabel["article_count"].fillna(0)

    # Lag-uri: sentimentul din zilele anterioare
    # shift(1) muta coloana "in jos" cu 1 pozitie
    # Adica: randul de azi primeste valoarea de ieri
    tabel["sentiment_lag_1zi"] = tabel["sentiment_score"].shift(1)
    tabel["sentiment_lag_2zi"] = tabel["sentiment_score"].shift(2)
    tabel["sentiment_lag_3zi"] = tabel["sentiment_score"].shift(3)

    # Media sentimentului pe ultima saptamana si ultima luna
    tabel["sentiment_medie_5zi"]  = tabel["sentiment_score"].rolling(ZILE_SCURT).mean()
    tabel["sentiment_medie_20zi"] = tabel["sentiment_score"].rolling(ZILE_LUNG).mean()

    # Schimbarea sentimentului fata de ieri
    # Un salt brusc de la negativ la pozitiv poate fi un semnal important
    tabel["schimbare_sentiment"] = tabel["sentiment_score"].diff()

    print("Features de sentiment adaugate!")
    return tabel


# ─────────────────────────────────────────────
# PASUL 4: DEFINIM TARGETUL
# ─────────────────────────────────────────────

def adauga_target(tabel):
    """
    Definim ce vrem sa prezica modelul.

    Intrebarea: Va fi pretul de inchidere de MAINE mai mare decat cel de AZI?
      1 = da, piata creste
      0 = nu, piata scade sau stagneaza

    Cum calculam?
    pct_change() calculeaza schimbarea procentuala fata de ziua precedenta.
    shift(-1) muta coloana "in sus" cu o pozitie.
    Rezultat: pentru fiecare zi, avem randamentul zilei URMATOARE.
    Asta e corect: vrem sa stim ce se intampla MAINE bazandu-ne pe AZI.

    ATENTIE: shift(-1) este operatia critica a intregului proiect.
    Daca nu o facem, dam modelului raspunsul direct si el nu invata nimic.
    """
    print("\nDefinim targetul...")

    # Randamentul zilei de maine
    # pct_change() = (pret_curent - pret_ieri) / pret_ieri
    # shift(-1) muta valorile "in sus", deci valoarea de la randul i
    # devine randamentul zilei i+1 (a doua zi)
    tabel["randament_maine"] = tabel["Close"].pct_change().shift(-1)

    # Binarizam: 1 daca creste, 0 daca scade sau stagneaza
    # (tabel["randament_maine"] > 0) returneaza True/False
    # .astype(int) transforma True in 1 si False in 0
    tabel["target"] = (tabel["randament_maine"] > 0).astype(int)

    # Afisam distributia targetului
    nr_creste = tabel["target"].sum()
    nr_scade  = len(tabel) - nr_creste - tabel["target"].isna().sum()

    print("Zile in care piata a CRESCUT:", nr_creste,
          "(", round(nr_creste / len(tabel) * 100, 1), "% )")
    print("Zile in care piata a SCAZUT: ", nr_scade,
          "(", round(nr_scade / len(tabel) * 100, 1), "% )")

    return tabel


# ─────────────────────────────────────────────
# PASUL 5: CURATAM SI SALVAM
# ─────────────────────────────────────────────

def curata_si_salveaza(tabel):
    """
    Eliminam coloanele brute si randurile cu prea multe valori lipsa.
    Salvam tabelul final.
    """
    print("\nCuratam tabelul final...")

    # Eliminam coloanele brute - avem deja derivatele lor ca features
    coloane_de_sters = ["Open", "High", "Low", "Volume",
                        "randament_maine", "pct_pozitiv", "pct_negativ"]

    # Stergem doar coloanele care chiar exista in tabel
    for coloana in coloane_de_sters:
        if coloana in tabel.columns:
            tabel = tabel.drop(columns=[coloana])

    # Eliminam ultimul rand (nu are target - nu stim ce se intampla maine)
    tabel = tabel.dropna(subset=["target"])

    # Eliminam primele randuri unde indicatorii nu s-au putut calcula inca
    # (primele 26 de zile nu au suficient istoric pentru MACD de exemplu)
    # Cream o lista cu coloanele de features (toate mai putin Date, Close, target)
    coloane_features = [col for col in tabel.columns
                       if col not in ["Date", "Close", "target"]]

    # Calculam ce procent din features lipseste pentru fiecare rand
    procent_lipsa = tabel[coloane_features].isna().mean(axis=1)

    # Pastram doar randurile unde mai putin de 30% din features lipsesc
    tabel = tabel[procent_lipsa < 0.3].copy()

    # Pentru valorile lipsa izolate ramase, le umplem cu valoarea anterioara
    tabel[coloane_features] = tabel[coloane_features].ffill()

    # Resetam indexul
    tabel = tabel.reset_index(drop=True)

    # Salvam
    tabel.to_csv("data/processed/features.csv", index=False)

    print("Tabel final:", len(tabel), "zile x", len(coloane_features), "features")
    print("Salvat in: data/processed/features.csv")
    print("\nFeatures incluse:")
    for f in sorted(coloane_features):
        print("  -", f)

    return tabel


# ─────────────────────────────────────────────
# RULAM TOT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("ETAPA 3: FEATURE ENGINEERING")
    print("=" * 50)

    tabel = citeste_si_combina()
    tabel = adauga_indicatori_tehnici(tabel)
    tabel = adauga_features_sentiment(tabel)
    tabel = adauga_target(tabel)
    tabel = curata_si_salveaza(tabel)

    print("\n" + "=" * 50)
    print("ETAPA 3 COMPLETA!")
    print("Urmatorul pas: python src/4_train_model.py")
    print("=" * 50)