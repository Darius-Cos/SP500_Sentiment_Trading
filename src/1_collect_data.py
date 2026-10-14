"""
1_collect_data.py
=================
Ce face acest fisier:
  - Descarca preturile S&P 500 din ultimii 2 ani
  - Descarca titluri de stiri financiare
  - Salveaza totul in fisiere CSV

De ce avem nevoie de aceste date:
  - Preturile ne spun CE s-a intamplat pe piata
  - Stirile ne spun DE CE s-a intamplat
  - Combinand cele doua, modelul poate invata tipare
"""

import os
import time
import feedparser
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# SETARI GENERALE
# ─────────────────────────────────────────────

# Vrem date din ultimii 2 ani
data_de_azi = datetime.today()
data_de_start = data_de_azi - timedelta(days=730)

# Transformam datele in text cu formatul YYYY-MM-DD
# Exemplu: "2024-01-15"
DATA_START = data_de_start.strftime("%Y-%m-%d")
DATA_FINAL = data_de_azi.strftime("%Y-%m-%d")

# Cream folderul unde salvam datele brute
# exist_ok=True inseamna: daca folderul exista deja, nu arunca eroare
os.makedirs("data/raw", exist_ok=True)

print("Perioada de date:", DATA_START, "pana la", DATA_FINAL)




def descarca_preturi():
    """
    Descarca preturile zilnice ale indicelui S&P 500.

    Ce este ^GSPC?
    Este codul (ticker-ul) indicelui S&P 500 pe Yahoo Finance.
    Ca un cod de bare unic pentru acest indice.

    Ce primim inapoi?
    Un tabel cu coloanele: Date, Open, High, Low, Close, Volume
      - Open   = pretul de deschidere al zilei
      - High   = cel mai mare pret din ziua respectiva
      - Low    = cel mai mic pret din ziua respectiva
      - Close  = pretul de inchidere (cel mai important pentru noi)
      - Volume = cate actiuni s-au tranzactionat in ziua respectiva
    """
    print("\nDescarcam preturile S&P 500...")

    # yfinance descarca datele direct de pe Yahoo Finance
    # auto_adjust=True corecteaza preturile pentru split-uri de actiuni
    tabel_preturi = yf.download(
        tickers="^GSPC",
        start=DATA_START,
        end=DATA_FINAL,
        progress=False,
        auto_adjust=True
    )

    # Dupa download, Date este indexul tabelului
    # reset_index() o transforma intr-o coloana normala
    tabel_preturi = tabel_preturi.reset_index()

    # Pastram doar coloanele de care avem nevoie
    tabel_preturi = tabel_preturi[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Transformam coloana Date in format text simplu YYYY-MM-DD
    tabel_preturi["Date"] = pd.to_datetime(tabel_preturi["Date"]).dt.strftime("%Y-%m-%d")

    # Salvam in fisier CSV
    tabel_preturi.to_csv("data/raw/prices_raw.csv", index=False)

    print("Am descarcat", len(tabel_preturi), "zile de preturi")
    print("Salvat in: data/raw/prices_raw.csv")

    return tabel_preturi




# Acestea sunt adresele RSS ale marilor publicatii financiare
# RSS = un format prin care site-urile publica automat titlurile articolelor noi
# Ca un feed de Instagram, dar pentru stiri, si putem citi cu Python
SURSE_STIRI = {
    "Reuters":       "https://feeds.reuters.com/reuters/businessNews",
    "CNBC":          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "MarketWatch":   "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Yahoo Finance": "https://finance.yahoo.com/rss/headline?s=^GSPC",
}


def citeste_un_feed(nume_sursa, adresa_url):
    """
    Citeste un singur feed RSS si returneaza articolele ca lista.

    Ce este un feed RSS?
    Fiecare publicatie mare are o adresa speciala unde publica
    automat titlurile articolelor noi. feedparser citeste acea adresa
    si ne da articolele intr-un format usor de folosit in Python.

    Ce returnam:
    O lista de dictionare, fiecare dictionar = un articol
    Exemplu: {"data": "2024-01-15", "titlu": "Apple profits rise", "sursa": "Reuters"}
    """
    lista_articole = []

    try:
        # feedparser citeste feed-ul RSS de la adresa respectiva
        feed = feedparser.parse(adresa_url)

        # Parcurgem fiecare articol din feed
        for articol in feed.entries:

            # Verificam daca articolul are data publicarii
            # Unele articole nu au data - le sarim
            if not hasattr(articol, "published_parsed"):
                continue

            if articol.published_parsed is None:
                continue

            # published_parsed este un tuplu cu valorile datei
            # Pozitia 0 = anul, 1 = luna, 2 = ziua
            # Pozitia 3 = ora, 4 = minutul, 5 = secunda
            # Exemplu: (2024, 1, 15, 10, 30, 0, ...)
            an      = articol.published_parsed[0]
            luna    = articol.published_parsed[1]
            zi      = articol.published_parsed[2]
            ora     = articol.published_parsed[3]
            minut   = articol.published_parsed[4]
            secunda = articol.published_parsed[5]

            # Cream un obiect datetime din partile de mai sus
            data_publicare = datetime(an, luna, zi, ora, minut, secunda)

            # Transformam in text format YYYY-MM-DD
            data_text = data_publicare.strftime("%Y-%m-%d")

            # Sarim articolele din afara perioadei noastre de interes
            if data_text < DATA_START:
                continue
            if data_text > DATA_FINAL:
                continue

            # Adaugam articolul in lista
            lista_articole.append({
                "data":  data_text,
                "titlu": articol.title.strip(),
                "sursa": nume_sursa
            })

    except Exception as eroare:
        # Daca un feed nu functioneaza, nu oprim tot programul
        # Doar afisam eroarea si continuam cu urmatoarele surse
        print("  Eroare la", nume_sursa, ":", eroare)

    return lista_articole


def descarca_stiri():
    """
    Parcurge toate sursele RSS si combina stirile intr-un singur tabel.
    """
    print("\nDescarcam stiri financiare")

    toate_articolele = []

    for nume, url in SURSE_STIRI.items():
        print("  Citim:", nume, end=" ")

        articole = citeste_un_feed(nume, url)
        toate_articolele.extend(articole)

        print(len(articole), "articole")

        
        time.sleep(0.5)

    # Transformam lista de dictionare intr-un DataFrame pandas
    tabel_stiri = pd.DataFrame(toate_articolele)

    if tabel_stiri.empty:
        print("Nu am gasit niciun articol!")
        return tabel_stiri

    # Eliminam stirile duplicate - acelasi titlu din surse diferite
    tabel_stiri = tabel_stiri.drop_duplicates(subset=["titlu"])

    # Sortam cronologic - de la cele mai vechi la cele mai noi
    tabel_stiri = tabel_stiri.sort_values("data")

    # Resetam indexul dupa sortare
    tabel_stiri = tabel_stiri.reset_index(drop=True)

    # Salvam
    tabel_stiri.to_csv("data/raw/news_raw.csv", index=False)

    print("Am descarcat", len(tabel_stiri), "articole unice")
    print("Salvat in: data/raw/news_raw.csv")

    return tabel_stiri


if __name__ == "__main__":
    print("=" * 50)
    print("ETAPA 1: COLECTARE DATE")
    print("=" * 50)

    preturi = descarca_preturi()
    stiri   = descarca_stiri()

    print("\n" + "=" * 50)
    print("REZULTAT FINAL:")
    print("  Zile de preturi descarcate:", len(preturi))
    print("  Articole de stiri descarcate:", len(stiri))
    print("\nUrmatorul pas: python src/2_sentiment_analysis.py")
    print("=" * 50)
