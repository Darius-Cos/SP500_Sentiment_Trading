"""
2_sentiment_analysis.py
=======================
Ce face acest fisier:
  - Ia fiecare titlu de stire din fisierul news_raw.csv
  - Il trimite catre modelul FinBERT
  - FinBERT returneaza: pozitiv, negativ sau neutru + cat de sigur e
  - La final calculam un scor mediu per zi

De ce FinBERT si nu ceva mai simplu?
  Instrumentele simple numara cuvinte pozitive si negative.
  Dar in finante, contextul conteaza enorm.
  "Earnings miss" suna neutru dar e negativ in finante.
  FinBERT a fost antrenat special pe texte financiare si intelege asta.
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────
# SETARI
# ─────────────────────────────────────────────

os.makedirs("data/processed", exist_ok=True)

# Numele modelului pe Hugging Face
# Hugging Face este un site unde cercetatorii publica modele AI gratuit
# ProsusAI/finbert este cel mai popular model de sentiment financiar
NUME_MODEL = "ProsusAI/finbert"

# Cate titluri procesam odata
# 16 = un numar bun pentru un laptop obisnuit
MARIME_BATCH = 16


# ─────────────────────────────────────────────
# PASUL 1: INCARCAM MODELUL
# ─────────────────────────────────────────────

def incarca_modelul():
    """
    Descarca si incarca FinBERT din internet.

    Prima rulare: descarca ~440MB (se face o singura data, apoi e in cache)
    Urmatoarele rulari: foloseste versiunea salvata local, instant

    Ce sunt tokenizer si model?
      - Tokenizer = transforma textul in numere pe care modelul le intelege
        Exemplu: "profit rises" -> [4921, 7835] (coduri numerice)
      - Model = reteaua neuronala care analizeaza acele numere si da verdictul
    """
    print("Incarcam modelul FinBERT...")
    print("(Prima rulare descarca ~440MB - normal, se face o singura data)")

    # Descarcam tokenizer-ul
    tokenizer = AutoTokenizer.from_pretrained(NUME_MODEL)

    # Descarcam modelul
    model = AutoModelForSequenceClassification.from_pretrained(NUME_MODEL)

    # Verificam daca avem GPU (placa video NVIDIA)
    # GPU = procesare de ~10x mai rapida decat CPU
    # Daca nu ai GPU, merge si pe CPU, doar mai incet
    if torch.cuda.is_available():
        device = "cuda"
        print("Folosim GPU pentru procesare rapida")
    else:
        device = "cpu"
        print("Folosim CPU (mai lent, dar merge)")

    # Mutam modelul pe device-ul ales
    model = model.to(device)

    # Punem modelul in modul de evaluare (nu de antrenament)
    # In modul de evaluare, modelul nu isi modifica parametrii
    model.eval()

    print("Model incarcat cu succes!")
    return tokenizer, model, device


# ─────────────────────────────────────────────
# PASUL 2: ANALIZAM SENTIMENTUL
# ─────────────────────────────────────────────

def analizeaza_un_batch(lista_titluri, tokenizer, model, device):
    """
    Analizeaza un grup de titluri si returneaza sentimentul fiecaruia.

    Cum functioneaza pe scurt:
      1. Tokenizare: transformam titlurile in numere
      2. Trimitem numerele prin model
      3. Modelul returneaza 3 scoruri pentru fiecare titlu:
         cat de probabil e sa fie pozitiv, negativ, neutru
      4. Alegem sentimentul cu probabilitatea cea mai mare

    Parametri:
      lista_titluri = o lista cu titluri de stiri (max 16)
      tokenizer     = instrumentul care transforma text in numere
      model         = FinBERT
      device        = "cuda" sau "cpu"

    Returnam:
      O lista de dictionare cu sentimentul fiecarui titlu
    """

    # Tokenizare: transformam titlurile in numere
    # padding=True: adaugam zerouri ca toate titlurile sa aiba aceeasi lungime
    # truncation=True: taiem titlurile prea lungi (max 128 de cuvinte)
    # return_tensors="pt": returnam in format PyTorch (pt = PyTorch)
    input_tokenizat = tokenizer(
        lista_titluri,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Mutam datele pe acelasi device ca modelul (CPU sau GPU)
    input_pe_device = {}
    for cheie, valoare in input_tokenizat.items():
        input_pe_device[cheie] = valoare.to(device)

    # torch.no_grad() inseamna: nu calcula gradienti
    # La analiza (nu antrenament) nu avem nevoie de gradienti
    # Economiseste memorie si e mai rapid
    with torch.no_grad():
        rezultat_model = model(
                                input_ids=input_pe_device["input_ids"],
                                attention_mask=input_pe_device["attention_mask"],
                                token_type_ids=input_pe_device["token_type_ids"]
                            )

    # rezultat_model.logits sunt scorurile brute ale modelului
    # Sunt numere care pot fi orice valoare, pozitiva sau negativa
    # Trebuie sa le transformam in probabilitati (intre 0 si 1)
    logits = rezultat_model.logits

    # Softmax transforma scorurile brute in probabilitati
    # Exemplu: [-1.2, 2.3, 0.5] -> [0.05, 0.82, 0.13]
    # Suma probabilitatilor = intotdeauna 1.0
    probabilitati = torch.softmax(logits, dim=1)

    # Mutam pe CPU si transformam in lista Python obisnuita
    probabilitati = probabilitati.cpu().tolist()

    # Etichetele in ordinea in care le-a invatat FinBERT
    # IMPORTANT: aceasta ordine e specifica modelului ProsusAI/finbert
    etichete = ["pozitiv", "negativ", "neutru"]

    # Construim rezultatul pentru fiecare titlu
    rezultate = []
    for prob in probabilitati:

        # prob este o lista cu 3 valori: [prob_pozitiv, prob_negativ, prob_neutru]
        # Gasim indexul valorii maxime
        index_maxim = prob.index(max(prob))

        # Eticheta cu probabilitatea cea mai mare
        sentiment = etichete[index_maxim]

        # Probabilitatea cea mai mare = cat de sigur e modelul
        siguranta = prob[index_maxim]

        rezultate.append({
            "sentiment":  sentiment,
            "siguranta":  siguranta,
            "prob_poz":   prob[0],
            "prob_neg":   prob[1],
            "prob_neu":   prob[2],
        })

    return rezultate


def analizeaza_toate_stirile(tabel_stiri, tokenizer, model, device):
    """
    Analizeaza toate titlurile din tabel, batch cu batch.

    De ce batch-uri?
    Nu putem trimite 5000 de titluri odata - nu incap in memorie.
    Le trimitem grupuri de cate 16.

    """
    print("\nAnalizam sentimentul titlurilor...")
    print("Numar total de titluri:", len(tabel_stiri))

    toate_titlurile = tabel_stiri["titlu"].tolist()
    toate_rezultatele = []

    # Parcurgem titlurile in grupuri de MARIME_BATCH
    # range(start, stop, pas) - mergem din 16 in 16
    for i in range(0, len(toate_titlurile), MARIME_BATCH):

        # Luam un grup de titluri
        # Daca suntem la final si au ramas mai putine de 16, lua ce a ramas
        # Cand facem slice nu o sa iasa out of index daca e peste ia cate elemente mai raman
        grup_titluri = toate_titlurile[i : i + MARIME_BATCH]

        # Analizam grupul
        rezultate_grup = analizeaza_un_batch(grup_titluri, tokenizer, model, device)

        # Adaugam rezultatele la lista totala
        toate_rezultatele.extend(rezultate_grup)

    # Transformam lista de dictionare intr-un DataFrame
    tabel_rezultate = pd.DataFrame(toate_rezultatele)

    # Lipim coloanele de sentiment langa tabelul original
    # reset_index() asigura ca indexii se potrivesc
    tabel_final = pd.concat(
        [tabel_stiri.reset_index(drop=True), tabel_rezultate],
        axis=1
    )

    # Salvam versiunea detaliata (un rand per articol)
    tabel_final.to_csv("data/processed/news_with_sentiment.csv", index=False)
    print("Salvat detaliat in: data/processed/news_with_sentiment.csv")

    return tabel_final


# ─────────────────────────────────────────────
# PASUL 3: AGREGAM PE ZILE
# ─────────────────────────────────────────────

def calculeaza_scor_zilnic(tabel_cu_sentiment):
    """
    Calculam un singur scor de sentiment per zi.

    De ce avem nevoie de un scor per zi?
    Modelul nostru (XGBoost) va prezice miscarea pietei pentru ziua urmatoare.
    El are nevoie de un singur numar per zi, nu de sute de articole separate.

    Cum calculam scorul zilnic?
    Folosim media ponderata:
      - Fiecare articol are un scor net = prob_pozitiv - prob_negativ
        Daca e pozitiv 0.9 si negativ 0.05 -> scor net = 0.9 - 0.05 = 0.85
        Daca e negativ 0.8 si pozitiv 0.1  -> scor net = 0.1 - 0.8 = -0.70
      - Ponderea fiecarui articol = cat de sigur e modelul (siguranta)
        Un articol cu siguranta 95% conteaza mai mult decat unul cu 51%

    Scorul final e intre -1 si +1:
      +1 = toate stirile zilei sunt extrem de pozitive
      -1 = toate stirile zilei sunt extrem de negative
       0 = neutru sau mixt
    """
    print("\nCalculam scorul de sentiment pe zile...")

    # Calculam scorul net pentru fiecare articol
    tabel_cu_sentiment["scor_net"] = (
        tabel_cu_sentiment["prob_poz"] - tabel_cu_sentiment["prob_neg"]
    )

    # Grupam articolele pe zile
    rezultate_zilnice = []

    # tabel_cu_sentiment["data"].unique() = lista tuturor zilelor unice
    for zi in sorted(tabel_cu_sentiment["data"].unique()):

        # Luam toate articolele din ziua respectiva
        articole_zi = tabel_cu_sentiment[tabel_cu_sentiment["data"] == zi]

        # Calculam media ponderata a scorului net
        # Media ponderata = suma(scor * pondere) / suma(ponderi)
        suma_ponderi = articole_zi["siguranta"].sum()

        if suma_ponderi == 0:
            # Daca nu avem ponderi (caz extrem), folosim media simpla
            scor_zilnic = articole_zi["scor_net"].mean()
        else:
            scor_zilnic = (articole_zi["scor_net"] * articole_zi["siguranta"]).sum() / suma_ponderi

        # Procentul de articole pozitive, negative, neutre in ziua respectiva
        nr_total = len(articole_zi)
        pct_pozitiv = (articole_zi["sentiment"] == "pozitiv").sum() / nr_total
        pct_negativ = (articole_zi["sentiment"] == "negativ").sum() / nr_total

        rezultate_zilnice.append({
            "Date":           zi,
            "sentiment_score": round(scor_zilnic, 4),
            "article_count":  nr_total,
            "pct_pozitiv":    round(pct_pozitiv, 3),
            "pct_negativ":    round(pct_negativ, 3),
        })

    # Transformam in DataFrame
    tabel_zilnic = pd.DataFrame(rezultate_zilnice)

    # Salvam
    tabel_zilnic.to_csv("data/processed/sentiment_daily.csv", index=False)
    print("Salvat in: data/processed/sentiment_daily.csv")
    print("Zile acoperite:", len(tabel_zilnic))

    return tabel_zilnic



# ─────────────────────────────────────────────
# RULAM TOT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("ETAPA 2: ANALIZA SENTIMENT (FinBERT)")
    print("=" * 50)

    # Citim stirile din etapa anterioara
    if not os.path.exists("data/raw/news_raw.csv"):
        print("EROARE: news_raw.csv nu exista!")
        print("Ruleaza mai intai: python src/1_collect_data.py")
        exit()

    tabel_stiri = pd.read_csv("data/raw/news_raw.csv")
    print("Am citit", len(tabel_stiri), "articole din news_raw.csv")

    # Incarcam modelul
    tokenizer, model, device = incarca_modelul()

    # Analizam toate titlurile
    tabel_cu_sentiment = analizeaza_toate_stirile(tabel_stiri, tokenizer, model, device)

    # Calculam scorul zilnic
    tabel_zilnic = calculeaza_scor_zilnic(tabel_cu_sentiment)

    print("\n" + "=" * 50)
    print("ETAPA 2 COMPLETA!")

