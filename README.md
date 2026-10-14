# S&P 500 Sentiment Trading

Sistem de predicție a direcției indicelui S&P 500 pentru ziua următoare,
bazat pe analiza sentimentului știrilor financiare (FinBERT) combinată cu indicatori tehnici clasici.

---

## Ce face proiectul

Răspunde la o singură întrebare:

> **"Bazându-mă pe știrile de azi și pe cum s-a mișcat piața recent — va crește sau va scădea S&P 500 mâine?"**

Majoritatea proiectelor de predicție bursieră folosesc doar prețurile istorice.
Acesta adaugă un strat de NLP cu un model antrenat special pe texte financiare (FinBERT),
combinând două surse de informație complet diferite: **date de piață** și **știri**.

---

## Structura proiectului

```
sp500-sentiment-trading/
│
├── src/
│   ├── 1_collect_data.py           # Descarcă prețuri și știri
│   ├── 2_sentiment_analysis.py     # Analizează sentimentul cu FinBERT
│   ├── 3_feature_engineering.py    # Creează features pentru model
│   ├── 4_train_model.py            # Antrenează XGBoost
│   └── 5_evaluate.py               # Grafice și backtest
│
├── data/
│   ├── raw/                        # Date brute descărcate
│   │   ├── prices_raw.csv          # Prețuri zilnice S&P 500
│   │   └── news_raw.csv            # Titluri știri financiare
│   └── processed/                  # Date procesate
│       ├── news_with_sentiment.csv # Știri cu scor sentiment per articol
│       ├── sentiment_daily.csv     # Scor sentiment agregat per zi
│       ├── features.csv            # Matricea finală pentru model
│       ├── predictions.csv         # Predicțiile walk-forward
│       ├── walk_forward_results.csv# Metrici per fold
│       ├── feature_importance.csv  # Importanța features
│       └── backtest_data.csv       # Datele backtestului
│
├── models/
│   ├── xgb_classifier.pkl          # Modelul XGBoost antrenat
│   └── feature_cols.pkl            # Lista de features (ordinea contează)
│
├── reports/
│   ├── confusion_matrix.png        # Câte predicții corecte vs greșite
│   ├── feature_importance.png      # Ce variabile contează cel mai mult
│   ├── fold_performance.png        # Performanța în timp per fold
│   └── backtest_returns.png        # Strategia noastră vs buy & hold
│
├── run_all.py                      # Rulează tot pipeline-ul dintr-o comandă
├── requirements.txt                # Toate dependențele Python
├── .env.example                    # Template pentru chei API
└── .gitignore                      # Ce nu urcăm pe GitHub
```

---

## Ce face fiecare fișier

### `src/1_collect_data.py` — Colectare date
Descarcă datele brute din două surse gratuite:
- **Prețuri S&P 500** prin librăria `yfinance` — ticker `^GSPC`, ultimii 2 ani, coloanele Open/High/Low/Close/Volume
- **Știri financiare** prin feed-uri RSS publice (Reuters, CNBC, MarketWatch, Yahoo Finance) — fiecare articol cu data și titlul

Salvează: `data/raw/prices_raw.csv` și `data/raw/news_raw.csv`

---

### `src/2_sentiment_analysis.py` — Analiza sentimentului (FinBERT)
Trece fiecare titlu de știre prin modelul **FinBERT** (un BERT antrenat special pe texte financiare) și obține trei probabilități: pozitiv, negativ, neutru.

De ce FinBERT și nu ceva mai simplu? Instrumentele generice ca VADER nu înțeleg contextul financiar. "Earnings miss" sună neutru dar e negativ. FinBERT știe asta.

Agregă scorurile per articol într-un **scor zilnic unic** prin medie ponderată — articolele unde modelul e mai sigur contează mai mult.

Scorul final e între **-1** (toate știrile extrem de negative) și **+1** (toate pozitive).

Salvează: `data/processed/news_with_sentiment.csv` și `data/processed/sentiment_daily.csv`

---

### `src/3_feature_engineering.py` — Pregătirea features
Combină prețurile cu sentimentul zilnic și calculează variabilele pe care le va primi modelul:

**Features din prețuri:**
- `randament_zilnic` — cât % a crescut/scăzut prețul față de ieri
- `randament_5zile`, `randament_20zile` — randamentul pe o săptămână / lună
- `rsi_14` — Relative Strength Index: e piața supraîncălzită (>70) sau supravândută (<30)?
- `volatilitate_5zile`, `volatilitate_20zile` — cât de mult oscilează prețul
- `atr_14` — Average True Range: volatilitatea intraday
- `macd` — diferența dintre două medii mobile (semnal de trend)
- `raport_volum` — volumul de azi față de media pe 20 de zile
- `pret_vs_medie20` — prețul față de media mobilă pe 20 de zile

**Features din sentiment:**
- `sentiment_score` — scorul zilnic din FinBERT
- `sentiment_lag_1zi`, `_2zi`, `_3zi` — sentimentul din zilele anterioare
- `sentiment_medie_5zi`, `_20zi` — media mobilă a sentimentului
- `schimbare_sentiment` — saltul de sentiment față de ieri
- `article_count` — câte știri au apărut în ziua respectivă

**Target (ce prezice modelul):**
- `1` dacă S&P 500 a închis mai sus a doua zi
- `0` dacă a închis mai jos

Salvează: `data/processed/features.csv`

---

### `src/4_train_model.py` — Antrenarea modelului
Antrenează un clasificator **XGBoost** pe matricea de features.

XGBoost construiește sute de arbori de decizie mici, fiecare corectând greșelile celui anterior. La final, decizia colectivă e mult mai precisă decât a oricărui arbore individual.

**Grid search** — testează 6 combinații de hiperparametri și o alege pe cea mai bună.

**Walk-forward validation** — metodă corectă de evaluare pe date de timp:
```
Fold 1: Antrenament [ziua 1 → 252]      Test [ziua 253 → 315]
Fold 2: Antrenament [ziua 1 → 315]      Test [ziua 316 → 378]
Fold 3: Antrenament [ziua 1 → 378]      Test [ziua 379 → 441]
...
```
La fiecare pas, testul se face pe date pe care modelul nu le-a văzut niciodată — exact cum funcționează în realitate.

Salvează: `models/xgb_classifier.pkl`, `data/processed/predictions.csv`, `data/processed/feature_importance.csv`

---

### `src/5_evaluate.py` — Evaluare și vizualizare
Generează 4 grafice în folderul `reports/`:

- **confusion_matrix.png** — câte zile de creștere/scădere am prezis corect vs greșit
- **feature_importance.png** — ce variabile a folosit cel mai mult XGBoost (dacă sentimentul apare primul, NLP-ul aduce valoare reală)
- **fold_performance.png** — accuracy și F1 per fold în timp (modelul e consistent?)
- **backtest_returns.png** — simulare: dacă am fi tranzacționat pe semnalele modelului vs buy & hold

---

### `run_all.py` — Rulează tot
Rulează toate cele 5 etape în ordine. Dacă o etapă eșuează, oprește și afișează eroarea.

---

## Cum pornești proiectul

### 1. Instalează dependențele (o singură dată)
```bash
pip install -r requirements.txt
```

Pe Mac, dacă apare eroare cu XGBoost:
```bash
brew install libomp
```

### 2. Rulează tot pipeline-ul
```bash
python run_all.py
```

Sau etapă cu etapă:
```bash
python src/1_collect_data.py
python src/2_sentiment_analysis.py
python src/3_feature_engineering.py
python src/4_train_model.py
python src/5_evaluate.py
```

### 3. Timp aproximativ de rulare

| Etapă | Durată |
|---|---|
| 1. Colectare date | ~1 minut |
| 2. Sentiment FinBERT | ~10-20 minute |
| 3. Feature engineering | ~10 secunde |
| 4. Antrenare model | ~3-5 minute |
| 5. Evaluare și grafice | ~30 secunde |

### 4. Rezultatele finale
Graficele se găsesc în `reports/` după ce rulezi etapa 5.

---

## Tehnologii folosite

| Categorie | Librărie | De ce |
|---|---|---|
| Date prețuri | `yfinance` | Gratuit, direct de pe Yahoo Finance |
| Date știri | `feedparser` | Citește RSS feeds fără API key |
| NLP / Sentiment | `transformers` (FinBERT) | Model specializat pe texte financiare |
| Model clasificare | `xgboost` | Standard de industrie pentru date tabulare |
| Evaluare | `scikit-learn` | Metrici, confusion matrix |
| Vizualizare | `matplotlib`, `seaborn` | Grafice profesionale |

---

## Concepte cheie învățate

- **FinBERT inference** — cum se folosește un model NLP pre-antrenat pentru clasificare de sentiment
- **Feature engineering** — cum se transformă date brute (prețuri, text) în variabile numerice utile
- **RSI, ATR, MACD** — indicatori tehnici clasici implementați manual
- **Walk-forward validation** — cum se evaluează corect un model pe date de timp fără data leakage
- **scale_pos_weight** — cum se gestionează dezechilibrul de clase în XGBoost
- **Backtesting** — cum se simulează o strategie de tranzacționare pentru a valida semnalele modelului

---

## Disclaimer

Acest proiect este realizat în scop educațional.
Predicțiile modelului nu reprezintă sfaturi financiare și nu trebuie folosite pentru decizii reale de investiții.

---

## Autor

**[Numele tău]**
[LinkedIn](#) · [GitHub](#)
