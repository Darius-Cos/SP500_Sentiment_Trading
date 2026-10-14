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


today      = datetime.today()
start_date = today - timedelta(days=730)

# Why convert to string?
# yfinance expects dates as plain text "YYYY-MM-DD", not datetime objects.
# strftime("%Y-%m-%d") → datetime(2024, 1, 5) becomes "2024-01-05"
TODAY      = today.strftime("%Y-%m-%d")
START_DATE = start_date.strftime("%Y-%m-%d")

# exist_ok=True → if the folder already exists, do nothing instead of crashing
os.makedirs("data/raw", exist_ok=True)

NEWS_SOURCES = {
    "Reuters":       "https://feeds.reuters.com/reuters/businessNews",
    "CNBC":          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "MarketWatch":   "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Yahoo Finance": "https://finance.yahoo.com/rss/headline?s=^GSPC",
}


def download_prices():
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
    raw = yf.download(
        tickers="^GSPC",
        start=START_DATE,
        end=TODAY,
        progress=False,    # suppresses the download progress bar
        auto_adjust=True   # adjusts prices for stock splits automatically
    )

    # After download(), Date is the *index* of the DataFrame, not a regular column.
    # reset_index() promotes it back to a normal column so we can work with it.
    raw = raw.reset_index()
    #we only save columns that we need
    raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Why .dt.strftime() and not just .strftime()?
    # "Date" is a pandas Series (a whole column), not a single datetime.
    # .dt is a special accessor that lets you call datetime methods on every
    # row at once. Without .dt you get: AttributeError: 'Series' has no attribute 'strftime'
    #
    # Why convert to string if we already have datetime?
    # The news dates come out as plain strings. When we later merge prices + news
    # on the date column, both sides must be in the exact same format —
    # otherwise pandas finds zero matches and the merged table is silently empty.
    raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")

    raw.to_csv("data/raw/prices.csv", index=False)
    print(f"Downloaded {len(raw)} trading days → data/raw/prices.csv")

    return raw


def read_one_feed(source_name, url):
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
    articles = []

    try:
        feed = feedparser.parse(url)
        # feed.status          → HTTP code (200 = ok, 404 = not found)
        # feed.feed.title      → name of the publication, e.g. "Reuters Business News"
        # feed.feed.link       → main website of the publication
        # feed.entries         → list of all articles

        for entry in feed.entries:
            # entry.title            → article headline
            # entry.link             → full URL of the article
            # entry.summary          → short description (not all feeds include this)
            # entry.author           → author name (rarely available)
            # entry.published_parsed → publish date as a tuple (year, month, day, hour, minute, second, ...)

            # Some entries are missing a publication date entirely.
            # We skip entries without dates because we can't place them on a timeline.
            if  entry.published_parsed is None:
                continue

            t = entry.published_parsed
            pub_date_str = datetime(t[0], t[1], t[2], t[3], t[4], t[5]).strftime("%Y-%m-%d")

            if pub_date_str < START_DATE or pub_date_str > TODAY:
                continue

            articles.append({
                "date":   pub_date_str,
                "title":  entry.title.strip(),   # .strip() removes leading/trailing whitespace
                "source": source_name
            })

    except Exception as e:
        # If one feed fails (network timeout, malformed XML, etc.),
        # we don't crash the whole script — just print the error and move on.
        print(f"  Error reading {source_name}: {e}")

    return articles


def download_news():
    """
    Parcurge toate sursele RSS si combina stirile intr-un singur tabel.
    """
    all_articles = []

    for name, url in NEWS_SOURCES.items():
        batch = read_one_feed(name, url)
        all_articles.extend(batch)   # extend() adds each item, not the list itself
        time.sleep(0.5)              # small pause so we don't get rate-limited

    news_df = pd.DataFrame(all_articles)

    if news_df.empty:
        print("No articles found!")
        return news_df

    # Why drop_duplicates with subset=["title"]?
    # The same headline often appears in multiple RSS feeds (e.g. Reuters republishes
    # on Yahoo Finance). Without deduplication, the model would see the same article
    # multiple times and treat it as stronger evidence than it really is.
    # subset=["title"] means: only compare the "title" column when deciding if two
    # rows are duplicates — ignore differences in "source" or "date".
    news_df = news_df.drop_duplicates(subset=["title"])

    # Sort chronologically so the CSV reads like a timeline
    news_df = news_df.sort_values("date")

    # Why reset_index(drop=True)?
    # After drop_duplicates() and sort_values(), the row index numbers are jumbled
    # (e.g. 0, 3, 7, 2, ...). reset_index() renumbers them cleanly from 0 upward.
    # drop=True means: throw away the old index instead of saving it as a new column.
    # Without drop=True, pandas would add an extra "index" column to your DataFrame
    # containing the old messy numbers — which we don't need.
    news_df = news_df.reset_index(drop=True)

    news_df.to_csv("data/raw/news.csv", index=False)
    print(f"Downloaded {len(news_df)} unique articles → data/raw/news.csv")

    return news_df


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: DATA COLLECTION")
    print("=" * 50)

    prices = download_prices()
    news   = download_news()

    print("\n" + "=" * 50)
    print(f"Trading days : {len(prices)}")
    print(f"News articles: {len(news)}")
    print("Next step: python src/2_sentiment_analysis.py")
    print("=" * 50)