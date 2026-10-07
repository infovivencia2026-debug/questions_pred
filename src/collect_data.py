"""Step 1: collect papers from the arXiv API (no API key needed), sampled PER YEAR so that
authors have multi-year histories and the temporal split has data in every period.
Output: data/raw/papers.csv with id, title, abstract, authors, date, year, category."""
import time, re, requests, pandas as pd, xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import load_config

NS = {"a": "http://www.w3.org/2005/Atom"}
API = "http://export.arxiv.org/api/query"
PAGE = 200


def parse(text):
    rows = []
    for e in ET.fromstring(text).findall("a:entry", NS):
        aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.split("/abs/")[-1])
        rows.append(dict(
            id=aid,
            title=" ".join(e.find("a:title", NS).text.split()),
            abstract=" ".join(e.find("a:summary", NS).text.split()),
            authors="|".join(a.find("a:name", NS).text for a in e.findall("a:author", NS)),
            date=e.find("a:published", NS).text[:10]))
    return rows


def fetch(category, year, total):
    rows = []
    q = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"
    for start in tqdm(range(0, total, PAGE), desc=f"{category} {year}", leave=False):
        params = dict(search_query=q, start=start, max_results=PAGE, sortBy="submittedDate", sortOrder="descending")
        got = []
        for attempt in range(5):
            try:
                r = requests.get(API, params=params, timeout=60)
                if r.status_code == 200:
                    got = parse(r.text)
                    if got:
                        break
            except requests.RequestException:
                pass
            time.sleep(5 * (attempt + 1))
        if not got:
            break
        for g in got:
            g["category"] = category
        rows += got
        time.sleep(3)  # arXiv asks for ~3 s between requests
    return rows


if __name__ == "__main__":
    cfg = load_config()
    rows = []
    for cat in cfg["arxiv_categories"]:
        for year in range(cfg["year_from"], cfg["year_to"] + 1):
            rows += fetch(cat, year, cfg["papers_per_category_year"])
            print(f"{cat} {year}: total so far {len(rows)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["year"] = df["date"].str[:4].astype(int)
    df.to_csv("data/raw/papers.csv", index=False)
    print(f"saved {len(df)} papers")
    print(df.groupby("year").size().to_string())
