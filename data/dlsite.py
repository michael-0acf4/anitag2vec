import argparse
import json
import hashlib
import requests
from typing import List, Dict, Set
from tqdm import tqdm
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def load_urls(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        urls = set([line.strip() for line in f if line.strip()])
        with_en = set()
        for url in urls:
            with_en.add(url)
            with_en.add(f"{url}?locale=en_US")
        return with_en

def clean(tag: str):
    tag = tag.strip()
    if not tag or len(tag) < 2:
        return None
    return tag

def extract_dom_tags(soup: BeautifulSoup) -> Set[str]:
    tags = set()
    selectors = [
        "div.main_genre a",
        "div.m-main_genre a",
        "table.m_work_genre a",
        "a.m-genre",
    ]
    for sel in selectors:
        for el in soup.select(sel):
            t = clean(el.get_text())
            if t:
                tags.add(t)
    return tags

def extract_meta_tags(soup: BeautifulSoup) -> Set[str]:
    tags = set()
    meta = soup.find("meta", {"name": "keywords"})
    if meta and meta.get("content"):
        for part in meta["content"].split(","):
            t = clean(part)
            if t:
                tags.add(t)
    return tags

def extract_jsonld_tags(soup: BeautifulSoup) -> Set[str]:
    tags = set()
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    for s in scripts:
        if not s.string:
            continue
        try:
            data = json.loads(s.string)
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "keywords":
                        if isinstance(v, str):
                            for t in v.split(","):
                                tt = clean(t)
                                if tt:
                                    tags.add(tt)
                        elif isinstance(v, list):
                            for t in v:
                                if isinstance(t, str):
                                    tt = clean(t)
                                    if tt:
                                        tags.add(tt)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for x in obj:
                    walk(x)
        walk(data)
    return tags


def extract_tags_from_url(url: str) -> List[str]:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tags = set()
    tags |= extract_dom_tags(soup)
    tags |= extract_meta_tags(soup)
    tags |= extract_jsonld_tags(soup)
    return sorted(tags)

def build_hash(urls: List[str]) -> str:
    joined = "".join(sorted(set(urls)))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser(description="DLsite batch tag extractor")
    parser.add_argument("file", help="Text file with DLsite URLs (one per line)")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    urls = load_urls(args.file)
    results = []
    seen = set()
    for url in tqdm(urls, desc="Processing DLsite"):
        if url in seen:
            continue
        seen.add(url)
        try:
            tags = extract_tags_from_url(url)
            if args.normalize:
                tags = [t.lower().replace(" ", "_") for t in tags]
            key = " ".join(tags)
            if key in seen or key == "":
                continue
            seen.add(key)
            results.append(tags)
        except Exception as e:
            print(f"[!] Failed: {url} -> {e}")

    out_file = args.out or f"dlsite_tags_{build_hash(urls)}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} entries -> {out_file}")

if __name__ == "__main__":
    main()
