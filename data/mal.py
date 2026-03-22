import argparse
import hashlib
import requests
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


def fetch_entries(is_manga: bool, username: str) -> List[Dict[str, Any]]:
    categ = "mangalist" if is_manga else "animelist"
    url = f"https://myanimelist.net/{categ}/{username}/load.json?status=7&offset=0"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def extract_entry_tags(entry: Dict[str, Any], is_manga: bool) -> Tuple[str, List[str]]:
    if is_manga:
        title = entry.get("manga_title_eng") or entry.get("manga_title")
        alt_title = entry.get("manga_english")  # fallback alt
        media_type = entry.get("manga_media_type_string")
        unique_id = f"manga:{entry.get('manga_id')}"
        category = "Manga"
    else:
        title = entry.get("anime_title_eng") or entry.get("anime_title")
        alt_title = entry.get("anime_title_eng")
        media_type = entry.get("anime_media_type_string")
        unique_id = f"anime:{entry.get('anime_id')}"
        category = "Anime"

    if not title:
        return None, []

    tags = [title, category]

    if alt_title and alt_title != title:
        tags.append(alt_title)

    if media_type:
        tags.append(media_type)
    for g in entry.get("genres", []):
        if g.get("name"):
            tags.append(g["name"])
    for d in entry.get("demographics", []):
        if d.get("name"):
            tags.append(d["name"])

    # dedupe per entry (preserve order)
    tags = list(dict.fromkeys(tags))

    return unique_id, tags

def build_hash(usernames):
    canonical = sorted(set(usernames))
    joined = "".join(canonical)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser(description="Build per-entry tag sets from MAL users")
    parser.add_argument("usernames", nargs="+")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--anime", action="store_true")
    parser.add_argument("--manga", action="store_true")
    args = parser.parse_args()

    fetch_anime = args.anime or not args.manga
    fetch_manga = args.manga or not args.anime
    # key = unique entry id => tags
    entries_map: Dict[str, List[str]] = {}
    for username in tqdm(args.usernames, desc="Processing user"):
        try:
            if fetch_anime:
                anime_entries = fetch_entries(False, username)
                for entry in anime_entries:
                    uid, tags = extract_entry_tags(entry, is_manga=False)
                    if uid and uid not in entries_map:
                        entries_map[uid] = tags

            if fetch_manga:
                manga_entries = fetch_entries(True, username)
                for entry in manga_entries:
                    uid, tags = extract_entry_tags(entry, is_manga=True)
                    if uid and uid not in entries_map:
                        entries_map[uid] = tags

        except Exception as e:
            print(f"[!] Failed for {username}: {e}")

    result = list(entries_map.values())
    if args.normalize:
        result = [
            [t.lower().replace(" ", "_") for t in tags]
            for tags in result
        ]
    filename = f"mal_{build_hash(args.usernames)}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        print(f"Saved {len(result)} entries to {filename}")

if __name__ == "__main__":
    main()
