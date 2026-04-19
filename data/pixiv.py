import hashlib
from urllib.parse import urlparse

import requests
import json
import argparse
from tqdm import tqdm

def fetch_pixiv_user_data(user_id, phpsessid):
    url = f"https://www.pixiv.net/ajax/user/{user_id}/profile/top?sensitiveFilterMode=userSetting&lang=en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://www.pixiv.net/users/{user_id}"
    }
    if phpsessid:
        headers["Cookie"] = f"PHPSESSID={phpsessid}"
    extracted = []
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("body", {})
        for category in ["illusts", "manga", "novels", "collections"]:
            content = data.get(category)
            if isinstance(content, dict):
                for item_id, details in content.items():
                    if details:
                        item = {
                            "id": details.get("id"),
                            "title": details.get("title"),
                            "type": category,
                            "user_id": details.get("userId"),
                            "user_name": details.get("userName"),
                            "tags": details.get("tags", []),
                        }
                        extracted.append(list(set([
                            item["title"],
                            item["user_name"],
                            item["type"],
                            item["title"],
                            *item["tags"]
                        ])))
    except Exception as e:
        tqdm.write(f"Error fetching user {user_id}: {e}")
    return extracted

def extract_pixiv_user_ids(ids_or_urls):
    ids = set()
    for item in ids_or_urls:
        if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
            ids.add(str(item))
            continue
        path = urlparse(item).path  # /en/users/{id}
        parts = path.strip("/").split("/")
        if "users" in parts:
            i = parts.index("users")
            if i + 1 < len(parts):
                ids.add(parts[i + 1])
    return sorted(ids)

def build_hash(ids):
    canonical = sorted(set(ids))
    joined = "".join(canonical)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]

def load_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

def main():
    parser = argparse.ArgumentParser(description="Fetch Pixiv user entries via CLI.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Pixiv user IDs, URLs, or file paths"
    )
    parser.add_argument("--cookie", required=False, help="Your PHPSESSID cookie")
    args = parser.parse_args()

    raw_items = []
    for inp in args.inputs:
        if inp.endswith(".txt"):
            raw_items.extend(load_from_file(inp))
        else:
            raw_items.append(inp)
    ids = extract_pixiv_user_ids(raw_items)

    all_results = []
    for uid in tqdm(ids, desc="Fetching users", unit="user"):
        all_results += fetch_pixiv_user_data(uid, args.cookie)
    output_filename = f"pixiv_{build_hash(ids)}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\nSaved to {output_filename}")

if __name__ == "__main__":
    main()
