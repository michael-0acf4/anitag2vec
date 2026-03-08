import requests
import csv
import os
import os
import urllib.parse

TAGS_PER_PAGE = 1000
PAGE = 1
PAGE_END = 5


def proxy(url: str) -> str:
    prefix = os.getenv("POORMAN_PROXY")

    if not prefix:
        return url
    if "?" not in prefix:
        prefix = prefix.rstrip("/") + "?url="
    elif not prefix.endswith(("=", "&")):
        prefix += "&url="

    return prefix + urllib.parse.quote(url, safe="")


all_tags = []
page = 1
while True:
    api = proxy("https://danbooru.donmai.us/tags.json")
    url = f"{api}?limit={TAGS_PER_PAGE}&page={page}&order=post_count&search="
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if page == PAGE_END:
        break
    tags = [tag["name"] for tag in data]
    all_tags.extend(tags)
    print(f"Page: {page}")
    print(url)
    page += 1

with open("all_tags.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Tag"])
    for tag in all_tags:
        writer.writerow([tag])

print("Done!")
