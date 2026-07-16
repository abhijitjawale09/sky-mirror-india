from __future__ import annotations

import argparse
import html.parser
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DatasetSource:
    name: str
    landing_page: str
    output_dir: str


DATASETS = [
    DatasetSource(
        name="rainfall",
        landing_page="https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
        output_dir="data/raw/imd/rainfall",
    ),
    DatasetSource(
        name="max_temperature",
        landing_page="https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html",
        output_dir="data/raw/imd/max_temperature",
    ),
    DatasetSource(
        name="min_temperature",
        landing_page="https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html",
        output_dir="data/raw/imd/min_temperature",
    ),
]


class LinkCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_text_parts: list[str] = []
        self._current_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_map = {key.lower(): value for key, value in attrs}
        self._current_href = attrs_map.get("href")
        self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = " ".join(part.strip() for part in self._current_text_parts).strip()
        self.links.append((self._current_href, text))
        self._current_href = None
        self._current_text_parts = []


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IMD rainfall and temperature datasets if they are not already present.")
    parser.add_argument("--force", action="store_true", help="Redownload even if files already exist locally.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    ssl_context = ssl._create_unverified_context()

    for dataset in DATASETS:
        target_dir = project_root / dataset.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if not args.force and any(target_dir.iterdir()):
            print(f"Skipping {dataset.name}: files already exist in {target_dir}")
            continue

        page_html = fetch_text(dataset.landing_page, ssl_context)
        collector = LinkCollector()
        collector.feed(page_html)

        candidate_links = select_candidate_links(dataset.landing_page, collector.links)
        if not candidate_links:
            (target_dir / "landing_page.html").write_text(page_html, encoding="utf-8")
            print(f"No direct download link found for {dataset.name}; saved landing page HTML to {target_dir / 'landing_page.html'}")
            continue

        for index, link in enumerate(candidate_links, start=1):
            file_name = derive_file_name(link, index)
            file_path = target_dir / file_name
            if file_path.exists() and not args.force:
                print(f"Skipping existing file {file_path}")
                continue
            binary = fetch_bytes(link, ssl_context)
            file_path.write_bytes(binary)
            print(f"Downloaded {link} -> {file_path}")

    return 0


def select_candidate_links(base_url: str, links: list[tuple[str, str]]) -> list[str]:
    chosen: list[str] = []
    for href, text in links:
        absolute_url = urljoin(base_url, href)
        parsed = urlparse(absolute_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        content_hint = f"{href} {text}".lower()
        if any(token in content_hint for token in ["download", "binary", "bin", ".zip", ".csv", ".txt", ".dat"]):
            chosen.append(absolute_url)

    deduped: list[str] = []
    seen: set[str] = set()
    for link in chosen:
        if link not in seen:
            seen.add(link)
            deduped.append(link)
    return deduped


def derive_file_name(link: str, index: int) -> str:
    parsed = urlparse(link)
    tail = Path(parsed.path).name
    if tail:
        return tail
    return f"dataset_{index}.bin"


def fetch_text(url: str, ssl_context: ssl.SSLContext) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, context=ssl_context, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_bytes(url: str, ssl_context: ssl.SSLContext) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, context=ssl_context, timeout=120) as response:
        return response.read()


if __name__ == "__main__":
    raise SystemExit(main())
