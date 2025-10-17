from __future__ import annotations
import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple
from urllib.parse import urljoin, urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm


BASE_URL = "https://www.nhs.uk"
INDEX_URL = f"{BASE_URL}/conditions/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3 (Educational crawler)"
    )
}
SECTION_NAMES = ("Symptoms", "Treatment", "Treating")
ALLOWED_SECTIONS = {"symptoms", "treatment"}
SECTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Symptoms": ("symptom", "symptoms"),
    "Treatment": ("treat"),
    "Types of": ("type", "form", "kind", "category"),
    "Treatments": ("treat", "therapy", "option"),
    "Treating": ("treat", "therapy", "option"),
    "Side effects": ("side effect", "adverse"),
}
SECTION_HEADING_PATTERNS: Dict[str, Tuple[Pattern[str], ...]] = {
    "Symptoms": (
        re.compile(r"\bsigns?\s*(?:&|and)\s*symptoms?\b", re.IGNORECASE),
        re.compile(r"\bsymptoms?\s*(?:&|and)\s*signs?\b", re.IGNORECASE),
        re.compile(r"\brecognis(?:e|ing)\b.*\bsymptoms?\b", re.IGNORECASE),
    ),
    "Treatment": (
        re.compile(r"\bhow\b.*\btreated\b", re.IGNORECASE),
        re.compile(r"\btreatments?\b.*\bself-care\b", re.IGNORECASE),
        re.compile(r"\bmanag(?:e|ing|ement)\b.*", re.IGNORECASE),
        re.compile(r"\byour\b.*\btreatment\b", re.IGNORECASE),
    ),
}
FALLBACK_PATH_SEGMENTS: Dict[str, Tuple[str, ...]] = {
    "Symptoms": ("symptoms",),
    "Treatment": ("treatment",),
}
SECTION_SLUGS = {
    "overview",
    "symptoms",
    "treatment",
    "prevention",
    "causes",
    "getting-diagnosed",
    "self-care",
    "diagnosis",
    "complications",
}
BLOCK_TAGS = {"script", "style", "aside", "nav", "footer", "header", "figure"}
ALLOWED_TAGS = {"p", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "a"}
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 4.0
SUBPAGE_EXCLUDED_SUFFIXES = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".mp3",
    ".mp4",
    ".zip",
    ".webp",
}
URL_PATTERN = re.compile(r"https?://\S+")


logger = logging.getLogger(__name__)


@dataclass
class ConditionSection:
    condition: str
    section: str
    url: str
    content: Sequence[str]

    def to_record(self) -> Dict[str, str]:
        return {
            "condition": self.condition,
            "section": self.section,
            "url": self.url,
            "content": "\n\n".join(self.content),
        }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def heading_level(tag: Tag) -> int:
    if not tag.name or not tag.name.startswith("h"):
        return 7
    try:
        return int(tag.name[1])
    except (IndexError, ValueError):
        return 7


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def collect_condition_urls(index_soup: BeautifulSoup) -> List[str]:
    urls: set[str] = set()
    main = index_soup.find("main") or index_soup
    for anchor in main.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue
        full_url = urljoin(INDEX_URL, href)
        parsed = urlparse(full_url)
        if parsed.netloc != urlparse(BASE_URL).netloc:
            continue
        segments = [seg for seg in parsed.path.split("/") if seg]
        if not segments or segments[0] != "conditions" or len(segments) < 2:
            continue
        while segments and segments[-1] in SECTION_SLUGS:
            segments.pop()
        if len(segments) < 2:
            continue
        condition_path = "/".join(segments)
        condition_url = urljoin(BASE_URL, condition_path + "/")
        urls.add(condition_url)
    return sorted(urls)


def normalize_section_key(value: str) -> str:
    return re.sub(r"[\s\-]+", "-", value.strip().lower())


def extract_title(soup: BeautifulSoup) -> str:
    main = soup.find("main") or soup
    heading = main.find("h1")
    if heading:
        text = normalize_whitespace(heading.get_text(" ", strip=True))
        if text:
            return text
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return normalize_whitespace(title_tag.string.split("-", 1)[0])
    return ""


def find_section_heading(soup: BeautifulSoup, section_name: str, anchor_id: Optional[str]) -> Optional[Tag]:
    main = soup.find("main") or soup
    if anchor_id:
        tag = main.find(id=anchor_id)
        if tag:
            if tag.name and tag.name.lower().startswith("h"):
                return tag
            heading = tag.find(lambda t: isinstance(t, Tag) and is_heading(t) and matches_section_name(t, section_name))
            if heading:
                return heading
    return main.find(lambda t: isinstance(t, Tag) and is_heading(t) and matches_section_name(t, section_name))


def is_heading(tag: Tag) -> bool:
    return tag.name in {"h2", "h3", "h4", "h5", "h6"}


def matches_section_name(tag: Tag, section_name: str) -> bool:
    text = normalize_whitespace(tag.get_text(" ", strip=True))
    lower_text = text.lower()
    target = section_name.lower()
    if lower_text == target or lower_text.startswith(target):
        return True
    patterns = SECTION_HEADING_PATTERNS.get(section_name)
    if patterns and any(pattern.search(text) for pattern in patterns):
        return True
    return False


def matches_section_keyword(value: Optional[str], section_name: str) -> bool:
    if not value:
        return False
    text = normalize_whitespace(value).lower()
    return any(keyword in text for keyword in SECTION_KEYWORDS[section_name])


def anchor_matches_section(anchor: Tag, section_name: str) -> bool:
    texts = [
        anchor.get_text(" ", strip=True),
        anchor.get("aria-label"),
        anchor.get("title"),
        anchor.get("data-track-label"),
    ]
    if any(matches_section_keyword(text, section_name) for text in texts if text):
        return True
    href = anchor.get("href", "")
    if not href:
        return False
    href_lower = href.lower()
    return any(keyword in href_lower for keyword in SECTION_KEYWORDS[section_name])


def ancestors_contain_block(tag: Tag) -> bool:
    return any(parent.name in BLOCK_TAGS for parent in tag.parents if isinstance(parent, Tag))


def extract_text_from_tag(tag: Tag) -> Iterable[str]:
    if tag.name in {"p", "h3", "h4", "h5", "h6"}:
        text = normalize_whitespace(tag.get_text(" ", strip=True))
        if text:
            yield text
    elif tag.name in {"ul", "ol"}:
        for item in tag.find_all("li"):
            text = normalize_whitespace(item.get_text(" ", strip=True))
            if text:
                yield f"- {text}"
    elif tag.name == "li":
        text = normalize_whitespace(tag.get_text(" ", strip=True))
        if text:
            yield f"- {text}"


def extract_section_content(heading: Tag) -> List[str]:
    level = heading_level(heading)
    content: List[str] = []
    heading_text = normalize_whitespace(heading.get_text(" ", strip=True))
    if heading_text:
        content.append(heading_text)
    for element in heading.next_elements:
        if element is heading:
            continue
        if isinstance(element, Tag):
            if element.name in BLOCK_TAGS:
                continue
            if is_heading(element) and heading_level(element) <= level:
                break
            if ancestors_contain_block(element):
                continue
            if element.name in ALLOWED_TAGS:
                for text in extract_text_from_tag(element):
                    content.append(text)
        elif isinstance(element, NavigableString):
            continue
    return content


def collect_condition_subpages(condition_url: str, soup: BeautifulSoup) -> List[str]:
    main = soup.find("main") or soup
    base_parsed = urlparse(condition_url)
    base_domain = urlparse(BASE_URL).netloc
    root_path = base_parsed.path.rstrip("/")
    root_segments = [segment for segment in root_path.split("/") if segment]
    seen: set[str] = set()
    for anchor in main.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue
        resolved = urljoin(condition_url, href)
        parsed = urlparse(resolved)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc and parsed.netloc != base_domain:
            continue
        if parsed.path.endswith("/"):
            target_path = parsed.path.rstrip("/")
        else:
            target_path = parsed.path
        if not target_path or target_path == root_path:
            continue
        if not target_path.startswith(root_path + "/"):
            continue
        suffix = Path(parsed.path).suffix.lower()
        if suffix and suffix in SUBPAGE_EXCLUDED_SUFFIXES:
            continue
        relative_segments = [segment for segment in target_path.split("/") if segment][len(root_segments):]
        if len(relative_segments) != 1:
            continue
        clean_url = parsed._replace(query="", fragment="").geturl().rstrip("/") + "/"
        seen.add(clean_url)
    return sorted(seen)


def extract_main_content(soup: BeautifulSoup) -> List[str]:
    main = soup.find("main") or soup
    content: List[str] = []
    for element in main.descendants:
        if isinstance(element, Tag):
            if element.name in BLOCK_TAGS or element.name in {"figure", "figcaption", "img"}:
                continue
            if ancestors_contain_block(element):
                continue
            if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                text = normalize_whitespace(element.get_text(" ", strip=True))
                if text and not URL_PATTERN.search(text):
                    content.append(text)
            elif element.name in {"p", "ul", "ol"}:
                for text in extract_text_from_tag(element):
                    if text and not URL_PATTERN.search(text):
                        content.append(text)
            elif element.name == "li":
                parent = element.parent
                if parent and isinstance(parent, Tag) and parent.name in {"ul", "ol"}:
                    continue
                for text in extract_text_from_tag(element):
                    if text and not URL_PATTERN.search(text):
                        content.append(text)
    return content


def should_use_fallback(url: str, section_name: str) -> bool:
    segments = FALLBACK_PATH_SEGMENTS.get(section_name)
    if not segments:
        return False
    path = urlparse(url).path.lower()
    trimmed = path.rstrip("/")
    for segment in segments:
        slug = segment.lower()
        if f"/{slug}/" in path or trimmed.endswith(f"/{slug}"):
            return True
    return False


def extract_fallback_content(container: Tag, section_name: str) -> List[str]:
    content: List[str] = [section_name]
    for element in container.descendants:
        if isinstance(element, Tag):
            if element.name in BLOCK_TAGS:
                continue
            if ancestors_contain_block(element):
                continue
            if is_heading(element):
                heading_text = normalize_whitespace(element.get_text(" ", strip=True))
                if heading_text:
                    content.append(heading_text)
            elif element.name in ALLOWED_TAGS:
                if element.name == "li" and element.parent and isinstance(element.parent, Tag) and element.parent.name in {"ul", "ol"}:
                    continue
                for text in extract_text_from_tag(element):
                    if text:
                        content.append(text)
        elif isinstance(element, NavigableString):
            continue
    return [item for item in content if item]


def resolve_section_target(base_url: str, soup: BeautifulSoup, section_name: str) -> Tuple[str, Optional[str]]:
    main = soup.find("main") or soup
    base_parsed = urlparse(base_url)
    base_domain = urlparse(BASE_URL).netloc
    condition_path = base_parsed.path.rstrip("/")
    best: Optional[Tuple[int, str, Optional[str]]] = None
    for anchor in main.find_all("a", href=True):
        if not anchor_matches_section(anchor, section_name):
            continue
        href = anchor["href"].strip()
        if not href:
            continue
        resolved = urljoin(base_url, href)
        parsed = urlparse(resolved)
        if parsed.netloc and parsed.netloc != base_domain:
            continue
        target_path = parsed.path.rstrip("/")
        if target_path and not target_path.startswith(condition_path):
            continue
        fragment = parsed.fragment or None
        target_url = resolved.split("#")[0]
        priority = 2
        if href.startswith("#"):
            priority = 0
        elif parsed.path.rstrip("/").startswith(condition_path):
            priority = 1
        candidate = (priority, target_url, fragment)
        if best is None or candidate < best:
            best = candidate
    if best:
        _, target_url, fragment = best
        return target_url, fragment
    return base_url, None


def extract_sections_for_condition(condition_url: str) -> List[ConditionSection]:
    sections: List[ConditionSection] = []
    soup_cache: Dict[str, BeautifulSoup] = {}

    def get_soup(url: str) -> BeautifulSoup:
        if url not in soup_cache:
            if soup_cache:
                time.sleep(REQUEST_DELAY)
            soup_cache[url] = fetch_soup(url)
        return soup_cache[url]

    base_soup = get_soup(condition_url)
    title = extract_title(base_soup)
    existing_urls: set[str] = set()
    for section_name in SECTION_NAMES:
        target_url, anchor_id = resolve_section_target(condition_url, base_soup, section_name)
        section_soup = get_soup(target_url)
        heading = find_section_heading(section_soup, section_name, anchor_id)
        if heading:
            content = extract_section_content(heading)
        else:
            fallback_content: List[str] = []
            slug = urlparse(target_url).path.rstrip("/").split("/")[-1].lower()
            if should_use_fallback(target_url, section_name) and (
                slug in ALLOWED_SECTIONS or section_name.lower() in ALLOWED_SECTIONS
            ):
                main_tag = section_soup.find("main") or section_soup
                fallback_content = extract_fallback_content(main_tag, section_name)
                if len(fallback_content) <= 1:
                    fallback_content = []
            if not fallback_content:
                logger.debug("No heading for %s at %s", section_name, target_url)
                continue
            content = fallback_content
        if not content:
            continue
        sections.append(
            ConditionSection(
                condition=title or content[0],
                section=section_name,
                url=target_url,
                content=content,
            )
        )
        existing_urls.add(target_url.rstrip("/") + "/")
    for subpage_url in collect_condition_subpages(condition_url, base_soup):
        normalized_subpage = subpage_url.rstrip("/") + "/"
        if normalized_subpage in existing_urls:
            continue
        section_slug = urlparse(subpage_url).path.rstrip("/").split("/")[-1].lower()
        if section_slug not in ALLOWED_SECTIONS:
            continue
        if not section_slug:
            continue
        subpage_soup = get_soup(subpage_url)
        content = extract_main_content(subpage_soup)
        if not content:
            continue
        sections.append(
            ConditionSection(
                condition=title or content[0],
                section=section_slug.replace("-", " "),
                url=subpage_url,
                content=content,
            )
        )
        existing_urls.add(normalized_subpage)
    return sections


def crawl_conditions(limit: Optional[int] = None) -> List[ConditionSection]:
    index_soup = fetch_soup(INDEX_URL)
    condition_urls = collect_condition_urls(index_soup)
    if limit is not None:
        condition_urls = condition_urls[:limit]
    results: List[ConditionSection] = []
    for url in tqdm(condition_urls, desc="Conditions", unit="condition"):
        try:
            results.extend(extract_sections_for_condition(url))
        except requests.RequestException as exc:
            logger.warning("Failed to process %s: %s", url, exc)
            continue
    return results


def save_results(sections: Sequence[ConditionSection], output_path: Path) -> None:
    records = [section.to_record() for section in sections]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl NHS condition pages.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of conditions to crawl for testing.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nhs_conditions.csv"),
        help="Output file path (.csv or .jsonl).",
    )
    return parser.parse_args(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logger.info("Starting crawl from %s", INDEX_URL)
    sections = crawl_conditions(limit=args.limit)
    logger.info("Extracted %d sections", len(sections))
    save_results(sections, args.output)
    logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()