from dataclasses import dataclass
from typing import List
import re
import requests
from bs4 import BeautifulSoup


WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_PAGE_TITLE = "Manchester United F.C."
DEFAULT_USER_AGENT = "ManchesterUnitedWikiRAG/0.1 (portfolio demo; contact: example@example.com)"


@dataclass
class Document:
    doc_id: str
    title: str
    section: str
    level: int
    url: str
    text: str


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "section"


class WikiSectionLoader:
    """Loads Wikipedia sections as individual RAG documents."""

    def __init__(self, page_title: str = DEFAULT_PAGE_TITLE, user_agent: str = DEFAULT_USER_AGENT):
        self.page_title = page_title
        self.headers = {"User-Agent": user_agent}

    def _get_json(self, params: dict) -> dict:
        response = requests.get(WIKI_API_URL, params=params, headers=self.headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    def get_section_metadata(self) -> List[dict]:
        params = {
            "action": "parse",
            "page": self.page_title,
            "prop": "sections",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        }
        data = self._get_json(params)
        return data["parse"].get("sections", [])

    def get_section_html(self, section_index: str) -> str:
        params = {
            "action": "parse",
            "page": self.page_title,
            "prop": "text",
            "section": section_index,
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        }
        data = self._get_json(params)
        return data["parse"]["text"]

    @staticmethod
    def clean_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        remove_selectors = [
            "table", "style", "script", "sup.reference", ".mw-editsection",
            ".shortdescription", ".navbox", ".infobox", ".ambox", ".hatnote",
            "figure", ".thumb", ".toc", ".metadata",
        ]
        for selector in remove_selectors:
            for tag in soup.select(selector):
                tag.decompose()

        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_documents(self, min_chars: int = 250, include_lead: bool = True) -> List[Document]:
        sections = self.get_section_metadata()
        docs: List[Document] = []

        if include_lead:
            lead_text = self.clean_html(self.get_section_html("0"))
            if len(lead_text) >= min_chars:
                docs.append(
                    Document(
                        doc_id="lead",
                        title=self.page_title,
                        section="Lead",
                        level=0,
                        url="https://en.wikipedia.org/wiki/Manchester_United_F.C.",
                        text=lead_text,
                    )
                )

        excluded_sections = {
            "See also", "References", "Bibliography", "Further reading", "External links", "Notes"
        }

        for section in sections:
            section_name = section.get("line", "Untitled")
            if section_name in excluded_sections:
                continue

            section_index = section["index"]
            level = int(section.get("level", 2))
            anchor = section.get("anchor", _slug(section_name))
            text = self.clean_html(self.get_section_html(section_index))

            if len(text) < min_chars:
                continue

            docs.append(
                Document(
                    doc_id=f"{section_index}-{_slug(section_name)}",
                    title=self.page_title,
                    section=section_name,
                    level=level,
                    url=f"https://en.wikipedia.org/wiki/Manchester_United_F.C.#{anchor}",
                    text=text,
                )
            )

        return docs
