from typing import List, Dict
import re


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are",
    "was", "were", "be", "been", "being", "as", "at", "from", "that", "this", "it", "its", "their",
    "what", "when", "where", "who", "why", "how", "did", "do", "does", "have", "has", "had",
}


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]


def keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def generate_grounded_answer(query: str, retrieved_docs: List[Dict], max_sentences: int = 5) -> str:
    """
    A lightweight extractive answer generator.

    This is deliberately simple for a first portfolio version. It keeps answers grounded by only using
    sentences from the retrieved Wikipedia sections.
    """
    query_terms = keywords(query)
    scored_sentences = []

    for result in retrieved_docs:
        section_score = result["score"]
        section = result["document"]["section"]
        for sentence in split_sentences(result["document"]["text"]):
            sentence_terms = keywords(sentence)
            overlap = len(query_terms.intersection(sentence_terms))
            score = section_score + (0.08 * overlap)
            if overlap > 0 or section_score > 0.45:
                scored_sentences.append((score, section, sentence))

    if not scored_sentences:
        return "I could not find enough relevant information in the retrieved sections to answer that confidently."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = scored_sentences[:max_sentences]

    lines = ["Based on the retrieved Wikipedia sections:"]
    for _, section, sentence in selected:
        lines.append(f"- [{section}] {sentence}")
    return "\n".join(lines)


def format_sources(retrieved_docs: List[Dict]) -> str:
    lines = ["Sources used:"]
    for result in retrieved_docs:
        doc = result["document"]
        lines.append(f"{result['rank']}. {doc['section']} - score {result['score']:.3f} - {doc['url']}")
    return "\n".join(lines)
