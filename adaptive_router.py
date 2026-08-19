import math
import re
from dataclasses import dataclass, field

import numpy as np
from google import genai


@dataclass
class QueryRouteDecision:
    tier: str  # "direct", "multi_query", "hyde"
    specificity_score: float
    score_margin: float
    entropy: float
    reason: str
    expanded_queries: list[str] = field(default_factory=list)
    hypothetical_doc: str | None = None


# Common English stop words for IDF estimation when ES corpus statistics are unavailable
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "what",
    "how",
    "why",
    "can",
    "could",
    "should",
    "does",
    "do",
    "explain",
    "show",
    "tell",
    "give",
    "me",
    "this",
    "these",
    "those",
}


class AdaptiveQueryRouter:
    """
    Mathematical query complexity router using Mean IDF and Fast-Probe Score Entropy
    to allocate LLM compute across Direct, Multi-Query, and HyDE retrieval tiers.
    """

    def __init__(
        self,
        gemini_client: genai.Client | None = None,
        specificity_threshold: float = 3.5,
        margin_threshold: float = 0.08,
        entropy_threshold: float = 1.3,
    ):
        self._client = gemini_client
        self.specificity_threshold = specificity_threshold
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def compute_mean_idf(self, query: str, es_store=None) -> float:
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
        if not words:
            return 0.0

        scores = []
        for w in words:
            if w in STOP_WORDS:
                scores.append(0.5)
            else:
                char_entropy = len(set(w)) / max(len(w), 1)
                rarity = math.log(len(w) + 1) * 2.0 * char_entropy
                scores.append(rarity)
        return float(np.mean(scores))

    def compute_probe_metrics(
        self, probe_scores: list[float], temperature: float = 0.15
    ) -> tuple[float, float]:
        if not probe_scores:
            return 0.0, 0.0
        if len(probe_scores) == 1:
            return float(probe_scores[0]), 0.0

        scores = np.array(probe_scores, dtype=np.float64)
        margin = float(scores[0] - scores[1])

        exp_scores = np.exp((scores - np.max(scores)) / max(temperature, 1e-4))
        probs = exp_scores / np.sum(exp_scores)

        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        return margin, entropy

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> list[str]:
        prompt = (
            f"Generate {num_queries} diverse search engine sub-queries or rephrasings for the following question. "
            f"Provide one query per line without bullet points or numbering.\n\n"
            f"Question: {query}"
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        cleaned = [re.sub(r"^\d+[\.\)]\s*", "", line) for line in lines]
        return cleaned[:num_queries]

    def generate_hyde_doc(self, query: str) -> str:
        prompt = (
            f"Write a short, direct, factual passage (1 paragraph) that answers the question: '{query}'. "
            f"Focus purely on factual content that would likely appear in an authoritative document."
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text.strip()

    def route(self, query: str, es_store, embed_fn) -> QueryRouteDecision:
        mean_idf = self.compute_mean_idf(query, es_store=es_store)

        query_vector = embed_fn(query)
        probe_hits = es_store.search_dense_knn(query_vector, top_k=5)
        probe_scores = [hit.get("score", 0.0) for hit in probe_hits]
        margin, entropy = self.compute_probe_metrics(probe_scores)

        if mean_idf >= self.specificity_threshold or (
            margin >= self.margin_threshold and entropy <= self.entropy_threshold
        ):
            return QueryRouteDecision(
                tier="direct",
                specificity_score=mean_idf,
                score_margin=margin,
                entropy=entropy,
                reason=f"High query specificity ({mean_idf:.2f}) or strong probe confidence (margin={margin:.2f}, entropy={entropy:.2f}).",
            )
        elif entropy > self.entropy_threshold and margin < self.margin_threshold:
            hyde_doc = self.generate_hyde_doc(query)
            return QueryRouteDecision(
                tier="hyde",
                specificity_score=mean_idf,
                score_margin=margin,
                entropy=entropy,
                reason=f"High retrieval ambiguity (entropy={entropy:.2f}, margin={margin:.2f}). Generating synthetic HyDE passage.",
                hypothetical_doc=hyde_doc,
            )
        else:
            sub_queries = self.generate_multi_queries(query, num_queries=3)
            return QueryRouteDecision(
                tier="multi_query",
                specificity_score=mean_idf,
                score_margin=margin,
                entropy=entropy,
                reason=f"Moderate query complexity ({mean_idf:.2f}). Expanding into {len(sub_queries)} sub-queries.",
                expanded_queries=sub_queries,
            )
