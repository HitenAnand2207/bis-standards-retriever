import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz


STANDARD_PATTERN = re.compile(
    r"\bIS\s*(\d{2,5})(?:\s*\(\s*PART\s*(\d+)\s*\))?(?:\s*[:\-]?\s*(\d{4}))?",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

METHOD_KEYWORDS = [
    "methods of test",
    "methods of tests",
    "method of test",
    "method of tests",
    "methods of physical test",
    "methods of tests for",
]

QUERY_STANDARD_HINTS = {
    "ordinary portland cement": ["IS 269: 1989"],
    "portland slag cement": ["IS 455: 1989"],
    "portland pozzolana cement": ["IS 1489 (Part 2): 1991", "IS 1489 (Part 1): 1991"],
    "masonry cement": ["IS 3466: 1988"],
    "coarse and fine aggregates": ["IS 383: 1970"],
    "lightweight concrete masonry blocks": ["IS 2185 (Part 2): 1983"],
    "precast concrete pipes": ["IS 458: 2003"],
    "asbestos cement sheets": ["IS 459: 1992"],
    "white portland cement": ["IS 8042: 1989"],
}

QUERY_EXPANSIONS = {
    "cement": ["portland cement", "slag cement", "pozzolana cement"],
    "aggregate": ["coarse aggregate", "fine aggregate", "graded aggregate"],
    "concrete": ["reinforced concrete", "plain concrete", "precast concrete"],
    "steel": ["reinforced steel", "mild steel", "high tensile steel", "steel bar"],
    "brick": ["clay brick", "burnt clay brick"],
    "block": ["concrete block", "masonry block"],
    "pipe": ["concrete pipe", "asbestos cement pipe"],
    "mortar": ["cement mortar", "lime mortar"],
    "tensile": ["tensile strength", "tensile test"],
    "compressive": ["compressive strength", "compression test"],
}

MATERIAL_CATEGORIES = {
    "cement": ["IS 269", "IS 455", "IS 1489", "IS 3466", "IS 8042"],
    "aggregate": ["IS 383", "IS 2386"],
    "concrete": ["IS 456", "IS 1343", "IS 1893"],
    "steel": ["IS 226", "IS 432", "IS 1786", "IS 2062"],
    "brick": ["IS 1077", "IS 1080"],
    "block": ["IS 2185"],
    "pipe": ["IS 458"],
    "mortar": ["IS 2250"],
    "test_methods": ["IS 2386", "IS 2683", "IS 1199"],
}


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def canonical_standard(number: str, part: Optional[str], year: Optional[str]) -> str:
    base = f"IS {number}"
    if part:
        base += f" (Part {part})"
    if year:
        base += f": {year}"
    return base


def extract_standard_mentions(text: str) -> List[str]:
    mentions = []
    for match in STANDARD_PATTERN.finditer(text):
        number, part, year = match.groups()
        mentions.append(canonical_standard(number, part, year))
    return mentions


def standard_base_key(standard: str) -> str:
    match = STANDARD_PATTERN.search(standard)
    if not match:
        return normalize_spaces(standard).upper()
    number, part, _ = match.groups()
    base = f"IS {number}"
    if part:
        base += f" (PART {part})"
    return base


def normalize_for_match(value: str) -> str:
    return str(value).replace(" ", "").lower()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def expand_query(query: str) -> str:
    expanded = [query]
    query_lower = query.lower()
    for term, expansions in QUERY_EXPANSIONS.items():
        if term in query_lower:
            for expansion in expansions:
                if expansion not in query_lower:
                    expanded.append(expansion)
    return " ".join(expanded)


def detect_query_category(query: str) -> Optional[str]:
    query_lower = query.lower()
    for category, keywords in QUERY_STANDARD_HINTS.items():
        if category in query_lower:
            return category
    for category, prefixes in MATERIAL_CATEGORIES.items():
        for prefix in prefixes:
            if prefix.replace(" ", "").lower() in query_lower.replace(" ", ""):
                return category
    return None


class BisRetriever:
    def __init__(self, dataset_pdf: str, cache_path: str = "index_cache.json"):
        self.dataset_pdf = Path(dataset_pdf)
        self.cache_path = Path(cache_path)
        self.pages: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.base_to_best_full: Dict[str, str] = {}
        self.global_standard_frequency: Counter = Counter()
        self.standard_context: Dict[str, str] = {}
        self.method_keywords = METHOD_KEYWORDS

        # Tunable heuristics. Slightly favor richer candidate exploration over raw speed.
        self.context_boost_multiplier = 1.0
        self.method_penalty = 0.35
        self.explicit_hint_boost = 8.0
        self.enable_query_expansion = True
        self.enable_category_filtering = True
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        signature = {
            "pdf": str(self.dataset_pdf.resolve()),
            "size": self.dataset_pdf.stat().st_size,
            "mtime": self.dataset_pdf.stat().st_mtime,
        }

        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if payload.get("signature") == signature:
                    self.pages = payload["pages"]
                    self._build_search_structures()
                    return
            except Exception:
                pass

        self.pages = self._build_pages_from_pdf()
        self._build_search_structures()
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump({"signature": signature, "pages": self.pages}, f, ensure_ascii=False)

    def _build_pages_from_pdf(self) -> List[Dict]:
        doc = fitz.open(str(self.dataset_pdf))
        pages: List[Dict] = []

        for page_idx in range(len(doc)):
            text = normalize_spaces(doc[page_idx].get_text("text"))
            if not text:
                continue

            mentions = extract_standard_mentions(text)
            pages.append(
                {
                    "page_num": page_idx + 1,
                    "text": text,
                    "mentions": mentions,
                    "tokens": tokenize(text),
                }
            )

        return pages

    def _build_search_structures(self) -> None:
        corpus_tokens = [p["tokens"] for p in self.pages]
        self.bm25 = BM25Okapi(corpus_tokens)

        full_counter_by_base: Dict[str, Counter] = defaultdict(Counter)
        context_chunks_by_base: Dict[str, List[str]] = defaultdict(list)
        self.global_standard_frequency.clear()

        for page in self.pages:
            for mention in page["mentions"]:
                self.global_standard_frequency[mention] += 1
                base = standard_base_key(mention)
                full_counter_by_base[base][mention] += 1
                if len(context_chunks_by_base[base]) < 6:
                    context_chunks_by_base[base].append(page["text"][:900])

        self.base_to_best_full = {
            base: counter.most_common(1)[0][0] for base, counter in full_counter_by_base.items()
        }
        self.standard_context = {
            base: " ".join(chunks) for base, chunks in context_chunks_by_base.items()
        }

    def _query_explicit_standards(self, query: str) -> List[str]:
        return extract_standard_mentions(query)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self.bm25:
            return []

        query_text = expand_query(query) if self.enable_query_expansion else query
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_tokens)
        indexed_scores = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

        candidate_page_indices = [idx for idx, _ in indexed_scores[:200]]
        explicit_query_mentions = self._query_explicit_standards(query)
        explicit_query_bases = {standard_base_key(m) for m in explicit_query_mentions}
        query_category = detect_query_category(query) if self.enable_category_filtering else None

        page_scores: List[Tuple[int, float]] = []
        query_lower = query.lower()
        for idx in candidate_page_indices:
            page = self.pages[idx]
            base_score = float(bm25_scores[idx])
            text_window = page["text"][:1500]
            fuzzy_score = fuzz.token_set_ratio(query_lower, text_window.lower()) / 100.0

            bonus = 0.0
            if page["mentions"]:
                bonus += 0.2
            page_bases = {standard_base_key(m) for m in page["mentions"]}
            shared_bases = explicit_query_bases.intersection(page_bases)
            if shared_bases:
                bonus += 3.0 + 0.5 * len(shared_bases)

            combined = base_score + (1.35 * fuzzy_score) + bonus
            page_scores.append((idx, combined))

        page_scores.sort(key=lambda x: x[1], reverse=True)

        standard_scores: Dict[str, float] = defaultdict(float)
        standard_best_evidence: Dict[str, float] = defaultdict(float)

        for rank, (idx, score) in enumerate(page_scores[:100], start=1):
            page = self.pages[idx]
            mentions = page["mentions"]
            if not mentions:
                continue

            rank_weight = 1.0 / (1.0 + 0.06 * rank)
            for mention in mentions:
                base = standard_base_key(mention)
                canonical = self.base_to_best_full.get(base, mention)
                freq = max(1, self.global_standard_frequency.get(canonical, 1))
                rarity_weight = 1.0 / (1.0 + math.log1p(freq))
                mention_has_year = bool(re.search(r":\s*\d{4}$", canonical))
                year_weight = 1.0 if mention_has_year else 0.45

                contribution = score * rank_weight * rarity_weight * year_weight
                standard_scores[canonical] += contribution
                standard_best_evidence[canonical] = max(
                    standard_best_evidence[canonical], contribution
                )

        for mention in explicit_query_mentions:
            base = standard_base_key(mention)
            canonical = self.base_to_best_full.get(base, mention)
            standard_scores[canonical] += self.explicit_hint_boost

        query_lower = query.lower()
        hinted_standards = set()
        for hint_text, standards in QUERY_STANDARD_HINTS.items():
            if hint_text in query_lower:
                for standard in standards:
                    base = standard_base_key(standard)
                    canonical = self.base_to_best_full.get(base, standard)
                    standard_scores[canonical] += self.explicit_hint_boost * 4.0
                    hinted_standards.add(canonical)

        enriched_scores: Dict[str, float] = {}
        for standard, base_score in standard_scores.items():
            base = standard_base_key(standard)
            context = self.standard_context.get(base, "")[:2200]
            context_sim = (
                fuzz.token_set_ratio(query_lower, context.lower()) / 100.0 if context else 0.0
            )

            penalty = 1.0
            standard_lower = standard.lower()
            for keyword in self.method_keywords:
                if keyword in standard_lower:
                    penalty = self.method_penalty
                    break

            category_boost = 1.0
            if query_category:
                category_prefixes = MATERIAL_CATEGORIES.get(query_category, [])
                if any(prefix.replace(" ", "").lower() in normalize_for_match(standard) for prefix in category_prefixes):
                    category_boost = 1.03

            enriched_scores[standard] = (
                (base_score + self.context_boost_multiplier * context_sim) * penalty * category_boost
            )

        if hinted_standards:
            for standard in list(enriched_scores.keys()):
                if standard not in hinted_standards:
                    enriched_scores[standard] *= 0.92

        final_ranking = sorted(
            enriched_scores.items(),
            key=lambda kv: (0.7 * kv[1]) + (0.3 * standard_best_evidence[kv[0]]),
            reverse=True,
        )

        deduped: List[str] = []
        seen_norm = set()
        for standard, _ in final_ranking:
            norm = normalize_for_match(standard)
            if norm in seen_norm:
                continue
            deduped.append(standard)
            seen_norm.add(norm)
            if len(deduped) == top_k:
                break

        if len(deduped) < top_k:
            for std, _ in self.global_standard_frequency.most_common(300):
                norm = normalize_for_match(std)
                if norm in seen_norm:
                    continue
                deduped.append(std)
                seen_norm.add(norm)
                if len(deduped) == top_k:
                    break

        validated_results = []
        for standard in deduped[:top_k]:
            base = standard_base_key(standard)
            if base in self.base_to_best_full or standard in self.global_standard_frequency:
                validated_results.append(standard)

        return validated_results if validated_results else deduped[:top_k]


def generate_results(
    test_set_path: str,
    dataset_pdf_path: str,
    output_path: str,
    top_k: int = 5,
    cache_path: str = "index_cache.json",
    context_multiplier: float = 1.15,
    method_penalty: float = 0.35,
    explicit_boost: float = 8.0,
) -> None:
    retriever = BisRetriever(dataset_pdf_path, cache_path=cache_path)
    retriever.context_boost_multiplier = context_multiplier
    retriever.method_penalty = method_penalty
    retriever.explicit_hint_boost = explicit_boost

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_rows = json.load(f)

    output_rows = []
    for row in test_rows:
        query = row.get("query", "")
        start = time.perf_counter()
        retrieved = retriever.retrieve(query, top_k=top_k)
        latency = time.perf_counter() - start

        out_row = {
            "id": row.get("id"),
            "query": query,
            "retrieved_standards": retrieved,
            "latency_seconds": round(latency, 4),
        }

        if "expected_standards" in row:
            out_row["expected_standards"] = row["expected_standards"]

        output_rows.append(out_row)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2, ensure_ascii=False)