"""Per-article processing pipeline (Spec §3.2; Ref 3.5).

Step 1  Duplicates: articles become sets of five-word shingles; Jaccard similarity
        J(A, B) = |A ∩ B| / |A ∪ B| is estimated with MinHash and locality-sensitive hashing;
        J > 0.8 marks the same story and only the earliest article is kept.
Step 2  Timestamps are converted to UTC and assigned to a trading session: news after the 16:00
        New York close belongs to the next trading day's return.
Step 3  Entity linking: candidate names are matched to the security master's alias table, then
        confirmed by comparing the article with the company description; matches need confidence > 0.8.
Step 4  Topic classification gives probabilities p_topic (zero-shot model or keyword fallback).
Step 5  Sentiment: s = P(positive) - P(negative) from FinBERT.
Step 6  Novelty = 1 - max cosine similarity with articles from the previous 72 hours.
Step 7  Source weights from each source's historical information coefficient, normalised so the
        best source has weight 1.0.

Heavy NLP models (transformers, sentence-transformers) are optional. Without them, the pipeline uses
transparent fallbacks: a small financial sentiment lexicon, keyword topics and TF–IDF similarity.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from atlas._optional import optional_import

TOPICS = {
    "war": ["war", "invasion", "missile", "military", "troops", "conflict"],
    "sanctions": ["sanction", "embargo", "blacklist"],
    "tariffs": ["tariff", "duties", "trade war"],
    "export_controls": ["export control", "export ban", "chip restrictions", "entity list"],
    "elections": ["election", "poll", "candidate", "ballot", "vote"],
    "regulation": ["regulation", "regulator", "antitrust", "bill", "legislation", "sec charges"],
    "mna": ["acquire", "acquisition", "merger", "takeover", "buyout"],
    "earnings": ["earnings", "revenue", "profit", "guidance", "quarterly results", "eps"],
}

# A compact finance-specific lexicon (Loughran–McDonald style) used only when FinBERT is unavailable.
POSITIVE = {"beat", "beats", "surge", "record", "growth", "upgrade", "strong", "gain", "gains", "raise",
            "raised", "profit", "outperform", "expand", "widen", "approval", "rally"}
NEGATIVE = {"miss", "misses", "plunge", "loss", "losses", "downgrade", "weak", "cut", "cuts", "lawsuit",
            "probe", "ban", "sanction", "decline", "default", "warning", "recall", "fraud", "fall", "falls"}


# ---- Step 1: deduplication --------------------------------------------------------------------
def shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class MinHasher:
    """MinHash signatures with banded locality-sensitive hashing (datasketch is not required)."""

    def __init__(self, num_perm: int = 128, bands: int = 32, seed: int = 1):
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2 ** 61 - 1, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 2 ** 61 - 1, num_perm, dtype=np.uint64)

    def signature(self, sh: set[str]) -> np.ndarray:
        hashes = np.array([int(hashlib.md5(s.encode()).hexdigest()[:15], 16) for s in sh], dtype=np.uint64)
        # (a x + b) mod p over all shingles, minimum per permutation
        p = np.uint64(2 ** 61 - 1)
        vals = (self.a[:, None] * hashes[None, :] + self.b[:, None]) % p
        return vals.min(axis=1)

    def candidate_pairs(self, sigs: list[np.ndarray]) -> set[tuple[int, int]]:
        buckets = defaultdict(list)
        for i, s in enumerate(sigs):
            for b in range(self.bands):
                buckets[(b, tuple(s[b * self.rows:(b + 1) * self.rows]))].append(i)
        pairs = set()
        for ids in buckets.values():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs.add((ids[x], ids[y]))
        return pairs


def deduplicate(articles: pd.DataFrame, threshold: float = 0.8, k: int = 5) -> pd.DataFrame:
    """Drop later near-duplicates (Jaccard > threshold), keeping the earliest version of each story."""
    a = articles.sort_values("published_utc").reset_index(drop=True)
    sh = [shingles(f"{t} {b}", k) for t, b in zip(a["title"].fillna(""), a.get("body", pd.Series("", index=a.index)).fillna(""))]
    mh = MinHasher()
    sigs = [mh.signature(s) for s in sh]
    drop = set()
    for i, j in sorted(mh.candidate_pairs(sigs)):
        if i in drop or j in drop:
            continue
        if jaccard(sh[i], sh[j]) > threshold:   # confirm candidates with the exact Jaccard
            drop.add(max(i, j))                 # later article (sorted by time) is removed
    return a.drop(index=sorted(drop)).reset_index(drop=True)


# ---- Step 2: trading-session alignment -----------------------------------------------------
def assign_session(published_utc: pd.Series, close_hour_ny: int = 16) -> pd.Series:
    """Map UTC timestamps to the trading date whose return they can affect.

    Articles after the New York close (or on weekends) are assigned to the next business day.
    Exchange holidays should be applied with a proper trading calendar in production.
    """
    ny = pd.to_datetime(published_utc, utc=True).dt.tz_convert("America/New_York")
    day = ny.dt.normalize().dt.tz_localize(None)
    after_close = ny.dt.hour >= close_hour_ny
    day = day.where(~after_close, day + pd.Timedelta(days=1))
    return day.apply(lambda d: d if d.weekday() < 5 else d + pd.offsets.BDay(1))


# ---- Step 3: entity linking --------------------------------------------------------------------
def link_entities(text: str, aliases: dict[str, str], descriptions: dict[str, str] | None = None,
                  min_confidence: float = 0.8) -> list[str]:
    """Return internal IDs mentioned in the text.

    `aliases` maps lower-case names and tickers to internal IDs. Confidence combines an exact
    alias match (0.7) with context similarity to the company description (up to 0.3, reached at a
    TF–IDF cosine of 1/3 or more, because short articles rarely share many words). Ambiguous
    names (such as "Apple") only pass when the context supports the company meaning.
    """
    tl = text.lower()
    hits = {}
    for alias, eid in aliases.items():
        if re.search(rf"\b{re.escape(alias.lower())}\b", tl):
            conf = 0.7
            if descriptions and eid in descriptions:
                conf += 0.3 * min(1.0, 3 * _tfidf_similarity(text, descriptions[eid]))
            else:
                conf += 0.15  # no description available: rely on the alias match
            hits[eid] = max(hits.get(eid, 0), conf)
    return [e for e, c in hits.items() if c > min_confidence]


def _tfidf_similarity(a: str, b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words="english").fit_transform([a, b])
    return float((X[0] @ X[1].T).toarray()[0, 0])


# ---- Step 4: topics -------------------------------------------------------------------------
def topic_probabilities(text: str) -> dict[str, float]:
    """Keyword-count topic probabilities (fallback for the zero-shot classifier)."""
    tl = text.lower()
    counts = {t: sum(tl.count(k) for k in kws) for t, kws in TOPICS.items()}
    total = sum(counts.values())
    return {t: (c / total if total else 0.0) for t, c in counts.items()}


# ---- Step 5: sentiment -----------------------------------------------------------------------
class SentimentModel:
    """FinBERT (ProsusAI/finbert) when transformers is installed, otherwise the lexicon fallback."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        tr = optional_import("transformers")
        self.pipe = None
        if tr is not None:
            try:
                self.pipe = tr.pipeline("text-classification", model=model_name, top_k=None, truncation=True)
            except Exception:
                self.pipe = None

    def score(self, texts: list[str]) -> np.ndarray:
        """s = P(positive) - P(negative), in [-1, 1]."""
        if self.pipe is not None:
            out = self.pipe(texts)
            return np.array([{d["label"].lower(): d["score"] for d in o}.get("positive", 0)
                             - {d["label"].lower(): d["score"] for d in o}.get("negative", 0) for o in out])
        scores = []
        for t in texts:
            w = re.findall(r"[a-z]+", t.lower())
            pos, neg = sum(x in POSITIVE for x in w), sum(x in NEGATIVE for x in w)
            scores.append((pos - neg) / (pos + neg) if pos + neg else 0.0)
        return np.array(scores)


# ---- Step 6: novelty -------------------------------------------------------------------------
def novelty_scores(articles: pd.DataFrame, window_hours: int = 72) -> pd.Series:
    """1 - max cosine similarity with articles published in the previous `window_hours`."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    a = articles.sort_values("published_utc")
    texts = (a["title"].fillna("") + " " + a.get("body", pd.Series("", index=a.index)).fillna("")).tolist()
    st = optional_import("sentence_transformers")
    if st is not None:
        E = np.asarray(st.SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True))
    else:
        E = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray()
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    t = pd.to_datetime(a["published_utc"], utc=True).values
    out = np.ones(len(a))
    for i in range(len(a)):
        lo = t[i] - np.timedelta64(window_hours, "h")
        prev = np.where((t < t[i]) & (t >= lo))[0]
        if len(prev):
            out[i] = 1 - float(np.max(E[prev] @ E[i]))
    return pd.Series(out, index=a.index).reindex(articles.index)


# ---- Step 7: source weights --------------------------------------------------------------------
def source_weights(source_ic: pd.Series) -> pd.Series:
    """Weights proportional to positive historical IC, normalised so the best source = 1.0."""
    w = source_ic.clip(lower=0)
    return w / w.max() if w.max() > 0 else w
