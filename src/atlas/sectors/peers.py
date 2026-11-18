"""Text-based peers (Spec §6.2; Ref 3.8).

1. Each company's 10-K business description (Item 1) is embedded as a vector e_i.
2. Pairwise cosine similarity S_ij = e_i . e_j / (|e_i| |e_j|).
3. A company's peers are the companies above a similarity threshold (for example the top 1% of all
   pairs).
4. Peer groups are rebuilt every year from the latest 10-K available at each date.

Embeddings come from sentence-transformers when installed. Without it, a TF–IDF representation
(scikit-learn) is used, which captures shared vocabulary and works well for business descriptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas._optional import optional_import


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Return one embedding row per text (L2-normalised)."""
    st = optional_import("sentence_transformers")
    if st is not None:
        model = st.SentenceTransformer(model_name)
        return np.asarray(model.encode(texts, normalize_embeddings=True))
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True).fit_transform(texts)
    X = X.toarray()
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def cosine_matrix(E: np.ndarray) -> np.ndarray:
    """S_ij for normalised embeddings is simply the dot product."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return En @ En.T


def peer_sets(ids: list[str], S: np.ndarray, top_pct: float = 0.01) -> dict[str, list[str]]:
    """Peers = pairs in the top `top_pct` of all pairwise similarities (excluding self-pairs)."""
    n = len(ids)
    iu = np.triu_indices(n, 1)
    thr = np.quantile(S[iu], 1 - top_pct) if len(iu[0]) else 1.0
    out = {}
    for i, eid in enumerate(ids):
        mask = (S[i] >= thr) & (np.arange(n) != i)
        out[eid] = [ids[j] for j in np.where(mask)[0]]
    return out


def peer_table(ids: list[str], texts: list[str], top_pct: float = 0.01) -> pd.DataFrame:
    """Long table (id_i, id_j, similarity) of peer pairs, ready for storage."""
    S = cosine_matrix(embed_texts(texts))
    peers = peer_sets(ids, S, top_pct)
    idx = {e: i for i, e in enumerate(ids)}
    rows = [(a, b, float(S[idx[a], idx[b]])) for a, ps in peers.items() for b in ps]
    return pd.DataFrame(rows, columns=["id_i", "id_j", "similarity"])
