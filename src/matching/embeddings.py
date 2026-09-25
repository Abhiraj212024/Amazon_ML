"""
Optional dense-embedding stage: a sixth blocking channel plus a pairwise
feature.

The lexical channels are strong on typos and abbreviations but weak on
transliteration and wording that shares no character n-grams. That is exactly
where the measured India/US gap sits (0.9425 vs 0.9740), and it is the only
lever available for the unseen test country.

This module is opt-in. It needs `sentence-transformers` (encoding) and
`hnswlib` (approximate search); an exact search over ~100k x 900k records is
not tractable, so the ANN index is not optional in practice. Both are imported
lazily, so the rest of the pipeline runs unchanged when they are absent.

Model choice is left to the caller via `model_name`. Verify the licence on the
model card before committing to one: the challenge requires an MIT/Apache-2.0
model of at most 8B parameters. Multilingual models are the right family here,
because the training data is transliterated Indian and US text and the test set
adds France.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def serialise_records(df):
    """
    One string per record for the encoder.

    Name, address and country are kept as labelled fields rather than blindly
    concatenated, so the model can tell a town from a company.
    """
    def column(name):
        if name not in df.columns:
            return np.full(len(df), "", dtype=object)
        return df[name].fillna("").astype(str).to_numpy(dtype=object)

    name_col = "business_name_clean" if "business_name_clean" in df.columns else "business_name"
    names = column(name_col)
    addresses = column("business_address_clean")
    countries = column("country_clean")

    return [
        f"name: {n} | address: {a} | country: {c}"
        for n, a, c in zip(names, addresses, countries)
    ]


class SentenceTransformerEncoder:
    """
    Thin wrapper returning L2-normalised float32 vectors.

    Normalising at encode time means a dot product is the cosine, which both
    the ANN index and the pairwise feature rely on.
    """

    def __init__(self, model_name=DEFAULT_MODEL, batch_size=256, device=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on environment
            raise ImportError(
                "the embedding stage needs sentence-transformers: "
                "pip install sentence-transformers"
            ) from error

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.model_name = model_name

    def encode(self, texts):
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)


def _exact_topk(query, pool, k, min_score):
    """Exact search, used only for inputs small enough that ANN is pointless."""
    hits = {}
    similarity = query @ pool.T
    k = min(k, pool.shape[0])
    for row in range(similarity.shape[0]):
        order = np.argpartition(-similarity[row], k - 1)[:k]
        row_hits = {
            int(col): float(similarity[row, col])
            for col in order
            if similarity[row, col] >= min_score
        }
        if row_hits:
            hits[row] = row_hits
    return hits


def embedding_topk(query_vectors, pool_vectors, k=25, min_score=0.5,
                   ef_construction=200, m=32, ef_search=None, n_threads=-1,
                   exact_below=2000):
    """
    Top-k pool rows per query row by cosine similarity.

    Returns {query_index: {pool_index: score}}, matching the shape the other
    blocking channels produce.
    """
    if len(query_vectors) == 0 or len(pool_vectors) == 0:
        return {}
    if len(pool_vectors) <= exact_below:
        return _exact_topk(query_vectors, pool_vectors, k, min_score)

    try:
        import hnswlib
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "the embedding channel needs hnswlib for approximate search: "
            "pip install hnswlib"
        ) from error

    dim = pool_vectors.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(pool_vectors), ef_construction=ef_construction, M=m)
    if n_threads and n_threads > 0:
        index.set_num_threads(n_threads)
    index.add_items(pool_vectors, np.arange(len(pool_vectors)))
    # ef must be at least k, and higher ef trades query time for recall.
    # Measured against exact search on structureless vectors (the worst case
    # for HNSW): ef=2k gave 0.82 recall, ef=8k gave 0.999 for ~15% more query
    # time. Blocking recall is the entire point of this channel, so pay it.
    index.set_ef(max(ef_search or 8 * k, k + 1))

    k = min(k, len(pool_vectors))
    labels, distances = index.knn_query(query_vectors, k=k)

    hits = {}
    for row in range(labels.shape[0]):
        # hnswlib returns cosine *distance* = 1 - cosine similarity
        row_hits = {
            int(col): float(1.0 - dist)
            for col, dist in zip(labels[row], distances[row])
            if (1.0 - dist) >= min_score
        }
        if row_hits:
            hits[row] = row_hits
    return hits


def build_embedding_lookup(df, encoder):
    """entity_id -> row index, plus the matching matrix of vectors."""
    texts = serialise_records(df)
    vectors = encoder.encode(texts)
    index = {entity_id: row for row, entity_id in enumerate(df["entity_id"].astype(str))}
    logger.info("encoded %d records into %d dimensions", len(vectors), vectors.shape[1])
    return index, vectors


def pair_cosines(pair_index, s1_index, s1_vectors, pool_index, pool_vectors,
                 chunk_size=200_000):
    """
    Cosine similarity for every candidate pair, in order.

    Computed in chunks: gathering all pairs' vectors at once would need tens of
    gigabytes at full scale, while a chunked gather is a few hundred megabytes.

    Pairs whose records have no vector score 0.0.
    """
    s1_ids = pair_index["s1_entity_id"].to_numpy(dtype=object)
    cand_ids = pair_index["candidate_entity_id"].to_numpy(dtype=object)
    out = np.zeros(len(s1_ids), dtype=np.float32)
    if len(out) == 0:
        return out

    s1_rows = np.array([s1_index.get(i, -1) for i in s1_ids], dtype=np.int64)
    cand_rows = np.array([pool_index.get(i, -1) for i in cand_ids], dtype=np.int64)
    valid = (s1_rows >= 0) & (cand_rows >= 0)

    for start in range(0, len(out), chunk_size):
        stop = min(start + chunk_size, len(out))
        mask = valid[start:stop]
        if not mask.any():
            continue
        left = s1_vectors[s1_rows[start:stop][mask]]
        right = pool_vectors[cand_rows[start:stop][mask]]
        chunk = np.einsum("ij,ij->i", left, right)
        block = out[start:stop]
        block[mask] = chunk
        out[start:stop] = block

    return out
