# Tier 3: retrieval quality on BEIR

nDCG@10 and Recall@100 on [BEIR](https://github.com/beir-cellar/beir) test queries, scored
the way BEIR / trec_eval do (linear gains, log2 discount, ideal from the qrels). Docs are
loaded with `dqadb import` (title -> `rdfs:label`, text -> `beir:text`, both full-text
indexed; the embedding text is title + text), then searched with `Kb::search`.

Modes:

| mode | query |
|---|---|
| full-text BM25 (any word) | the query's words joined with `OR` (as the BM25 baselines do) |
| full-text (raw query) | the query as typed: dqaDB's default (any word, BM25; AND until 2026-11-11) |
| semantic | bge-m3 (ONNX, CLS, cosine) over the HNSW segments |
| hybrid | RRF of the any-word BM25 and semantic rankings (`--text-mode fuse`) |

```sh
B=dqadb-bench-retrieval   # CARGO_TARGET_DIR=... cargo build --release
$B subset beir/scifact beir/scifact-912 912      # optional, see below
$B load beir/scifact-912 work/scifact-912        # import + embed (the embedding cache is kept)
$B eval beir/scifact-912 work/scifact-912
```

## Results (2026-11-11)

**SciFact, full corpus (5,183 docs, 300 queries), full-text only:**

| | nDCG@10 | Recall@100 |
|---|---|---|
| dqaDB full-text, any word | **0.661** | 0.889 |
| dqaDB full-text, raw query (default since 2026-11-11) | 0.651 | 0.874 |
| dqaDB full-text, all words (the old default) | 0.018 | 0.018 |
| Pyserini BM25 flat (BEIR leaderboard) | 0.679 | |
| Pyserini BM25 multifield | 0.665 | |

dqaDB's BM25 (tantivy) is within 0.02 of Lucene's with no stemming or English stop words:
the analyzer is language-neutral (accent / đ folding only) so Vietnamese works.

**SciFact subset: the first 912 docs, the 55 test queries whose relevant docs are all in
it.** bge-m3 runs at ~5 s per doc on the benchmark laptop (i5-4288U, 2 cores, no GPU), so
the full corpus would take ~7 h to embed. A 912-doc corpus is easier than 5,183 docs:
compare the rows with each other, not with the full-corpus numbers.

| | nDCG@10 | Recall@100 | ms/query |
|---|---|---|---|
| full-text, any word | 0.796 | 0.945 | 2 |
| full-text, raw query (default) | 0.804 | 0.945 | 2 |
| full-text, all words (old default) | 0.036 | 0.036 | 1 |
| semantic (bge-m3) | 0.735 | 0.982 | 404 |
| hybrid (RRF) | **0.811** | **0.982** | 333 |

Semantic query time is the bge-m3 query embedding on this CPU, not the index.

**Zalo legal text retrieval (Vietnamese; 61,425 law articles, 788 test questions;
[GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), MIT), full-text only:**

Title ("Điều 1. Phạm vi áp dụng") -> `rdfs:label`, article text -> `beir:text`. Raw query
(dqaDB default), nDCG@10 / Recall@100:

| | Zalo legal (vi) | SciFact (en) |
|---|---|---|
| words only (before) | 0.551 / 0.910 | 0.651 / 0.874 |
| + word bigrams x0.1 | 0.654 / 0.948 | 0.657 / 0.874 |
| + bigrams x0.2 | 0.695 / 0.959 | 0.659 / 0.871 |
| + bigrams x0.3 | 0.710 / 0.967 | 0.652 / 0.871 |
| **+ bigrams x0.4 (default)** | **0.713 / 0.970** | 0.650 / 0.870 |
| + bigrams x0.5 | 0.713 / 0.971 | 0.642 / 0.874 |
| + bigrams x1 | 0.709 / 0.968 | 0.604 / 0.866 |
| + bigrams x2 | 0.703 / 0.964 | 0.552 / 0.837 |

With bigrams x0.4, title weight (`rdfs:label` `dqadb_index: fts^w`):

| title weight | Zalo legal | SciFact |
|---|---|---|
| 1 (default) | 0.713 / 0.970 | 0.650 / 0.870 |
| **1.5** | **0.730** / 0.968 | **0.652** / 0.879 |
| 2 | 0.713 / 0.953 | 0.620 / 0.879 |
| 3 | 0.659 / 0.916 | 0.566 / 0.859 |

Bigrams are the word in Vietnamese (syllables are written apart: "hợp đồng" = contract),
in English they are mostly phrases, hence the different optimum; 0.4 sits on the flat
top for Vietnamese at no cost in English. The full-text index grows ~3.6x (bigram terms;
59 MB for the 61k articles vs 506 MB for the SQLite index). ~20 ms per query at 61k
articles on this laptop. Sweeps: `DQADB_BIGRAM_BOOST=<x>` overrides the weight.

**ALQAC 2026 task 1 (the 18 Vietnamese laws kb ingests: 3,352 articles; 50 case
questions with their gold articles), full-text only.** Gold articles are often procedural
(court fees, Civil Procedure Code) and share few words with the case: hard lexically for
both systems. kb = art-ws `score-retrieval.ts`, run a26b (LanceDB, 2026-11-11), which
searches the question and the case facts together; 50 questions, so differences of a
few hundredths are noise.

| | R@10 | R@20 | MRR | nDCG@10 |
|---|---|---|---|---|
| kb (LanceDB), question + facts | **0.065** | **0.090** | 0.106 | |
| dqaDB, question | 0.040 | 0.076 | **0.141** | 0.053 |
| dqaDB, question + facts (RRF of two searches) | 0.035 | 0.072 | 0.124 | 0.045 |
| dqaDB, question and facts as one query | 0.014 | 0.029 | 0.040 | 0.017 |

## What this found

- **Bug, fixed:** `fold()` lower-cased the whole query, so `a OR b` searched for the word
  "or" (OR behaved as AND). Operators `AND` / `OR` / `NOT` now stay as they are
  (`fts::fold_query`), as in Lucene / tantivy syntax.
- **AND by default was wrong for natural-language queries** (0.018): a long question
  rarely has every word in one abstract. Changed to any word, ranked by BM25, as in
  Lucene / Elasticsearch `match` / Solr / tantivy (Meilisearch: all words first, then
  drops words). Words can still be required: `a AND b`, `+a`, `"a b"`. The raw query is
  0.01 below the bare-words query because characters like `-` and `:` in a question are
  read as query syntax.
- **Hybrid > either alone** on the subset (+0.015 over BM25, +0.076 over semantic),
  as in the BEIR / Qdrant hybrid results. With 55 queries the difference between
  full-text and hybrid is within noise; the semantic gap is not.

- **Vietnamese needs word bigrams:** +0.16 nDCG@10 on Zalo legal (0.551 -> 0.713), +0.18
  with the title weighted 1.5 (0.730).

Not yet run: NFCorpus, FiQA, ArguAna (the Hugging Face CDN is blocked from this network;
download the BEIR zips by hand into `beir/<name>/`).
