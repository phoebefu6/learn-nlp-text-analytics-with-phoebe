<!-- learn-with-phoebe hub banner -->
> ### 📚 Part of [**Learn with Phoebe**](https://phoebefu6.github.io/learn-with-phoebe/)
> The shelf of 205 free, hands-on courses on AI, data, and the craft around them. **[Browse every course ↗](https://phoebefu6.github.io/learn-with-phoebe/)**
<!-- /learn-with-phoebe hub banner -->

# Learn NLP & Text Analytics with Phoebe

Text as data, before and without a large language model. Six 45-minute sessions that tokenise,
weight, classify and search real complaint text in the browser, and ask when a small, readable
classifier beats the model everyone wants to buy. By Phoebe Fu.

## Sessions

1. Text is a table you have not built yet - tokens, vocabulary, normalisation, Zipf's law
2. Weight the words - counts, TF-IDF, n-grams, vocabulary size
3. Route it: a classifier you can read - naive Bayes, logistic regression, errors as routing cost
4. The text bench - six rungs on 600 held-out complaints, two traps, a modelled LLM rung
5. Search and themes - BM25 and cosine search, k-means themes and their instability, drift
6. When the big model earns its place - cost, audit, the contested evidence, a build-or-buy note

## The bench

`assets/ntx-live.js` computes everything in the browser on `assets/ntx-sample.js`: 3,000 real
narratives from the CFPB Consumer Complaint Database (calendar 2023, six product queues, rebalanced,
first 50 words each, 600 held out). The narratives come from a February 2024 snapshot of the public
data (CC0); the current CFPB export no longer includes the narrative field. The running case,
Ledgerly, is constructed. The large-language-model rung is modelled (priced with stated assumptions,
never scored).

`materials/ntx_reference.py` reimplements the engine in plain Python; `materials/compare-ladder.py`
checks the two agree value by value (with a planted-fault self-test). `materials/build_ntx_sample.py`
rebuilds the sample from the snapshot shard.

## Licence notes

Complaint narratives: CFPB Consumer Complaint Database, public domain (CC0 1.0), published by the
CFPB with consumer consent and scrubbed of personal information.
