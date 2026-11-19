# Official course map - learn-nlp-text-analytics-with-phoebe ("NLP & Text Analytics")

Internal build document (never linked from an audience page). Phase 2, written 2026-11-19 before any
page. Every number a page quotes must appear here, with its tier. Tiers: **read-at-source** (we read
the page, PDF or abstract ourselves), **reported** (seen second-hand only: say "reported"),
**computed** (printed by the bench on the shipped sample; Python and node agree), **constructed**
(the Ledgerly case: fictional, label "Constructed Ledgerly example").

Audience: analysts and data scientists handed free text (complaints, tickets, survey answers,
reviews) and told to "just use an LLM". Question across all six sessions: what can be counted,
classified and searched cheaply, auditably and reproducibly, and where does a large model earn its
cost? Single track, 6 x 45 min, 22 min hands-on each. Builder audience. Light Python on the page;
every build-along also runs in the browser.

## The data (read-at-source, 2026-11-19)

- **CFPB Consumer Complaint Database.** The API `_meta` block reads `"license":"CC0"`,
  `total_record_count` 19,251,022, `last_updated` 2026-11-18. The current field list on
  consumerfinance.gov/complaint/data-use/ ("How we share complaint data", page last modified
  Sept. 14, 2026) has 15 fields and **no "Consumer complaint narrative" field**; the API search
  results and the bulk CSV header (complaints.csv.zip, 368,685,609 bytes, last-modified 18 Nov 2026)
  carry no narrative column either. So the narratives can only be studied today from an earlier
  public release. Teach this as a fact about the source: text you depend on can be withdrawn.
  We do not know why or when it was removed; do not speculate on the page.
- **Rows used:** the CFPB export as mirrored on Hugging Face by BEE-spoke-data,
  `consumer-finance-complaints`, config `has-text` (1,689,573 rows with a narrative, dates
  2015-03-19 to 2024-02-09; the card says "Pulled Feb 2024"; licence CC0 1.0; original CFPB column
  names incl. "Consumer complaint narrative" and "Consumer consent provided?"). Shard
  `has-text/train-00000-of-00003.parquet`, sha256 60902c32f0a6b3db25a378570d421765d0a15ced396c0c471a9d60e49b38d833
  (matches the LFS oid). Narratives are published only with the consumer's consent and CFPB scrubs
  personal information; redactions show as XXXX (visible in the rows: 8.1% of shipped tokens).
- **Sample** (`materials/build_ntx_sample.py`, seed 2023): calendar 2023 complaints in the shard, six
  queues, fixed quotas, 3,000 narratives, stratified holdout 600 (20%), 2,400 train.

| Queue key | Label | Quota | Test | Shard 2023 rows (one third of the mirror) |
|---|---|---|---|---|
| report | Credit reporting | 900 | 180 | 206,282 |
| collect | Debt collection | 600 | 120 | 19,822 |
| card | Credit card | 450 | 90 | 19,091 |
| bank | Checking or savings | 450 | 90 | 15,947 |
| student | Student loan | 300 | 60 | 3,655 |
| auto | Vehicle loan or lease | 300 | 60 | 4,154 |

  Real 2023 mix in the shard: credit reporting is 206,282 of 268,951 = 76.7% (computed from the
  shard counts above). The sample is REBALANCED (credit reporting 30%); say so wherever accuracy is
  quoted. Export choices (each a normalisation decision): lowercase; apostrophes removed; numbers,
  money amounts and punctuation dropped; every XX/XXXX redaction becomes the token `xxxx`; only the
  first 50 word tokens kept (84.1% of narratives were longer; median full length 136 words). Ships
  as 388 KB of packed token ids.
- **Label taxonomy changed mid-year (read in the data):** in August 2023 the product "Credit card or
  prepaid card" stops and "Credit card" and "Prepaid card" begin; "Credit reporting, credit repair
  services, or other personal consumer reports" becomes "Credit reporting or other personal consumer
  reports" the same month. The builder maps both names to one queue. Teaching use: s1 (corpus card
  records the mapping), s5 (drift), s6 (a label schema is part of what you audit).

## Canon numbers (computed; node = Python on 857 values, planted 0.1% fault caught)

Run: `node materials/ntx-node.js > canon-node.json`, `python3 -I materials/ntx_reference.py
assets/ntx-sample.js > canon-python.json`, `python3 -I materials/compare-ladder.py canon-node.json
canon-python.json` (exit 0). All on the 600-row holdout unless stated.

**Corpus card:** 3,000 documents; 141,198 tokens; vocabulary 5,250 word types; 2,094 seen once
(hapax, 39.9% of the vocabulary); redaction token `xxxx` = 8.1% of all tokens and the single most
frequent token (11,461); median full narrative 136 words; mean full length of the 600 test narratives
191.8 words.

**Zipf (top 1,000 ranks, least-squares slope of log10 frequency on log10 rank):** raw tokens slope
-1.11, top 10 types = 31.4% of all tokens (xxxx, i, the, to, my, a, and, of, on, that). After stop
words + light stemmer + redaction removal: slope -0.98, vocabulary 4,050, top types account, credit,
report, payment, loan, information, bank, card, call, debt.

**Normaliser (s1; widget prints vocabulary and tokens kept for every toggle mix):** raw 5,250 /
141,198; xxxx out 5,249 / 129,737; stop words out 5,131 / 76,313 (stop words = 46% of all tokens);
stemmer only 4,159 / 141,198; all three 4,050 / 64,852. Example complaint 6400037 (sample row 18,
credit reporting): 50 tokens in; 47 after xxxx out; 26 after stop words; stemmer merges reporting and
report, and also produces "dat" (dated), "previou" (previous), "receiv" (received). Stemmer errors on
the corpus: "wells" -> "well" (Wells Fargo), "charged" -> "charg" but "charge" stays (split),
"violated" -> "violat" but "violation" stays. A row has at most 50 non-zero cells of 5,250 columns:
more than 99% zeros.

**The ladder (s4 bench; s3 quotes r2-r4):**

| Rung | Accuracy | Correct / 600 | Macro F1 |
|---|---|---|---|
| r0 Majority queue (credit reporting) | 0.300 | 180 | 0.077 |
| r1 Keyword rules | 0.610 | 366 | 0.615 |
| r2 Counts + naive Bayes (alpha 1) | 0.762 | 457 | 0.760 |
| r3 TF-IDF + naive Bayes (alpha 1) | 0.608 | 365 | 0.569 |
| r4 TF-IDF + bigrams (min df 2) + logistic (10 epochs SGD, lr 1) | 0.778 | 467 | 0.779 |
| r5 Tuned on a validation split (picks grid row 8: logistic, stemmer on, stop words kept, lr 2) | 0.790 | 474 | 0.791 |

- r3 is WORSE than r2: TF-IDF with l2 normalisation shrinks every value below 1, so naive Bayes'
  class prior dominates and 175 of 180 credit reporting complaints are right but recall collapses
  elsewhere (auto 0.283, student 0.450). A better weighting is not better for every model.
- r1 keyword rules: credit reporting recall 0.922 but precision 0.466 (everything that falls through
  lands there); card recall 0.333.
- Tuned grid validation accuracies (val = every 4th training row, 600 rows): NB counts a1 u 0.768,
  a1 uni+bi 0.748, a0.3 u 0.760, a0.3 uni+bi 0.735; LR (stop,stem,lr): off/off/1 0.777, off/off/2
  0.778, off/on/1 0.767, **off/on/2 0.787 (chosen)**, on/off/1 0.768, on/off/2 0.777, on/on/1 0.768,
  on/on/2 0.768. Stop-word removal never helped the logistic model here.
- Per-class r5 (precision / recall / F1): report .773/.906/.834; collect .760/.608/.676; card
  .770/.744/.757; bank .773/.833/.802; student .877/.833/.855; auto .885/.767/.821.
- r5 confusion: 29 of 120 debt collection complaints routed to credit reporting (collectors report
  debts to the bureaus, the vocabulary overlaps). r4: 23 of 120.
- **Traps:** score on the training rows: r2 0.890, r4 0.975 (looks wonderful, measures memory).
  Shuffle the training labels (break button, seed 13): r2 0.247, r4 0.195, below the 0.300 majority
  rung. Misroutes per 1,000 at r5: 210.

**Top terms per queue (s2):** counts with stop words kept: every queue's top 7 is xxxx, i, the, to,
my, a, and (in varying order). TF-IDF alone still leads with xxxx, i, the. TF-IDF + stop words +
redaction removed + bigrams (min df 2; 9,033 features vs 5,250 unigram): report: credit, report,
reporting, credit_report, information, accounts, section, consumer; collect: debt, credit,
collection, account, report, company, credit_report, received; card: card, credit_card, credit, late,
account, payment, never, bank; bank: account, bank, checking, check, money, checking_account,
deposit, funds; student: loans, loan, student, mohela, payments, payment, plan, nelnet; auto: car,
vehicle, loan, payment, auto, purchased, finance, payments.

**Search (s5; whole corpus, stop + stem + redaction off, BM25 k1 1.2 b 0.75, Lucene idf
ln(1+(N-df+0.5)/(df+0.5))):** "late fee refund" 274 docs match a term; "repossessed my car" 139;
"identity theft inquiry" 284 (BM25 and cosine agree on all five top ids); "zelle scam" 13 (top 1
agrees, the rest reorder). Top ids are real CFPB complaint ids (e.g. 6594476, 6475003, 6697815,
6551957).

**Themes (s5; spherical k-means on TF-IDF, stop + stem + redaction removed, min df 5 = 1,365
features, k 6, 20 iterations):** seed 1 sizes 722 / 522 / 333 / 214 / 489 / 720; top terms include
[report credit information item account theft] (purity 0.75 credit reporting), [call car well fargo
time vehicle] (the stemmer turned "Wells" into "well"), [right u section violat consumer privacy] (a
legal-template theme: FCRA section citations), [payment loan late student month account] (student
loan 0.50 purity). Seed 2 gives different clusters; pair agreement between seed 1 and seed 2 (pairs
together in both / pairs together in either) = 0.546. Teach: themes are a view, not a truth.
- Drift: the [payment loan late student] theme in seed 1 by month Jan-Dec 2023: 35, 34, 35, 23, 21,
  18, 32, 46, 64, 79, 61, 41 documents: it more than quadruples from June (18) to October (79). FSA:
  interest resumed September 1, 2023 and payments resumed in October 2023 (read-at-source, below).
  Say "in this sample" and "coincides with"; do not claim cause.

**LLM comparison rung (MODELLED, s4/s6):** nothing is sent anywhere. Assumptions printed on the
widget: 1.3 tokens per English word, a 120-token instruction naming six queues, 4 output tokens. The
600 test narratives at FULL length = 115,069 words, 221,590 input tokens, 2,400 output tokens. At
placeholder prices $1 per million input and $5 per million output tokens: $0.39 per 1,000
complaints. Placeholder prices are not a quote: re-verify the vendor's price list before delivery.
Accuracy of an LLM on this holdout is NOT measured (no API key, no paid tool): pages teach the
published disagreement instead.

## Verified sources (read 2026-11-19)

1. Jurafsky, D. and Martin, J. H. *Speech and Language Processing*, 3rd ed. draft, online manuscript
   released August 19, 2026. https://web.stanford.edu/~jurafsky/slp3/ (read-at-source). Ch 4
   "Logistic Regression and Text Classification"; Ch 5 "Embeddings" (cosine, Eq 5.10, "By far the
   most common similarity metric is the cosine"); Ch 11 "Information Retrieval and Retrieval-Augmented
   Generation" (tf = 1 + log10 count, idf = log10 N/df, Eq 11.4-11.6; footnote: log10(count+1) "used
   in earlier editions"). **The current draft has no naive Bayes chapter.** Naive Bayes is cited from
   the August 20, 2024 draft, Ch 4 "Naive Bayes, Text Classification, and Sentiment" (bag of words,
   s4.1; conditional independence assumption; add-one smoothing Eq 4.14) - read via a Wayback copy;
   cite as the 2024 draft.
2. Manning, C. D., Raghavan, P. and Schutze, H. (2008) *Introduction to Information Retrieval*,
   Cambridge University Press. https://nlp.stanford.edu/IR-book/ (read-at-source via Wayback HTML).
   BM25 page: "reasonable values are to set k1 and k3 to a value between 1.2 and 2 and b = 0.75";
   "A k1 value of 0 corresponds to a binary model". Tf-idf weighting is section 6.2.2. Zipf: "the
   collection frequency cf_i of the i th most common term is proportional to 1/i". Section numbers
   11.4.3 and 5.1.2 not visible on the pages: do not print them.
3. Robertson, S. and Zaragoza, H. (2009) "The Probabilistic Relevance Framework: BM25 and Beyond",
   Foundations and Trends in Information Retrieval 3(4): 333-389 (read-at-source, author PDF). b = 1
   full length normalisation, b = 0 none; "A common combination would be b = 0.5 and k1 = 2"; "many
   experiments suggest a somewhat lower value of k1 and a somewhat higher value of b". **Differs from
   IIR's default: teach that defaults disagree; the bench uses 1.2 / 0.75 and lets you move both.**
4. scikit-learn user guide, "Text feature extraction" and API pages (read-at-source):
   https://scikit-learn.org/stable/modules/feature_extraction.html - "There are several known issues
   in our provided 'english' stop word list"; stop lists "may include words that are highly
   informative to some tasks"; TfidfTransformer defaults norm='l2', use_idf=True, smooth_idf=True,
   sublinear_tf=False; smooth idf = log((1+n)/(1+df(t))) + 1 (natural log in the code; the guide
   writes "log"). MultinomialNB alpha default 1.0.
5. Blei, D. M., Ng, A. Y. and Jordan, M. I. (2003) "Latent Dirichlet Allocation", JMLR 3: 993-1022
   (read-at-source): "a generative probabilistic model for collections of discrete data such as text
   corpora".
6. Lee, D. D. and Seung, H. S. (1999) "Learning the parts of objects by non-negative matrix
   factorization", Nature 401: 788-791 (abstract read-at-source; pages from Crossref): non-negativity
   "allow[s] only additive, not subtractive, combinations"; learns "semantic features of text".
7. Gilardi, F., Alizadeh, M. and Kubli, M. (2023) "ChatGPT outperforms crowd workers for
   text-annotation tasks", PNAS 120(30): e2305016120 (abstract read-at-source): 6,183 tweets and news
   articles; zero-shot accuracy "exceeds that of crowd workers by about 25 percentage points on
   average"; per-annotation cost "less than $0.003", about thirty times cheaper than MTurk. (The
   arXiv preprint says twenty times and 2,382 tweets: cite PNAS.)
8. Bucher, M. J. J. and Martini, M. (2024) "Fine-Tuned 'Small' LLMs (Still) Significantly Outperform
   Zero-Shot Generative AI Models in Text Classification", arXiv:2406.08660 (read-at-source):
   fine-tuned smaller models "consistently and significantly outperform larger, zero-shot prompted
   models" (GPT-3.5, GPT-4, Claude Opus); fine-tuning wins "in all cases". No numeric headline in the
   abstract.
9. Ziems, C. et al. (2024) "Can Large Language Models Transform Computational Social Science?",
   Computational Linguistics (arXiv:2305.03514, read-at-source): 13 models, 25 benchmarks; on
   classification LLMs "fail to outperform the best fine-tuned models" (quote checked); their
   explanations "often exceed the quality of crowdworkers' gold references". Volume and pages not read.
10. Kocon, J. et al. (2023) "ChatGPT: Jack of all trades, master of none", arXiv:2302.10724
    (abstract read-at-source): "the average loss in quality of the ChatGPT model was about 25% for
    zero-shot and few-shot evaluation" against the best models, across 25 tasks. Venue not read.
11. Porter, M. F. (1980) "An algorithm for suffix stripping", Program 14(3): 130-137
    (read-at-source, Emerald). The bench's stemmer is a lighter suffix stripper, NOT Porter's.
12. US Department of Education, Federal Student Aid, FY2024 Annual Report, p. 45
    (https://studentaid.gov/sites/default/files/fy2024-fsa-annual-report.pdf, read-at-source):
    "student loan interest would resume on September 1, 2023, and payments resumed in October 2023".
13. CFPB, "How we share complaint data", https://www.consumerfinance.gov/complaint/data-use/
    (read-at-source, page modified Sept. 14, 2026); Consumer Complaint Database API
    https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/ (`_meta` licence
    CC0, read 2026-11-19).
14. BEE-spoke-data, `consumer-finance-complaints` (Hugging Face dataset card, CC0 1.0, "Pulled Feb
    2024"), https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints (read-at-source).

Reported only (say "reported" or leave out): the 12-month student loan "on-ramp" (Oct 2023 to Sep
2024); any current LLM price; IIR section numbers 11.4.3 and 5.1.2; Ziems volume/pages; Kocon venue.

## The contested question (s6 teaches the disagreement, never resolves it)

- For zero-shot LLM labelling: Gilardi et al. 2023 (beats crowd workers by ~25 points, ~30x cheaper
  per label than MTurk).
- Against: Bucher and Martini 2024 (fine-tuned smaller models win in all cases tested); Kocon et al.
  2023 (~25% average quality loss vs the best models); Ziems et al. 2024 (fails to beat the best
  fine-tuned models on classification, useful as a collaborator).
- Why both can be true: different baselines (crowd workers vs fine-tuned models), different tasks,
  different model generations. The bench adds the cost and audit columns and leaves accuracy for the
  learner to measure on their own labelled sample.

## Seams (pick the side out loud, link absolute https, never reteach)

| Sibling | They own | This course owns | Link from |
|---|---|---|---|
| learn-classification-regression c6 | naive Bayes as a METHOD (derivation) | NB as one rung on text | s3 https://phoebefu6.github.io/learn-classification-regression-with-phoebe/courses/c6-naive-bayes.html |
| learn-ship-internal-ai-assistant b2 | BM25 by hand for clause retrieval | search as an analytics tool over a corpus | s5 https://phoebefu6.github.io/learn-ship-internal-ai-assistant-with-phoebe/courses/b2-chunk-index-retrieve.html |
| learn-rag b1, b4 | embeddings, retrieval for an LLM | one comparison rung only | s6 https://phoebefu6.github.io/learn-rag-with-phoebe/courses/b1-embeddings.html |
| learn-recommendation b5 | TF-IDF item profiles | TF-IDF as document weighting for classification and themes | s2 https://phoebefu6.github.io/learn-recommendation-with-phoebe/courses/b5-content-based.html |
| learn-data-literacy 07 | reading model output as a consumer | building and checking it | s1 https://phoebefu6.github.io/learn-data-literacy-with-phoebe/courses/07-ml-basics-segment.html |
| learn-intro-ml b2, b7 | train/test split, classification metrics | applied to text only | s3 https://phoebefu6.github.io/learn-intro-ml-with-phoebe/courses/b7-classification-metrics.html (and b2-train-test-split.html) |
| learn-survey-design | questionnaire design | none; coding open-text answers = homework transfer | s5/s6 homework https://phoebefu6.github.io/learn-survey-design-with-phoebe/ |

## Running case (constructed)

**Ledgerly** - a constructed US consumer-lending fintech (cards, student refinance, auto loans, a
deposit account; it also furnishes data to credit bureaus and uses collectors). Its complaints team
must route each complaint to one of six product queues and spot rising issues. Label every use
"Constructed Ledgerly example". The complaints themselves are REAL CFPB narratives about many
companies; Ledgerly is the fictional team doing the analysis. Running artifact: **Ledgerly's text
playbook**, one page that grows each session: s1 corpus card -> s2 vocabulary and weighting choice
-> s3 routing classifier with its error table -> s4 bench result with the trap rows -> s5 search
setup + theme watchlist -> s6 signed build-or-buy note.

## Per-session coverage

1. **Text is a table you have not built yet** (green). Documents, tokens, vocabulary; normalisation
   (case, stop words, stemming, redaction marks) and what each throws away; Zipf on the real corpus;
   the vanished narrative field. Build-along: Ledgerly's corpus card (widgets: corpus card, normaliser,
   Zipf). Sources 1 (2024 draft ch 4 bag of words), 2 (Zipf), 4 (stop list caveat), 11, 13, 14.
2. **Weight the words** (yellow). Counts vs TF-IDF vs n-grams; sparse vectors; vocabulary size vs
   signal; smooth idf formula; sublinear tf mentioned. Build-along: top terms per queue under three
   weightings. Sources 1 (ch 11 tf-idf), 2 (6.2.2), 4.
3. **Route it: a classifier you can read** (yellow). Naive Bayes and logistic regression on the
   vectors; reading weights; per-class precision/recall; the confusion table as a routing cost.
   Build-along: routing classifier r2 vs r4 and the error table. Sources 1 (2024 ch 4, 2026 ch 4), 4.
4. **The text bench** (orange). Whole ladder r0-r5, validation tuning, traps (score on train,
   shuffled labels), the modelled LLM rung (cost and audit only). Sources 1, 4, 7-10.
5. **Search and themes** (orange). BM25 vs cosine; k1/b defaults disagree (IIR vs R&Z); k-means themes,
   seed instability (0.546), NMF and LDA named and described as alternatives (not run); theme drift
   by month (June 18 -> October 79). Build-along: Ledgerly's theme watchlist. Sources 2, 3, 5, 6, 12.
6. **When the big model earns its place** (red). Cost per thousand, latency, audit and
   reproducibility vs accuracy; the contested evidence; a decision rule; capstone: Ledgerly's signed
   build-or-buy note. Sources 7-10, 13.

## Not covered (honest list)

Word embeddings and transformers beyond one comparison mention (learn-rag owns them); fine-tuning
any neural model; LDA and NMF are described, not computed (the bench runs k-means only); named-entity
recognition; sentiment lexicons; languages other than English; LLM accuracy on this holdout (not
measured: no API key); any current LLM price (placeholder only); the reasons the CFPB removed the
narrative field.

## Citation appendix (exact strings for pages)

- Jurafsky and Martin (2026 draft), Speech and Language Processing, 3rd ed., ch 4, 5, 11.
- Jurafsky and Martin (2024 draft), ch 4 "Naive Bayes, Text Classification, and Sentiment".
- Manning, Raghavan and Schutze (2008), Introduction to Information Retrieval, Cambridge UP.
- Robertson and Zaragoza (2009), Foundations and Trends in IR 3(4): 333-389.
- scikit-learn documentation, "Text feature extraction" (read 2026-11-19).
- Blei, Ng and Jordan (2003), JMLR 3: 993-1022.
- Lee and Seung (1999), Nature 401: 788-791.
- Gilardi, Alizadeh and Kubli (2023), PNAS 120(30): e2305016120.
- Bucher and Martini (2024), arXiv:2406.08660.
- Ziems et al. (2024), Computational Linguistics (arXiv:2305.03514).
- Kocon et al. (2023), arXiv:2302.10724.
- Porter (1980), Program 14(3): 130-137.
- Federal Student Aid, FY2024 Annual Report, p. 45.
- CFPB Consumer Complaint Database (CC0), narratives from the Feb 2024 snapshot.

## Extra canon for sessions 2, 3, 5, 6 (computed 2026-11-19; node = Python on 1,001 values)

**What the logistic model learned (s3; bench "What the logistic model learned" panel, six heaviest
weights per queue).** R4: report: report 4.40, credit 4.12, experian 3.53, bureaus 3.44, these 3.33,
credit_bureaus 2.95 · collect: debt 8.96, collection 6.35, company 3.26, a_debt 3.21, collections 3.05,
report 2.66 · card: card 11.08, credit_card 6.68, for 2.74, credit 2.70, late 2.42, case 2.23 · bank:
bank 6.81, account 5.48, deposit 5.02, money 4.14, checking 4.10, my_account 3.90 · student: loans
7.32, loan 6.85, mohela 6.59, student 5.21, nelnet 4.47, aidvantage 3.71 · auto: car 7.93, vehicle
6.34, loan 4.45, auto 4.27, finance 4.23, my_car 3.51. Note: "report" is a heavy word for debt
collection too (2.66) - the overlap behind the 29 misroutes. Company names (mohela, nelnet,
aidvantage = federal student loan servicers; experian = a credit bureau) are among the strongest
signals: a model that routes by company name breaks when the company mix changes.
R5 (stemmer on): report: report 7.00, experian 5.00, these 4.75, credit_bureau 4.48 ... · collect:
debt 10.90, collection 9.48, company 3.97, apartment 3.91 ... · card: card 13.71, credit_card 7.84,
balance 3.75 ... · bank: bank 8.80, deposit 8.52, check 6.76 ... · student: loan 12.21, mohela 8.56,
nelnet 6.22 ... · auto: car 9.72, vehicle 8.50, finance 5.79 ...

**Search details (s5; widget, BM25 k1 1.2 b 0.75).**
- "late fee refund": 274 complaints contain a query term. BM25 top 5: 6594476 (Vehicle loan),
  7914608 (Credit card, "i am furious about why i was charged xxxx late fee then xxxx late"),
  7668748 (Credit card), 7449382 (Checking or savings, an annual fee on an IRA), 6608852 (Credit
  card). Cosine top 5: 7449382, 7914608, 6426167 (Student loan, "refund"), 7668748, 7170260. Same
  query, two rankings: cosine rewards short texts dense in one term; BM25 saturates repeats.
- k1 = 0 (binary, IIR) for "late fee refund": top 5 becomes 7668748, 6594476, 6608852, 6645804,
  7142738. b = 0 or b = 1 also reorder positions 4-5.
- "repossessed my car": 139 match; all top 5 under both rankings are Vehicle loan complaints
  (6475003 "car was repossessed by wells fargo bank even after had made payments").
- "identity theft inquiry": 284 match; the top 5 under BOTH rankings are five different complaints
  (6697815, 7166396, 7215809, 7252034, 7270412) whose first words are identical: "i am a victim of
  identity theft please remove this fraud inquiry from my", tied at BM25 11.825. Templated
  complaints are real and common in credit reporting; search exposes them (and they are the
  near-duplicate risk from s4).
- "zelle scam": only "scam" survives (13 complaints); "zelle" is not in the vocabulary of the first
  50 words of any complaint. Teach: a query can only find words the index kept.

**Themes, more seeds (s5).** Pair agreement with seed 1 at the same k: k 4: seed 2 0.487, seed 3
0.355; k 6: seed 2 0.546, seed 3 0.496; k 8: seed 2 0.487, seed 3 0.387. k 8 seed 1 separates a
student loan theme [loan student payment mohela plan interest] (318, purity 0.63) and an identity
theft theme [credit report item information theft identity] (481, purity 0.84). Stemmer artefacts
visible as theme words: well (Wells), receiv, violat, u and s (from "U.S.C." citations).

**Drift (s5).** Student loan complaints per month in the sample, Jan-Dec 2023 (the quota sample draws
uniformly from the queue's whole year, so the monthly SHAPE is the real one; the level is not):
17, 14, 10, 5, 6, 7, 17, 24, 49, 67, 47, 37. April 5 -> October 67. FSA: interest resumed 1 Sep
2023, payments resumed October 2023. The [payment loan late student] theme (k 6, seed 1) by month:
35, 34, 35, 23, 21, 18, 32, 46, 64, 79, 61, 41. Say "coincides with", never "caused by".

**Cost scale for s6 (constructed volume).** Constructed Ledgerly volume: 30,000 complaints a month
(label constructed). At the placeholder $1 / $5 per million tokens: 30 x $0.39 = about $12 a month
(computed: 0.3893 x 30 = $11.68). The money is small; the s6 argument is audit, repeatability,
latency, drift and data handling, not the bill. Prices: placeholder, re-verify before delivery.

**Weights (s2; document frequency over all 3,000, smooth idf = ln((1+n)/(1+df)) + 1, n = 3,000).**
xxxx df 1,966 idf 1.422 · the 2,331 / 1.252 · i 2,567 / 1.156 · credit 1,218 / 1.901 · report 647 /
2.533 · card 328 / 3.211 · debt 300 / 3.300 · loan 287 / 3.344 · mohela 72 / 4.716 · repossessed 24 /
5.788 · scam 11 / 6.522 · zelle 0 (absent). Feature counts: unigrams 5,250; unigrams min df 2: 3,054;
stop words + xxxx out, min df 2: 2,937; unigrams + bigrams min df 2: 15,741; stop + xxxx out,
unigrams + bigrams, min df 1: 42,055; same with min df 2: 9,033 (the widget's third weighting).
Bigrams quadruple the raw feature count; min df 2 cuts the bigram table by 79% (42,055 -> 9,033).

**Training priors (s3).** Training rows per queue: 720, 480, 360, 360, 240, 240 of 2,400 (0.30, 0.20,
0.15, 0.15, 0.10, 0.10). R2 confusion (rows true report, collect, card, bank, student, auto):
[157,16,2,1,1,3] [21,80,3,10,2,4] [7,6,60,16,0,1] [3,8,8,68,0,3] [1,6,1,0,51,1] [3,10,0,6,0,41].
R4: [161,12,5,1,0,1] [23,80,7,7,1,2] [9,5,62,14,0,0] [2,6,7,73,0,2] [4,3,0,2,50,1] [5,9,1,4,0,41].
R2 per class P/R/F1: report .818/.872/.844; collect .635/.667/.650; card .811/.667/.732; bank
.673/.756/.712; student .944/.850/.895; auto .774/.683/.726. R4: report .789/.894/.839; collect
.696/.667/.681; card .756/.689/.721; bank .723/.811/.764; student .980/.833/.901; auto .872/.683/.766.
R2 misroutes card -> bank 16; R4 card -> bank 14.
