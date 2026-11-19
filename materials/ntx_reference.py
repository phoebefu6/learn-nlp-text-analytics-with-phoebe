"""ntx_reference.py - plain-Python reference for assets/ntx-live.js (learn-nlp-text-analytics-with-phoebe).

Same arithmetic, same operation order, same seeded shuffles, no numpy, so the browser bench and this
script can be compared number by number (materials/compare-ladder.py).

Usage: python3 -I ntx_reference.py ../assets/ntx-sample.js > canon-python.json
The scikit-learn equivalents are noted beside each piece; they are close but not identical
(scikit-learn's LogisticRegression uses a different solver), which is why the bench ships its own.
"""
import sys, json, math

M32 = 0xFFFFFFFF


def rng(seed):
    st = [seed & M32]

    def imul(a, b):
        return (a * b) & M32

    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        t = st[0]
        t = imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + imul(t ^ (t >> 7), t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return nxt


def shuffle(arr, r):
    for i in range(len(arr) - 1, 0, -1):
        j = math.floor(r() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr


STOP_LIST = ("a about above after again against all am an and any are as at be because been before being below "
             "between both but by can could did do does doing down during each few for from further had has have having he her "
             "here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of "
             "off on once only or other our ours ourselves out over own same she should so some such than that the their theirs "
             "them themselves then there these they this those through to too under until up very was we were what when where "
             "which while who whom why will with would you your yours yourself yourselves").split(" ")
STOP = set(STOP_LIST)


def stem(w):  # light suffix stripper, not Porter (sklearn has no stemmer; nltk's PorterStemmer differs)
    if len(w) <= 4:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) - 3 >= 3:
        return w[:-3]
    if w.endswith("ed") and len(w) - 2 >= 3:
        return w[:-2]
    if w.endswith("ly") and len(w) > 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalise(tokens, o):
    out = []
    for w in tokens:
        if o.get("redact") and w == "xxxx":
            continue
        if o.get("stop") and w in STOP:
            continue
        if o.get("stem"):
            w = stem(w)
        out.append(w)
    return out


def terms(tokens, o):
    n = normalise(tokens, o)
    if o.get("ngram", 1) < 2:
        return n
    return n + [n[i] + "_" + n[i + 1] for i in range(len(n) - 1)]


class Corpus:
    def __init__(self, S):
        V = S["vocab"].split(" ")
        ix = {c: i for i, c in enumerate(S["alphabet"])}
        self.S = S
        self.docs = [[V[ix[s[k]] * 90 + ix[s[k + 1]]] for k in range(0, len(s), 2)] for s in S["docs"]]
        self.y, self.m, self.len, self.id = S["y"], S["m"], S["len"], S["id"]
        self.K = len(S["queues"])
        self.queues = S["queues"]
        self.train = [d for d in range(len(self.docs)) if not S["test"][d]]
        self.test = [d for d in range(len(self.docs)) if S["test"][d]]


C = None


class Space:
    def __init__(self, fit_idx, o):
        self.o = o
        self.cache = {}
        df = {}
        for d in fit_idx:
            for w in set(self.T(d)):
                df[w] = df.get(w, 0) + 1
        keep = sorted(w for w in df if df[w] >= o.get("minDf", 1))
        n = len(fit_idx)
        self.terms = keep
        self.index = {w: j for j, w in enumerate(keep)}
        self.idf = [math.log((1 + n) / (1 + df[w])) + 1 for w in keep]
        self.p = len(keep)

    def T(self, d):
        if d not in self.cache:
            self.cache[d] = terms(C.docs[d], self.o)
        return self.cache[d]


def vec(space, d):
    tf = {}
    for w in space.T(d):
        j = space.index.get(w)
        if j is not None:
            tf[j] = tf.get(j, 0) + 1
    js = sorted(tf)
    vs = []
    for j in js:
        v = float(tf[j])
        if space.o.get("weight") == "tfidf":
            v = v * space.idf[j]
        vs.append(v)
    if space.o.get("weight") == "tfidf":
        ss = 0.0
        for v in vs:
            ss += v * v
        nr = math.sqrt(ss)
        if nr > 0:
            vs = [v / nr for v in vs]
    return js, vs


def majority_of(idx, y):
    cnt = [0] * C.K
    for d in idx:
        cnt[y[d]] += 1
    best = 0
    for c in range(1, C.K):
        if cnt[c] > cnt[best]:
            best = c
    return best


RULES = [
    ("student", ["student", "navient", "nelnet", "mohela", "aidvantage", "forgiveness"]),
    ("auto", ["car", "vehicle", "auto", "repossessed", "repossession", "dealership", "dealer"]),
    ("collect", ["collector", "collection", "collections", "debt"]),
    ("bank", ["checking", "savings", "deposit", "overdraft", "atm", "zelle", "branch"]),
    ("card", ["card"]),
    ("report", ["report", "bureau", "equifax", "experian", "transunion", "inquiry", "inquiries"]),
]


def keyword_predict(d, fallback):
    have = set(C.docs[d])
    for q, ws in RULES:
        for w in ws:
            if w in have:
                return C.queues.index(q)
    return fallback


def train_nb(space, idx, y, alpha):  # sklearn MultinomialNB(alpha=alpha) on the same matrix
    K, p = C.K, space.p
    fs = [[0.0] * p for _ in range(K)]
    cnt = [0] * K
    for d in idx:
        js, vs = vec(space, d)
        c = y[d]
        cnt[c] += 1
        row = fs[c]
        for j, v in zip(js, vs):
            row[j] += v
    logp, prior = [], []
    for c in range(K):
        s = 0.0
        for j in range(p):
            s += fs[c][j]
        den = s + alpha * p
        logp.append([math.log((fs[c][j] + alpha) / den) for j in range(p)])
        prior.append(math.log(cnt[c] / len(idx)))

    def f(d):
        js, vs = vec(space, d)
        best, bs = 0, -math.inf
        for c in range(K):
            z = prior[c]
            lp = logp[c]
            for j, v in zip(js, vs):
                z += v * lp[j]
            if z > bs:
                bs, best = z, c
        return best
    return f


def train_lr(space, idx, y, epochs, lr0, seed):  # multinomial logistic regression, seeded SGD
    K, p = C.K, space.p
    W = [[0.0] * p for _ in range(K)]
    b = [0.0] * K
    X = {d: vec(space, d) for d in idx}
    r = rng(seed)
    for e in range(epochs):
        order = shuffle(list(idx), r)
        lr = lr0 / (1 + e)
        for d in order:
            js, vs = X[d]
            z = [0.0] * K
            m = -math.inf
            for c in range(K):
                s = 0.0
                Wc = W[c]
                for j, v in zip(js, vs):
                    s += Wc[j] * v
                z[c] = b[c] + s
                if z[c] > m:
                    m = z[c]
            ez = [0.0] * K
            S = 0.0
            for c in range(K):
                ez[c] = math.exp(z[c] - m)
                S += ez[c]
            for c in range(K):
                g = ez[c] / S - (1 if y[d] == c else 0)
                if g != 0:
                    st = lr * g
                    Wc = W[c]
                    for j, v in zip(js, vs):
                        Wc[j] -= st * v
                    b[c] -= st

    def f(d):
        js, vs = vec(space, d)
        best, bs = 0, -math.inf
        for c in range(K):
            s = 0.0
            for j, v in zip(js, vs):
                s += W[c][j] * v
            zz = b[c] + s
            if zz > bs:
                bs, best = zz, c
        return best
    f.W, f.space = W, space
    return f


def score(pred, idx, y):
    K = C.K
    cm = [[0] * K for _ in range(K)]
    ok = 0
    for d, pr in zip(idx, pred):
        cm[y[d]][pr] += 1
        ok += y[d] == pr
    per, mf = [], 0.0
    for c in range(K):
        tp = cm[c][c]
        rs = sum(cm[c])
        cs = sum(cm[k][c] for k in range(K))
        P = tp / cs if cs else 0
        R = tp / rs if rs else 0
        F = 2 * P * R / (P + R) if (P + R) else 0
        per.append({"p": P, "r": R, "f1": F, "n": rs})
        mf += F
    return {"acc": ok / len(idx), "correct": ok, "n": len(idx), "macroF1": mf / K, "per": per, "cm": cm}


BASE = {"stop": False, "stem": False, "redact": False, "ngram": 1, "minDf": 1}


def ext(a, b):
    o = dict(a)
    o.update(b)
    return o


RUNGS = [
    ("r0", {"model": "majority"}),
    ("r1", {"model": "keywords"}),
    ("r2", ext(BASE, {"model": "nb", "weight": "count", "alpha": 1})),
    ("r3", ext(BASE, {"model": "nb", "weight": "tfidf", "alpha": 1})),
    ("r4", ext(BASE, {"model": "lr", "weight": "tfidf", "ngram": 2, "minDf": 2, "epochs": 10, "lr": 1})),
    ("r5", {"model": "tuned"}),
]
GRID = []
for al in (1, 0.3):
    for ng in (1, 2):
        GRID.append(ext(BASE, {"model": "nb", "weight": "count", "alpha": al, "ngram": ng, "minDf": 2 if ng == 2 else 1}))
for st in (False, True):
    for sm in (False, True):
        for l in (1, 2):
            GRID.append(ext(BASE, {"model": "lr", "weight": "tfidf", "ngram": 2, "minDf": 2, "epochs": 10, "lr": l, "stop": st, "stem": sm}))


LAST = [None]


def top_weights(model, k=6):
    if model is None or not hasattr(model, "W"):
        return None
    out = []
    for w in model.W:
        order = sorted(range(len(w)), key=lambda j: (-w[j], j))
        out.append([[model.space.terms[j], w[j]] for j in order[:k]])
    return out


def fit_predict(cfg, fit_idx, y, eval_idx):
    LAST[0] = None
    if cfg["model"] == "majority":
        mj = majority_of(fit_idx, y)
        return [mj for _ in eval_idx]
    if cfg["model"] == "keywords":
        fb = majority_of(fit_idx, y)
        return [keyword_predict(d, fb) for d in eval_idx]
    space = Space(fit_idx, cfg)
    f = train_nb(space, fit_idx, y, cfg["alpha"]) if cfg["model"] == "nb" else train_lr(space, fit_idx, y, cfg["epochs"], cfg["lr"], 7)
    LAST[0] = f
    return [f(d) for d in eval_idx]


def tune(y):
    fit, val = [], []
    for i, d in enumerate(C.train):
        (val if i % 4 == 0 else fit).append(d)
    rows = [score(fit_predict(cfg, fit, y, val), val, y)["acc"] for cfg in GRID]
    best = 0
    for i in range(1, len(rows)):
        if rows[i] > rows[best]:
            best = i
    return rows, best


def run_rung(rid, on="test", shuf=False):
    y = list(C.y)
    if shuf:
        lab = [C.y[d] for d in C.train]
        shuffle(lab, rng(13))
        for d, l in zip(C.train, lab):
            y[d] = l
    cfg = dict(RUNGS)[rid]
    tuned = None
    if cfg["model"] == "tuned":
        rows, best = tune(y)
        tuned = rows + [best]
        cfg = GRID[best]
    ev = C.train if on == "train" else C.test
    res = score(fit_predict(cfg, C.train, y, ev), ev, C.y)
    res["tuned"] = tuned
    res["weights"] = top_weights(LAST[0])
    return res


def corpus_card():
    n_tok = n_red = 0
    freq, per_q = {}, [0] * C.K
    for d, t in enumerate(C.docs):
        per_q[C.y[d]] += 1
        for w in t:
            n_tok += 1
            n_red += w == "xxxx"
            freq[w] = freq.get(w, 0) + 1
    lens = sorted(C.len)
    L = len(lens)
    med = lens[(L - 1) // 2] if L % 2 else (lens[L // 2 - 1] + lens[L // 2]) / 2
    cut = sum(1 for l in C.len if l > C.S["T"])
    return {"docs": len(C.docs), "train": len(C.train), "test": len(C.test), "tokens": n_tok, "vocab": len(freq),
            "hapax": sum(1 for w in freq if freq[w] == 1), "redactShare": n_red / n_tok, "perQueue": per_q,
            "medianFullWords": med, "truncatedShare": cut / len(C.docs), "shard2023": C.S["shard2023"]}


def zipf(o):
    freq = {}
    for t in C.docs:
        for w in normalise(t, o):
            freq[w] = freq.get(w, 0) + 1
    lst = sorted(freq.items(), key=lambda e: (-e[1], e[0]))
    n = min(1000, len(lst))
    sx = sy = sxx = sxy = 0.0
    total = sum(e[1] for e in lst)
    for i in range(n):
        x = math.log10(i + 1)
        yv = math.log10(lst[i][1])
        sx += x; sy += yv; sxx += x * x; sxy += x * yv
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    top10 = sum(e[1] for e in lst[:10])
    hapax = sum(1 for e in lst if e[1] == 1)
    return {"top": [list(e) for e in lst[:10]], "vocab": len(lst), "tokens": total, "slope": slope,
            "top10Share": top10 / total, "hapaxShare": hapax / len(lst)}


def top_terms(o, q, k):
    alld = list(range(len(C.docs)))
    space = Space(alld, o)
    s = [0.0] * space.p
    n = 0
    for d in alld:
        if C.y[d] != q:
            continue
        n += 1
        js, vs = vec(space, d)
        for j, v in zip(js, vs):
            s[j] += v
    order = sorted(range(space.p), key=lambda j: (-s[j], j))
    return space.p, [space.terms[j] for j in order[:k]]


SEARCH_O = {"stop": True, "stem": True, "redact": True, "ngram": 1, "minDf": 1}


def search_index(o=SEARCH_O):
    N = len(C.docs)
    df, tfs, dl, tot = {}, [], [], 0
    for d in range(N):
        t = normalise(C.docs[d], o)
        m = {}
        for w in t:
            m[w] = m.get(w, 0) + 1
        for w in m:
            df[w] = df.get(w, 0) + 1
        tfs.append(m); dl.append(len(t)); tot += len(t)
    return {"o": o, "N": N, "df": df, "tf": tfs, "dl": dl, "avgdl": tot / N}


def qtokens(query, ix):
    import re
    return normalise(re.findall(r"[a-z]+", query.lower().replace("'", "")), ix["o"])


def bm25(ix, query, k1, b, k):
    qs = []
    for w in qtokens(query, ix):
        if w not in qs and w in ix["df"]:
            qs.append(w)
    res = []
    for d in range(ix["N"]):
        s, hit = 0.0, False
        for w in qs:
            f = ix["tf"][d].get(w)
            if not f:
                continue
            hit = True
            df = ix["df"][w]
            idf = math.log(1 + (ix["N"] - df + 0.5) / (df + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * ix["dl"][d] / ix["avgdl"]))
        if hit:
            res.append((d, s))
    res.sort(key=lambda e: (-e[1], e[0]))
    return len(res), res[:k]


def cosine_search(ix, query, k):
    qm = {}
    for w in qtokens(query, ix):
        if w in ix["df"]:
            qm[w] = qm.get(w, 0) + 1
    N = ix["N"]

    def idf(w):
        return math.log((1 + N) / (1 + ix["df"][w])) + 1
    qw = sorted(qm)
    qn = 0.0
    for w in qw:
        qn += (qm[w] * idf(w)) ** 2
    qn = math.sqrt(qn)
    res = []
    for d in range(N):
        m = ix["tf"][d]
        dot, hit = 0.0, False
        for w in qw:
            if m.get(w):
                hit = True
                dot += qm[w] * idf(w) * m[w] * idf(w)
        if not hit:
            continue
        dn = 0.0
        for w in sorted(m):
            dn += (m[w] * idf(w)) ** 2
        res.append((d, dot / (qn * math.sqrt(dn))))
    res.sort(key=lambda e: (-e[1], e[0]))
    return len(res), res[:k]


THEME_O = {"stop": True, "stem": True, "redact": True, "ngram": 1, "minDf": 5, "weight": "tfidf"}


def kmeans(k, seed, iters=20):
    alld = list(range(len(C.docs)))
    space = Space(alld, THEME_O)
    p = space.p
    X = [vec(space, d) for d in alld]
    r = rng(seed)
    picked = []
    while len(picked) < k:
        cand = math.floor(r() * len(X))
        if cand not in picked and len(X[cand][0]) > 0:
            picked.append(cand)
    cent = []
    for d in picked:
        c = [0.0] * p
        for j, v in zip(*X[d]):
            c[j] = v
        cent.append(c)
    asg = [-1] * len(X)
    for _ in range(iters):
        for n, (js, vs) in enumerate(X):
            best, bs = 0, -math.inf
            for c in range(k):
                s = 0.0
                ce = cent[c]
                for j, v in zip(js, vs):
                    s += ce[j] * v
                if s > bs:
                    bs, best = s, c
            asg[n] = best
        for c in range(k):
            sm = [0.0] * p
            cnt = 0
            for n, (js, vs) in enumerate(X):
                if asg[n] == c:
                    cnt += 1
                    for j, v in zip(js, vs):
                        sm[j] += v
            if not cnt:
                continue
            ss = 0.0
            for j in range(p):
                ss += sm[j] * sm[j]
            nr = math.sqrt(ss)
            cent[c] = [x / nr for x in sm]
    clusters = []
    for c in range(k):
        mem = [n for n in range(len(X)) if asg[n] == c]
        byq = [0] * C.K
        bym = [0] * 12
        for n in mem:
            byq[C.y[n]] += 1
            bym[C.m[n]] += 1
        ce = cent[c]
        order = sorted(range(p), key=lambda j: (-ce[j], j))
        dom = 0
        for q in range(1, C.K):
            if byq[q] > byq[dom]:
                dom = q
        clusters.append({"size": len(mem), "top": [space.terms[j] for j in order[:6]], "dominant": dom,
                         "purity": byq[dom] / len(mem) if mem else 0, "byMonth": bym})
    return asg, clusters, p


def pair_agreement(a1, a2):
    tab, r1, r2 = {}, {}, {}
    for x, z in zip(a1, a2):
        tab[(x, z)] = tab.get((x, z), 0) + 1
        r1[x] = r1.get(x, 0) + 1
        r2[z] = r2.get(z, 0) + 1

    def pairs(o):
        return sum(v * (v - 1) / 2 for v in o.values())
    both, t1, t2 = pairs(tab), pairs(r1), pairs(r2)
    return both / (t1 + t2 - both)


def term_by_month(term, q, o=SEARCH_O):
    hit, tot = [0] * 12, [0] * 12
    for d in range(len(C.docs)):
        if q is not None and C.y[d] != q:
            continue
        tot[C.m[d]] += 1
        if term in normalise(C.docs[d], o):
            hit[C.m[d]] += 1
    return [h / t if t else 0 for h, t in zip(hit, tot)]


def llm_cost(in_price, out_price, idx):
    words = sum(C.len[d] for d in idx)
    in_tok = math.ceil(words * 1.3) + 120 * len(idx)
    out_tok = 4 * len(idx)
    per1k = (in_tok / 1e6 * in_price + out_tok / 1e6 * out_price) / len(idx) * 1000
    return {"docs": len(idx), "words": words, "inTok": in_tok, "outTok": out_tok, "per1k": per1k, "meanWords": words / len(idx)}


def canon():
    out = {"card": corpus_card(), "zipfRaw": zipf({}), "zipfNorm": zipf({"stop": True, "stem": True, "redact": True}),
           "ladder": {}, "traps": {}}
    for rid, _ in RUNGS:
        res = run_rung(rid)
        out["ladder"][rid] = {k: res[k] for k in ("acc", "correct", "macroF1", "per", "cm")}
        if res["weights"]:
            out["ladder"][rid]["weights"] = res["weights"]
        if res["tuned"]:
            out["tunedGrid"] = res["tuned"]
    for rid in ("r2", "r4"):
        out["traps"][rid + "_train"] = run_rung(rid, on="train")["acc"]
        out["traps"][rid + "_shuffle"] = run_rung(rid, shuf=True)["acc"]
    out["topTerms"] = {}
    for name, w in (("count", {"weight": "count"}), ("tfidf", {"weight": "tfidf"}),
                    ("tfidf_stop_bi", {"weight": "tfidf", "stop": True, "redact": True, "ngram": 2, "minDf": 2})):
        o = ext(BASE, w)
        out["topTerms"][name] = [top_terms(o, q, 8)[1] for q in range(C.K)]
        out["topTerms"][name + "_p"] = top_terms(o, 0, 1)[0]
    ix = search_index()
    out["search"] = {}
    for qq in ("late fee refund", "repossessed my car", "identity theft inquiry", "zelle scam"):
        n1, a = bm25(ix, qq, 1.2, 0.75, 5)
        n2, c = cosine_search(ix, qq, 5)
        out["search"][qq] = {"matched": n1, "bm25": [C.id[e[0]] for e in a], "cos": [C.id[e[0]] for e in c]}
    a1, k1, p = kmeans(6, 1)
    a2, k2, _ = kmeans(6, 2)
    out["themes"] = {"s1": k1, "s2": [{"size": c["size"], "top": c["top"]} for c in k2],
                     "agree": pair_agreement(a1, a2), "p": p}
    out["drift"] = {"student_payment": term_by_month("payment", 4), "card_prepaid": term_by_month("prepaid", None)}
    out["llm"] = llm_cost(1, 5, C.test)
    return out


def load(path):
    global C
    s = open(path).read()
    C = Corpus(json.loads(s[s.index("{"):s.rindex("}") + 1]))
    return C


if __name__ == "__main__":
    load(sys.argv[1])
    json.dump(canon(), sys.stdout, indent=1)
