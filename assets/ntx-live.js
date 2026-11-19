/* ntx-live.js - the text bench for learn-nlp-text-analytics-with-phoebe, by Phoebe Fu.
   Everything below runs for real in the browser on assets/ntx-sample.js (3,000 real CFPB complaint
   narratives, 2023, first 50 word tokens each). materials/ntx_reference.py implements the same
   arithmetic in plain Python; materials/compare-ladder.py checks the two agree by value.
   Engine: NTX (pure functions, also runs under node). UI: NTX_UI mounts [data-ntx] widgets. */
(function (root) {
  "use strict";

  // ---------- deterministic randomness (mulberry32, identical in the Python reference) ----------
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
      t = (t ^ (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0))) >>> 0;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(arr, r) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(r() * (i + 1));
      var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // ---------- normalisation ----------
  var STOP_LIST = ("a about above after again against all am an and any are as at be because been before being below " +
    "between both but by can could did do does doing down during each few for from further had has have having he her " +
    "here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of " +
    "off on once only or other our ours ourselves out over own same she should so some such than that the their theirs " +
    "them themselves then there these they this those through to too under until up very was we were what when where " +
    "which while who whom why will with would you your yours yourself yourselves").split(" ");
  var STOP = {};
  STOP_LIST.forEach(function (w) { STOP[w] = 1; });

  // A light suffix stripper (NOT Porter): teaches what stemming merges and what it breaks.
  function stem(w) {
    if (w.length <= 4) return w;
    if (w.slice(-3) === "ies") return w.slice(0, -3) + "y";
    if (w.slice(-3) === "ing" && w.length - 3 >= 3) return w.slice(0, -3);
    if (w.slice(-2) === "ed" && w.length - 2 >= 3) return w.slice(0, -2);
    if (w.slice(-2) === "ly" && w.length > 5) return w.slice(0, -2);
    if (w.slice(-1) === "s" && w.slice(-2) !== "ss") return w.slice(0, -1);
    return w;
  }

  function normalise(tokens, o) {
    var out = [];
    for (var i = 0; i < tokens.length; i++) {
      var w = tokens[i];
      if (o.redact && w === "xxxx") continue;
      if (o.stop && STOP[w] === 1) continue;
      if (o.stem) w = stem(w);
      out.push(w);
    }
    return out;
  }
  function terms(tokens, o) {
    var n = normalise(tokens, o);
    if ((o.ngram || 1) < 2) return n;
    var t = n.slice();
    for (var i = 0; i + 1 < n.length; i++) t.push(n[i] + "_" + n[i + 1]);
    return t;
  }

  // ---------- corpus ----------
  var C = null;
  function load(S) {
    var V = S.vocab.split(" "), A = S.alphabet, ix = {};
    for (var i = 0; i < A.length; i++) ix[A.charAt(i)] = i;
    var docs = S.docs.map(function (s) {
      var t = [];
      for (var k = 0; k < s.length; k += 2) t.push(V[ix[s.charAt(k)] * 90 + ix[s.charAt(k + 1)]]);
      return t;
    });
    var train = [], test = [];
    for (var d = 0; d < docs.length; d++) (S.test[d] ? test : train).push(d);
    C = { S: S, docs: docs, y: S.y, m: S.m, len: S.len, id: S.id, K: S.queues.length, train: train, test: test,
          labels: S.labels, queues: S.queues };
    return C;
  }

  // ---------- features ----------
  function cmpStr(a, b) { return a < b ? -1 : a > b ? 1 : 0; }
  function buildSpace(fitIdx, o) {
    var df = {}, termsOf = {};
    for (var i = 0; i < C.docs.length; i++) termsOf[i] = null;
    function T(d) { if (termsOf[d] === null) termsOf[d] = terms(C.docs[d], o); return termsOf[d]; }
    fitIdx.forEach(function (d) {
      var seen = {};
      T(d).forEach(function (w) { if (!seen[w]) { seen[w] = 1; df[w] = (df[w] || 0) + 1; } });
    });
    var keep = Object.keys(df).filter(function (w) { return df[w] >= (o.minDf || 1); }).sort(cmpStr);
    var index = {}, idf = new Float64Array(keep.length), n = fitIdx.length;
    keep.forEach(function (w, j) { index[w] = j; idf[j] = Math.log((1 + n) / (1 + df[w])) + 1; });
    return { o: o, index: index, terms: keep, idf: idf, T: T, p: keep.length };
  }
  function vec(space, d) {
    var tf = {}, o = space.o, list = space.T(d);
    for (var i = 0; i < list.length; i++) {
      var j = space.index[list[i]];
      if (j !== undefined) tf[j] = (tf[j] || 0) + 1;
    }
    var js = Object.keys(tf).map(Number).sort(function (a, b) { return a - b; });
    var vs = new Array(js.length);
    for (var k = 0; k < js.length; k++) {
      var v = tf[js[k]];
      if (o.weight === "tfidf") v = v * space.idf[js[k]];
      vs[k] = v;
    }
    if (o.weight === "tfidf") {
      var ss = 0;
      for (k = 0; k < vs.length; k++) ss += vs[k] * vs[k];
      var nr = Math.sqrt(ss);
      if (nr > 0) for (k = 0; k < vs.length; k++) vs[k] = vs[k] / nr;
    }
    return { j: js, v: vs };
  }

  // ---------- models ----------
  function majorityOf(idx, y) {
    var cnt = new Array(C.K).fill(0);
    idx.forEach(function (d) { cnt[y[d]]++; });
    var best = 0;
    for (var c = 1; c < C.K; c++) if (cnt[c] > cnt[best]) best = c;
    return best;
  }
  // Keyword rules a complaints lead might write on day one (checked in this order, first hit wins).
  var RULES = [
    ["student", ["student", "navient", "nelnet", "mohela", "aidvantage", "forgiveness"]],
    ["auto", ["car", "vehicle", "auto", "repossessed", "repossession", "dealership", "dealer"]],
    ["collect", ["collector", "collection", "collections", "debt"]],
    ["bank", ["checking", "savings", "deposit", "overdraft", "atm", "zelle", "branch"]],
    ["card", ["card"]],
    ["report", ["report", "bureau", "equifax", "experian", "transunion", "inquiry", "inquiries"]]
  ];
  function keywordPredict(d, fallback) {
    var toks = C.docs[d], have = {};
    toks.forEach(function (w) { have[w] = 1; });
    for (var r = 0; r < RULES.length; r++) {
      var ws = RULES[r][1];
      for (var k = 0; k < ws.length; k++) if (have[ws[k]] === 1) return C.queues.indexOf(RULES[r][0]);
    }
    return fallback;
  }
  function trainNB(space, idx, y, alpha) {
    var K = C.K, p = space.p, fs = [], tot = new Float64Array(K), cnt = new Array(K).fill(0);
    for (var c = 0; c < K; c++) fs.push(new Float64Array(p));
    idx.forEach(function (d) {
      var x = vec(space, d), c2 = y[d];
      cnt[c2]++;
      for (var k = 0; k < x.j.length; k++) fs[c2][x.j[k]] += x.v[k];
    });
    var logp = [], prior = [];
    for (c = 0; c < K; c++) {
      var s = 0;
      for (var j = 0; j < p; j++) s += fs[c][j];
      tot[c] = s;
      var lp = new Float64Array(p), den = s + alpha * p;
      for (j = 0; j < p; j++) lp[j] = Math.log((fs[c][j] + alpha) / den);
      logp.push(lp);
      prior.push(Math.log(cnt[c] / idx.length));
    }
    return function (d) {
      var x = vec(space, d), best = 0, bs = -Infinity;
      for (var c3 = 0; c3 < K; c3++) {
        var z = prior[c3];
        for (var k = 0; k < x.j.length; k++) z += x.v[k] * logp[c3][x.j[k]];
        if (z > bs) { bs = z; best = c3; }
      }
      return best;
    };
  }
  function trainLR(space, idx, y, epochs, lr0, seed) {
    var K = C.K, p = space.p, W = [], b = new Float64Array(K), X = {};
    for (var c = 0; c < K; c++) W.push(new Float64Array(p));
    idx.forEach(function (d) { X[d] = vec(space, d); });
    var r = rng(seed), z = new Float64Array(K), ez = new Float64Array(K);
    for (var e = 0; e < epochs; e++) {
      var order = shuffle(idx.slice(), r), lr = lr0 / (1 + e);
      for (var t = 0; t < order.length; t++) {
        var d = order[t], x = X[d], m = -Infinity;
        for (c = 0; c < K; c++) {
          var s = 0, Wc = W[c];
          for (var k = 0; k < x.j.length; k++) s += Wc[x.j[k]] * x.v[k];
          z[c] = b[c] + s;
          if (z[c] > m) m = z[c];
        }
        var S = 0;
        for (c = 0; c < K; c++) { ez[c] = Math.exp(z[c] - m); S += ez[c]; }
        for (c = 0; c < K; c++) {
          var g = ez[c] / S - (y[d] === c ? 1 : 0);
          if (g !== 0) {
            var st = lr * g, Wc2 = W[c];
            for (k = 0; k < x.j.length; k++) Wc2[x.j[k]] -= st * x.v[k];
            b[c] -= st;
          }
        }
      }
    }
    var model = function (d2) {
      var x2 = vec(space, d2), best = 0, bs = -Infinity;
      for (var c4 = 0; c4 < K; c4++) {
        var s2 = 0;
        for (var k2 = 0; k2 < x2.j.length; k2++) s2 += W[c4][x2.j[k2]] * x2.v[k2];
        var zz = b[c4] + s2;
        if (zz > bs) { bs = zz; best = c4; }
      }
      return best;
    };
    model.W = W; model.b = b; model.space = space;
    return model;
  }

  // ---------- metrics ----------
  function score(pred, idx, y) {
    var K = C.K, cm = [];
    for (var a = 0; a < K; a++) cm.push(new Array(K).fill(0));
    var ok = 0;
    idx.forEach(function (d, i) { cm[y[d]][pred[i]]++; if (y[d] === pred[i]) ok++; });
    var per = [], mf = 0;
    for (var c = 0; c < K; c++) {
      var tp = cm[c][c], rs = 0, cs = 0;
      for (var k = 0; k < K; k++) { rs += cm[c][k]; cs += cm[k][c]; }
      var P = cs ? tp / cs : 0, R = rs ? tp / rs : 0, F = (P + R) ? 2 * P * R / (P + R) : 0;
      per.push({ p: P, r: R, f1: F, n: rs });
      mf += F;
    }
    return { acc: ok / idx.length, correct: ok, n: idx.length, macroF1: mf / K, per: per, cm: cm };
  }

  // ---------- the ladder ----------
  var BASE = { stop: false, stem: false, redact: false, ngram: 1, minDf: 1 };
  function ext(a, b) { var o = {}, k; for (k in a) o[k] = a[k]; for (k in b) o[k] = b[k]; return o; }
  var RUNGS = [
    { id: "r0", name: "Majority queue", cfg: { model: "majority" } },
    { id: "r1", name: "Keyword rules", cfg: { model: "keywords" } },
    { id: "r2", name: "Counts + naive Bayes", cfg: ext(BASE, { model: "nb", weight: "count", alpha: 1 }) },
    { id: "r3", name: "TF-IDF + naive Bayes", cfg: ext(BASE, { model: "nb", weight: "tfidf", alpha: 1 }) },
    { id: "r4", name: "TF-IDF + bigrams + logistic", cfg: ext(BASE, { model: "lr", weight: "tfidf", ngram: 2, minDf: 2, epochs: 10, lr: 1 }) },
    { id: "r5", name: "Tuned on a validation split", cfg: { model: "tuned" } }
  ];
  var GRID = (function () {
    var g = [];
    [1, 0.3].forEach(function (al) { [1, 2].forEach(function (ng) {
      g.push(ext(BASE, { model: "nb", weight: "count", alpha: al, ngram: ng, minDf: ng === 2 ? 2 : 1 }));
    }); });
    [false, true].forEach(function (st) { [false, true].forEach(function (sm) { [1, 2].forEach(function (l) {
      g.push(ext(BASE, { model: "lr", weight: "tfidf", ngram: 2, minDf: 2, epochs: 10, lr: l, stop: st, stem: sm }));
    }); }); });
    return g;
  })();

  var LAST_MODEL = null;
  function topWeights(model, k) {
    if (!model || !model.W) return null;
    return model.W.map(function (w) {
      var idx = [];
      for (var j = 0; j < w.length; j++) idx.push(j);
      idx.sort(function (a, b) { return w[b] - w[a] || a - b; });
      return idx.slice(0, k || 6).map(function (j2) { return [model.space.terms[j2], w[j2]]; });
    });
  }
  function fitPredict(cfg, fitIdx, y, evalIdx) {
    LAST_MODEL = null;
    if (cfg.model === "majority") {
      var mj = majorityOf(fitIdx, y);
      return evalIdx.map(function () { return mj; });
    }
    if (cfg.model === "keywords") {
      var fb = majorityOf(fitIdx, y);
      return evalIdx.map(function (d) { return keywordPredict(d, fb); });
    }
    var space = buildSpace(fitIdx, cfg);
    var f = cfg.model === "nb" ? trainNB(space, fitIdx, y, cfg.alpha) : trainLR(space, fitIdx, y, cfg.epochs, cfg.lr, 7);
    LAST_MODEL = f;
    return evalIdx.map(function (d) { return f(d); });
  }
  function tune(y) {
    var fit = [], val = [];
    C.train.forEach(function (d, i) { (i % 4 === 0 ? val : fit).push(d); });
    var rows = GRID.map(function (cfg) {
      return { cfg: cfg, val: score(fitPredict(cfg, fit, y, val), val, y).acc };
    });
    var best = 0;
    for (var i = 1; i < rows.length; i++) if (rows[i].val > rows[best].val) best = i;
    return { rows: rows, best: best, cfg: rows[best].cfg };
  }
  // opts: { on: "test" | "train", shuffle: bool }
  function runRung(id, opts) {
    opts = opts || {};
    var y = C.y;
    if (opts.shuffle) {
      y = C.y.slice();
      var lab = C.train.map(function (d) { return C.y[d]; });
      shuffle(lab, rng(13));
      C.train.forEach(function (d, i) { y[d] = lab[i]; });
    }
    var rung = RUNGS.filter(function (r) { return r.id === id; })[0], cfg = rung.cfg, tuned = null;
    if (cfg.model === "tuned") { tuned = tune(y); cfg = tuned.cfg; }
    var evalIdx = opts.on === "train" ? C.train : C.test;
    var t0 = Date.now();
    var pred = fitPredict(cfg, C.train, y, evalIdx);
    var ms = Date.now() - t0;
    var res = score(pred, evalIdx, C.y);
    res.weights = topWeights(LAST_MODEL, 6);
    res.id = id; res.name = rung.name; res.cfg = cfg; res.tuned = tuned; res.ms = ms; res.pred = pred; res.on = opts.on || "test";
    return res;
  }

  // ---------- corpus card + Zipf ----------
  function corpusCard() {
    var nTok = 0, nRed = 0, freq = {}, perQ = new Array(C.K).fill(0);
    C.docs.forEach(function (t, d) {
      perQ[C.y[d]]++;
      t.forEach(function (w) { nTok++; if (w === "xxxx") nRed++; freq[w] = (freq[w] || 0) + 1; });
    });
    var lens = C.len.slice().sort(function (a, b) { return a - b; });
    var med = lens.length % 2 ? lens[(lens.length - 1) / 2] : (lens[lens.length / 2 - 1] + lens[lens.length / 2]) / 2;
    var cut = 0;
    C.len.forEach(function (l) { if (l > C.S.T) cut++; });
    var V = Object.keys(freq).length, hapax = 0;
    for (var w in freq) if (freq[w] === 1) hapax++;
    return { docs: C.docs.length, train: C.train.length, test: C.test.length, tokens: nTok, vocab: V, hapax: hapax,
             redactShare: nRed / nTok, perQueue: perQ, medianFullWords: med, truncatedShare: cut / C.docs.length,
             shard2023: C.S.shard2023 };
  }
  function zipf(o) {
    var freq = {};
    C.docs.forEach(function (t) { normalise(t, o || {}).forEach(function (w) { freq[w] = (freq[w] || 0) + 1; }); });
    var list = Object.keys(freq).map(function (w) { return [w, freq[w]]; });
    list.sort(function (a, b) { return b[1] - a[1] || cmpStr(a[0], b[0]); });
    var n = Math.min(1000, list.length), sx = 0, sy = 0, sxx = 0, sxy = 0, total = 0;
    list.forEach(function (e) { total += e[1]; });
    for (var i = 0; i < n; i++) {
      var x = Math.log10(i + 1), yv = Math.log10(list[i][1]);
      sx += x; sy += yv; sxx += x * x; sxy += x * yv;
    }
    var slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    var top10 = 0;
    for (i = 0; i < Math.min(10, list.length); i++) top10 += list[i][1];
    var hapax = list.filter(function (e) { return e[1] === 1; }).length;
    return { top: list.slice(0, 20), vocab: list.length, tokens: total, slope: slope, top10Share: top10 / total,
             hapaxShare: hapax / list.length };
  }

  // ---------- top terms per queue (session 2) ----------
  function topTerms(o, q, k) {
    var all = [];
    for (var d = 0; d < C.docs.length; d++) all.push(d);
    var space = buildSpace(all, o), sum = new Float64Array(space.p), n = 0;
    all.forEach(function (d2) {
      if (C.y[d2] !== q) return;
      n++;
      var x = vec(space, d2);
      for (var i = 0; i < x.j.length; i++) sum[x.j[i]] += x.v[i];
    });
    var idx = [];
    for (var j = 0; j < space.p; j++) idx.push(j);
    idx.sort(function (a, b) { return sum[b] - sum[a] || a - b; });
    return { p: space.p, top: idx.slice(0, k || 8).map(function (j2) { return [space.terms[j2], sum[j2] / n]; }) };
  }

  // ---------- search (session 5) ----------
  var SEARCH_O = { stop: true, stem: true, redact: true, ngram: 1, minDf: 1 };
  function searchIndex(o) {
    o = o || SEARCH_O;
    var N = C.docs.length, df = {}, tf = [], dl = [], tot = 0;
    for (var d = 0; d < N; d++) {
      var t = normalise(C.docs[d], o), m = {};
      t.forEach(function (w) { m[w] = (m[w] || 0) + 1; });
      Object.keys(m).forEach(function (w) { df[w] = (df[w] || 0) + 1; });
      tf.push(m); dl.push(t.length); tot += t.length;
    }
    return { o: o, N: N, df: df, tf: tf, dl: dl, avgdl: tot / N };
  }
  function bm25(ix, query, k1, b, k) {
    var q = normalise(query.toLowerCase().replace(/'/g, "").match(/[a-z]+/g) || [], ix.o), seen = {}, qs = [];
    q.forEach(function (w) { if (!seen[w] && ix.df[w]) { seen[w] = 1; qs.push(w); } });
    var res = [];
    for (var d = 0; d < ix.N; d++) {
      var s = 0, hit = false;
      for (var i = 0; i < qs.length; i++) {
        var f = ix.tf[d][qs[i]];
        if (!f) continue;
        hit = true;
        var idf = Math.log(1 + (ix.N - ix.df[qs[i]] + 0.5) / (ix.df[qs[i]] + 0.5));
        s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * ix.dl[d] / ix.avgdl));
      }
      if (hit) res.push([d, s]);
    }
    res.sort(function (a, c) { return c[1] - a[1] || a[0] - c[0]; });
    return { terms: qs, matched: res.length, top: res.slice(0, k || 5) };
  }
  function cosineSearch(ix, query, k) {
    var q = normalise(query.toLowerCase().replace(/'/g, "").match(/[a-z]+/g) || [], ix.o), qm = {};
    q.forEach(function (w) { if (ix.df[w]) qm[w] = (qm[w] || 0) + 1; });
    function idf(w) { return Math.log((1 + ix.N) / (1 + ix.df[w])) + 1; }
    var qn = 0, qw = Object.keys(qm).sort(cmpStr);
    qw.forEach(function (w) { qn += Math.pow(qm[w] * idf(w), 2); });
    qn = Math.sqrt(qn);
    var res = [];
    for (var d = 0; d < ix.N; d++) {
      var dot = 0, hit = false, m = ix.tf[d];
      for (var i = 0; i < qw.length; i++) if (m[qw[i]]) { hit = true; dot += qm[qw[i]] * idf(qw[i]) * m[qw[i]] * idf(qw[i]); }
      if (!hit) continue;
      var dn = 0;
      Object.keys(m).sort(cmpStr).forEach(function (w) { dn += Math.pow(m[w] * idf(w), 2); });
      res.push([d, dot / (qn * Math.sqrt(dn))]);
    }
    res.sort(function (a, c) { return c[1] - a[1] || a[0] - c[0]; });
    return { terms: qw, matched: res.length, top: res.slice(0, k || 5) };
  }

  // ---------- themes: spherical k-means (session 5) ----------
  var THEME_O = { stop: true, stem: true, redact: true, ngram: 1, minDf: 5, weight: "tfidf" };
  function kmeans(k, seed, iters) {
    var all = [];
    for (var d = 0; d < C.docs.length; d++) all.push(d);
    var space = buildSpace(all, THEME_O), p = space.p, X = all.map(function (d2) { return vec(space, d2); });
    var r = rng(seed), picked = [], cent = [];
    while (picked.length < k) {
      var cand = Math.floor(r() * X.length);
      if (picked.indexOf(cand) < 0 && X[cand].j.length > 0) picked.push(cand);
    }
    picked.forEach(function (d3) {
      var c = new Float64Array(p);
      for (var i = 0; i < X[d3].j.length; i++) c[X[d3].j[i]] = X[d3].v[i];
      cent.push(c);
    });
    var asg = new Array(X.length).fill(-1);
    for (var it = 0; it < (iters || 20); it++) {
      for (var n = 0; n < X.length; n++) {
        var best = 0, bs = -Infinity;
        for (var c2 = 0; c2 < k; c2++) {
          var s = 0;
          for (var i2 = 0; i2 < X[n].j.length; i2++) s += cent[c2][X[n].j[i2]] * X[n].v[i2];
          if (s > bs) { bs = s; best = c2; }
        }
        asg[n] = best;
      }
      for (c2 = 0; c2 < k; c2++) {
        var sum = new Float64Array(p), cnt = 0;
        for (n = 0; n < X.length; n++) if (asg[n] === c2) {
          cnt++;
          for (i2 = 0; i2 < X[n].j.length; i2++) sum[X[n].j[i2]] += X[n].v[i2];
        }
        if (!cnt) continue;
        var ss = 0;
        for (var j = 0; j < p; j++) ss += sum[j] * sum[j];
        var nr = Math.sqrt(ss);
        for (j = 0; j < p; j++) sum[j] = sum[j] / nr;
        cent[c2] = sum;
      }
    }
    var clusters = [];
    for (c2 = 0; c2 < k; c2++) {
      var members = [], byQ = new Array(C.K).fill(0), byM = new Array(12).fill(0);
      for (n = 0; n < X.length; n++) if (asg[n] === c2) { members.push(n); byQ[C.y[n]]++; byM[C.m[n]]++; }
      var ord = [];
      for (j = 0; j < p; j++) ord.push(j);
      var ce = cent[c2];
      ord.sort(function (a, b2) { return ce[b2] - ce[a] || a - b2; });
      var dom = 0;
      for (var q = 1; q < C.K; q++) if (byQ[q] > byQ[dom]) dom = q;
      clusters.push({ size: members.length, top: ord.slice(0, 6).map(function (j2) { return space.terms[j2]; }),
                      dominant: dom, purity: members.length ? byQ[dom] / members.length : 0, byQueue: byQ, byMonth: byM });
    }
    return { k: k, seed: seed, assign: asg, clusters: clusters, p: p };
  }
  function pairAgreement(a1, a2) {
    var tab = {}, r1 = {}, r2 = {};
    for (var i = 0; i < a1.length; i++) {
      var key = a1[i] + "," + a2[i];
      tab[key] = (tab[key] || 0) + 1; r1[a1[i]] = (r1[a1[i]] || 0) + 1; r2[a2[i]] = (r2[a2[i]] || 0) + 1;
    }
    function pairs(o) { var s = 0; for (var k in o) s += o[k] * (o[k] - 1) / 2; return s; }
    var both = pairs(tab), t1 = pairs(r1), t2 = pairs(r2);
    return both / (t1 + t2 - both);
  }
  // share of a queue's complaints, per month, whose normalised tokens contain a term
  function termByMonth(term, q, o) {
    o = o || SEARCH_O;
    var hit = new Array(12).fill(0), tot = new Array(12).fill(0);
    for (var d = 0; d < C.docs.length; d++) {
      if (q !== null && q !== undefined && C.y[d] !== q) continue;
      tot[C.m[d]]++;
      if (normalise(C.docs[d], o).indexOf(term) >= 0) hit[C.m[d]]++;
    }
    return { hit: hit, tot: tot, share: hit.map(function (h, i) { return tot[i] ? h / tot[i] : 0; }) };
  }

  // ---------- the modelled LLM rung (cost and audit only; nothing is sent anywhere) ----------
  // Assumptions are printed on the widget: 1.3 tokens per English word, a 120-token instruction
  // naming the six queues, 4 output tokens. Prices are placeholders the learner sets.
  function llmCost(inPrice, outPrice, idx) {
    idx = idx || C.test;
    var words = 0;
    idx.forEach(function (d) { words += C.len[d]; });
    var inTok = Math.ceil(words * 1.3) + 120 * idx.length, outTok = 4 * idx.length;
    var per1k = (inTok / 1e6 * inPrice + outTok / 1e6 * outPrice) / idx.length * 1000;
    return { docs: idx.length, words: words, inTok: inTok, outTok: outTok, per1k: per1k, meanWords: words / idx.length };
  }

  function canon() {
    var out = { card: corpusCard(), zipfRaw: zipf({}), zipfNorm: zipf({ stop: true, stem: true, redact: true }), ladder: {}, traps: {} };
    out.zipfRaw.top = out.zipfRaw.top.slice(0, 10); out.zipfNorm.top = out.zipfNorm.top.slice(0, 10);
    RUNGS.forEach(function (r) {
      var res = runRung(r.id, {});
      out.ladder[r.id] = { acc: res.acc, correct: res.correct, macroF1: res.macroF1, per: res.per, cm: res.cm };
      if (res.weights) out.ladder[r.id].weights = res.weights;
      if (res.tuned) out.tunedGrid = res.tuned.rows.map(function (x) { return x.val; }).concat([res.tuned.best]);
    });
    ["r2", "r4"].forEach(function (id) {
      out.traps[id + "_train"] = runRung(id, { on: "train" }).acc;
      out.traps[id + "_shuffle"] = runRung(id, { shuffle: true }).acc;
    });
    out.topTerms = {};
    [["count", { weight: "count" }], ["tfidf", { weight: "tfidf" }], ["tfidf_stop_bi", { weight: "tfidf", stop: true, redact: true, ngram: 2, minDf: 2 }]].forEach(function (w) {
      out.topTerms[w[0]] = C.queues.map(function (q, qi) { return topTerms(ext(BASE, w[1]), qi, 8).top.map(function (e) { return e[0]; }); });
      out.topTerms[w[0] + "_p"] = topTerms(ext(BASE, w[1]), 0, 1).p;
    });
    var ix = searchIndex();
    out.search = {};
    ["late fee refund", "repossessed my car", "identity theft inquiry", "zelle scam"].forEach(function (qq) {
      var a = bm25(ix, qq, 1.2, 0.75, 5), c = cosineSearch(ix, qq, 5);
      out.search[qq] = { matched: a.matched, bm25: a.top.map(function (e) { return C.id[e[0]]; }), cos: c.top.map(function (e) { return C.id[e[0]]; }) };
    });
    var k1 = kmeans(6, 1, 20), k2 = kmeans(6, 2, 20);
    out.themes = { s1: k1.clusters.map(function (c) { return { size: c.size, top: c.top, dominant: c.dominant, purity: c.purity, byMonth: c.byMonth }; }),
                   s2: k2.clusters.map(function (c) { return { size: c.size, top: c.top }; }),
                   agree: pairAgreement(k1.assign, k2.assign), p: k1.p };
    out.drift = { student_payment: termByMonth("payment", 4).share, card_prepaid: termByMonth("prepaid", null).share };
    out.llm = llmCost(1, 5);
    return out;
  }

  var NTX = { rng: rng, stem: stem, normalise: normalise, STOP: STOP_LIST, RULES: RULES, load: load, RUNGS: RUNGS, GRID: GRID,
              runRung: runRung, corpusCard: corpusCard, zipf: zipf, topTerms: topTerms, searchIndex: searchIndex,
              bm25: bm25, cosineSearch: cosineSearch, kmeans: kmeans, pairAgreement: pairAgreement,
              termByMonth: termByMonth, llmCost: llmCost, canon: canon, ext: ext, BASE: BASE,
              get C() { return C; } };
  root.NTX = NTX;
  if (typeof module !== "undefined") module.exports = NTX;
})(typeof window !== "undefined" ? window : globalThis);

/* ================= NTX_UI: the widgets. Each mounts on a <div data-ntx="..."> ================= */
(function () {
  "use strict";
  if (typeof document === "undefined") return;
  var NTX = window.NTX;
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
  function pct(x, d) { return (x * 100).toFixed(d === undefined ? 1 : d) + "%"; }
  function f3(x) { return x.toFixed(3); }
  function ready() { if (!NTX.C) NTX.load(window.NTX_SAMPLE); return NTX.C; }
  var RAIL = "Real CFPB complaint narratives (2023, CC0, first 50 words each), 3,000 rows, 600 held out. Computed in your browser.";
  var EXAMPLES = [18, 2, 26, 86];

  // ---------- corpus card ----------
  function mountCard(host) {
    var C = ready(), k = NTX.corpusCard(), max = Math.max.apply(null, k.perQueue);
    var rows = C.labels.map(function (l, q) {
      var share = k.shard2023[q] / k.shard2023.reduce(function (a, b) { return a + b; }, 0);
      return "<tr><td>" + l + "</td><td class='ntx-num'><b>" + k.perQueue[q] + "</b></td><td><span class='ntx-bar' style='width:" + (k.perQueue[q] / max * 100).toFixed(0) + "%'></span></td><td class='ntx-num'>" + pct(share) + "</td></tr>";
    }).join("");
    host.innerHTML = "<div class='ntx-head'><h4>Ledgerly's corpus card</h4><p class='ntx-rail'>" + RAIL + "</p></div>" +
      "<div class='ntx-stats'>" +
      "<div><b>" + k.docs.toLocaleString("en-US") + "</b><span>documents</span></div>" +
      "<div><b>" + k.tokens.toLocaleString("en-US") + "</b><span>word tokens</span></div>" +
      "<div><b>" + k.vocab.toLocaleString("en-US") + "</b><span>distinct words</span></div>" +
      "<div><b>" + pct(k.hapax / k.vocab) + "</b><span>words seen once</span></div>" +
      "<div><b>" + pct(k.redactShare) + "</b><span>tokens that are xxxx</span></div>" +
      "<div><b>" + k.medianFullWords + "</b><span>median words, full text</span></div></div>" +
      "<table class='clean ntx-table'><thead><tr><th>Queue</th><th>In sample</th><th></th><th>Real 2023 share</th></tr></thead><tbody>" + rows + "</tbody></table>" +
      "<p class='ntx-note'>The sample is rebalanced: credit reporting is 30% here and " + pct(k.shard2023[0] / k.shard2023.reduce(function (a, b) { return a + b; }, 0)) + " of the real 2023 rows. " + pct(k.truncatedShare) + " of narratives were cut to their first 50 words. Train " + k.train + " · held out " + k.test + ".</p>";
  }

  // ---------- normaliser ----------
  function mountNormalise(host) {
    var C = ready(), state = { stop: false, stem: false, redact: false, ex: 0 };
    host.innerHTML = "<div class='ntx-head'><h4>The normaliser</h4><p class='ntx-rail'>" + RAIL + "</p></div>" +
      "<div class='ntx-bar-row'><button type='button' class='ntx-tog' data-k='redact'>Drop xxxx redactions</button><button type='button' class='ntx-tog' data-k='stop'>Remove stop words</button><button type='button' class='ntx-tog' data-k='stem'>Light stemmer</button><button type='button' class='ntx-btn' data-k='next'>Another complaint</button></div>" +
      "<div class='ntx-out'></div>";
    var out = host.querySelector(".ntx-out");
    function draw() {
      var d = EXAMPLES[state.ex], raw = C.docs[d], o = { stop: state.stop, stem: state.stem, redact: state.redact };
      var norm = NTX.normalise(raw, o), z = NTX.zipf(o);
      var rawHtml = raw.map(function (w) {
        var gone = (o.redact && w === "xxxx") || (o.stop && NTX.STOP.indexOf(w) >= 0);
        var st = !gone && o.stem && NTX.stem(w) !== w;
        return "<span class='ntx-tok" + (gone ? " is-gone" : st ? " is-stem" : "") + "'>" + (st ? NTX.stem(w) : w) + "</span>";
      }).join(" ");
      out.innerHTML = "<p class='ntx-meta'>Complaint " + C.id[d] + " · " + C.labels[C.y[d]] + " · " + raw.length + " tokens in, <b>" + norm.length + "</b> out</p>" +
        "<div class='ntx-doc'>" + rawHtml + "</div>" +
        "<div class='ntx-stats ntx-stats-3'><div><b>" + z.vocab.toLocaleString("en-US") + "</b><span>vocabulary, whole corpus</span></div><div><b>" + z.tokens.toLocaleString("en-US") + "</b><span>tokens kept</span></div><div><b>" + pct(z.top10Share) + "</b><span>of tokens are the top 10 words</span></div></div>" +
        "<p class='ntx-note'>Top 10 now: " + z.top.slice(0, 10).map(function (e) { return "<code>" + e[0] + "</code> " + e[1]; }).join(", ") + "</p>";
    }
    host.addEventListener("click", function (ev) {
      var b = ev.target.closest("button"); if (!b) return;
      var k = b.getAttribute("data-k");
      if (k === "next") state.ex = (state.ex + 1) % EXAMPLES.length;
      else { state[k] = !state[k]; b.classList.toggle("is-on", state[k]); }
      draw();
    });
    draw();
  }

  // ---------- Zipf ----------
  function mountZipf(host) {
    ready();
    var state = { norm: false };
    host.innerHTML = "<div class='ntx-head'><h4>Zipf on the complaints</h4><p class='ntx-rail'>" + RAIL + "</p></div>" +
      "<div class='ntx-bar-row'><button type='button' class='ntx-tog'>Normalise first (stop words, stemmer, xxxx out)</button></div><div class='ntx-out'></div>";
    var out = host.querySelector(".ntx-out");
    function draw() {
      var z = NTX.zipf(state.norm ? { stop: true, stem: true, redact: true } : {}), top = z.top[0][1];
      out.innerHTML = "<div class='ntx-stats ntx-stats-3'><div><b>" + z.slope.toFixed(2) + "</b><span>log-log slope, top 1,000 ranks</span></div><div><b>" + pct(z.top10Share) + "</b><span>tokens from the top 10 words</span></div><div><b>" + pct(z.hapaxShare) + "</b><span>of the vocabulary seen once</span></div></div>" +
        "<div class='ntx-zipf'>" + z.top.slice(0, 12).map(function (e, i) {
          return "<div class='ntx-zrow'><span class='ntx-zr'>" + (i + 1) + "</span><code>" + e[0] + "</code><span class='ntx-zbar'><span style='width:" + (e[1] / top * 100).toFixed(1) + "%'></span></span><span class='ntx-num'>" + e[1].toLocaleString("en-US") + "</span><span class='ntx-zr'>1/" + (i + 1) + " = " + Math.round(top / (i + 1)).toLocaleString("en-US") + "</span></div>";
        }).join("") + "</div><p class='ntx-note'>Last column: what an exact 1/rank law would predict from the top word's count.</p>";
    }
    host.querySelector("button").addEventListener("click", function (ev) { state.norm = !state.norm; ev.target.classList.toggle("is-on", state.norm); draw(); });
    draw();
  }

  // ---------- top terms (session 2) ----------
  var WEIGHTS = [
    ["Raw counts", { weight: "count" }],
    ["TF-IDF", { weight: "tfidf" }],
    ["TF-IDF, stop words and xxxx out, plus bigrams", { weight: "tfidf", stop: true, redact: true, ngram: 2, minDf: 2 }]
  ];
  function mountTerms(host) {
    var C = ready(), state = { w: 0, q: 0 };
    host.innerHTML = "<div class='ntx-head'><h4>Top words per queue, three weightings</h4><p class='ntx-rail'>" + RAIL + "</p></div>" +
      "<div class='ntx-bar-row ntx-wrow'>" + WEIGHTS.map(function (w, i) { return "<button type='button' class='ntx-tog' data-w='" + i + "'>" + w[0] + "</button>"; }).join("") + "</div>" +
      "<div class='ntx-bar-row ntx-qrow'>" + C.labels.map(function (l, i) { return "<button type='button' class='ntx-chip' data-q='" + i + "'>" + l + "</button>"; }).join("") + "</div><div class='ntx-out'></div>";
    var out = host.querySelector(".ntx-out");
    function draw() {
      host.querySelectorAll("[data-w]").forEach(function (b) { b.classList.toggle("is-on", +b.getAttribute("data-w") === state.w); });
      host.querySelectorAll("[data-q]").forEach(function (b) { b.classList.toggle("is-on", +b.getAttribute("data-q") === state.q); });
      var r = NTX.topTerms(NTX.ext(NTX.BASE, WEIGHTS[state.w][1]), state.q, 8), max = r.top[0][1];
      out.innerHTML = "<p class='ntx-meta'>" + r.p.toLocaleString("en-US") + " features · mean weight per " + C.labels[state.q] + " complaint</p><div class='ntx-zipf'>" +
        r.top.map(function (e, i) { return "<div class='ntx-zrow'><span class='ntx-zr'>" + (i + 1) + "</span><code>" + e[0] + "</code><span class='ntx-zbar'><span style='width:" + (e[1] / max * 100).toFixed(1) + "%'></span></span><span class='ntx-num'>" + e[1].toFixed(3) + "</span></div>"; }).join("") + "</div>";
    }
    host.addEventListener("click", function (ev) {
      var b = ev.target.closest("button"); if (!b) return;
      if (b.hasAttribute("data-w")) state.w = +b.getAttribute("data-w");
      if (b.hasAttribute("data-q")) state.q = +b.getAttribute("data-q");
      draw();
    });
    draw();
  }

  // ---------- the bench (sessions 3 and 4) ----------
  var BENCHES = [];
  function mountBench(host) {
    var C = ready();
    var ids = (host.getAttribute("data-rungs") || "r0,r1,r2,r3,r4,r5").split(",");
    var showLLM = host.getAttribute("data-llm") !== "off";
    var state = { id: host.getAttribute("data-start") || ids[0], train: false, shuffle: false, inP: 1, outP: 5 };
    host.innerHTML = "<div class='ntx-head'><h4>The routing bench</h4><p class='ntx-rail'>" + RAIL + " The LLM row is modelled: nothing is sent anywhere.</p></div>" +
      "<div class='ntx-bar-row'>" + ids.map(function (id) {
        var r = NTX.RUNGS.filter(function (x) { return x.id === id; })[0];
        return "<button type='button' class='ntx-rung' data-id='" + id + "'><i>" + id.toUpperCase() + "</i> " + r.name + "</button>";
      }).join("") + "</div>" +
      "<div class='ntx-bar-row'><button type='button' class='ntx-trap' data-t='train'>Trap: score on the training rows</button><button type='button' class='ntx-trap' data-t='shuffle'>Break: shuffle the training labels</button></div>" +
      "<div class='ntx-out' aria-live='polite'></div>" + (showLLM ? "<div class='ntx-llm'></div>" : "");
    var out = host.querySelector(".ntx-out"), last = null;
    function draw() {
      host.querySelectorAll("[data-id]").forEach(function (b) { b.classList.toggle("is-on", b.getAttribute("data-id") === state.id); });
      host.querySelectorAll("[data-t]").forEach(function (b) { b.classList.toggle("is-on", !!state[b.getAttribute("data-t")]); });
      out.innerHTML = "<p class='ntx-meta'>Computing on " + (state.train ? C.train.length + " training" : C.test.length + " held-out") + " complaints...</p>";
      setTimeout(function () {
        var r = NTX.runRung(state.id, { on: state.train ? "train" : "test", shuffle: state.shuffle });
        last = r;
        var warn = state.train ? "<p class='ntx-warn'>Scored on rows the model has already seen. This number measures memory, not routing.</p>" :
          state.shuffle ? "<p class='ntx-warn'>Trained on shuffled labels, scored honestly on the held-out rows. If the bench measures anything, this must fall.</p>" : "";
        var per = r.per.map(function (p, q) {
          return "<tr><td>" + C.labels[q] + "</td><td class='ntx-num'>" + p.n + "</td><td class='ntx-num'>" + f3(p.p) + "</td><td class='ntx-num'>" + f3(p.r) + "</td><td class='ntx-num'><b>" + f3(p.f1) + "</b></td></tr>";
        }).join("");
        var cm = "<tr><th>True \\ routed to</th>" + C.queues.map(function (q) { return "<th>" + q + "</th>"; }).join("") + "</tr>" + r.cm.map(function (row, i) {
          return "<tr><th>" + C.queues[i] + "</th>" + row.map(function (v, j) { return "<td class='ntx-num" + (i === j ? " ntx-diag" : v >= 10 ? " ntx-hot" : "") + "'>" + v + "</td>"; }).join("") + "</tr>";
        }).join("");
        var tuned = r.tuned ? "<details class='ntx-grid'><summary>Validation grid: " + r.tuned.rows.length + " settings tried on 600 training rows held back, best = row " + (r.tuned.best + 1) + "</summary><table class='clean ntx-table'><thead><tr><th>#</th><th>Setting</th><th>Validation accuracy</th></tr></thead><tbody>" +
          r.tuned.rows.map(function (x, i) {
            var c = x.cfg, s = c.model === "nb" ? "Counts + NB, alpha " + c.alpha + (c.ngram === 2 ? ", + bigrams" : "") : "Logistic, lr " + c.lr + (c.stop ? ", stop words out" : "") + (c.stem ? ", stemmer" : "");
            return "<tr" + (i === r.tuned.best ? " class='ntx-best'" : "") + "><td>" + (i + 1) + "</td><td>" + s + "</td><td class='ntx-num'>" + f3(x.val) + "</td></tr>";
          }).join("") + "</tbody></table></details>" : "";
        out.innerHTML = warn + "<div class='ntx-headline'><div class='ntx-big'><b>" + f3(r.acc) + "</b><span>accuracy (" + r.correct + " of " + r.n + ")</span></div>" +
          "<div class='ntx-two'><span>Macro F1 <b>" + f3(r.macroF1) + "</b></span><span>Misroutes per 1,000 <b>" + Math.round((1 - r.acc) * 1000) + "</b></span><span>Train + predict <b>" + r.ms + " ms</b></span></div></div>" +
          "<table class='clean ntx-table'><thead><tr><th>Queue</th><th>Rows</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>" + per + "</tbody></table>" +
          "<div class='ntx-tablewrap'><table class='clean ntx-table ntx-cm'>" + cm + "</table></div>" + tuned +
          (r.weights ? "<details class='ntx-grid'><summary>What the logistic model learned: the six heaviest words per queue</summary><table class='clean ntx-table'><tbody>" +
            r.weights.map(function (ws, q) { return "<tr><td>" + C.labels[q] + "</td><td class='ntx-terms'>" + ws.map(function (e) { return "<code>" + e[0] + "</code> " + e[1].toFixed(2); }).join(" ") + "</td></tr>"; }).join("") + "</tbody></table></details>" : "");
        drawLLM();
      }, 30);
    }
    function drawLLM() {
      var box = host.querySelector(".ntx-llm"); if (!box) return;
      var c = NTX.llmCost(state.inP, state.outP);
      box.innerHTML = "<h5>Modelled rung: send each complaint to a large language model</h5>" +
        "<p class='ntx-note'>Accuracy on this holdout: <b>not measured here</b> (no API call is made). Published comparisons disagree; session 6 lays them out. What can be computed honestly is the bill and the audit trail.</p>" +
        "<div class='ntx-bar-row ntx-prices'><label>Input $ per million tokens <input type='number' min='0' step='0.05' value='" + state.inP + "' data-p='inP'></label><label>Output $ per million tokens <input type='number' min='0' step='0.05' value='" + state.outP + "' data-p='outP'></label></div>" +
        "<table class='clean ntx-table'><tbody><tr><td>Held-out complaints, full length</td><td class='ntx-num'>" + c.docs + " · " + c.words.toLocaleString("en-US") + " words</td></tr>" +
        "<tr><td>Input tokens (1.3 per word + 120-token instruction each)</td><td class='ntx-num'>" + c.inTok.toLocaleString("en-US") + "</td></tr>" +
        "<tr><td>Output tokens (4 per label)</td><td class='ntx-num'>" + c.outTok.toLocaleString("en-US") + "</td></tr>" +
        "<tr><td><b>Modelled cost per 1,000 complaints</b></td><td class='ntx-num'><b>$" + c.per1k.toFixed(2) + "</b></td></tr>" +
        "<tr><td>Same labels again tomorrow?</td><td>Only if the model version, prompt and settings are pinned and logged</td></tr></tbody></table>" +
        "<p class='ntx-note'>Prices are placeholders you set, not a quote. Re-verify the vendor's current price list before any decision.</p>";
    }
    host.addEventListener("click", function (ev) {
      var b = ev.target.closest("button"); if (!b) return;
      if (b.hasAttribute("data-id")) state.id = b.getAttribute("data-id");
      if (b.hasAttribute("data-t")) { var t = b.getAttribute("data-t"); state[t] = !state[t]; if (state.train && state.shuffle) state[t === "train" ? "shuffle" : "train"] = false; }
      draw();
    });
    host.addEventListener("change", function (ev) {
      var p = ev.target.getAttribute("data-p"); if (!p) return;
      var v = parseFloat(ev.target.value); state[p] = isFinite(v) && v >= 0 ? v : 0; drawLLM();
    });
    var api = { set: function (id, o) { state.id = id; state.train = !!(o && o.train); state.shuffle = !!(o && o.shuffle); draw(); },
                last: function () { return last; } };
    BENCHES.push(api);
    draw();
  }

  // ---------- search (session 5) ----------
  var IX = null;
  function mountSearch(host) {
    var C = ready(); if (!IX) IX = NTX.searchIndex();
    var state = { q: host.getAttribute("data-q") || "late fee refund", mode: "bm25", k1: 1.2, b: 0.75 };
    host.innerHTML = "<div class='ntx-head'><h4>Search the complaints</h4><p class='ntx-rail'>" + RAIL + " All 3,000 rows indexed; stop words, xxxx and the light stemmer applied to text and query.</p></div>" +
      "<div class='ntx-bar-row'><input type='text' class='ntx-q' value='" + state.q + "' aria-label='Search query'><button type='button' class='ntx-tog is-on' data-m='bm25'>BM25</button><button type='button' class='ntx-tog' data-m='cos'>TF-IDF cosine</button></div>" +
      "<div class='ntx-bar-row ntx-sliders'><label>k1 <input type='range' min='0' max='3' step='0.1' value='1.2' data-s='k1'> <b class='ntx-k1'>1.2</b></label><label>b <input type='range' min='0' max='1' step='0.05' value='0.75' data-s='b'> <b class='ntx-b'>0.75</b></label></div><div class='ntx-out'></div>";
    var out = host.querySelector(".ntx-out");
    function draw() {
      var r = state.mode === "bm25" ? NTX.bm25(IX, state.q, state.k1, state.b, 5) : NTX.cosineSearch(IX, state.q, 5);
      host.querySelector(".ntx-k1").textContent = state.k1; host.querySelector(".ntx-b").textContent = state.b;
      out.innerHTML = "<p class='ntx-meta'>Query terms after normalising: " + (r.terms.map(function (t) { return "<code>" + t + "</code>"; }).join(" ") || "none") + " · " + r.matched + " complaints contain at least one</p>" +
        "<ol class='ntx-hits'>" + r.top.map(function (e) {
          return "<li><span class='ntx-where'>" + C.labels[C.y[e[0]]] + " · complaint " + C.id[e[0]] + " · score " + e[1].toFixed(3) + "</span><span class='ntx-snip'>" + C.docs[e[0]].slice(0, 30).join(" ") + "...</span></li>";
        }).join("") + "</ol>";
    }
    host.addEventListener("click", function (ev) {
      var b = ev.target.closest("button[data-m]"); if (!b) return;
      state.mode = b.getAttribute("data-m");
      host.querySelectorAll("[data-m]").forEach(function (x) { x.classList.toggle("is-on", x === b); });
      draw();
    });
    host.addEventListener("input", function (ev) {
      if (ev.target.classList.contains("ntx-q")) state.q = ev.target.value;
      var s = ev.target.getAttribute("data-s"); if (s) state[s] = parseFloat(ev.target.value);
      draw();
    });
    draw();
  }

  // ---------- themes (session 5) ----------
  var MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
  function mountThemes(host) {
    var C = ready(), state = { seed: 1, k: 6 }, base = null;
    host.innerHTML = "<div class='ntx-head'><h4>Themes by k-means</h4><p class='ntx-rail'>" + RAIL + " Spherical k-means on TF-IDF (stop words, xxxx out, light stemmer, words in 5+ complaints), 20 rounds.</p></div>" +
      "<div class='ntx-bar-row'><span class='ntx-lab'>Seed</span>" + [1, 2, 3].map(function (s) { return "<button type='button' class='ntx-tog' data-seed='" + s + "'>" + s + "</button>"; }).join("") +
      "<span class='ntx-lab'>Themes</span>" + [4, 6, 8].map(function (k) { return "<button type='button' class='ntx-tog' data-k='" + k + "'>" + k + "</button>"; }).join("") + "</div><div class='ntx-out' aria-live='polite'></div>";
    var out = host.querySelector(".ntx-out");
    function draw() {
      host.querySelectorAll("[data-seed]").forEach(function (b) { b.classList.toggle("is-on", +b.getAttribute("data-seed") === state.seed); });
      host.querySelectorAll("[data-k]").forEach(function (b) { b.classList.toggle("is-on", +b.getAttribute("data-k") === state.k); });
      out.innerHTML = "<p class='ntx-meta'>Clustering 3,000 complaints...</p>";
      setTimeout(function () {
        var r = NTX.kmeans(state.k, state.seed, 20);
        if (!base || base.k !== state.k) base = NTX.kmeans(state.k, 1, 20);
        var agree = NTX.pairAgreement(base.assign, r.assign);
        out.innerHTML = "<p class='ntx-meta'>Pair agreement with seed 1 at the same k: <b>" + f3(agree) + "</b> (1 = the same groups)</p><div class='ntx-themes'>" +
          r.clusters.map(function (c, i) {
            var mx = Math.max.apply(null, c.byMonth) || 1;
            return "<div class='ntx-theme'><p class='ntx-tt'>Theme " + (i + 1) + " · " + c.size + " complaints</p><p class='ntx-terms'>" + c.top.map(function (t) { return "<code>" + t + "</code>"; }).join(" ") + "</p>" +
              "<p class='ntx-note'>Largest queue: " + C.labels[c.dominant] + " (" + pct(c.purity, 0) + ")</p><div class='ntx-months' aria-label='complaints per month, January to December 2023: " + c.byMonth.join(", ") + "'>" +
              c.byMonth.map(function (v, m) { return "<span title='" + MONTHS[m] + " " + v + "'><i style='height:" + Math.max(2, v / mx * 100).toFixed(0) + "%'></i></span>"; }).join("") + "</div><p class='ntx-mlab'>Jan to Dec 2023</p></div>";
          }).join("") + "</div>";
      }, 30);
    }
    host.addEventListener("click", function (ev) {
      var b = ev.target.closest("button"); if (!b) return;
      if (b.hasAttribute("data-seed")) state.seed = +b.getAttribute("data-seed");
      if (b.hasAttribute("data-k")) state.k = +b.getAttribute("data-k");
      draw();
    });
    draw();
  }

  var MOUNT = { card: mountCard, normalise: mountNormalise, zipf: mountZipf, terms: mountTerms, bench: mountBench, search: mountSearch, themes: mountThemes };
  function boot() {
    if (!window.NTX_SAMPLE) return;
    document.querySelectorAll("[data-ntx]").forEach(function (h) {
      var f = MOUNT[h.getAttribute("data-ntx")];
      if (f && !h.getAttribute("data-mounted")) { h.setAttribute("data-mounted", "1"); h.classList.add("ntx-widget"); f(h); }
    });
  }
  window.NTX_UI = { benches: BENCHES, boot: boot };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
