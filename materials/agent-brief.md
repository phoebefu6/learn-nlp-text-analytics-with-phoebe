# Agent brief - shared by every fan-out page of learn-nlp-text-analytics-with-phoebe

Internal build document (filled from course-builder/references/agent-brief.template.md, 2026-11-19).
Never linked from an audience page.

You are writing ONE static HTML session page. No servers, no npm. **If your target file already
exists on disk, do not write it; report that and stop.** Write the file, return its path and one
line of coverage. No HTML in your reply.

## Read first, in this order

1. The template page. Copy its structure, classes, SVG grammar and quiz markup EXACTLY, including
   4 options per quiz question:
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-nlp-text-analytics-with-phoebe/courses/01-text-is-a-table.html`
   For how a bench widget sits inside a page and how its numbers are quoted, also skim the bench page:
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-nlp-text-analytics-with-phoebe/courses/04-the-text-bench.html`
2. The source map: every verified number, its evidence tier, per-session coverage, the seams. Use
   ONLY its numbers; never invent a statistic; if a fact is missing, teach the uncertainty.
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-nlp-text-analytics-with-phoebe/materials/official-course-map.md`
3. The stylesheet `:root` block for the palette tokens (first 30 lines):
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-nlp-text-analytics-with-phoebe/assets/style.css`

## Page skeleton (keep every component)

toolbar (crumb EXACTLY `<a href="../index.html">learn-nlp-text-analytics-with-phoebe</a> / Session N of 6`,
#toggle-all, #zoom-toggle) · masthead (eyebrow "Learn NLP &amp; Text Analytics with Phoebe · Session N
of 6", h1 with one `<span class="accent">`, .sub, .chip-row with the level chip (given in your outline),
the two `.chip.audience` chips copied from the template, `.chip.time` "45 min", .agenda a1-a4) ·
main.wrap · section#intro (Part 0: kicker, .lede, .legend pills, .callout.win "★ What you walk out
with tonight.") · 3 Parts, each `section.section#part-N` with section-kicker (klabel "Part N · covers
...", h2, `.tag.concept "N min live"`), a `.lede`, ONE figure, `details.card` accordions (summary:
`.mode.live` or `.mode.self`, title, `.mini`, `.caret ▶`), at least one `.callout.example` with
`span.ex-pill` "Real world" on the page · section#demo-1 Build-along (kicker `.tag.demo "★ 22 min ·
everyone builds"`, .lede, ONE figure, `.steps > .step`, each with a `.prompt-box.good` carrying a
`span.label`; the widget mounts named in your outline; the last step writes this session's page of
Ledgerly's text playbook as a prompt-box labelled "(Constructed Ledgerly example)") · section#exercise
Homework (ol, 4 items) · section#quiz (3 x `.quiz-q data-answer="0-based"`, `p.qtext`, 4 x
`button.qopt` "A · ...", `p.qwhy`; one `p.quiz-score` after the last; vary the correct letter) ·
section#official, h2 EXACTLY "What this session teaches, and where it came from", `.covered >
.covered-row` (pill solid ✓ / light ◐ + name + note), then the `.mono` line EXACTLY "Every fact on
this page, and its verification tier, is recorded in the course's source map." ·
section.cheat#cheatsheet (h3 "Session N cheat sheet <span>· pin this</span>", .grid-2 of six
.cheat-item) · `.callout.next` with `.nx-pill` "Next session" (session 6: "Where to go next") ·
footer.pagefoot · then, in this order:
`<script src="../assets/ntx-sample.js?v=2"></script>` `<script src="../assets/ntx-live.js?v=2"></script>`
`<script src="../assets/app.js?v=1"></script>`.

Head: the template's social meta block with this page's own title/description/url (url
`https://phoebefu6.github.io/learn-nlp-text-analytics-with-phoebe/courses/<file>`);
`<title>Session N · <Title> - learn NLP and text analytics with phoebe</title>`;
`<link rel="stylesheet" href="../assets/style.css?v=2">`. Nothing else external.

First `details.card` in the FIRST Part is `open`; no other. Sentence case headings. Warm
practitioner voice, concrete, never dry. Inside prompt-boxes escape `&` `<` `>`. 450 to 650 lines
is guidance about depth, never a target: never collapse whitespace, dissolve a list into a
paragraph, or drop a component to fit.

## Widgets (real, computed in the browser by assets/ntx-live.js; do not write any JS)

Mount with `<div class="ntx-mount" data-ntx="NAME"></div>`. Available NAMEs: `card`, `normalise`,
`zipf` (session 1), `terms` (top words per queue under three weightings), `bench` (attributes:
`data-rungs="r2,r4"` limits the rung buttons, `data-start="r4"` picks the first rung, `data-llm="off"`
hides the modelled LLM panel), `search` (`data-q="late fee refund"` sets the first query), `themes`.
Every number you quote from a widget must be one the map lists as printed by it. Never claim the
widget shows something it does not. The bench's rung names are: R0 Majority queue, R1 Keyword rules,
R2 Counts + naive Bayes, R3 TF-IDF + naive Bayes, R4 TF-IDF + bigrams + logistic, R5 Tuned on a
validation split. Trap button "Trap: score on the training rows"; break button "Break: shuffle the
training labels"; the logistic rungs have a "What the logistic model learned" panel.

## Hard rules (a violation is rework)

- NEVER an em dash or en dash, anywhere (prose, code, aria-labels, comments). Hyphen only.
- No meta text: never "this course", "in this course", "the course teaches", "banned here". State
  the professional norm directly with its reason. The two exact estate phrases above are the only
  self-references; "session 5" cross-references are fine.
- Attribution "by Phoebe Fu". Never "built with" a tool.
- Every number comes from the map or is labelled constructed. For constructed data print "your
  numbers will differ", never invented outputs as if run. Python snippets: never print an output
  the map does not give; write "# your numbers will be close, not identical" instead.
- Contested or missing evidence: teach the disagreement; never resolve what the literature has not.
- Citations in the exact form of the map's appendix; anything marked reported is "reported".
- NEVER "lottery" or "lotteries"; say the mechanism ("decided by row order", "arbitrary", "a
  random draw").
- Default to the English word. No Chinese terms.
- No cross marks anywhere (✗ ✘ ✕ ❌ ×): label states in words. Ticks only in the coverage pills.
- Titles, widget ids and class names must not collide with siblings. Use only existing classes from
  style.css; new figure classes only with your figure prefix.
- The running case is "Constructed Ledgerly example", fictional; the complaints are REAL CFPB
  narratives (2023, CC0). Never name Phoebe's employers, clients or projects. Never mention any other
  course's case names.
- The LLM rung is MODELLED: never say an LLM scored X on this bench. Prices are placeholders; every
  mention of price says to re-verify the vendor's current list.
- Never mention `materials/`, file paths in the repo, or "source map" except in the fixed `.mono` line.
- Links to sibling courses are absolute https URLs from the map's seams table; link, never reteach.

## Figure grammar (hand-drawn, every figure)

Palette, ONLY these hexes (no invented greys): #592A22 (accent) · #3B1A15 (deep) · #8A4A3C (mid) ·
#DDB8AD (soft) · #F8EEEB (accent-50) · #2B1C18 (ink) · #5E4A44 (muted) · #D3C3BE (faint) · #EDE3E0
(hairline) · #A9C8E3 (contrast fill, ink text only) · #1E4566 (contrast ink, text) · #EAF2F9 (contrast
tint) · #FDFBFA (paper) · #FFFFFF · universal reds `#991B1B` `#FEF2F2` `#FCA5A5` only for a wrong-way
panel.

- `<figure class="zoomable">` > `<svg viewBox="0 0 880 H" xmlns="http://www.w3.org/2000/svg" role="img"
  aria-label="the data, not the shape">` > `<defs>` + `<style>` + content, then `<figcaption>🔍 Click
  to zoom - takeaway</figcaption>`. Grow H, never W.
- Prefix unique per figure, used for every class and id: `s<N><letter>` in page order, e.g. s2a,
  s2b, s2c, s2d (Part 1, 2, 3, build-along).
- `<defs>` holds three things with the figure prefix P: a wobble filter `id="PSk"`
  (`feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="<int>"` +
  `feDisplacementMap scale="2.4" xChannelSelector="R" yChannelSelector="G"`, `x="-3%" y="-3%"
  width="106%" height="106%"`), a hachure pattern `id="PHc"` (7x7 userSpaceOnUse, rotate(-38), one
  #592A22 line, opacity .5), an open arrowhead `id="PAr"` (path `M1 1 L9 5 L1 9`, fill none, #2B1C18
  stroke 1.6). ALL shapes sit inside ONE `<g filter="url(#PSk)" fill="none" stroke="#2B1C18"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round">`; rects carry a tiny rotation
  (-4 to 4 degrees for hand-placed items, under 1 for panels). Fills: white, #F8EEEB, the hachure for
  "the pile" or "the data" (never put text on hachure), and the contrast fill #A9C8E3 ONLY for the
  one thing the figure is about. One doodle anchor per figure, simple strokes, never a mascot.
  Text classes: `.PH` 800 12px #2B1C18 heading · `.PL` 600 12px ink label · `.PS` 400 11px #5E4A44 ·
  `.PB` 800 11px #1E4566 · `.PV` 800 16-20px #3B1A15 value · `.PW` 800 12px white on #592A22 or
  #8A4A3C only · `.PA` 700 11px #592A22 axis caption · `.PN` 400 12px #5E4A44 note. Never muted (#5E4A44)
  text on #DDB8AD or darker. Hand-stacked items must not overlap as painted rects.
- ALL `<text>` outside any filtered group, Inter sans stack, never below 10.5px.
- Fit: max chars ≈ (box width - 20) / 7 at 12px, 6.4px/char at 11px; full-width note under 110
  chars; 40px between neighbouring point labels; no line, arrow or curve may cross a label; bottom
  note 22px below the last row, H clears it by 8px. When in doubt, shorten.
- Floor: one figure per Part plus one in the build-along. Draw the MECHANISM (how a weight is
  computed, how a probability multiplies, how BM25 saturates, how a seed changes clusters), never a
  metaphor literally, never decoration.

## Voice and honesty

Every Part gets a real-world story (a `.callout.example`), generic and plausible, never a named real
company scandal unless it is in the map. Ledgerly is "Constructed Ledgerly example" every time it
appears in a prompt-box label or a figure. The rebalanced sample (credit reporting 30% here, 76.7% in
the real 2023 rows) must be mentioned wherever accuracy is interpreted. Stemmer artefacts are real
outputs of the bench's light stemmer, not Porter.

## Cross-links (absolute URLs)

- naive Bayes derivation: https://phoebefu6.github.io/learn-classification-regression-with-phoebe/courses/c6-naive-bayes.html
- BM25 by hand for retrieval: https://phoebefu6.github.io/learn-ship-internal-ai-assistant-with-phoebe/courses/b2-chunk-index-retrieve.html
- embeddings: https://phoebefu6.github.io/learn-rag-with-phoebe/courses/b1-embeddings.html
- retrieval for an LLM: https://phoebefu6.github.io/learn-rag-with-phoebe/courses/b4-retrieval-basics.html
- TF-IDF item profiles: https://phoebefu6.github.io/learn-recommendation-with-phoebe/courses/b5-content-based.html
- classification metrics: https://phoebefu6.github.io/learn-intro-ml-with-phoebe/courses/b7-classification-metrics.html
- train/test split: https://phoebefu6.github.io/learn-intro-ml-with-phoebe/courses/b2-train-test-split.html
- survey design: https://phoebefu6.github.io/learn-survey-design-with-phoebe/
- Hub: https://phoebefu6.github.io/learn-with-phoebe/

## Footer chain and session titles

Footer left: `Session N of 6 · learn-nlp-text-analytics-with-phoebe · by Phoebe Fu &nbsp;·&nbsp; 📚 <a href="https://phoebefu6.github.io/learn-with-phoebe/">Learn with Phoebe ↗</a>`
Footer right: `<a href="PREV">← Prev: PREV TITLE</a> &nbsp;·&nbsp; <a href="NEXT">Next: NEXT TITLE →</a>`
(session 6: `<a href="05-search-and-themes.html">← Prev: Search and themes</a> &nbsp;·&nbsp; <a href="../index.html">Course home</a>`).

Session titles and files (exact):
1. 01-text-is-a-table.html · Text is a table you have not built yet
2. 02-weight-the-words.html · Weight the words
3. 03-route-it.html · Route it: a classifier you can read
4. 04-the-text-bench.html · The text bench
5. 05-search-and-themes.html · Search and themes
6. 06-when-the-big-model-earns-its-place.html · When the big model earns its place

Ledgerly's text playbook pages: 1 corpus card (s1) · 2 vocabulary and weighting choice (s2) · 3 routing
classifier and error table (s3) · 4 bench result with trap rows (s4) · 5 search setup and theme
watchlist (s5) · 6 signed build-or-buy note (s6).
