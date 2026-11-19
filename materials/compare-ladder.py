"""Compare the node canon (assets/ntx-live.js) with the Python canon (ntx_reference.py) BY VALUE.
Integers, strings and ids must match exactly; floats within 1e-9 relative. Keys only one side has
(timings starting with "_") are ignored. Self-test: a planted 0.1% fault in one value must fail.
Usage: python3 -I compare-ladder.py canon-node.json canon-python.json
"""
import sys, json, copy

def walk(a, b, path, bad, count):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k.startswith("_"):
                continue
            if k not in a or k not in b:
                bad.append(path + "/" + k + " missing on one side")
                continue
            walk(a[k], b[k], path + "/" + k, bad, count)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append("%s length %d vs %d" % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), bad, count)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        count[0] += 1
        if a != b:
            bad.append("%s %r vs %r" % (path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        count[0] += 1
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                bad.append("%s %r vs %r" % (path, a, b))
        elif abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            bad.append("%s %r vs %r" % (path, a, b))
    else:
        if a != b:
            bad.append("%s %r vs %r" % (path, a, b))

def compare(a, b):
    bad, count = [], [0]
    walk(a, b, "", bad, count)
    return bad, count[0]

node, py = (json.load(open(p)) for p in sys.argv[1:3])
bad, n = compare(node, py)
planted = copy.deepcopy(py)
planted["ladder"]["r4"]["acc"] *= 1.001
pbad, _ = compare(node, planted)
print("values compared:", n, "| mismatches:", len(bad))
for line in bad[:20]:
    print("  ", line)
print("planted 0.1% fault caught:", len(pbad) > len(bad))
sys.exit(0 if (not bad and len(pbad) > len(bad)) else 1)
