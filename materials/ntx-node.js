// Run the in-browser engine under node and print the canon JSON.
// Usage: node materials/ntx-node.js > canon-node.json
const fs = require("fs"), path = require("path"), vm = require("vm");
const root = path.join(__dirname, "..", "assets");
const ctx = { window: {}, Math, Date, JSON, Object, Array, Float64Array, Number, String, Infinity, console };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root, "ntx-sample.js"), "utf8"), ctx);
vm.runInContext(fs.readFileSync(path.join(root, "ntx-live.js"), "utf8"), ctx);
const NTX = ctx.window.NTX;
NTX.load(ctx.window.NTX_SAMPLE);
const t0 = Date.now();
const out = NTX.canon();
out._ms = Date.now() - t0;
process.stdout.write(JSON.stringify(out, null, 1));
