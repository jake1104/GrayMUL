import init, { multiply } from "../../crates/graymul-wasm/pkg/graymul_wasm.js";

const wasmUrl = new URL(
  "../../crates/graymul-wasm/pkg/graymul_wasm_bg.wasm",
  import.meta.url,
);
await init(wasmUrl);

self.onmessage = (event) => {
  const { id, a, b, budgetMs } = event.data;
  const start = Date.now();
  let payload;
  try {
    payload = multiply(a, b, budgetMs);
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error), wallMs: Date.now() - start });
    return;
  }
  self.postMessage({ id, ok: true, body: payload, wallMs: Date.now() - start });
};
