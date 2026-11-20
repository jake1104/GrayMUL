const inputAEl = document.getElementById("inputA");
const inputBEl = document.getElementById("inputB");
const budgetEl = document.getElementById("budget");
const multiplyBtn = document.getElementById("multiply");
const cancelBtn = document.getElementById("cancel");
const statusEl = document.getElementById("status");
const wallEl = document.getElementById("wall");
const productEl = document.getElementById("product");
const methodEl = document.getElementById("method");
const limbsEl = document.getElementById("limbs");
const elapsedEl = document.getElementById("elapsed");
const rawEl = document.getElementById("raw");

let worker = null;
let requestId = 0;
let runStarted = 0;

function spawnWorker() {
  if (worker) {
    worker.terminate();
  }
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (event) => {
    setStatus("error: " + (event.message || "worker failed"));
    setRunning(false);
  };
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setRunning(running) {
  multiplyBtn.disabled = running;
  cancelBtn.disabled = !running;
}

function showResult(payload, wallMs) {
  let data;
  try {
    data = JSON.parse(payload);
  } catch {
    setStatus("error: invalid JSON from worker");
    return;
  }
  rawEl.textContent = JSON.stringify(data, null, 2);
  wallEl.textContent = wallMs + " ms";
  if (!data.ok) {
    setStatus("failed: " + data.error);
    return;
  }
  productEl.textContent = data.product;
  methodEl.textContent = data.method;
  limbsEl.textContent = String(data.limbs);
  elapsedEl.textContent = String(data.elapsedMs) + " ms";
  try {
    if (BigInt(data.product) === BigInt(data.inputA) * BigInt(data.inputB)) {
      setStatus("done: product verifies");
    } else {
      setStatus("done: PRODUCT MISMATCH");
    }
  } catch {
    setStatus("done: product not verifiable");
  }
  setRunning(false);
}

function onWorkerMessage(event) {
  const { id, ok, body, error, wallMs } = event.data;
  if (id !== requestId) {
    return;
  }
  if (!ok) {
    setStatus("failed: " + error);
    setRunning(false);
    return;
  }
  showResult(body, wallMs);
}

spawnWorker();

multiplyBtn.addEventListener("click", () => {
  spawnWorker();
  const a = inputAEl.value.trim();
  const b = inputBEl.value.trim();
  const budgetMs = Number(budgetEl.value);
  if (!a || !b) {
    setStatus("error: empty input");
    return;
  }
  requestId += 1;
  runStarted = performance.now();
  setRunning(true);
  setStatus("running: no progress events, wait for done");
  worker.postMessage({ id: requestId, a, b, budgetMs });
});

cancelBtn.addEventListener("click", () => {
  spawnWorker();
  const waited = Math.round(performance.now() - runStarted);
  wallEl.textContent = waited + " ms (cancelled)";
  setStatus("cancelled by worker termination");
  setRunning(false);
});
