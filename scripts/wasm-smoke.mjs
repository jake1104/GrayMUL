import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(here, "..", "crates", "graymul-wasm", "pkg");
const gluePath = resolve(pkgDir, "graymul_wasm.js");
const wasmPath = resolve(pkgDir, "graymul_wasm_bg.wasm");

const SUCCESS_KEYS = [
  "ok",
  "inputA",
  "inputB",
  "product",
  "method",
  "limbs",
  "elapsedMs",
  "stages",
];
const STAGE_KEYS = ["stage", "degree", "ms", "detail"];
const FAILURE_KEYS = ["ok", "inputA", "inputB", "error"];

const MEDIUM_A =
  "578964009337337874187360781092024505713778130725625315039426694259848288593038490104106364031657339731126988579026769215394097305034050787316311861802113091065295322707102605992775493513473767026023590777587678135624990873917354601609674034493922431556445515586865401840097711823467817583264137541257113771036211218623864154994859976239959919561520395643529793580358827587349302049925731225688418929589285713569468469974511430715289221698159613704143658834413050098547857147458650713756470774456134307736509113621450231040101303277051594265056528074727116924730504215587973635960351044694499555281558";
const MEDIUM_B =
  "933771120216228746025170036994693749676259258415339690792533757098528690952921487367218944289960368445720520173150795393784232028282940949270276061485149743909443002248523460042398410534417469905544854231311344463407962365625168022061091549134433606505808779218589414503133058786341286388967253868193938723753031893753376031992961496301933439710285601366523262038838126449555905150988380187147286016989093054725640393030600680290710577332129649790025380641422342521950001804850081298069959594420880485414486285087944606761743713889010039711954099016068783002083870781004405942026296415783571340277211";

const CASES = [
  {
    name: "small-schoolbook",
    a: "12345",
    b: "67890",
    budgetMs: 5000,
    expectedProduct: "838102050",
  },
  {
    name: "medium-600digit",
    a: MEDIUM_A,
    b: MEDIUM_B,
    budgetMs: 10000,
  },
  {
    name: "invalid-input",
    a: "abc",
    b: "12",
    budgetMs: 5000,
    expectFailure: "input",
  },
];

function sameKeys(object, expected) {
  const actual = Object.keys(object).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
}

const glue = await import(pathToFileURL(gluePath).href);
const init = glue.default;
const bytes = readFileSync(wasmPath);

let initForm = "options-object";
if (typeof init !== "function") {
  initForm = "sync-auto";
} else {
  try {
    await init({ module_or_path: bytes });
  } catch (error) {
    initForm = "positional";
    await init(bytes);
  }
}

console.log(`version: ${glue.version()}`);
console.log(`init-form: ${initForm}`);
console.log(`wasm-bytes: ${bytes.length}`);

if (typeof glue.version() !== "string" || !/^\d+\.\d+\.\d+$/.test(glue.version())) {
  console.log(`version-shape: FAIL ${String(glue.version())}`);
  process.exitCode = 1;
}

let passed = 0;
let failed = 0;

for (const testCase of CASES) {
  const started = Date.now();
  const raw = glue.multiply(testCase.a, testCase.b, testCase.budgetMs);
  const wallMs = Date.now() - started;
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (error) {
    failed += 1;
    console.log(`${testCase.name}: FAIL invalid-json ${String(error)}`);
    console.log(`raw: ${String(raw).slice(0, 200)}`);
    continue;
  }
  console.log(`${testCase.name}: keys ${Object.keys(payload).sort().join(",")}`);
  console.log(`${testCase.name}: wall-ms ${wallMs}`);

  if (testCase.expectFailure) {
    if (payload.ok !== false) {
      failed += 1;
      console.log(`${testCase.name}: FAIL expected-failure-got-success`);
      continue;
    }
    if (!sameKeys(payload, FAILURE_KEYS)) {
      failed += 1;
      console.log(`${testCase.name}: FAIL failure-key-set ${Object.keys(payload).join(",")}`);
      continue;
    }
    if (payload.inputA !== testCase.a || payload.inputB !== testCase.b) {
      failed += 1;
      console.log(`${testCase.name}: FAIL failure-echo`);
      continue;
    }
    if (typeof payload.error !== "string" || !payload.error.includes(testCase.expectFailure)) {
      failed += 1;
      console.log(`${testCase.name}: FAIL failure-error ${String(payload.error)}`);
      continue;
    }
    passed += 1;
    console.log(`${testCase.name}: PASS failure-path`);
    continue;
  }

  if (payload.ok !== true) {
    failed += 1;
    console.log(`${testCase.name}: FAIL ok-false ${String(payload.error)}`);
    continue;
  }
  if (!sameKeys(payload, SUCCESS_KEYS)) {
    failed += 1;
    console.log(`${testCase.name}: FAIL success-key-set ${Object.keys(payload).join(",")}`);
    continue;
  }
  if (payload.inputA !== testCase.a || payload.inputB !== testCase.b) {
    failed += 1;
    console.log(`${testCase.name}: FAIL input-echo`);
    continue;
  }
  const expected = testCase.expectedProduct ?? (BigInt(testCase.a) * BigInt(testCase.b)).toString();
  if (payload.product !== expected) {
    failed += 1;
    console.log(`${testCase.name}: FAIL product`);
    continue;
  }
  if (BigInt(payload.product) !== BigInt(testCase.a) * BigInt(testCase.b)) {
    failed += 1;
    console.log(`${testCase.name}: FAIL product-verify`);
    continue;
  }
  if (typeof payload.method !== "string" || payload.method.length === 0) {
    failed += 1;
    console.log(`${testCase.name}: FAIL method ${String(payload.method)}`);
    continue;
  }
  if (typeof payload.limbs !== "number" || payload.limbs < 1) {
    failed += 1;
    console.log(`${testCase.name}: FAIL limbs ${String(payload.limbs)}`);
    continue;
  }
  const stages = Array.isArray(payload.stages) ? payload.stages : null;
  if (!stages || stages.some((stage) => !sameKeys(stage, STAGE_KEYS))) {
    failed += 1;
    console.log(`${testCase.name}: FAIL stage-key-set`);
    continue;
  }
  const elapsedMs = payload.elapsedMs;
  if (typeof elapsedMs !== "number" || !Number.isFinite(elapsedMs) || elapsedMs <= 0) {
    failed += 1;
    console.log(`${testCase.name}: FAIL elapsedMs ${String(payload.elapsedMs)}`);
    continue;
  }
  if (elapsedMs > testCase.budgetMs + 2000) {
    failed += 1;
    console.log(`${testCase.name}: FAIL elapsedMs-over-budget ${String(payload.elapsedMs)}`);
    continue;
  }
  if (elapsedMs > wallMs + 2000) {
    failed += 1;
    console.log(`${testCase.name}: FAIL elapsedMs-over-wall ${String(payload.elapsedMs)}`);
    continue;
  }
  const badStage = stages.find(
    (stage) =>
      typeof stage.ms !== "number" ||
      !Number.isFinite(stage.ms) ||
      stage.ms < 0 ||
      stage.ms > testCase.budgetMs + 2000 ||
      stage.ms > wallMs + 2000,
  );
  if (badStage) {
    failed += 1;
    console.log(`${testCase.name}: FAIL stage-ms ${String(badStage.ms)}`);
    continue;
  }
  passed += 1;
  console.log(`${testCase.name}: PASS method=${payload.method} limbs=${payload.limbs}`);
}

console.log(`summary: ${passed}/${CASES.length} PASS failed ${failed}`);
if (failed > 0) {
  process.exitCode = 1;
}
