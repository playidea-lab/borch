// 카탈로그를 읽어 표를 그리고, 가장 작은 모델 하나를 실제로 받아 검증한다.
//
// **목록을 페이지에 적지 않는다.** 모델은 늘어나고, 손으로 옮겨 적은 목록은
// 늘어난 것을 모른 채 맞는 것처럼 보인다. 이 저장소가 그 실패로 하루를 썼다.
const CATALOG = "https://models.pilab.kr/index.json";

// 문구는 페이지 언어를 따른다. 두 페이지가 같은 스크립트를 쓴다.
const KO = document.documentElement.lang === "ko";
const T = KO ? {
  loading: "목차 받는 중…", idle: "대기", noCat: "목차가 응답하지 않았다",
  model: "모델", adapter: "어댑터", env: "환경", loaded: "받아서 싣기까지",
  verify: "검증", stopped: "멈춤", hint: "어댑터가 없다고 나오면, GPU 페이지에 브라우저가 무엇을 요구하는지 적혀 있다.",
} : {
  loading: "loading the catalog…", idle: "idle", noCat: "the catalog did not answer",
  model: "model", adapter: "adapter", env: "environment", loaded: "downloaded and loaded",
  verify: "verify", stopped: "stopped", hint: "If this says no adapter, the GPU page has what your browser is asking for.",
};

const $ = (id) => document.getElementById(id);
const mb = (n) => (n / 1e6).toFixed(1) + " MB";

// 라이선스는 목차가 아니라 매니페스트에 있다. 표를 먼저 그리고 나중에 채운다 —
// 여섯 번의 왕복 때문에 목록 전체가 기다릴 이유가 없다.
async function licence(url) {
  try {
    const m = await (await fetch(url)).json();
    const l = m.license || {};
    return [l.weights, l.data].filter(Boolean).join(" · ") || "—";
  } catch {
    return "—";
  }
}

async function catalogue() {
  const body = document.querySelector("#cat tbody");
  let index;
  try {
    index = await (await fetch(CATALOG)).json();
  } catch (e) {
    body.innerHTML = `<tr><td colspan="6" class="small">${T.noCat} (${e})</td></tr>`;
    return null;
  }
  const rows = index.models.slice().sort((a, b) => a.bytes - b.bytes);
  body.innerHTML = rows.map((m, i) => `<tr>
      <td><a href="${m.manifestUrl}"><code>${m.name}</code></a><br><span class="small">${m.version}</span></td>
      <td class="small">${m.task}</td>
      <td class="small">${m.dataset}</td>
      <td class="small">${mb(m.bytes)}</td>
      <td class="small"><code>${m.origin}</code></td>
      <td class="small" id="lic${i}">…</td>
    </tr>`).join("");
  rows.forEach((m, i) => licence(m.manifestUrl).then((t) => { $("lic" + i).textContent = t; }));
  return rows;
}

// 순서가 요점이다 — 환경 판정을 마지막에 하면 10MB 를 받은 뒤에 안 된다고 말하게 된다.
async function run(smallest) {
  const out = $("out");
  const say = (s) => { out.textContent += s + "\n"; };
  out.textContent = "";
  $("go").disabled = true;
  try {
    const [{ init, Device }, hub] = await Promise.all([
      import("borch-ts"),
      import("borch-hub"),
    ]);
    say(`${T.model}    ${smallest.name} ${smallest.version} · ${mb(smallest.bytes)}`);
    await init();
    $("badge").textContent = Device.adapterInfo || "";
    say(`${T.adapter}  ${Device.adapterInfo || "?"}`);

    if (hub.checkEnvironment) {
      const manifest = await hub.fetchManifest(smallest.manifestUrl);
      const env = await hub.checkEnvironment(manifest);
      say(`${T.env}  ${JSON.stringify(env)}`);
    }
    const t0 = performance.now();
    const { model, manifest } = await hub.load(smallest.manifestUrl);
    say(`${T.loaded}  ${Math.round(performance.now() - t0)} ms`);
    const badge = await hub.verify(model, manifest, smallest.manifestUrl);
    say(`${T.verify}  ${JSON.stringify(badge, null, 1)}`);
  } catch (e) {
    say(`${T.stopped}: ${e && e.message ? e.message : e}`);
    say("");
    say(T.hint);
  } finally {
    $("go").disabled = false;
  }
}

const rows = await catalogue();
if (rows && rows.length) {
  $("go").addEventListener("click", () => run(rows[0]));
} else {
  $("go").disabled = true;
}
