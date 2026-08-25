/**
 * The living parts of the landing page — the hero demo and the device badge.
 *
 * An explainer that writes "it runs in the browser" and then does not show it has
 * written an advertisement. So the first screen offers one run, right there.
 */

import { HERO_CODE } from "./examples.js";
import { t } from "./i18n.js";
import { describeError, highlight, probeDevice, runCode } from "./runner.js";

const codeEl = document.getElementById("hero-code");
const outEl = document.getElementById("hero-out");
const runBtn = document.getElementById("hero-run");
const badge = document.getElementById("device-badge");
const badgeText = document.getElementById("device-text");

codeEl.innerHTML = highlight(HERO_CODE);

function say(text, kind = "") {
  const line = document.createElement("div");
  if (kind) line.className = kind;
  line.textContent = text;
  outEl.append(line);
  outEl.scrollTop = outEl.scrollHeight;
}

/** The same line, with a way out of it. `textContent` everywhere else keeps user output
 *  from becoming markup; this one place builds the anchor itself rather than parsing a
 *  string, so that stays true. */
function sayLink(text, href) {
  const line = document.createElement("div");
  line.className = "note";
  const a = document.createElement("a");
  a.href = href;
  a.textContent = text;
  line.append(a);
  outEl.append(line);
  outEl.scrollTop = outEl.scrollHeight;
}


/** Is this adapter a CPU? The four names `tests/browser/launch.py` refuses on — one rule,
 *  two places, and the day they disagree is the day one of them lies about a number. */
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|software/i;

/** Linux, where Chrome's blocklist is what stands between the page and the driver. */
const ON_LINUX = /linux/i.test(navigator.userAgent) && !/android/i.test(navigator.userAgent);

(async () => {
  try {
    const p = await probeDevice();
    if (p.ok) {
      badge.className = SOFTWARE.test(p.adapter) ? "badge off" : "badge on";
      badgeText.textContent = p.adapter;
      // **One screen, one sentence.** Saying both — "that adapter is a CPU" and "Run
      // executes on this tab's GPU" — is the confusion this site exists to refuse, and it
      // was here for a day: the warning was added above the ready line without the ready
      // line being asked whether it was still true.
      if (SOFTWARE.test(p.adapter)) {
        say(t("device.software"), "err");
        sayLink(t("device.setupSay"), t("device.setupHref"));
      } else {
        say(t("device.ready"), "note");
      }
    } else {
      badge.className = "badge off";
      badgeText.textContent = t(p.why === "no-api" ? "device.noApi" : "device.noAdapter");
      say(p.message, "err");
      if (p.why === "no-adapter" && ON_LINUX) say(t("device.linuxFlags"), "note");
      say(t("device.noFallback"), "note");
      sayLink(t("device.setupSay"), t("device.setupHref"));
      runBtn.disabled = true;
    }
  } catch (err) {
    badge.className = "badge off";
    badgeText.textContent = t("device.notLoaded");
    say(String(err.message ?? err), "err");
    runBtn.disabled = true;
  }
})();

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  outEl.textContent = "";
  const t0 = performance.now();
  try {
    await runCode(HERO_CODE, { onLog: (t, k) => say(t, k) });
    say("");
    say(t("run.doneLocal", (performance.now() - t0).toFixed(0)), "ok");
  } catch (err) {
    say(describeError(err), "err");
  } finally {
    runBtn.disabled = false;
  }
});
