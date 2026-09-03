/**
 * The living parts of the landing page — the hero demo and the device badge.
 *
 * An explainer that writes "it runs in the browser" and then does not show it has
 * written an advertisement. So the first screen offers one run, right there.
 */

import { HERO_CODE } from "./examples.js";
import { t } from "./i18n.js";
import { describeError, highlight, loadBorch, probeDevice, runCode } from "./runner.js";

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


/**
 * Is this adapter a CPU? **The library answers now.**
 *
 * This file kept its own copy of the four names, and so did `playground.js`, and
 * `tests/browser/launch.py` has a third. Three copies of one judgement, in the shape
 * that drifts — and the direction nobody reports is a visitor with a real GPU being
 * told it is software. `probe()` carries `software` with the name, so the rule lives
 * once in `borch-ts/src/device.ts` and this reads it.
 */

/** Linux, where Chrome's blocklist is what stands between the page and the driver. */
const ON_LINUX = /linux/i.test(navigator.userAgent) && !/android/i.test(navigator.userAgent);

(async () => {
  try {
    const p = await probeDevice();
    if (p.ok) {
      badge.className = p.software ? "badge off" : "badge on";
      badgeText.textContent = p.adapter;
      // **The device is made now, while the visitor is still reading.** On Linux with
      // the NVIDIA driver, asking for the adapter and the device at the click cost
      // three seconds; moved here it overlaps the reading, and the click pays for the
      // shaders and the readback only. `init()` is idempotent, so the click's own
      // `init()` joins this one rather than making a second device.
      if (!p.software) {
        const warm = () => loadBorch().then((b) => b.init()).catch(() => {});
        if ("requestIdleCallback" in window) requestIdleCallback(warm); else setTimeout(warm, 0);
      }
      // **One screen, one sentence.** Saying both — "that adapter is a CPU" and "Run
      // executes on this tab's GPU" — is the confusion this site exists to refuse, and it
      // was here for a day: the warning was added above the ready line without the ready
      // line being asked whether it was still true.
      if (p.software) {
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
