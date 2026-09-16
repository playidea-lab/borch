/**
 * The "Copy embed code" buttons in the landing's embed showcase.
 *
 * The snippet is read from the card's own iframe — `iframe.src` is the address the browser
 * already resolved, absolute and correct on every origin and from either language's page,
 * so nothing here hard-codes a host or a path prefix. Kept in its own file, not inline,
 * because the site's broken-link check reads page HTML and would misread a `src="` literal
 * in an inline script as a link.
 */
const COPIED = document.documentElement.lang === "ko" ? "복사됨 ✓" : "Copied ✓";

for (const button of document.querySelectorAll("[data-copy-embed]")) {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    const iframe = button.closest(".card")?.querySelector("iframe");
    if (!iframe) return;
    const code = '<iframe src="' + iframe.src
      + '" width="100%" height="480" style="border:0" title="borch widget"></iframe>';
    const flash = () => {
      const was = button.textContent;
      button.textContent = COPIED;
      setTimeout(() => { button.textContent = was; }, 1500);
    };
    if (navigator.clipboard) navigator.clipboard.writeText(code).then(flash, flash);
    else flash();
  });
}
