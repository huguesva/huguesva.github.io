// Build the site first, then run with Node 24+ and Chrome/Chromium installed.
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const site = join(root, "_site");
const articlePath = "/blog/2026/world-models/";
const canonical = "https://huguesva.github.io" + articlePath;
const output = join(root, "blog/2026/world-models/world-models.pdf");
const chromePath =
  process.env.CHROME_BIN || (process.platform === "darwin" ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "chromium");
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".gif": "image/gif",
  ".png": "image/png",
  ".woff2": "font/woff2",
  ".bib": "text/plain",
};
const server = createServer(async (req, res) => {
  try {
    let pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (pathname.endsWith("/")) pathname += "index.html";
    const path = resolve(site, "." + pathname);
    if (!path.startsWith(site + "/")) throw new Error("Invalid path");
    res.setHeader("Content-Type", mime[extname(path)] || "application/octet-stream");
    res.end(await readFile(path));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const profile = await mkdtemp(join(tmpdir(), "world-model-pdf-"));
const chrome = spawn(chromePath, ["--headless", "--disable-gpu", "--no-first-run", "--remote-debugging-pipe", `--user-data-dir=${profile}`], {
  stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"],
});
let nextId = 0;
const pending = new Map();
let buffer = "";
chrome.stdio[4].on("data", (chunk) => {
  buffer += chunk;
  let end;
  while ((end = buffer.indexOf("\0")) !== -1) {
    const message = JSON.parse(buffer.slice(0, end));
    buffer = buffer.slice(end + 1);
    const callbacks = pending.get(message.id);
    if (callbacks) {
      pending.delete(message.id);
      message.error ? callbacks.reject(new Error(JSON.stringify(message.error))) : callbacks.resolve(message.result);
    }
  }
});
function send(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Timed out: ${method}`));
    }, 60000);
    pending.set(id, {
      resolve: (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      reject: (e) => {
        clearTimeout(timer);
        reject(e);
      },
    });
    chrome.stdio[3].write(JSON.stringify({ id, method, params, sessionId }) + "\0");
  });
}
try {
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const call = (method, params) => send(method, params, sessionId);
  const evaluate = async (expression) => {
    const result = await call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call("Page.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
  await call("Page.navigate", { url: `http://127.0.0.1:${server.address().port}${articlePath}` });
  await evaluate(`new Promise((resolve, reject) => {
    const start = Date.now();
    const timer = setInterval(() => {
      if (document.querySelector('mjx-container') && document.querySelector('d-citation-list li')) { clearInterval(timer); resolve(true); }
      else if (Date.now() - start > 45000) { clearInterval(timer); reject(new Error('Math or references did not render')); }
    }, 100);
  })`);
  console.log(
    await evaluate(
      `JSON.stringify({math: document.querySelectorAll('mjx-container').length, references: document.querySelectorAll('d-citation-list li').length, details: document.querySelectorAll('details').length})`
    )
  );
  const printCSS = await readFile(join(root, "scripts/world-models-print.css"), "utf8");
  await evaluate(`(async () => {
    document.documentElement.setAttribute('data-theme', 'light');
    document.title = 'Feature Selection in World Models: A Spectral View';
    document.querySelector('a[href$="world-models.pdf"]')?.closest('p')?.remove();
    const article = document.querySelector('d-article');
    const groups = [
      { letter: 'A', title: 'Probabilistic objectives and loss approximations', proof: false },
      { letter: 'B', title: 'Mode-selection derivations', proof: true }
    ];
    const details = [...article.querySelectorAll('details')];
    for (const group of groups) {
      const appendix = document.createElement('section');
      appendix.className = 'pdf-appendix';
      const heading = document.createElement('h2');
      heading.textContent = 'Appendix ' + group.letter + '. ' + group.title;
      appendix.append(heading);
      const sections = details.filter(detail => detail.querySelector('summary') && detail.querySelector('summary').textContent.startsWith('Derivation') === group.proof);
      if (sections.length !== 5) throw new Error('Expected five sections in Appendix ' + group.letter);
      sections.forEach((detail, index) => {
        const label = group.letter + '.' + (index + 1);
        const id = 'appendix-' + group.letter.toLowerCase() + '-' + (index + 1);
        const section = document.createElement('section');
        section.id = id;
        const title = document.createElement('h3');
        const summary = detail.querySelector('summary');
        title.textContent = label + '. ' + summary.textContent;
        summary.remove();
        section.append(title, ...detail.childNodes);
        appendix.append(section);
        const crossref = document.createElement('p');
        crossref.className = 'pdf-crossref';
        const link = document.createElement('a');
        link.href = '#' + id;
        link.dataset.pdfInternal = 'true';
        link.textContent = (group.proof ? 'Derivation' : 'Loss approximation') + ': Appendix ' + label + '.';
        crossref.append(link);
        detail.replaceWith(crossref);
      });
      article.append(appendix);
    }
    for (const paragraph of article.querySelectorAll('p')) {
      for (const node of paragraph.childNodes) {
        if (node.nodeType === Node.TEXT_NODE) node.textContent = node.textContent.replace('The dropdowns provide', 'Appendix A provides');
      }
    }
    await MathJax.startup.promise;
    await MathJax.typesetPromise();
    if (document.querySelector('mjx-merror')) throw new Error('Math rendering error');
    document.querySelectorAll('img').forEach(img => img.loading = 'eager');
    await Promise.all([...document.images].map(img => img.decode().catch(() => {})));
    const broken = [...document.images].filter(img => !img.naturalWidth);
    if (broken.length) throw new Error('Missing figures: ' + broken.map(img => img.src));
    // Freeze animations at their first frame; the PDF links readers to the full animations.
    for (const img of document.querySelectorAll('img[src$=".gif"]')) {
      const response = await fetch(img.src);
      const decoder = new ImageDecoder({ data: await response.arrayBuffer(), type: 'image/gif' });
      const { image } = await decoder.decode({ frameIndex: 0 });
      const canvas = document.createElement('canvas'); canvas.width = image.displayWidth; canvas.height = image.displayHeight;
      canvas.getContext('2d').drawImage(image, 0, 0); img.src = canvas.toDataURL('image/png'); image.close(); decoder.close();
    }
    document.querySelector('d-byline').innerHTML = '<p class="pdf-author">Hugues Van Assel</p><p>October 2, 2026 · Research note · PDF version: October 13, 2026</p>';
    const note = document.createElement('p'); note.className = 'pdf-version';
    note.innerHTML = 'Full text and derivations of the blog post. Animated figures are shown as still frames. Interactive version: <a href="${canonical}">${canonical}</a>.';
    document.querySelector('d-article').prepend(note);
    document.querySelectorAll('a[href]').forEach(a => { if (a.origin === location.origin && !a.dataset.pdfInternal) a.href = 'https://huguesva.github.io' + a.pathname + a.search + a.hash; });
    const style = document.createElement('style'); style.textContent = ${JSON.stringify(printCSS)}; document.head.append(style);
    await document.fonts.ready;
  })()`);
  await call("Emulation.setEmulatedMedia", { media: "print" });
  // Optional first-page-like screenshot for visual inspection during maintenance.
  if (process.env.PDF_PREVIEW) {
    const shot = await call("Page.captureScreenshot", { format: "png" });
    await writeFile(process.env.PDF_PREVIEW, Buffer.from(shot.data, "base64"));
  }
  const pdf = await call("Page.printToPDF", {
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate:
      '<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    marginTop: 0.65,
    marginBottom: 0.65,
    marginLeft: 0.65,
    marginRight: 0.65,
  });
  const bytes = Buffer.from(pdf.data, "base64");
  if (bytes.length > 5_000_000) throw new Error(`PDF exceeds Scholar's 5 MB limit: ${bytes.length}`);
  await writeFile(output, bytes);
  console.log(`Wrote ${output} (${bytes.length} bytes)`);
} finally {
  await send("Browser.close").catch(() => chrome.kill());
  server.close();
  await new Promise((resolve) => {
    if (chrome.exitCode !== null) resolve();
    else chrome.once("exit", resolve);
  });
  await rm(profile, { recursive: true, force: true });
}
