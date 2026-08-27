import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the RestructAI stress-risk advisor", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>RestructAI — 90-Day MSME Stress Risk<\/title>/i);
  assert.match(html, /See repayment stress before it becomes default\./);
  assert.match(html, /Run 90-day risk analysis/);
  assert.match(html, /Selected on evidence, not assumption/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project/);
});
