import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("a denied kiosk entry waits for another gesture and still allows exit", async () => {
    const listeners = new Map();
    let attempts = 0;
    const canvasPage = {
        fullscreenEnabled: true,
        fullscreenElement: null,
        documentElement: {
            async requestFullscreen() {
                if (++attempts === 1) throw new Error("Activation expired");
                canvasPage.fullscreenElement = this;
            },
        },
        async exitFullscreen() { this.fullscreenElement = null; },
        addEventListener(key, handler) { listeners.set(key, handler); },
    };
    globalThis.document = canvasPage;
    const activation = { isActive: false };
    Object.defineProperty(globalThis, "navigator", {
        value: { userActivation: activation }, configurable: true,
    });
    const source = await readFile(new URL("../src/app/display/fullscreen.js", import.meta.url));
    const { sync_fullscreen } = await import("data:text/javascript;base64," + source.toString("base64"));

    assert.equal(sync_fullscreen(true, true), true);
    assert.equal(attempts, 0);
    activation.isActive = true;
    listeners.get("click")({ isTrusted: true });
    await new Promise(setImmediate);
    assert.equal(attempts, 1);
    assert.equal(sync_fullscreen(undefined, true), true);
    listeners.get("click")({ isTrusted: true });
    await new Promise(setImmediate);
    assert.equal(canvasPage.fullscreenElement, canvasPage.documentElement);
    assert.equal(attempts, 2);
    assert.equal(sync_fullscreen(false, true), false);
    await new Promise(setImmediate);
    assert.equal(canvasPage.fullscreenElement, null);
});
