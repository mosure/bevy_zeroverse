// The Fullscreen API consumes transient user activation. Request synchronously
// in the DOM event, including the first gesture after opening a fullscreen URL.
let installed = false, desired = false, shortcuts = true, pending = false;

function apply() {
    if (pending || desired === Boolean(document.fullscreenElement)) return;
    if (desired && !navigator.userActivation?.isActive) return;
    pending = true;
    const transition = desired
        ? document.documentElement.requestFullscreen()
        : document.exitFullscreen();
    transition.catch(() => { desired = Boolean(document.fullscreenElement); })
        .finally(() => { pending = false; apply(); });
}

export function sync_fullscreen(request, allowShortcuts) {
    if (!document.fullscreenEnabled) return false;
    if (!installed) {
        installed = true;
        document.addEventListener("fullscreenchange", () => {
            // Escape or a browser control can exit independently of the Studio.
            if (!pending) desired = Boolean(document.fullscreenElement);
        });
        document.addEventListener("keydown", event => {
            if (!event.isTrusted) return;
            if (event.key === "Escape" && document.fullscreenElement) {
                // Let the browser exit fullscreen without also closing the Bevy app.
                event.stopImmediatePropagation();
                desired = false;
                apply();
                return;
            }
            if (shortcuts && event.code === "F11") {
                event.preventDefault();
                if (event.repeat) return;
                desired = !Boolean(document.fullscreenElement);
            }
            apply();
        }, true);
        document.addEventListener("click", event => { if (event.isTrusted) apply(); }, true);
    }
    shortcuts = allowShortcuts;
    if (request != null) {
        desired = request;
        apply();
    }
    return desired;
}
