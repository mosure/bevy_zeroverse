// The script is bundled alongside its matching wasm-bindgen glue. The compiled
// WebAssembly.Module is structured-cloned once per worker, without a second fetch,
// SharedArrayBuffer, cross-origin isolation, canvas or graphics context.
let worker, active, idle, unavailable = false;
const queue = [];

function stop() {
    clearTimeout(idle);
    worker?.terminate();
    worker = undefined;
}
function complete(ticket, success, value) {
    if (active !== ticket) return;
    clearTimeout(ticket.timeout);
    active = undefined;
    ticket.done = true;
    if (success) ticket.resolve(value);
    else ticket.reject(value);
    dispatch();
}
function dispatch() {
    if (active) return;
    clearTimeout(idle);
    if (!queue.length) {
        // Release the separate wasm heap when automation stops.
        idle = setTimeout(stop, 15000);
        return;
    }
    const ticket = active = queue.shift();
    try {
        if (unavailable) throw Error('Preparation worker unavailable');
        if (!worker) {
            const url = new URL(import.meta.url);
            url.searchParams.set('zeroverse-preparation', '1');
            worker = new Worker(url, {type: 'module', name: 'Zeroverse scene preparation'});
            worker.postMessage({glue: globalThis.zeroverseViewerModule, module: ticket.module});
        }
        worker.onmessage = ({data}) => {
            if (active !== ticket) return;
            if (data.error) {
                // A seed rejection is a normal binary result. This branch means
                // the worker itself failed; avoid restarting it for every mesh.
                unavailable = true;
                stop();
                complete(ticket, false, Error(data.error));
            } else complete(ticket, true, data.result);
        };
        worker.onmessageerror = worker.onerror = event => {
            if (active !== ticket) return;
            event.preventDefault();
            unavailable = true;
            stop();
            complete(ticket, false, Error(event.message || 'Worker initialization failed'));
        };
        ticket.timeout = setTimeout(() => {
            if (active !== ticket) return;
            stop();
            // A slow seed does not invalidate module workers for later rooms.
            complete(ticket, false, new DOMException('Preparation worker timed out', 'TimeoutError'));
        }, 60000);
        const buffers = (Array.isArray(ticket.data) ? ticket.data : [ticket.data]).map(a => a.buffer);
        worker.postMessage({operation: ticket.operation, data: ticket.data}, buffers);
    } catch (error) {
        unavailable = true;
        stop();
        complete(ticket, false, error);
    }
}

export function studioWorkerEnabled() {
    return !unavailable && !!globalThis.zeroverseViewerModule && typeof Worker === 'function';
}
export function startStudioJob(module, operation, data) {
    if (!studioWorkerEnabled()) return null;
    const ticket = {module, operation, data};
    ticket.promise = new Promise((resolve, reject) => Object.assign(ticket, {resolve, reject}));
    queue.push(ticket);
    dispatch();
    return ticket;
}
export function studioJobResult(ticket) { return ticket.promise; }
export function cancelStudioJob(ticket) {
    if (ticket.done) return;
    if (active === ticket) {
        stop();
        complete(ticket, false, Error('Preparation cancelled'));
    } else {
        const index = queue.indexOf(ticket);
        if (index !== -1) queue.splice(index, 1);
        ticket.done = true;
        ticket.reject(Error('Preparation cancelled'));
    }
}
// Only the entry script installs the dispatcher. Importing the viewer glue also
// imports the unqualified copy of this module into the same worker.
if (typeof document === 'undefined' && new URL(import.meta.url).searchParams.has('zeroverse-preparation')) {
    let ready;
    globalThis.onmessage = async ({data}) => {
        if (data.module) {
            ready = import(data.glue).then(api => {
                api.initSync({module: data.module});
                return api;
            });
            // Defer failure delivery until the corresponding job is received.
            ready.catch(() => {});
            return;
        }
        try {
            const api = await ready;
            let result;
            switch (data.operation) {
                case 'plan': result = api.zeroversePlanRoom(data.data); break;
                case 'materialMaps': result = api.zeroverseMaterialMaps(data.data); break;
                case 'probeTree': result = api.zeroverseProbeTree(data.data); break;
                case 'tangents': result = api.zeroverseMeshTangents(...data.data); break;
                default: throw Error('Unknown Studio worker operation');
            }
            const transfers = Array.isArray(result) ? result.map(plane => plane.buffer) : [result.buffer];
            globalThis.postMessage({result}, transfers);
        } catch (e) {
            globalThis.postMessage({error: String(e)});
        }
    };
}
