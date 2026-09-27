
let channel;
const pending = [];
export function inferenceYield() {
    if (globalThis.scheduler?.yield) return globalThis.scheduler.yield();
    return new Promise(resolve => {
        if (!channel) {
            channel = new MessageChannel();
            channel.port1.onmessage = () => pending.shift()();
        }
        pending.push(resolve);
        channel.port2.postMessage(0);
    });
}
export function inferenceYieldCallback() { return inferenceYield; }
