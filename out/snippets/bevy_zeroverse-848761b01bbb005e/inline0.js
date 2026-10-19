
export async function zeroversePreparationYield() {
    // A prioritized continuation avoids nested-timer clamping. Keep a real
    // event-loop yield on browsers without the scheduling API as well.
    if (typeof globalThis.scheduler?.yield === 'function') {
        try {
            await globalThis.scheduler.yield();
            return;
        } catch (_) {
            // An enclosing aborted postTask must not strand preparation.
        }
    }
    await new Promise(resolve => setTimeout(resolve, 0));
}
