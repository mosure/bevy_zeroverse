/* Camera-count figures share one measured cohort; these controls only select a
 * baseline panel and never alter the room/annotation explorer's state. */
(() => {
  const image = document.getElementById('camera-study-heatmap');
  const link = document.getElementById('camera-study-heatmap-link');
  if (!image || !link) return;
  const buttons = [...document.querySelectorAll('[data-camera-study-baseline]')];
  for (const button of buttons) button.addEventListener('click', () => {
    const baseline = button.dataset.cameraStudyBaseline;
    if (!['0', '0.5', '1'].includes(baseline)) return;
    const source = `static/media/cameras/camera-count-heatmap-b${baseline}.svg`;
    image.src = source;
    link.href = source;
    image.alt = `Primary-room camera-path heatmaps for all camera counts from one through twelve at baseline ${baseline}`;
    for (const item of buttons) item.setAttribute('aria-pressed', String(item === button));
  });
})();
