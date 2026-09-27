const MAX_IMAGE_BYTES = 64 * 1024 * 1024;
const MAX_TEXTS = 256;
const MAX_TEXT_BYTES = 64 * 1024;
const MAX_TEXT_BATCH_BYTES = 1024 * 1024;
const SHA256_PATTERN = /^[0-9a-f]{64}$/u;
const TOKENIZER_SHA256 = "8a99220b2556b72893f261f733d71d403e0c4d047636b5a53581893df3771069";

const MODELS = Object.freeze({
  base: {
    label: "Base · patch16 · 224 px",
    variant: "base-patch16-224",
    imageSize: 224,
    projectionDim: 768,
    parts: 14,
    bundleBytes: 789_082_777,
    deviceBytes: 1_500_751_880,
    loadedWeightSha256: "6267af27dfa05ce21b73649f4d4f6cd4c6738acd4bcde5a0716051403c6f130e",
    tokenizerSha256: TOKENIZER_SHA256,
    upstreamModelId: "google/siglip2-base-patch16-224",
    upstreamRevision: "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
  },
  large: {
    label: "Large · patch16 · 256 px",
    variant: "large-patch16-256",
    imageSize: 256,
    projectionDim: 1024,
    parts: 35,
    bundleBytes: 1_801_817_417,
    deviceBytes: 3_526_107_144,
    loadedWeightSha256: "eedde6eafa2c9cb987183977cb3a8ea55dc75846f00fedc2ff523a1cc2548fec",
    tokenizerSha256: TOKENIZER_SHA256,
    upstreamModelId: "google/siglip2-large-patch16-256",
    upstreamRevision: "787800c8990e6f058423089178e718139608408c",
  },
  so400m: {
    label: "So400m · patch14 · 224 px",
    variant: "so400m-patch14-224",
    imageSize: 224,
    projectionDim: 1152,
    parts: 42,
    bundleBytes: 2_309_706_787,
    deviceBytes: 4_541_854_408,
    loadedWeightSha256: "70d77cf0bf06f29465bb98930f7a70c16edafec93835f29ef8e83085420c3e58",
    tokenizerSha256: TOKENIZER_SHA256,
    upstreamModelId: "google/siglip2-so400m-patch14-224",
    upstreamRevision: "78e403963a4f6a3640d07803284752326fdf4edf",
  },
});

const elements = Object.freeze({
  errorMessage: requiredElement("error-message"),
  errorPanel: requiredElement("error-panel"),
  imageFile: requiredElement("image-file"),
  imageMeta: requiredElement("image-meta"),
  imagePreview: requiredElement("image-preview"),
  inferenceFields: requiredElement("inference-fields"),
  inferenceStatus: requiredElement("inference-status"),
  inferenceTime: requiredElement("inference-time"),
  loadDetail: requiredElement("load-detail"),
  loadModel: requiredElement("load-model"),
  loadPanel: requiredElement("load-panel"),
  loadProgress: requiredElement("load-progress"),
  loadTitle: requiredElement("load-title"),
  loadedModel: requiredElement("loaded-model"),
  modelFacts: requiredElement("model-facts"),
  modelSize: requiredElement("model-size"),
  modelStatus: requiredElement("model-status"),
  previewFrame: requiredElement("preview-frame"),
  resultsBody: requiredElement("results-body"),
  resultsCard: requiredElement("results-card"),
  runInference: requiredElement("run-inference"),
  embeddingDetails: requiredElement("embedding-details"),
  textCandidates: requiredElement("text-candidates"),
  webgpuBadge: requiredElement("webgpu-badge"),
});

const state = {
  imageBytes: null,
  imageObjectUrl: null,
  loadedSize: null,
  model: null,
  modelIdentity: null,
  wasmClass: null,
};

function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required element #${id}`);
  return element;
}

function formatBytes(value) {
  const bytes = typeof value === "bigint" ? Number(value) : value;
  if (!Number.isFinite(bytes) || bytes < 0) return "unknown size";
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = bytes;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function formatSeconds(milliseconds) {
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

function selectedModel() {
  const model = MODELS[elements.modelSize.value];
  if (!model) throw new Error(`Unsupported model selection: ${elements.modelSize.value}`);
  return model;
}

function updateModelFacts() {
  const model = selectedModel();
  elements.modelFacts.textContent =
    `${model.parts} verified shards · ${formatBytes(model.bundleBytes)} CDN bundle · ` +
    `about ${formatBytes(model.deviceBytes)} of F32 weights before activations`;
}

function showError(error, context) {
  const detail = error instanceof Error ? error.message : String(error);
  elements.errorMessage.textContent = context ? `${context}: ${detail}` : detail;
  elements.errorPanel.hidden = false;
}

function clearError() {
  elements.errorPanel.hidden = true;
  elements.errorMessage.textContent = "";
}

function disposeModel() {
  if (state.model) {
    state.model.free?.();
  }
  state.model = null;
  state.modelIdentity = null;
  state.loadedSize = null;
  elements.inferenceFields.disabled = true;
  elements.loadedModel.textContent = "No model loaded";
  elements.loadedModel.className = "badge";
  elements.modelStatus.textContent = "";
  elements.resultsCard.hidden = true;
}

function setLoading(active) {
  elements.loadModel.disabled = active || !state.wasmClass;
  elements.modelSize.disabled = active;
  elements.loadPanel.hidden = !active;
  if (active) {
    elements.loadProgress.removeAttribute("value");
  }
}

function readAndValidateModelIdentity(model, expected) {
  const identity = Object.freeze({
    loaded_weight_sha256: model.loadedWeightSha256,
    tokenizer_sha256: model.tokenizerSha256,
    upstream_model_id: model.upstreamModelId,
    upstream_revision: model.upstreamRevision,
  });
  for (const [field, value] of [
    ["loaded_weight_sha256", identity.loaded_weight_sha256],
    ["tokenizer_sha256", identity.tokenizer_sha256],
  ]) {
    if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
      throw new Error(`Model returned an invalid ${field}: ${String(value)}.`);
    }
  }
  for (const [field, actual, wanted] of [
    ["loaded weight SHA-256", identity.loaded_weight_sha256, expected.loadedWeightSha256],
    ["tokenizer SHA-256", identity.tokenizer_sha256, expected.tokenizerSha256],
    ["upstream model ID", identity.upstream_model_id, expected.upstreamModelId],
    ["upstream revision", identity.upstream_revision, expected.upstreamRevision],
  ]) {
    if (actual !== wanted) throw new Error(`${field} is ${String(actual)}; expected ${wanted}.`);
  }
  return identity;
}

async function loadSelectedModel() {
  clearError();
  const size = elements.modelSize.value;
  const details = selectedModel();
  disposeModel();
  setLoading(true);
  elements.loadTitle.textContent = `Loading ${details.label}`;

  const startedAt = performance.now();
  const updateElapsed = () => {
    elements.loadDetail.textContent =
      `Streaming and verifying up to ${formatBytes(details.bundleBytes)} · ` +
      `${formatSeconds(performance.now() - startedAt)} elapsed`;
  };
  updateElapsed();
  const timer = window.setInterval(updateElapsed, 250);

  try {
    const model = await state.wasmClass.createFromModelSize(size);
    state.model = model;
    const actualModelSize = model.modelSize;
    if (actualModelSize !== details.variant) {
      throw new Error(`CDN returned ${actualModelSize}; expected ${details.variant}.`);
    }
    const actualEmbeddingSize = model.embeddingSize;
    if (actualEmbeddingSize !== details.projectionDim) {
      throw new Error(
        `Model embedding width is ${actualEmbeddingSize}; expected ${details.projectionDim}.`,
      );
    }
    const actualImageSize = model.imageSize;
    if (actualImageSize !== details.imageSize) {
      throw new Error(`Model image size is ${actualImageSize}; expected ${details.imageSize}.`);
    }
    state.modelIdentity = readAndValidateModelIdentity(model, details);
    state.loadedSize = size;
    const elapsed = performance.now() - startedAt;
    elements.loadProgress.max = 1;
    elements.loadProgress.value = 1;
    elements.modelStatus.textContent =
      `Ready in ${formatSeconds(elapsed)}. Applied ${model.loadedPartCount} shards ` +
      `(${formatBytes(model.loadedBytes)} of model data). ` +
      `Weights ${state.modelIdentity.loaded_weight_sha256.slice(0, 12)}… · ` +
      `upstream ${state.modelIdentity.upstream_revision.slice(0, 12)}…`;
    elements.loadedModel.textContent = details.label;
    elements.loadedModel.className = "badge badge-ready";
    elements.inferenceFields.disabled = false;
  } catch (error) {
    disposeModel();
    showError(error, `Could not load ${details.label}`);
  } finally {
    window.clearInterval(timer);
    setLoading(false);
  }
}

async function readSelectedImage() {
  clearError();
  const file = elements.imageFile.files?.[0];
  state.imageBytes = null;
  elements.previewFrame.hidden = true;

  if (state.imageObjectUrl) {
    URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = null;
  }
  if (!file) return;
  if (file.size === 0) {
    showError("The selected image is empty.");
    elements.imageFile.value = "";
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showError(`The selected image is ${formatBytes(file.size)}; the limit is 64 MiB.`);
    elements.imageFile.value = "";
    return;
  }

  try {
    state.imageBytes = new Uint8Array(await file.arrayBuffer());
    state.imageObjectUrl = URL.createObjectURL(file);
    elements.imagePreview.src = state.imageObjectUrl;
    elements.imageMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;
    elements.previewFrame.hidden = false;
  } catch (error) {
    showError(error, "Could not read the selected image");
  }
}

function candidateTexts() {
  const texts = elements.textCandidates.value
    .split(/\r?\n/u)
    .map((text) => text.trim())
    .filter(Boolean);
  if (texts.length === 0) throw new Error("Enter at least one text candidate.");
  if (texts.length > MAX_TEXTS) {
    throw new Error(`Enter at most ${MAX_TEXTS} text candidates.`);
  }

  const encoder = new TextEncoder();
  let totalBytes = 0;
  for (const [index, text] of texts.entries()) {
    const bytes = encoder.encode(text).byteLength;
    if (bytes > MAX_TEXT_BYTES) {
      throw new Error(`Text candidate ${index + 1} exceeds the 64 KiB limit.`);
    }
    totalBytes += bytes;
  }
  if (totalBytes > MAX_TEXT_BATCH_BYTES) {
    throw new Error("The text batch exceeds the 1 MiB limit.");
  }
  return texts;
}

function vectorNorm(values, offset, columns) {
  let squared = 0;
  for (let index = 0; index < columns; index += 1) {
    const value = values[offset + index];
    squared += value * value;
  }
  return Math.sqrt(squared);
}

function summarizeResponse(response) {
  const imageColumns = response.image_embedding_shape[1];
  const textColumns = response.text_embedding_shape[1];
  const textRows = response.text_embedding_shape[0];
  return {
    schema_version: response.schema_version,
    method: response.method,
    model_identity: state.modelIdentity,
    image_embedding_shape: response.image_embedding_shape,
    text_embedding_shape: response.text_embedding_shape,
    logits_shape: response.logits_shape,
    image_raw_l2_norm: vectorNorm(response.raw_image_embedding, 0, imageColumns),
    image_normalized_l2_norm: vectorNorm(response.normalized_image_embedding, 0, imageColumns),
    text_raw_l2_norms: Array.from({ length: textRows }, (_, row) =>
      vectorNorm(response.raw_text_embedding, row * textColumns, textColumns),
    ),
    text_normalized_l2_norms: Array.from({ length: textRows }, (_, row) =>
      vectorNorm(response.normalized_text_embedding, row * textColumns, textColumns),
    ),
    normalized_image_preview: response.normalized_image_embedding.slice(0, 8),
    normalized_first_text_preview: response.normalized_text_embedding.slice(0, 8),
  };
}

function renderResults(response, texts, elapsed) {
  if (
    !Array.isArray(response.logits_per_image) ||
    !Array.isArray(response.probabilities_per_image) ||
    response.logits_per_image.length !== texts.length ||
    response.probabilities_per_image.length !== texts.length
  ) {
    throw new Error("The model returned an unexpected score shape.");
  }

  const ranked = texts
    .map((text, index) => ({
      text,
      logit: response.logits_per_image[index],
      probability: response.probabilities_per_image[index],
    }))
    .sort((left, right) => right.probability - left.probability);

  elements.resultsBody.replaceChildren();
  for (const [index, item] of ranked.entries()) {
    const row = document.createElement("tr");
    const values = [
      String(index + 1),
      item.text,
      Number(item.logit).toFixed(5),
      `${(Number(item.probability) * 100).toFixed(4)}%`,
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.resultsBody.append(row);
  }

  elements.embeddingDetails.textContent = JSON.stringify(summarizeResponse(response), null, 2);
  elements.inferenceTime.textContent = formatSeconds(elapsed);
  elements.resultsCard.hidden = false;
  elements.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runInference() {
  clearError();
  if (!state.model || !state.loadedSize) {
    showError("Load a model before running inference.");
    return;
  }
  if (!state.imageBytes) {
    showError("Choose an image before running inference.");
    return;
  }

  let texts;
  try {
    texts = candidateTexts();
  } catch (error) {
    showError(error);
    return;
  }

  elements.runInference.disabled = true;
  elements.modelSize.disabled = true;
  elements.loadModel.disabled = true;
  elements.inferenceStatus.textContent = "Running both SigLIP2 towers on WebGPU…";
  elements.resultsCard.hidden = true;
  const startedAt = performance.now();

  try {
    const json = await state.model.scoreImageTextsJson(state.imageBytes, texts);
    const response = JSON.parse(json);
    renderResults(response, texts, performance.now() - startedAt);
    elements.inferenceStatus.textContent = "Inference complete.";
  } catch (error) {
    elements.inferenceStatus.textContent = "";
    showError(error, "Inference failed");
  } finally {
    elements.runInference.disabled = false;
    elements.modelSize.disabled = false;
    elements.loadModel.disabled = !state.wasmClass;
  }
}

async function initialize() {
  updateModelFacts();
  if (!window.isSecureContext) {
    throw new Error("WebGPU requires HTTPS or a localhost development server.");
  }
  if (!("gpu" in navigator)) {
    throw new Error("This browser does not expose WebGPU.");
  }

  const moduleUrl = new URL("./pkg/burn_siglip2.js", import.meta.url).href;
  const wasm = await import(moduleUrl);
  await wasm.default();
  state.wasmClass = wasm.WasmSiglip2;
  if (!state.wasmClass) throw new Error("The Wasm package does not export WasmSiglip2.");

  elements.webgpuBadge.textContent = "WebGPU available";
  elements.webgpuBadge.className = "badge badge-ready";
  elements.loadModel.disabled = false;
  elements.modelStatus.textContent = "Choose a model size, then load its verified CDN bundle.";
}

elements.modelSize.addEventListener("change", () => {
  clearError();
  if (state.loadedSize && state.loadedSize !== elements.modelSize.value) disposeModel();
  updateModelFacts();
});
elements.loadModel.addEventListener("click", loadSelectedModel);
elements.imageFile.addEventListener("change", readSelectedImage);
elements.runInference.addEventListener("click", runInference);
window.addEventListener("pagehide", () => {
  disposeModel();
  if (state.imageObjectUrl) URL.revokeObjectURL(state.imageObjectUrl);
});

initialize().catch((error) => {
  elements.webgpuBadge.textContent = "Unavailable";
  elements.webgpuBadge.className = "badge badge-error";
  showError(error, "Could not initialize the browser runtime");
});
