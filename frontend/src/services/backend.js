const API_BASE_URL = (import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

async function readJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Inference service request failed (${response.status}).`);
  }
  return payload;
}

export async function getHealth() {
  return readJsonResponse(await fetch(apiUrl("/api/health")));
}

export async function listScenes(source) {
  const payload = await readJsonResponse(
    await fetch(apiUrl(`/api/scenes?source=${encodeURIComponent(source)}`)),
  );
  return payload.scenes;
}

export async function reconstructScene({
  source,
  sceneKey,
  sensorIsLowResolution,
  settings,
  rgbFile,
  depthFile,
  referenceFile,
}) {
  const body = new FormData();
  body.append("source", source);
  body.append("settings_json", JSON.stringify(settings));
  body.append("sensor_is_low_resolution", String(sensorIsLowResolution));
  if (source !== "upload") body.append("scene_key", sceneKey);
  if (rgbFile) body.append("rgb_file", rgbFile);
  if (depthFile) body.append("depth_file", depthFile);
  if (referenceFile) body.append("reference_file", referenceFile);
  return readJsonResponse(
    await fetch(apiUrl("/api/reconstruct"), { method: "POST", body }),
  );
}

export async function getSurface(resultId, method) {
  return readJsonResponse(
    await fetch(
      apiUrl(
        `/api/results/${encodeURIComponent(resultId)}/surface/${encodeURIComponent(method)}`,
      ),
    ),
  );
}

export async function downloadDepth(resultId, method) {
  const response = await fetch(
    apiUrl(
      `/api/results/${encodeURIComponent(resultId)}/depth/${encodeURIComponent(method)}`,
    ),
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(
      payload.detail || `Depth export failed (${response.status}).`,
    );
  }
  const blobUrl = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = blobUrl;
  anchor.download = `${method.toLowerCase().replaceAll(" ", "_")}_depth.npy`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}
