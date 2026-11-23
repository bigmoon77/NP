const statusElement = document.querySelector("#capture-status");
const countElement = document.querySelector("#capture-count");
const recordButton = document.querySelector("#record");
const downloadButton = document.querySelector("#download-dataset");
const intervalSelect = document.querySelector("#capture-interval");
const durationSelect = document.querySelector("#capture-duration");

let recording = false;
let frameCounter = 0;
let targetDuration = 0;
let capturePeriod = 1 / 10;
let capturedFrames = [];
let captureInterval = 6;
let captureDt = 0;
let simulationFrame = 0;
let captureElapsed = 0;
let capturedElapsed = 0;

function buildEdges(geometry) {
  const index = geometry.index.array;
  const unique = new Set();
  const edges = [];
  for (let i = 0; i < index.length; i += 3) {
    const a = index[i];
    const b = index[i + 1];
    const c = index[i + 2];
    for (const pair of [[a, b], [b, c], [c, a]]) {
      const lo = Math.min(pair[0], pair[1]);
      const hi = Math.max(pair[0], pair[1]);
      const key = lo * geometry.attributes.position.count + hi;
      if (!unique.has(key)) {
        unique.add(key);
        edges.push(lo, hi);
      }
    }
  }
  return new Uint32Array(edges);
}

function readTarget(target, vertexCount) {
  const width = target.width;
  const height = target.height;
  const raw = new Float32Array(width * height * 4);
  window.npSimulation.renderer.readRenderTargetPixels(target, 0, 0, width, height, raw);
  const result = new Float32Array(vertexCount * 3);
  for (let row = 0; row < height; row += 1) {
    const sourceRow = height - 1 - row;
    for (let column = 0; column < width; column += 1) {
      const source = (sourceRow * width + column) * 4;
      const destination = (row * width + column) * 3;
      result[destination] = raw[source];
      result[destination + 1] = raw[source + 1];
      result[destination + 2] = raw[source + 2];
    }
  }
  return result;
}

function captureFrame() {
  const simulation = window.npSimulation;
  const vertexCount = simulation.geometry.attributes.position.count;
  const positions = readTarget(simulation.getPositionTarget(), vertexCount);
  const velocities = readTarget(simulation.getVelocityTarget(), vertexCount);
  capturedFrames.push({ positions, velocities, elapsed: captureElapsed });
  capturedElapsed += captureElapsed;
  captureElapsed = 0;
  captureDt = capturedFrames.length > 0 ? capturedElapsed / capturedFrames.length : captureDt;
  countElement.textContent = capturedFrames.length + " frames · " + capturedElapsed.toFixed(1) + " s";
  if (capturedElapsed >= targetDuration) {
    recording = false;
    recordButton.disabled = false;
    downloadButton.disabled = false;
    statusElement.textContent = "Capture complete";
  }
}

function makeDataset() {
  const simulation = window.npSimulation;
  const geometry = simulation.geometry;
  const vertexCount = geometry.attributes.position.count;
  const restPositions = new Float32Array(vertexCount * 3);
  restPositions.set(geometry.attributes.position.array);
  const edges = buildEdges(geometry);
  const header = {
    format: "NPDS",
    version: 1,
    coordinate_system: "threejs_y_up",
    vertex_count: vertexCount,
    edge_count: edges.length / 2,
    frame_count: capturedFrames.length,
    capture_dt: captureDt,
    arrays: ["rest_positions:f32[n,3]", "edges:u32[m,2]", "frames:{positions:f32[n,3],velocities:f32[n,3]}"]
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const restBytes = restPositions.byteLength;
  const edgeBytes = edges.byteLength;
  const frameBytes = vertexCount * 3 * 4 * 2;
  const totalBytes = 12 + headerBytes.byteLength + restBytes + edgeBytes + frameBytes * capturedFrames.length;
  const output = new Uint8Array(totalBytes);
  output.set(new TextEncoder().encode("NPDS0001"), 0);
  const view = new DataView(output.buffer);
  view.setUint32(8, headerBytes.byteLength, true);
  let offset = 12;
  output.set(headerBytes, offset);
  offset += headerBytes.byteLength;
  output.set(new Uint8Array(restPositions.buffer), offset);
  offset += restBytes;
  output.set(new Uint8Array(edges.buffer), offset);
  offset += edgeBytes;
  for (const frame of capturedFrames) {
    output.set(new Uint8Array(frame.positions.buffer), offset);
    offset += frame.positions.byteLength;
    output.set(new Uint8Array(frame.velocities.buffer), offset);
    offset += frame.velocities.byteLength;
  }
  return new Blob([output], { type: "application/octet-stream" });
}

function downloadDataset() {
  const blob = makeDataset();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "np-trajectory-" + new Date().toISOString().replaceAll(":", "-") + ".npds";
  anchor.click();
  URL.revokeObjectURL(url);
}

function startRecording() {
  if (!window.npSimulation || recording) return;
  capturedFrames = [];
  captureInterval = Number(intervalSelect.value);
  const duration = Number(durationSelect.value);
  targetDuration = duration;
  capturePeriod = captureInterval / 60;
  captureDt = capturePeriod;
  frameCounter = 0;
  captureElapsed = 0;
  capturedElapsed = 0;
  recording = true;
  recordButton.disabled = true;
  downloadButton.disabled = true;
  statusElement.textContent = "Recording… avoid resetting during capture";
  countElement.textContent = "0 frames";
}

function onAnimationFrame() {
  if (recording) {
    captureElapsed += window.npSimulation ? window.npSimulation.getDelta() : 1 / 60;
    frameCounter += 1;
    if (captureElapsed >= capturePeriod) {
      captureFrame();
    }
  }
  requestAnimationFrame(onAnimationFrame);
}

recordButton.addEventListener("click", startRecording);
downloadButton.addEventListener("click", downloadDataset);
window.addEventListener("load", () => {
  if (!window.npSimulation) {
    statusElement.textContent = "Waiting for GPU solver…";
  }
});
requestAnimationFrame(onAnimationFrame);
