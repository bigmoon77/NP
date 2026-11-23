import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GPUComputationRenderer } from "three/addons/misc/GPUComputationRenderer.js";

const WIDTH_SEGMENTS = 316;
const HEIGHT_SEGMENTS = 316;
const TEXTURE_WIDTH = WIDTH_SEGMENTS + 1;
const TEXTURE_HEIGHT = HEIGHT_SEGMENTS + 1;
const VERTEX_COUNT = TEXTURE_WIDTH * TEXTURE_HEIGHT;

const canvas = document.querySelector("#view");
const errorPanel = document.querySelector("#error-panel");
const statusElement = document.querySelector("#backend-status");
const statusContainer = document.querySelector(".status");
const vertexCountElement = document.querySelector("#vertex-count");
const fpsElement = document.querySelector("#fps");
const stepTimeElement = document.querySelector("#step-time");
const stateElement = document.querySelector("#sim-state");
const hintElement = document.querySelector("#hint");

const settings = {
  elasticity: 34,
  shapeStrength: 5,
  damping: 2.8,
  grabRadius: 0.34,
  grabStrength: 85
};

let renderer;
let scene;
let camera;
let controls;
let mesh;
let gpuCompute;
let positionVariable;
let velocityVariable;
let initialTexture;
let isGrabbing = false;
let grabCenter = new THREE.Vector3();
let grabTarget = new THREE.Vector3();
let grabPlane = new THREE.Plane();
let grabOffset = new THREE.Vector3();
let frameCount = 0;
let lastFpsTime = performance.now();
let fps = 0;
let lastFrameTime = performance.now();
let stepCpuMs = 0;

const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const planeHit = new THREE.Vector3();
const inverseWorld = new THREE.Matrix4();
const cameraDirection = new THREE.Vector3();

function showError(message) {
  errorPanel.hidden = false;
  errorPanel.textContent = message;
  statusElement.textContent = "Initialization failed";
  statusContainer.classList.remove("ready");
}

function readShader(path) {
  return fetch(path).then((response) => {
    if (!response.ok) {
      throw new Error("Unable to load shader: " + path);
    }
    return response.text();
  });
}

function setStatus(message, ready) {
  statusElement.textContent = message;
  statusContainer.classList.toggle("ready", ready);
}

function updateSetting(id, key, outputId, format) {
  const input = document.querySelector("#" + id);
  const output = document.querySelector("#" + outputId);
  input.addEventListener("input", () => {
    settings[key] = Number(input.value);
    output.value = format(settings[key]);
    output.textContent = format(settings[key]);
    syncComputeUniforms();
  });
  output.value = format(settings[key]);
  output.textContent = format(settings[key]);
}

function syncComputeUniforms() {
  if (!velocityVariable) return;
  const uniforms = velocityVariable.material.uniforms;
  uniforms.elasticity.value = settings.elasticity;
  uniforms.shapeStrength.value = settings.shapeStrength;
  uniforms.damping.value = settings.damping;
  uniforms.grabRadius.value = settings.grabRadius;
  uniforms.grabStrength.value = settings.grabStrength;
}

function makeInitialTexture(geometry) {
  const texture = gpuCompute.createTexture();
  const positions = geometry.attributes.position.array;
  const data = texture.image.data;
  for (let index = 0; index < VERTEX_COUNT; index += 1) {
    const row = Math.floor(index / TEXTURE_WIDTH);
    const column = index - row * TEXTURE_WIDTH;
    const source = index * 3;
    const target = ((TEXTURE_HEIGHT - 1 - row) * TEXTURE_WIDTH + column) * 4;
    data[target] = positions[source];
    data[target + 1] = positions[source + 1];
    data[target + 2] = positions[source + 2];
    data[target + 3] = 1;
  }
  return texture;
}

function createSurfaceMaterial(positionTexture) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uPosition: { value: positionTexture },
      uTexel: { value: new THREE.Vector2(1 / TEXTURE_WIDTH, 1 / TEXTURE_HEIGHT) },
      uLightDirection: { value: new THREE.Vector3(-0.45, 0.8, 0.55).normalize() }
    },
    vertexShader: `
      uniform sampler2D uPosition;
      uniform vec2 uTexel;
      uniform vec3 uLightDirection;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying float vHeight;
      void main() {
        vec3 p = texture2D(uPosition, uv).xyz;
        vec3 pL = texture2D(uPosition, uv - vec2(uTexel.x, 0.0)).xyz;
        vec3 pR = texture2D(uPosition, uv + vec2(uTexel.x, 0.0)).xyz;
        vec3 pD = texture2D(uPosition, uv - vec2(0.0, uTexel.y)).xyz;
        vec3 pU = texture2D(uPosition, uv + vec2(0.0, uTexel.y)).xyz;
        vec3 tangentU = pR - pL;
        vec3 tangentV = pU - pD;
        vec3 objectNormal = normalize(cross(tangentU, tangentV));
        if (dot(objectNormal, p) < 0.0) objectNormal = -objectNormal;
        vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
        vNormal = normalize(normalMatrix * objectNormal);
        vViewPosition = -mvPosition.xyz;
        vHeight = p.y;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying float vHeight;
      void main() {
        vec3 normal = normalize(vNormal);
        vec3 lightDirection = normalize(vec3(-0.45, 0.8, 0.55));
        float diffuse = max(dot(normal, lightDirection), 0.0);
        vec3 viewDirection = normalize(vViewPosition);
        float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);
        vec3 baseColor = mix(vec3(0.10, 0.43, 0.61), vec3(0.30, 0.75, 0.88), clamp(vHeight * 0.35 + 0.55, 0.0, 1.0));
        vec3 color = baseColor * (0.22 + diffuse * 0.82);
        color += vec3(0.30, 0.72, 0.92) * fresnel * 0.45;
        gl_FragColor = vec4(color, 1.0);
      }
    `
  });
}

function initializeCompute(geometry, velocityShader, positionShader) {
  gpuCompute = new GPUComputationRenderer(TEXTURE_WIDTH, TEXTURE_HEIGHT, renderer);
  initialTexture = makeInitialTexture(geometry);

  const positionTexture = gpuCompute.createTexture();
  positionTexture.image.data.set(initialTexture.image.data);
  const velocityTexture = gpuCompute.createTexture();

  positionVariable = gpuCompute.addVariable("texturePosition", positionShader, positionTexture);
  velocityVariable = gpuCompute.addVariable("textureVelocity", velocityShader, velocityTexture);

  gpuCompute.setVariableDependencies(positionVariable, [positionVariable, velocityVariable]);
  gpuCompute.setVariableDependencies(velocityVariable, [positionVariable, velocityVariable]);

  positionVariable.wrapS = THREE.RepeatWrapping;
  positionVariable.wrapT = THREE.ClampToEdgeWrapping;
  velocityVariable.wrapS = THREE.RepeatWrapping;
  velocityVariable.wrapT = THREE.ClampToEdgeWrapping;

  positionVariable.material.uniforms.delta = { value: 1 / 60 };
  positionVariable.material.uniforms.textureInitial = { value: initialTexture };
  positionVariable.material.uniforms.resetState = { value: 0 };

  velocityVariable.material.uniforms.delta = { value: 1 / 60 };
  velocityVariable.material.uniforms.textureInitial = { value: initialTexture };
  velocityVariable.material.uniforms.resetState = { value: 0 };
  velocityVariable.material.uniforms.elasticity = { value: settings.elasticity };
  velocityVariable.material.uniforms.shapeStrength = { value: settings.shapeStrength };
  velocityVariable.material.uniforms.damping = { value: settings.damping };
  velocityVariable.material.uniforms.grabRadius = { value: settings.grabRadius };
  velocityVariable.material.uniforms.grabStrength = { value: settings.grabStrength };
  velocityVariable.material.uniforms.grabCenter = { value: new THREE.Vector3() };
  velocityVariable.material.uniforms.grabTarget = { value: new THREE.Vector3() };
  velocityVariable.material.uniforms.isGrabbing = { value: 0 };

  const computeError = gpuCompute.init();
  if (computeError !== null) {
    throw new Error(computeError);
  }
  return gpuCompute.getCurrentRenderTarget(positionVariable).texture;
}

function updatePointerCoordinates(event) {
  const bounds = canvas.getBoundingClientRect();
  pointerNdc.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointerNdc.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
}

function beginGrab(event) {
  if (!mesh || event.button !== 0) return;
  updatePointerCoordinates(event);
  raycaster.setFromCamera(pointerNdc, camera);
  const intersections = raycaster.intersectObject(mesh, false);
  if (intersections.length === 0) return;

  const hit = intersections[0].point;
  grabCenter.copy(hit);
  grabTarget.copy(hit);
  camera.getWorldDirection(cameraDirection);
  grabPlane.setFromNormalAndCoplanarPoint(cameraDirection, hit);
  grabOffset.set(0, 0, 0);
  isGrabbing = true;
  controls.enabled = false;
  canvas.setPointerCapture(event.pointerId);
  hintElement.textContent = "Pulling surface · release to let the sphere recover";
  stateElement.textContent = "GRABBED";
  syncGrabUniforms();
}

function moveGrab(event) {
  if (!isGrabbing) return;
  updatePointerCoordinates(event);
  raycaster.setFromCamera(pointerNdc, camera);
  if (raycaster.ray.intersectPlane(grabPlane, planeHit)) {
    grabTarget.copy(planeHit).add(grabOffset);
    syncGrabUniforms();
  }
}

function endGrab() {
  if (!isGrabbing) return;
  isGrabbing = false;
  controls.enabled = true;
  if (velocityVariable) {
    velocityVariable.material.uniforms.isGrabbing.value = 0;
  }
  hintElement.textContent = "Drag the sphere to pull its surface · Orbit when not dragging";
  stateElement.textContent = "RUNNING";
}

function syncGrabUniforms() {
  if (!velocityVariable) return;
  const uniforms = velocityVariable.material.uniforms;
  uniforms.grabCenter.value.copy(grabCenter);
  uniforms.grabTarget.value.copy(grabTarget);
  uniforms.isGrabbing.value = isGrabbing ? 1 : 0;
}

function resetSimulation() {
  if (!gpuCompute || !positionVariable || !velocityVariable) return;
  positionVariable.material.uniforms.resetState.value = 1;
  velocityVariable.material.uniforms.resetState.value = 1;
  gpuCompute.compute();
  positionVariable.material.uniforms.resetState.value = 0;
  velocityVariable.material.uniforms.resetState.value = 0;
  mesh.material.uniforms.uPosition.value = gpuCompute.getCurrentRenderTarget(positionVariable).texture;
  endGrab();
  stateElement.textContent = "RESET";
}

function resize() {
  if (!renderer || !camera) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate(now) {
  requestAnimationFrame(animate);
  const elapsed = Math.min((now - lastFrameTime) / 1000, 1 / 30);
  lastFrameTime = now;
  const delta = Math.max(elapsed, 1 / 240);

  if (gpuCompute) {
    positionVariable.material.uniforms.delta.value = delta;
    velocityVariable.material.uniforms.delta.value = delta;
    const start = performance.now();
    gpuCompute.compute();
    stepCpuMs = performance.now() - start;
    mesh.material.uniforms.uPosition.value = gpuCompute.getCurrentRenderTarget(positionVariable).texture;
  }

  controls.update();
  renderer.render(scene, camera);

  frameCount += 1;
  if (now - lastFpsTime >= 500) {
    fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
    frameCount = 0;
    lastFpsTime = now;
    fpsElement.textContent = String(fps);
    stepTimeElement.textContent = stepCpuMs.toFixed(2) + " ms";
    if (!isGrabbing && stateElement.textContent !== "RESET") {
      stateElement.textContent = "RUNNING";
    }
  }
}

async function initialize() {
  try {
    if (!window.WebGL2RenderingContext) {
      throw new Error("WebGL2 is required for this demo.");
    }

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const gl = renderer.getContext();
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("This browser/GPU does not expose EXT_color_buffer_float, which is required for floating-point simulation textures.");
    }

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(38, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
    camera.position.set(3.4, 2.2, 4.4);
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 2.5;
    controls.maxDistance = 9;
    controls.target.set(0, 0, 0);

    const geometry = new THREE.SphereGeometry(1, WIDTH_SEGMENTS, HEIGHT_SEGMENTS);
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 3);
    const [velocityShader, positionShader] = await Promise.all([
      readShader("./src/shaders/velocity.glsl"),
      readShader("./src/shaders/position.glsl")
    ]);

    const startTexture = initializeCompute(geometry, velocityShader, positionShader);
    const material = createSurfaceMaterial(startTexture);
    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const grid = new THREE.GridHelper(8, 24, 0x263d50, 0x172838);
    grid.position.y = -1.25;
    scene.add(grid);

    const ambient = new THREE.HemisphereLight(0xb7dcf4, 0x182331, 1.4);
    scene.add(ambient);

    vertexCountElement.textContent = VERTEX_COUNT.toLocaleString("en-US");
    setStatus("WebGL2 GPU solver ready", true);
    stateElement.textContent = "RUNNING";

    updateSetting("elasticity", "elasticity", "elasticity-value", (value) => value.toFixed(0));
    updateSetting("shape", "shapeStrength", "shape-value", (value) => value.toFixed(1));
    updateSetting("damping", "damping", "damping-value", (value) => value.toFixed(1));
    updateSetting("grab-radius", "grabRadius", "grab-radius-value", (value) => value.toFixed(2));
    updateSetting("grab-strength", "grabStrength", "grab-strength-value", (value) => value.toFixed(0));

    document.querySelector("#reset").addEventListener("click", resetSimulation);
    canvas.addEventListener("pointerdown", beginGrab);
    canvas.addEventListener("pointermove", moveGrab);
    canvas.addEventListener("pointerup", endGrab);
    canvas.addEventListener("pointercancel", endGrab);
    window.addEventListener("resize", resize);

    resize();
    requestAnimationFrame((time) => {
      lastFrameTime = time;
      animate(time);
    });
  } catch (error) {
    console.error(error);
    showError(error instanceof Error ? error.message : String(error));
  }
}

initialize();