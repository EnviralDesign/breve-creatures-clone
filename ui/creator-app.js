import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const AXIS_TILT_GAIN = 1.9;
const EDGE_OUTWARD_GROWTH_MIN_DOT = 0.22;
const MAX_GRAPH_PARTS = 72;
const MAX_GRAPH_EDGES_PER_NODE = 4;

const backend = `${window.location.protocol}//${window.location.host}`;
const urls = {
  currentGenome: `${backend}/api/evolution/genome/current`,
  bestGenome: `${backend}/api/evolution/genome/best`,
  control: `${backend}/api/evolution/control`,
  evolutionState: `${backend}/api/evolution/state`,
  creatureList: `${backend}/api/creatures`,
  creatureSave: `${backend}/api/creatures/save`,
  creatureGet: (id) => `${backend}/api/creatures/${encodeURIComponent(id)}`,
};

const state = {
  currentGenome: null,
  editorMode: "visual",
};

const statusEl = document.getElementById("status");
const genomeEditor = document.getElementById("genomeEditor");
const creatureIdInput = document.getElementById("creatureIdInput");
const notesInput = document.getElementById("notesInput");
const creatureList = document.getElementById("creatureList");
const modeStatus = document.getElementById("modeStatus");
const visualPane = document.getElementById("visualPane");
const jsonPane = document.getElementById("jsonPane");
const showVisualBtn = document.getElementById("showVisualBtn");
const showJsonBtn = document.getElementById("showJsonBtn");
const rebuildVisualBtn = document.getElementById("rebuildVisualBtn");
const visualInfo = document.getElementById("visualInfo");
const visualViewportEl = document.getElementById("visualViewport");

const lockTopology = document.getElementById("lockTopology");
const lockJointTypes = document.getElementById("lockJointTypes");
const lockJointLimits = document.getElementById("lockJointLimits");
const lockSegmentDynamics = document.getElementById("lockSegmentDynamics");
const lockActuatorMechanics = document.getElementById("lockActuatorMechanics");
const lockControls = document.getElementById("lockControls");
const lockVisualHue = document.getElementById("lockVisualHue");
const presetGaitOnlyBtn = document.getElementById("presetGaitOnlyBtn");
const presetGaitAndLimitsBtn = document.getElementById("presetGaitAndLimitsBtn");

function setStatus(message, cls = "") {
  statusEl.className = `status ${cls}`.trim();
  statusEl.textContent = message;
}

function toFinite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampInt(value, min, max) {
  return Math.trunc(clamp(toFinite(value, min), min, max));
}

function deepCloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function parseGenomeEditor() {
  const raw = genomeEditor.value.trim();
  if (!raw) {
    throw new Error("Genome JSON is empty.");
  }
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Genome JSON must be an object.");
  }
  if (parsed.graph && Array.isArray(parsed.graph.nodes)) {
    return parsed;
  }
  if (parsed.genome?.graph && Array.isArray(parsed.genome.graph.nodes)) {
    return parsed.genome;
  }
  throw new Error("JSON must be a genome object with graph.nodes.");
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) {
    const message = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${message ? `: ${message}` : ""}`);
  }
  return await response.json();
}

function lockPayload() {
  return {
    lockTopology: Boolean(lockTopology.checked),
    lockJointTypes: Boolean(lockJointTypes.checked),
    lockJointLimits: Boolean(lockJointLimits.checked),
    lockSegmentDynamics: Boolean(lockSegmentDynamics.checked),
    lockActuatorMechanics: Boolean(lockActuatorMechanics.checked),
    lockControls: Boolean(lockControls.checked),
    lockVisualHue: Boolean(lockVisualHue.checked),
  };
}

function applyLocks(locks) {
  const src = locks || {};
  lockTopology.checked = Boolean(src.lockTopology);
  lockJointTypes.checked = Boolean(src.lockJointTypes);
  lockJointLimits.checked = Boolean(src.lockJointLimits);
  lockSegmentDynamics.checked = Boolean(src.lockSegmentDynamics);
  lockActuatorMechanics.checked = src.lockActuatorMechanics === undefined
    ? Boolean(src.lockSegmentDynamics)
    : Boolean(src.lockActuatorMechanics);
  lockControls.checked = Boolean(src.lockControls);
  lockVisualHue.checked = Boolean(src.lockVisualHue);
}

function applyGaitOnlyPreset() {
  lockTopology.checked = true;
  lockJointTypes.checked = true;
  lockJointLimits.checked = true;
  lockSegmentDynamics.checked = true;
  lockActuatorMechanics.checked = true;
  lockControls.checked = false;
  lockVisualHue.checked = true;
  setStatus("Applied preset: morphology fixed, neural controller evolvable (gait/controller-only).", "ok");
}

function applyGaitAndLimitsPreset() {
  lockTopology.checked = true;
  lockJointTypes.checked = true;
  lockJointLimits.checked = false;
  lockSegmentDynamics.checked = true;
  lockActuatorMechanics.checked = true;
  lockControls.checked = false;
  lockVisualHue.checked = true;
  setStatus("Applied preset: morphology fixed, controller + joint limits evolvable.", "ok");
}

function setEditorMode(mode) {
  state.editorMode = mode === "json" ? "json" : "visual";
  const visualActive = state.editorMode === "visual";
  visualPane.classList.toggle("active", visualActive);
  jsonPane.classList.toggle("active", !visualActive);
  showVisualBtn.classList.toggle("active", visualActive);
  showJsonBtn.classList.toggle("active", !visualActive);
  if (visualActive && visualView) {
    requestAnimationFrame(() => visualView.resize());
  }
}

function labelMorphMode(mode) {
  if (mode === "authored") return "authored";
  if (mode === "random") return "random";
  return String(mode || "unknown");
}

async function refreshModeStatus() {
  try {
    const response = await fetchJson(urls.evolutionState);
    const mode = labelMorphMode(response?.morphologyMode ?? response?.morphology_mode);
    const authoredId = response?.authoredCreatureId ?? response?.authored_creature_id;
    if (mode === "authored") {
      modeStatus.textContent = `Current backend mode: authored (${authoredId || "no creature id"})`;
    } else {
      modeStatus.textContent = `Current backend mode: ${mode}`;
    }
  } catch (error) {
    modeStatus.textContent = `Current backend mode: unavailable (${error.message})`;
  }
}

function setGenomeState(genome) {
  state.currentGenome = deepCloneJson(genome);
  genomeEditor.value = JSON.stringify(state.currentGenome, null, 2);
}

function rebuildVisualFromCurrentGenome() {
  if (!visualView) {
    return;
  }
  if (!state.currentGenome) {
    visualInfo.textContent = "No genome loaded.";
    return;
  }
  visualView.rebuild(state.currentGenome);
}

function rebuildVisualFromEditor() {
  try {
    const genome = parseGenomeEditor();
    setGenomeState(genome);
    rebuildVisualFromCurrentGenome();
    setStatus("Rebuilt visual from current JSON.", "ok");
  } catch (error) {
    setStatus(`Cannot rebuild visual: ${error.message}`, "err");
  }
}

async function loadGenome(url, label) {
  try {
    const genome = await fetchJson(url);
    setGenomeState(genome);
    rebuildVisualFromCurrentGenome();
    setStatus(`Loaded ${label} genome.`, "ok");
  } catch (error) {
    setStatus(`Failed loading ${label} genome: ${error.message}`, "err");
  }
}

async function refreshCreatureList(selectId) {
  try {
    const payload = await fetchJson(urls.creatureList);
    const creatures = Array.isArray(payload?.creatures) ? payload.creatures : [];
    creatureList.innerHTML = "";
    for (const creature of creatures) {
      const option = document.createElement("option");
      option.value = creature.id;
      option.textContent = creature.id;
      creatureList.appendChild(option);
    }
    if (selectId) {
      creatureList.value = selectId;
    }
    if (!creatureList.value && creatureList.options.length > 0) {
      creatureList.selectedIndex = 0;
    }
  } catch (error) {
    setStatus(`Failed refreshing creature list: ${error.message}`, "err");
  }
}

async function saveCreature() {
  const id = creatureIdInput.value.trim();
  if (!id) {
    setStatus("Creature id is required.", "warn");
    return;
  }
  let genome;
  try {
    genome = parseGenomeEditor();
  } catch (error) {
    setStatus(error.message, "warn");
    return;
  }
  setGenomeState(genome);
  try {
    const payload = await fetchJson(urls.creatureSave, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id,
        genome,
        notes: notesInput.value || "",
        locks: lockPayload(),
      }),
    });
    creatureIdInput.value = payload.id || id;
    await refreshCreatureList(payload.id || id);
    setStatus(`Saved creature '${payload.id}'.`, "ok");
  } catch (error) {
    setStatus(`Save failed: ${error.message}`, "err");
  }
}

async function loadSelectedCreature() {
  const id = creatureList.value.trim();
  if (!id) {
    setStatus("Select a creature first.", "warn");
    return;
  }
  try {
    const creature = await fetchJson(urls.creatureGet(id));
    creatureIdInput.value = creature.id || id;
    notesInput.value = creature.notes || "";
    applyLocks(creature.locks);
    setGenomeState(creature.genome);
    rebuildVisualFromCurrentGenome();
    setStatus(`Loaded creature '${id}'.`, "ok");
  } catch (error) {
    setStatus(`Load failed: ${error.message}`, "err");
  }
}

async function sendControl(payload) {
  await fetchJson(urls.control, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function activateAuthoredMode() {
  const id = creatureIdInput.value.trim() || creatureList.value.trim();
  if (!id) {
    setStatus("Creature id is required to activate authored mode.", "warn");
    return;
  }
  try {
    await sendControl({
      action: "set_morphology_mode",
      morphologyMode: "authored",
      creatureId: id,
    });
    await refreshModeStatus();
    setStatus(`Activated authored mode with '${id}'. Evolution restarted.`, "ok");
  } catch (error) {
    setStatus(`Failed to activate authored mode: ${error.message}`, "err");
  }
}

async function setRandomMode() {
  try {
    await sendControl({ action: "set_morphology_mode", morphologyMode: "random" });
    await refreshModeStatus();
    setStatus("Switched morphology mode to random. Evolution restarted.", "ok");
  } catch (error) {
    setStatus(`Failed switching to random mode: ${error.message}`, "err");
  }
}

function normalizedAxis(axisY, axisZ) {
  const axis = new THREE.Vector3(
    1.0,
    toFinite(axisY) * AXIS_TILT_GAIN,
    toFinite(axisZ) * AXIS_TILT_GAIN
  );
  if (axis.lengthSq() <= 1e-8) {
    return new THREE.Vector3(1, 0, 0);
  }
  return axis.normalize();
}

function outwardBiasedGrowthDir(rawDir, anchorHint) {
  const dir = rawDir.lengthSq() > 1e-8 ? rawDir.clone().normalize() : new THREE.Vector3(0, -1, 0);
  const outward = anchorHint.lengthSq() > 1e-8
    ? anchorHint.clone().normalize()
    : new THREE.Vector3(0, -1, 0);
  const dot = dir.dot(outward);
  if (dot < EDGE_OUTWARD_GROWTH_MIN_DOT) {
    dir.addScaledVector(outward, EDGE_OUTWARD_GROWTH_MIN_DOT - dot);
    if (dir.lengthSq() <= 1e-8) {
      return outward;
    }
    dir.normalize();
  }
  return dir;
}

function scaledPartSize(part, scale = 1) {
  const safeScale = Math.abs(toFinite(scale, 1.0));
  const w = clamp(Math.abs(toFinite(part?.w, 0.6) * safeScale), 0.14, 2.8);
  const h = clamp(Math.abs(toFinite(part?.h, 0.8) * safeScale), 0.22, 3.4);
  const d = clamp(Math.abs(toFinite(part?.d, 0.6) * safeScale), 0.14, 2.8);
  return [w, h, d];
}

function expandGraph(graph) {
  const nodes = asArray(graph?.nodes);
  if (!nodes.length) {
    return [{ nodeIndex: 0, parentIndex: null, incomingEdge: null }];
  }
  const root = clampInt(graph?.root ?? 0, 0, nodes.length - 1);
  const maxParts = clampInt(graph?.maxParts ?? 32, 1, MAX_GRAPH_PARTS);
  const expanded = [{ nodeIndex: root, parentIndex: null, incomingEdge: null }];
  const queue = [{ expandedIndex: 0, nodeIndex: root, ancestry: [root] }];
  while (queue.length && expanded.length < maxParts) {
    const current = queue.shift();
    const node = nodes[current.nodeIndex];
    if (!node) {
      continue;
    }
    const edges = asArray(node.edges).slice(0, MAX_GRAPH_EDGES_PER_NODE);
    for (const edge of edges) {
      if (expanded.length >= maxParts) {
        break;
      }
      const to = Math.trunc(toFinite(edge?.to, -1));
      if (to < 0 || to >= nodes.length) {
        continue;
      }
      const recursiveLimit = Math.max(1, Math.trunc(toFinite(edge?.recursiveLimit, 1)));
      const occurrences = current.ancestry.filter((value) => value === to).length;
      const isRecursive = occurrences > 0;
      if (isRecursive && occurrences >= recursiveLimit) {
        continue;
      }
      const terminalOnly = Boolean(edge?.terminalOnly);
      if (terminalOnly && (!isRecursive || occurrences + 1 < recursiveLimit)) {
        continue;
      }
      const childAncestry = current.ancestry.slice();
      childAncestry.push(to);
      const childExpandedIndex = expanded.length;
      expanded.push({
        nodeIndex: to,
        parentIndex: current.expandedIndex,
        incomingEdge: { ...edge },
      });
      queue.push({
        expandedIndex: childExpandedIndex,
        nodeIndex: to,
        ancestry: childAncestry,
      });
    }
  }
  if (expanded.length === 1) {
    const rootNode = nodes[root];
    const firstEdge = asArray(rootNode?.edges)[0];
    if (firstEdge) {
      const to = clampInt(firstEdge.to ?? 0, 0, nodes.length - 1);
      expanded.push({
        nodeIndex: to,
        parentIndex: 0,
        incomingEdge: { ...firstEdge },
      });
    }
  }
  return expanded;
}

function buildVisualModel(genome) {
  const graph = genome?.graph;
  const nodes = asArray(graph?.nodes);
  const expanded = expandGraph(graph);
  if (!nodes.length || !expanded.length) {
    return { parts: [], joints: [], nodeCount: nodes.length };
  }
  const parts = [];
  const joints = [];
  const partByExpanded = new Map();

  const rootExpanded = expanded[0];
  const rootNodeIndex = clampInt(rootExpanded.nodeIndex ?? 0, 0, nodes.length - 1);
  const rootNode = nodes[rootNodeIndex] || {};
  const rootSize = scaledPartSize(rootNode.part, 1.0);
  parts.push({
    nodeIndex: rootNodeIndex,
    parentPartIndex: null,
    size: rootSize,
    center: new THREE.Vector3(0, rootSize[1] * 0.5, 0),
    quaternion: new THREE.Quaternion(),
  });
  partByExpanded.set(0, 0);

  for (let expandedIndex = 1; expandedIndex < expanded.length; expandedIndex += 1) {
    const expandedPart = expanded[expandedIndex];
    const parentPartIndex = partByExpanded.get(expandedPart.parentIndex);
    if (parentPartIndex === undefined) {
      continue;
    }
    const parent = parts[parentPartIndex];
    const edge = expandedPart.incomingEdge || {};
    const nodeIndex = clampInt(expandedPart.nodeIndex ?? 0, 0, nodes.length - 1);
    const node = nodes[nodeIndex] || {};

    const reflectSign = Boolean(edge.reflectX) ? -1 : 1;
    const anchorX = toFinite(edge.anchorX, 0);
    const anchorY = toFinite(edge.anchorY, -0.8);
    const anchorZ = toFinite(edge.anchorZ, 0);

    const pivotFromParent = new THREE.Vector3(
      anchorX * parent.size[0] * 0.5,
      anchorY * parent.size[1] * 0.5,
      anchorZ * parent.size[2] * 0.5
    );
    const axisLocal = normalizedAxis(edge.axisY, edge.axisZ);
    const localGrowth = outwardBiasedGrowthDir(
      new THREE.Vector3(
        toFinite(edge.dirX, 0) * reflectSign,
        toFinite(edge.dirY, -1),
        toFinite(edge.dirZ, 0) * reflectSign
      ),
      new THREE.Vector3(anchorX * reflectSign, anchorY, anchorZ)
    );

    const anchorWorld = parent.center
      .clone()
      .add(pivotFromParent.applyQuaternion(parent.quaternion.clone()));
    const childSize = scaledPartSize(node.part, toFinite(edge.scale, 1));
    const growthWorld = localGrowth.clone().applyQuaternion(parent.quaternion.clone());
    const center = anchorWorld.clone().addScaledVector(growthWorld, childSize[1] * 0.5);
    const segLocalRot = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, -1, 0),
      localGrowth.clone().normalize()
    );
    const childRotation = parent.quaternion.clone().multiply(segLocalRot);

    const childPartIndex = parts.length;
    parts.push({
      nodeIndex,
      parentPartIndex,
      size: childSize,
      center,
      quaternion: childRotation,
    });
    partByExpanded.set(expandedIndex, childPartIndex);

    const axisWorld = axisLocal.applyQuaternion(parent.quaternion.clone()).normalize();
    joints.push({
      parentPartIndex,
      childPartIndex,
      parentCenter: parent.center.clone(),
      anchorWorld,
      axisWorld,
      jointType: String(edge.jointType || "hinge"),
    });
  }

  return {
    parts,
    joints,
    nodeCount: nodes.length,
  };
}

class CreatorVisualView {
  constructor(container, infoEl) {
    this.container = container;
    this.infoEl = infoEl;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x111922, 7, 50);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
    this.camera.position.set(4.5, 3.2, 5.2);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.target.set(0, 0.8, 0);

    this.modelGroup = new THREE.Group();
    this.segmentGroup = new THREE.Group();
    this.jointGroup = new THREE.Group();
    this.modelGroup.add(this.segmentGroup);
    this.modelGroup.add(this.jointGroup);
    this.scene.add(this.modelGroup);

    this.scene.add(new THREE.HemisphereLight(0xd8e7f2, 0x263025, 0.78));
    const sun = new THREE.DirectionalLight(0xffedcf, 1.22);
    sun.position.set(6.5, 9, 4.2);
    this.scene.add(sun);

    // Read-only morphology viewport: no ground plane/grid so geometry remains unobstructed.

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  clearGroup(group) {
    while (group.children.length) {
      const child = group.children.pop();
      if (child.geometry) {
        child.geometry.dispose();
      }
      if (child.material) {
        if (Array.isArray(child.material)) {
          for (const mat of child.material) {
            mat.dispose();
          }
        } else {
          child.material.dispose();
        }
      }
      if (child.children?.length) {
        for (const nested of child.children) {
          if (nested.geometry) {
            nested.geometry.dispose();
          }
          if (nested.material) {
            if (Array.isArray(nested.material)) {
              for (const mat of nested.material) {
                mat.dispose();
              }
            } else {
              nested.material.dispose();
            }
          }
        }
      }
    }
  }

  fitCamera() {
    const bounds = new THREE.Box3().setFromObject(this.modelGroup);
    if (!Number.isFinite(bounds.min.x) || !Number.isFinite(bounds.max.x)) {
      this.controls.target.set(0, 0.8, 0);
      this.camera.position.set(4.5, 3.2, 5.2);
      return;
    }
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bounds.getCenter(center);
    bounds.getSize(size);
    const radius = Math.max(size.length() * 0.5, 0.9);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const distance = radius / Math.sin(fov * 0.5);
    this.camera.position
      .copy(center)
      .add(new THREE.Vector3(distance * 0.82, distance * 0.62, distance * 0.92));
    this.camera.near = Math.max(distance / 300, 0.01);
    this.camera.far = Math.max(distance * 8, 80);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  rebuild(genome) {
    this.clearGroup(this.segmentGroup);
    this.clearGroup(this.jointGroup);

    const model = buildVisualModel(genome);
    if (!model.parts.length) {
      this.infoEl.textContent = "No graph nodes to render. Load a genome with graph nodes.";
      this.fitCamera();
      return;
    }

    for (let i = 0; i < model.parts.length; i += 1) {
      const part = model.parts[i];
      const hue = ((part.nodeIndex * 0.17) + 0.12) % 1;
      const color = new THREE.Color().setHSL(hue, 0.52, 0.43);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(part.size[0], part.size[1], part.size[2]),
        new THREE.MeshStandardMaterial({
          color,
          roughness: 0.82,
          metalness: 0.05,
        })
      );
      mesh.position.copy(part.center);
      mesh.quaternion.copy(part.quaternion);
      const edgeLines = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color: 0x0a1210,
          transparent: true,
          opacity: 0.6,
        })
      );
      mesh.add(edgeLines);
      this.segmentGroup.add(mesh);
    }

    for (const joint of model.joints) {
      const jointColor = joint.jointType === "ball" ? 0x72ccff : 0xf3bc67;
      const markerRadius = 0.085;
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(markerRadius, 14, 10),
        new THREE.MeshBasicMaterial({
          color: jointColor,
          transparent: true,
          opacity: 0.92,
          depthTest: false,
        })
      );
      marker.position.copy(joint.anchorWorld);
      marker.renderOrder = 20;
      this.jointGroup.add(marker);

      const stemGeo = new THREE.BufferGeometry().setFromPoints([
        joint.parentCenter,
        joint.anchorWorld,
      ]);
      const stem = new THREE.Line(
        stemGeo,
        new THREE.LineBasicMaterial({
          color: 0x86a594,
          transparent: true,
          opacity: 0.8,
          depthTest: false,
        })
      );
      stem.renderOrder = 19;
      this.jointGroup.add(stem);

      const axisSpan = joint.axisWorld.clone().multiplyScalar(0.32);
      const axisGeo = new THREE.BufferGeometry().setFromPoints([
        joint.anchorWorld.clone().sub(axisSpan),
        joint.anchorWorld.clone().add(axisSpan),
      ]);
      const axis = new THREE.Line(
        axisGeo,
        new THREE.LineBasicMaterial({
          color: 0x61d0c3,
          transparent: true,
          opacity: 0.95,
          depthTest: false,
        })
      );
      axis.renderOrder = 21;
      this.jointGroup.add(axis);
    }

    this.infoEl.textContent = `nodes: ${model.nodeCount} | expanded parts: ${model.parts.length} | joints: ${model.joints.length}
Read-only parity view: edit JSON then click "Rebuild visual from JSON".`;
    this.fitCamera();
  }
}

let visualView = null;
try {
  visualView = new CreatorVisualView(visualViewportEl, visualInfo);
} catch (error) {
  setStatus(`Visual mode unavailable: ${error.message}`, "err");
  visualInfo.textContent = `Failed to initialize Three.js viewport: ${error.message}`;
}

document.getElementById("loadCurrentBtn").addEventListener("click", () => {
  void loadGenome(urls.currentGenome, "current");
});
document.getElementById("loadBestBtn").addEventListener("click", () => {
  void loadGenome(urls.bestGenome, "best");
});
document.getElementById("saveCreatureBtn").addEventListener("click", () => {
  void saveCreature();
});
document.getElementById("refreshCreaturesBtn").addEventListener("click", () => {
  void refreshCreatureList();
});
document.getElementById("loadSelectedBtn").addEventListener("click", () => {
  void loadSelectedCreature();
});
document.getElementById("activateAuthoredBtn").addEventListener("click", () => {
  void activateAuthoredMode();
});
document.getElementById("setRandomBtn").addEventListener("click", () => {
  void setRandomMode();
});
showVisualBtn.addEventListener("click", () => {
  setEditorMode("visual");
});
showJsonBtn.addEventListener("click", () => {
  setEditorMode("json");
});
rebuildVisualBtn.addEventListener("click", () => {
  rebuildVisualFromEditor();
});
presetGaitOnlyBtn.addEventListener("click", () => {
  applyGaitOnlyPreset();
});
presetGaitAndLimitsBtn.addEventListener("click", () => {
  applyGaitAndLimitsPreset();
});

const initialGenome = {
  version: 2,
  graph: {
    root: 0,
    maxParts: 24,
    nodes: [],
  },
};
setGenomeState(initialGenome);
setEditorMode("visual");
rebuildVisualFromCurrentGenome();
void refreshCreatureList();
void refreshModeStatus();
