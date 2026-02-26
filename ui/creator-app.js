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
  visualModel: null,
  editorMode: "visual",
  selection: null,
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
const outlinerSearch = document.getElementById("outlinerSearch");
const outlinerList = document.getElementById("outlinerList");
const selectionBreadcrumb = document.getElementById("selectionBreadcrumb");
const inspectorContent = document.getElementById("inspectorContent");
const focusSelectedBtn = document.getElementById("focusSelectedBtn");

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

function selectionToken(selection) {
  if (!selection || !selection.kind || !selection.id) {
    return "";
  }
  return `${selection.kind}:${selection.id}`;
}

function fixed(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function formatVec3(v, digits = 3) {
  if (!v) {
    return "n/a";
  }
  return `[${fixed(v.x, digits)}, ${fixed(v.y, digits)}, ${fixed(v.z, digits)}]`;
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

function findSelectionEntity(model, selection) {
  if (!model || !selection) {
    return null;
  }
  if (selection.kind === "node") {
    return model.nodes.find((item) => item.id === selection.id) || null;
  }
  if (selection.kind === "part") {
    return model.parts.find((item) => item.id === selection.id) || null;
  }
  if (selection.kind === "joint") {
    return model.joints.find((item) => item.id === selection.id) || null;
  }
  return null;
}

function setSelection(selection, source = "app") {
  const model = state.visualModel;
  let nextSelection = null;
  if (selection && selection.kind && selection.id && findSelectionEntity(model, selection)) {
    nextSelection = { kind: selection.kind, id: selection.id };
  }
  const changed = selectionToken(state.selection) !== selectionToken(nextSelection);
  state.selection = nextSelection;
  if (visualView) {
    visualView.setSelection(state.selection);
  }
  renderOutliner();
  if (source === "viewport") {
    const selectedRow = outlinerList.querySelector(".outlinerItem.selected");
    selectedRow?.scrollIntoView({ block: "nearest" });
  }
  if (source === "outliner" && visualView && state.selection) {
    visualView.focusSelection(state.selection);
  }
  renderInspector();
  if (changed && source === "viewport") {
    setStatus(`Selected ${state.selection?.kind || "none"}.`, "ok");
  }
}

function syncSelectionToModel() {
  if (state.selection && !findSelectionEntity(state.visualModel, state.selection)) {
    state.selection = null;
  }
}

function rebuildVisualFromCurrentGenome() {
  if (!visualView) {
    return;
  }
  if (!state.currentGenome) {
    visualInfo.textContent = "No genome loaded.";
    state.visualModel = null;
    setSelection(null);
    return;
  }
  state.visualModel = visualView.rebuild(state.currentGenome);
  syncSelectionToModel();
  setSelection(state.selection);
}

function appendEmpty(container, message) {
  container.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "emptyState";
  empty.textContent = message;
  container.appendChild(empty);
}

function renderOutliner() {
  const model = state.visualModel;
  if (!model || (!model.nodes.length && !model.parts.length)) {
    appendEmpty(outlinerList, "No visual entities. Load or rebuild a genome first.");
    return;
  }

  const query = outlinerSearch.value.trim().toLowerCase();
  outlinerList.innerHTML = "";
  let rendered = 0;
  const selectedToken = selectionToken(state.selection);

  const addSection = (label) => {
    const section = document.createElement("div");
    section.className = "outlinerSectionTitle";
    section.textContent = label;
    outlinerList.appendChild(section);
  };

  const addItem = ({ kind, id, label, details, depth = 0, searchText = "" }) => {
    const haystack = `${label} ${details || ""} ${searchText}`.toLowerCase();
    if (query && !haystack.includes(query)) {
      return;
    }
    rendered += 1;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `outlinerItem ${kind}`;
    button.style.paddingLeft = `${7 + depth * 14}px`;
    button.dataset.kind = kind;
    button.dataset.id = id;

    const line1 = document.createElement("div");
    line1.className = "line1";
    const labelSpan = document.createElement("span");
    labelSpan.textContent = label;
    const idSpan = document.createElement("span");
    idSpan.className = "tiny";
    idSpan.textContent = id;
    line1.appendChild(labelSpan);
    line1.appendChild(idSpan);

    const line2 = document.createElement("div");
    line2.className = "line2";
    line2.textContent = details || "";

    button.appendChild(line1);
    button.appendChild(line2);
    if (selectedToken === selectionToken({ kind, id })) {
      button.classList.add("selected");
    }
    button.addEventListener("click", () => {
      setSelection({ kind, id }, "outliner");
    });
    outlinerList.appendChild(button);
  };

  addSection(`Graph Nodes (${model.nodes.length})`);
  for (const node of model.nodes) {
    const brainNeurons = asArray(node.brain?.neurons).length;
    addItem({
      kind: "node",
      id: node.id,
      label: `Node ${node.nodeIndex}`,
      details: `${node.partIds.length} part instances | ${node.edgeCount} edges | ${brainNeurons} local neurons`,
      depth: 0,
      searchText: `node ${node.nodeIndex}`,
    });
  }

  addSection(`Expanded Parts (${model.parts.length})`);
  for (const part of model.parts) {
    const parentText = part.parentPartId ? `parent ${part.parentPartId}` : "root";
    addItem({
      kind: "part",
      id: part.id,
      label: `${part.id} (node ${part.nodeIndex})`,
      details: `${parentText} | size ${fixed(part.size[0], 2)} x ${fixed(part.size[1], 2)} x ${fixed(part.size[2], 2)}`,
      depth: Math.max(0, part.depth || 0),
      searchText: `part node ${part.nodeIndex} ${parentText}`,
    });
  }

  addSection(`Joints (${model.joints.length})`);
  for (const joint of model.joints) {
    addItem({
      kind: "joint",
      id: joint.id,
      label: `${joint.id} (${joint.jointType})`,
      details: `${joint.parentPartId} -> ${joint.childPartId}`,
      depth: 0,
      searchText: `joint ${joint.jointType} ${joint.parentPartId} ${joint.childPartId}`,
    });
  }

  if (!rendered) {
    appendEmpty(outlinerList, "No matches for current outliner filter.");
  }
}

function appendField(container, key, value) {
  const row = document.createElement("div");
  row.className = "fieldRow";
  const keyEl = document.createElement("div");
  keyEl.className = "k";
  keyEl.textContent = key;
  const valEl = document.createElement("div");
  valEl.className = "v";
  valEl.textContent = value;
  row.appendChild(keyEl);
  row.appendChild(valEl);
  container.appendChild(row);
}

function renderInspector() {
  inspectorContent.innerHTML = "";
  const model = state.visualModel;
  const selection = state.selection;

  if (!model) {
    selectionBreadcrumb.textContent = "Nothing selected.";
    appendEmpty(inspectorContent, "No model loaded.");
    return;
  }

  if (!selection) {
    selectionBreadcrumb.textContent = "Nothing selected.";
    appendEmpty(inspectorContent, "Select a node, part, or joint in the viewport or outliner.");
    return;
  }

  const entity = findSelectionEntity(model, selection);
  if (!entity) {
    selectionBreadcrumb.textContent = "Nothing selected.";
    appendEmpty(inspectorContent, "Selection is stale after rebuild.");
    return;
  }

  selectionBreadcrumb.textContent = `Selected: ${selection.kind} / ${selection.id}`;

  if (selection.kind === "node") {
    appendField(inspectorContent, "node index", String(entity.nodeIndex));
    appendField(inspectorContent, "part instances", String(entity.partIds.length));
    appendField(inspectorContent, "edge genes", String(entity.edgeCount));
    appendField(inspectorContent, "part dims", `${fixed(entity.part?.w, 3)} x ${fixed(entity.part?.h, 3)} x ${fixed(entity.part?.d, 3)}`);
    appendField(inspectorContent, "part mass", fixed(toFinite(entity.part?.mass, entity.part?.m), 3));
    appendField(inspectorContent, "part ids", entity.partIds.join(", ") || "none");
    appendField(inspectorContent, "brain neurons", String(asArray(entity.brain?.neurons).length));
    appendField(inspectorContent, "brain effectors", String(asArray(entity.brain?.effectors).length));
    return;
  }

  if (selection.kind === "part") {
    appendField(inspectorContent, "part id", entity.id);
    appendField(inspectorContent, "expanded index", String(entity.expandedIndex));
    appendField(inspectorContent, "node index", String(entity.nodeIndex));
    appendField(inspectorContent, "parent", entity.parentPartId || "none (root)");
    appendField(inspectorContent, "depth", String(entity.depth || 0));
    appendField(inspectorContent, "size w/h/d", `${fixed(entity.size[0], 3)} x ${fixed(entity.size[1], 3)} x ${fixed(entity.size[2], 3)}`);
    appendField(inspectorContent, "center", formatVec3(entity.center, 3));
    if (entity.incomingEdge) {
      appendField(inspectorContent, "incoming joint", String(entity.incomingEdge.jointType || "hinge"));
      appendField(inspectorContent, "incoming scale", fixed(toFinite(entity.incomingEdge.scale, 1), 3));
      appendField(inspectorContent, "anchor", `[${fixed(toFinite(entity.incomingEdge.anchorX, 0), 3)}, ${fixed(toFinite(entity.incomingEdge.anchorY, 0), 3)}, ${fixed(toFinite(entity.incomingEdge.anchorZ, 0), 3)}]`);
      appendField(inspectorContent, "growth dir", `[${fixed(toFinite(entity.incomingEdge.dirX, 0), 3)}, ${fixed(toFinite(entity.incomingEdge.dirY, 0), 3)}, ${fixed(toFinite(entity.incomingEdge.dirZ, 0), 3)}]`);
      appendField(inspectorContent, "axis yz", `[${fixed(toFinite(entity.incomingEdge.axisY, 0), 3)}, ${fixed(toFinite(entity.incomingEdge.axisZ, 0), 3)}]`);
    }
    return;
  }

  if (selection.kind === "joint") {
    appendField(inspectorContent, "joint id", entity.id);
    appendField(inspectorContent, "type", entity.jointType);
    appendField(inspectorContent, "parent part", entity.parentPartId);
    appendField(inspectorContent, "child part", entity.childPartId);
    appendField(inspectorContent, "anchor world", formatVec3(entity.anchorWorld, 3));
    appendField(inspectorContent, "axis world", formatVec3(entity.axisWorld, 3));
    appendField(inspectorContent, "recursive limit", String(clampInt(entity.edge?.recursiveLimit ?? 1, 1, 9999)));
    appendField(inspectorContent, "terminal only", String(Boolean(entity.edge?.terminalOnly)));
    appendField(inspectorContent, "reflect x", String(Boolean(entity.edge?.reflectX)));
    appendField(inspectorContent, "motor strength", fixed(toFinite(entity.edge?.motorStrength, 0), 3));
    appendField(inspectorContent, "stiffness", fixed(toFinite(entity.edge?.stiffness, 0), 3));
    appendField(inspectorContent, "joint limit min xyz", `[${fixed(toFinite(entity.edge?.jointLimitMinX, 0), 3)}, ${fixed(toFinite(entity.edge?.jointLimitMinY, 0), 3)}, ${fixed(toFinite(entity.edge?.jointLimitMinZ, 0), 3)}]`);
    appendField(inspectorContent, "joint limit max xyz", `[${fixed(toFinite(entity.edge?.jointLimitMaxX, 0), 3)}, ${fixed(toFinite(entity.edge?.jointLimitMaxY, 0), 3)}, ${fixed(toFinite(entity.edge?.jointLimitMaxZ, 0), 3)}]`);
  }
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
  const expanded = [{ nodeIndex: root, parentIndex: null, incomingEdge: null, depth: 0 }];
  const queue = [{ expandedIndex: 0, nodeIndex: root, ancestry: [root], depth: 0 }];
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
        depth: current.depth + 1,
      });
      queue.push({
        expandedIndex: childExpandedIndex,
        nodeIndex: to,
        ancestry: childAncestry,
        depth: current.depth + 1,
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
        depth: 1,
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
    return { nodes: [], parts: [], joints: [], nodeCount: nodes.length };
  }
  const nodeEntries = nodes.map((node, nodeIndex) => ({
    id: `node-${nodeIndex}`,
    nodeIndex,
    edgeCount: asArray(node?.edges).length,
    part: node?.part || {},
    brain: node?.brain || null,
    partIds: [],
  }));
  const parts = [];
  const joints = [];
  const partByExpanded = new Map();

  const rootExpanded = expanded[0];
  const rootNodeIndex = clampInt(rootExpanded.nodeIndex ?? 0, 0, nodes.length - 1);
  const rootNode = nodes[rootNodeIndex] || {};
  const rootSize = scaledPartSize(rootNode.part, 1.0);
  const rootId = "part-0";
  parts.push({
    id: rootId,
    expandedIndex: 0,
    depth: 0,
    nodeId: nodeEntries[rootNodeIndex].id,
    nodeIndex: rootNodeIndex,
    parentPartId: null,
    parentPartIndex: null,
    incomingEdge: null,
    size: rootSize,
    center: new THREE.Vector3(0, rootSize[1] * 0.5, 0),
    quaternion: new THREE.Quaternion(),
  });
  nodeEntries[rootNodeIndex].partIds.push(rootId);
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
    const childPartId = `part-${childPartIndex}`;
    const incomingEdge = { ...edge };
    parts.push({
      id: childPartId,
      expandedIndex,
      depth: expandedPart.depth ?? 0,
      nodeId: nodeEntries[nodeIndex].id,
      nodeIndex,
      parentPartId: parent.id,
      parentPartIndex,
      incomingEdge,
      size: childSize,
      center,
      quaternion: childRotation,
    });
    nodeEntries[nodeIndex].partIds.push(childPartId);
    partByExpanded.set(expandedIndex, childPartIndex);

    const axisWorld = axisLocal.applyQuaternion(parent.quaternion.clone()).normalize();
    const jointId = `joint-${joints.length}`;
    joints.push({
      id: jointId,
      parentPartIndex,
      childPartIndex,
      parentPartId: parent.id,
      childPartId: childPartId,
      parentNodeIndex: parent.nodeIndex,
      childNodeIndex: nodeIndex,
      parentCenter: parent.center.clone(),
      anchorWorld,
      axisWorld,
      jointType: String(edge.jointType || "hinge"),
      edge: incomingEdge,
    });
  }

  return {
    nodes: nodeEntries,
    parts,
    joints,
    nodeCount: nodes.length,
  };
}

class CreatorVisualView {
  constructor(container, infoEl) {
    this.container = container;
    this.infoEl = infoEl;
    this.latestModel = null;
    this.currentSelection = null;
    this.onSelect = null;
    this.selectionObjects = new Map();
    this.pickables = [];
    this.pointerDown = null;
    this.raycaster = new THREE.Raycaster();
    this.pointerNdc = new THREE.Vector2(0, 0);

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
    this.installPointerHandlers();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  makeSelectionKey(kind, id) {
    return `${kind}:${id}`;
  }

  setSelectionCallback(callback) {
    this.onSelect = callback;
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

  focusBoundingBox(bounds) {
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

  fitCamera() {
    this.focusBoundingBox(new THREE.Box3().setFromObject(this.modelGroup));
  }

  registerSelectable(selection, object3d, pickable = true) {
    const key = this.makeSelectionKey(selection.kind, selection.id);
    object3d.userData.selection = { kind: selection.kind, id: selection.id };
    object3d.userData.selectionKey = key;
    object3d.userData.baseScale = object3d.scale.clone();
    if (object3d.material && !Array.isArray(object3d.material)) {
      const material = object3d.material;
      object3d.userData.baseColor = material.color?.getHex?.();
      object3d.userData.baseOpacity = material.opacity;
      object3d.userData.baseEmissive = material.emissive?.getHex?.();
    }
    if (!this.selectionObjects.has(key)) {
      this.selectionObjects.set(key, []);
    }
    this.selectionObjects.get(key).push(object3d);
    if (pickable) {
      this.pickables.push(object3d);
    }
  }

  installPointerHandlers() {
    const element = this.renderer.domElement;
    element.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      this.pointerDown = { x: event.clientX, y: event.clientY };
    });
    element.addEventListener("pointerup", (event) => {
      if (event.button !== 0) {
        return;
      }
      if (!this.pointerDown) {
        return;
      }
      const dx = event.clientX - this.pointerDown.x;
      const dy = event.clientY - this.pointerDown.y;
      this.pointerDown = null;
      if (Math.hypot(dx, dy) > 5) {
        return;
      }
      this.pickFromPointerEvent(event);
    });
    element.addEventListener("pointerleave", () => {
      this.pointerDown = null;
    });
  }

  pickFromPointerEvent(event) {
    if (!this.pickables.length) {
      return;
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) {
      return;
    }
    this.pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    if (!hits.length) {
      return;
    }
    const hit = hits[0].object;
    const selection = hit?.userData?.selection;
    if (!selection || !this.onSelect) {
      return;
    }
    this.onSelect(selection, "viewport");
  }

  resolveSelectionKeys(selection) {
    if (!selection) {
      return [];
    }
    if (selection.kind === "node") {
      const node = this.latestModel?.nodes?.find((item) => item.id === selection.id);
      if (!node) {
        return [];
      }
      return node.partIds.map((partId) => this.makeSelectionKey("part", partId));
    }
    return [this.makeSelectionKey(selection.kind, selection.id)];
  }

  setSelection(selection) {
    this.currentSelection = selection && selection.kind && selection.id
      ? { kind: selection.kind, id: selection.id }
      : null;
    this.applySelectionHighlight();
  }

  applySelectionHighlight() {
    for (const objects of this.selectionObjects.values()) {
      for (const object3d of objects) {
        object3d.scale.copy(object3d.userData.baseScale || new THREE.Vector3(1, 1, 1));
        const material = object3d.material;
        if (material && !Array.isArray(material)) {
          if (material.color && object3d.userData.baseColor !== undefined) {
            material.color.setHex(object3d.userData.baseColor);
          }
          if (material.emissive && object3d.userData.baseEmissive !== undefined) {
            material.emissive.setHex(object3d.userData.baseEmissive);
          }
          if (object3d.userData.baseOpacity !== undefined) {
            material.opacity = object3d.userData.baseOpacity;
          }
        }
      }
    }

    const selection = this.currentSelection;
    if (!selection) {
      return;
    }
    const highlightKeys = this.resolveSelectionKeys(selection);
    const tint = selection.kind === "joint" ? 0xffdb8a : 0x9fdba4;
    for (const key of highlightKeys) {
      const objects = this.selectionObjects.get(key) || [];
      for (const object3d of objects) {
        object3d.scale.multiplyScalar(1.08);
        const material = object3d.material;
        if (material && !Array.isArray(material)) {
          if (material.color) {
            material.color.setHex(tint);
          }
          if (material.emissive) {
            material.emissive.setHex(0x2f5936);
          }
          if (material.transparent) {
            material.opacity = Math.max(material.opacity, 0.95);
          }
        }
      }
    }
  }

  focusSelection(selection) {
    const selectionKeys = this.resolveSelectionKeys(selection);
    if (!selectionKeys.length) {
      return;
    }
    const bounds = new THREE.Box3();
    let hasAny = false;
    for (const key of selectionKeys) {
      const objects = this.selectionObjects.get(key) || [];
      for (const object3d of objects) {
        const objectBounds = new THREE.Box3().setFromObject(object3d);
        if (!Number.isFinite(objectBounds.min.x) || !Number.isFinite(objectBounds.max.x)) {
          continue;
        }
        if (!hasAny) {
          bounds.copy(objectBounds);
          hasAny = true;
        } else {
          bounds.union(objectBounds);
        }
      }
    }
    if (!hasAny) {
      return;
    }
    const size = new THREE.Vector3();
    bounds.getSize(size);
    if (size.lengthSq() < 1e-5) {
      const center = new THREE.Vector3();
      bounds.getCenter(center);
      const fallback = new THREE.Box3(
        center.clone().addScalar(-0.25),
        center.clone().addScalar(0.25)
      );
      this.focusBoundingBox(fallback);
      return;
    }
    this.focusBoundingBox(bounds);
  }

  rebuild(genome) {
    this.clearGroup(this.segmentGroup);
    this.clearGroup(this.jointGroup);
    this.selectionObjects.clear();
    this.pickables = [];

    const model = buildVisualModel(genome);
    this.latestModel = model;
    if (!model.parts.length) {
      this.infoEl.textContent = "No graph nodes to render. Load a genome with graph nodes.";
      this.fitCamera();
      return model;
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
      this.registerSelectable({ kind: "part", id: part.id }, mesh, true);
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
      this.registerSelectable({ kind: "joint", id: joint.id }, marker, true);
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
      this.registerSelectable({ kind: "joint", id: joint.id }, stem, true);
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
      this.registerSelectable({ kind: "joint", id: joint.id }, axis, true);
      this.jointGroup.add(axis);
    }

    this.infoEl.textContent = `nodes: ${model.nodeCount} | expanded parts: ${model.parts.length} | joints: ${model.joints.length}
Selection enabled: click viewport object or outliner row to inspect details.`;
    this.applySelectionHighlight();
    this.fitCamera();
    return model;
  }
}

let visualView = null;
try {
  visualView = new CreatorVisualView(visualViewportEl, visualInfo);
  visualView.setSelectionCallback((selection, source) => {
    setSelection(selection, source);
  });
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
outlinerSearch.addEventListener("input", () => {
  renderOutliner();
});
focusSelectedBtn.addEventListener("click", () => {
  if (!visualView || !state.selection) {
    return;
  }
  visualView.focusSelection(state.selection);
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
renderOutliner();
renderInspector();
void refreshCreatureList();
void refreshModeStatus();
