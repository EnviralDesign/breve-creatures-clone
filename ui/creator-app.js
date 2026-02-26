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
  inspectorError: null,
};

const JOINT_TYPE_OPTIONS = ["hinge", "ball"];
const FIELD_SCHEMAS = {
  "node.part.w": {
    type: "number",
    min: 0.14,
    max: 2.2,
    step: 0.01,
    description: "Base width for this node template part.",
  },
  "node.part.h": {
    type: "number",
    min: 0.2,
    max: 2.8,
    step: 0.01,
    description: "Base height for this node template part.",
  },
  "node.part.d": {
    type: "number",
    min: 0.14,
    max: 2.2,
    step: 0.01,
    description: "Base depth for this node template part.",
  },
  "node.part.mass": {
    type: "number",
    min: 0.08,
    max: 2.8,
    step: 0.01,
    description: "Mass used by physics for this node template part.",
  },
  "node.brain.effectorX.gain": {
    type: "number",
    min: 0.2,
    max: 2.0,
    step: 0.01,
    description: "Drive gain for local effector X.",
  },
  "node.brain.effectorY.gain": {
    type: "number",
    min: 0.2,
    max: 2.0,
    step: 0.01,
    description: "Drive gain for local effector Y.",
  },
  "node.brain.effectorZ.gain": {
    type: "number",
    min: 0.2,
    max: 2.0,
    step: 0.01,
    description: "Drive gain for local effector Z.",
  },
  "part.part.w": {
    type: "number",
    min: 0.14,
    max: 2.2,
    step: 0.01,
    description: "Width of the source node part for this selected part.",
  },
  "part.part.h": {
    type: "number",
    min: 0.2,
    max: 2.8,
    step: 0.01,
    description: "Height of the source node part for this selected part.",
  },
  "part.part.d": {
    type: "number",
    min: 0.14,
    max: 2.2,
    step: 0.01,
    description: "Depth of the source node part for this selected part.",
  },
  "part.part.mass": {
    type: "number",
    min: 0.08,
    max: 2.8,
    step: 0.01,
    description: "Mass of the source node part for this selected part.",
  },
  "part.edge.scale": {
    type: "number",
    min: 0.45,
    max: 1.55,
    step: 0.01,
    description: "Relative size scaling applied to this branch.",
  },
  "part.edge.anchorX": {
    type: "number",
    min: -0.86,
    max: 0.86,
    step: 0.01,
    description: "Attachment offset on parent part along local X.",
  },
  "part.edge.anchorY": {
    type: "number",
    min: -0.86,
    max: 0.86,
    step: 0.01,
    description: "Attachment offset on parent part along local Y.",
  },
  "part.edge.anchorZ": {
    type: "number",
    min: -0.86,
    max: 0.86,
    step: 0.01,
    description: "Attachment offset on parent part along local Z.",
  },
  "part.edge.dirX": {
    type: "number",
    min: -1.15,
    max: 1.15,
    step: 0.01,
    description: "Growth direction component along X.",
  },
  "part.edge.dirY": {
    type: "number",
    min: -1.2,
    max: 0.85,
    step: 0.01,
    description: "Growth direction component along Y.",
  },
  "part.edge.dirZ": {
    type: "number",
    min: -1.15,
    max: 1.15,
    step: 0.01,
    description: "Growth direction component along Z.",
  },
  "part.edge.axisY": {
    type: "number",
    min: -0.65,
    max: 0.65,
    step: 0.01,
    description: "Orientation helper axis component Y.",
  },
  "part.edge.axisZ": {
    type: "number",
    min: -0.65,
    max: 0.65,
    step: 0.01,
    description: "Orientation helper axis component Z.",
  },
  "joint.edge.jointType": {
    type: "enum",
    options: JOINT_TYPE_OPTIONS,
    description: "Constraint type used between parent and child parts.",
  },
  "joint.edge.limitX": {
    type: "number",
    min: 0.12,
    max: Math.PI * 0.95,
    step: 0.01,
    description: "Angular travel limit around local X (radians).",
  },
  "joint.edge.limitY": {
    type: "number",
    min: 0.1,
    max: Math.PI * 0.75,
    step: 0.01,
    description: "Angular travel limit around local Y (radians).",
  },
  "joint.edge.limitZ": {
    type: "number",
    min: 0.1,
    max: Math.PI * 0.75,
    step: 0.01,
    description: "Angular travel limit around local Z (radians).",
  },
  "joint.edge.motorStrength": {
    type: "number",
    min: 0.3,
    max: 4.5,
    step: 0.01,
    description: "Motor drive strength for this joint.",
  },
  "joint.edge.stiffness": {
    type: "number",
    min: 12.0,
    max: 160.0,
    step: 0.1,
    description: "Spring stiffness of the joint constraint.",
  },
  "joint.edge.recursiveLimit": {
    type: "int",
    min: 1,
    max: 5,
    step: 1,
    description: "Maximum branch recursion depth spawned from this edge.",
  },
  "joint.edge.terminalOnly": {
    type: "bool",
    description: "Spawn this edge only when the parent part is terminal.",
  },
  "joint.edge.reflectX": {
    type: "bool",
    description: "Mirror this branch across local X.",
  },
};

const statusEl = document.getElementById("status");
const genomeEditor = document.getElementById("genomeEditor");
const creatureIdInput = document.getElementById("creatureIdInput");
const notesInput = document.getElementById("notesInput");
const creatureList = document.getElementById("creatureList");
const modeStatus = document.getElementById("modeStatus");
const layoutRoot = document.getElementById("layoutRoot");
const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
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
const inspectorPanelErrorEl = document.getElementById("inspectorPanelError");
const inspectorApplyStatusEl = document.getElementById("inspectorApplyStatus");

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

function setInspectorPanelError(message = "") {
  if (!inspectorPanelErrorEl) {
    return;
  }
  inspectorPanelErrorEl.textContent = message ? String(message) : "";
}

function setInspectorApplyStatus(message, cls = "") {
  if (!inspectorApplyStatusEl) {
    return;
  }
  inspectorApplyStatusEl.className = `inspectorApplyStatus ${cls}`.trim();
  inspectorApplyStatusEl.textContent = String(message || "Inspector ready.");
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

function setSidebarCollapsed(collapsed) {
  const isCollapsed = Boolean(collapsed);
  if (!layoutRoot || !toggleSidebarBtn) {
    return;
  }
  layoutRoot.classList.toggle("sidebarCollapsed", isCollapsed);
  toggleSidebarBtn.textContent = isCollapsed ? "Show controls" : "Hide controls";
  toggleSidebarBtn.setAttribute("aria-pressed", isCollapsed ? "true" : "false");
  if (visualView) {
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
  state.inspectorError = null;
  setInspectorPanelError("");
  setInspectorApplyStatus("Genome loaded into editor.", "ok");
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
  if (changed) {
    state.inspectorError = null;
    setInspectorPanelError("");
    if (nextSelection) {
      setInspectorApplyStatus("Editing live: changes apply immediately.", "ok");
    } else {
      setInspectorApplyStatus("Selection required for editing.", "warn");
    }
  }
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

function rebuildVisualFromCurrentGenome(options = {}) {
  const shouldFitCamera = options.fitCamera !== false;
  if (!visualView) {
    return;
  }
  if (!state.currentGenome) {
    visualInfo.textContent = "No genome loaded.";
    state.visualModel = null;
    setSelection(null);
    setInspectorApplyStatus("Load a genome to edit.", "warn");
    return;
  }
  state.visualModel = visualView.rebuild(state.currentGenome, { fitCamera: shouldFitCamera });
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

  const nodeByIndex = new Map(model.nodes.map((node) => [node.nodeIndex, node]));
  const jointByChildPart = new Map(model.joints.map((joint) => [joint.childPartId, joint]));
  const childrenByParentPart = new Map();
  for (const part of model.parts) {
    const parentKey = part.parentPartId || "__root__";
    if (!childrenByParentPart.has(parentKey)) {
      childrenByParentPart.set(parentKey, []);
    }
    childrenByParentPart.get(parentKey).push(part);
  }
  for (const list of childrenByParentPart.values()) {
    list.sort((a, b) => a.expandedIndex - b.expandedIndex);
  }
  const firstPartForNode = new Map();
  for (const node of model.nodes) {
    if (node.partIds.length > 0) {
      firstPartForNode.set(node.nodeIndex, node.partIds[0]);
    }
  }

  addSection(`Creature Hierarchy (${model.parts.length} parts, ${model.joints.length} joints)`);
  const roots = childrenByParentPart.get("__root__") || [];
  const renderPartSubtree = (part, depth) => {
    const node = nodeByIndex.get(part.nodeIndex);
    addItem({
      kind: "part",
      id: part.id,
      label: `${part.id}`,
      details: `node ${part.nodeIndex} | size ${fixed(part.size[0], 2)} x ${fixed(part.size[1], 2)} x ${fixed(part.size[2], 2)}`,
      depth,
      searchText: `part node ${part.nodeIndex} expanded ${part.expandedIndex}`,
    });

    if (node && firstPartForNode.get(part.nodeIndex) === part.id) {
      const brainNeurons = asArray(node.brain?.neurons).length;
      addItem({
        kind: "node",
        id: node.id,
        label: `template node ${node.nodeIndex}`,
        details: `${node.partIds.length} part instances | ${node.edgeCount} edges | ${brainNeurons} local neurons`,
        depth: depth + 1,
        searchText: `node ${node.nodeIndex} template ${node.partIds.join(" ")}`,
      });
    }

    const children = childrenByParentPart.get(part.id) || [];
    for (const child of children) {
      const joint = jointByChildPart.get(child.id);
      if (joint) {
        addItem({
          kind: "joint",
          id: joint.id,
          label: `${joint.id} (${joint.jointType})`,
          details: `${joint.parentPartId} -> ${joint.childPartId}`,
          depth: depth + 1,
          searchText: `joint ${joint.jointType} ${joint.parentPartId} ${joint.childPartId}`,
        });
      }
      renderPartSubtree(child, depth + 2);
    }
  };

  for (const rootPart of roots) {
    renderPartSubtree(rootPart, 0);
  }

  const unusedNodes = model.nodes.filter((node) => node.partIds.length === 0);
  if (unusedNodes.length > 0) {
    addSection(`Unused Node Templates (${unusedNodes.length})`);
    for (const node of unusedNodes) {
      const brainNeurons = asArray(node.brain?.neurons).length;
      addItem({
        kind: "node",
        id: node.id,
        label: `template node ${node.nodeIndex}`,
        details: `0 part instances | ${node.edgeCount} edges | ${brainNeurons} local neurons`,
        depth: 1,
        searchText: `unused node ${node.nodeIndex}`,
      });
    }
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

function hasOwn(target, key) {
  return Object.prototype.hasOwnProperty.call(target || {}, key);
}

function graphNodesFromState() {
  return asArray(state.currentGenome?.graph?.nodes);
}

function getNodeGene(nodeIndex) {
  const nodes = graphNodesFromState();
  return nodes[nodeIndex] || null;
}

function getEdgeGeneByRef(edgeRef) {
  if (!edgeRef || !Number.isInteger(edgeRef.fromNodeIndex) || !Number.isInteger(edgeRef.edgeIndex)) {
    return null;
  }
  const parentNode = getNodeGene(edgeRef.fromNodeIndex);
  const edges = asArray(parentNode?.edges);
  return edges[edgeRef.edgeIndex] || null;
}

function getPartMass(partGene) {
  if (!partGene || typeof partGene !== "object") {
    return 0;
  }
  if (Number.isFinite(Number(partGene.mass))) {
    return toFinite(partGene.mass, 0);
  }
  if (Number.isFinite(Number(partGene.m))) {
    return toFinite(partGene.m, 0);
  }
  return 0;
}

function setPartMass(partGene, value) {
  if (!partGene || typeof partGene !== "object") {
    return;
  }
  if (hasOwn(partGene, "mass") || !hasOwn(partGene, "m")) {
    partGene.mass = value;
  } else {
    partGene.m = value;
  }
}

function getEdgeStiffness(edgeGene) {
  if (!edgeGene || typeof edgeGene !== "object") {
    return 0;
  }
  if (Number.isFinite(Number(edgeGene.jointStiffness))) {
    return toFinite(edgeGene.jointStiffness, 0);
  }
  if (Number.isFinite(Number(edgeGene.stiffness))) {
    return toFinite(edgeGene.stiffness, 0);
  }
  return 0;
}

function setEdgeStiffness(edgeGene, value) {
  if (!edgeGene || typeof edgeGene !== "object") {
    return;
  }
  if (hasOwn(edgeGene, "jointStiffness") || !hasOwn(edgeGene, "stiffness")) {
    edgeGene.jointStiffness = value;
  } else {
    edgeGene.stiffness = value;
  }
}

function getEdgeLimit(edgeGene, axis) {
  if (!edgeGene || typeof edgeGene !== "object") {
    return 0;
  }
  const suffix = String(axis || "").toUpperCase();
  const limitKey = `limit${suffix}`;
  const maxKey = `jointLimitMax${suffix}`;
  const minKey = `jointLimitMin${suffix}`;
  if (Number.isFinite(Number(edgeGene[limitKey]))) {
    return toFinite(edgeGene[limitKey], 0);
  }
  if (Number.isFinite(Number(edgeGene[maxKey]))) {
    return Math.abs(toFinite(edgeGene[maxKey], 0));
  }
  if (Number.isFinite(Number(edgeGene[minKey]))) {
    return Math.abs(toFinite(edgeGene[minKey], 0));
  }
  return 0;
}

function setEdgeLimit(edgeGene, axis, value) {
  if (!edgeGene || typeof edgeGene !== "object") {
    return;
  }
  const suffix = String(axis || "").toUpperCase();
  const limitKey = `limit${suffix}`;
  const maxKey = `jointLimitMax${suffix}`;
  const minKey = `jointLimitMin${suffix}`;
  if (hasOwn(edgeGene, limitKey) || (!hasOwn(edgeGene, maxKey) && !hasOwn(edgeGene, minKey))) {
    edgeGene[limitKey] = value;
    return;
  }
  if (hasOwn(edgeGene, maxKey)) {
    edgeGene[maxKey] = value;
  }
  if (hasOwn(edgeGene, minKey)) {
    edgeGene[minKey] = -value;
  }
}

function setInspectorFieldError(selection, field, message) {
  state.inspectorError = {
    token: selectionToken(selection),
    field,
    message: String(message || "Invalid value."),
  };
  setInspectorPanelError(state.inspectorError.message);
  setInspectorApplyStatus("Edit rejected.", "err");
}

function clearInspectorFieldError(selection, field) {
  const current = state.inspectorError;
  if (!current) {
    return;
  }
  const token = selectionToken(selection);
  if (current.token !== token) {
    return;
  }
  if (field && current.field !== field) {
    return;
  }
  state.inspectorError = null;
  setInspectorPanelError("");
}

function inspectorFieldError(selection, field) {
  const current = state.inspectorError;
  if (!current) {
    return "";
  }
  if (current.token !== selectionToken(selection)) {
    return "";
  }
  if (current.field !== field) {
    return "";
  }
  return current.message || "";
}

function parseBooleanValue(rawValue) {
  if (typeof rawValue === "boolean") {
    return { ok: true, value: rawValue, clamped: false };
  }
  const text = String(rawValue ?? "").trim().toLowerCase();
  if (["true", "1", "yes", "on"].includes(text)) {
    return { ok: true, value: true, clamped: false };
  }
  if (["false", "0", "no", "off"].includes(text)) {
    return { ok: true, value: false, clamped: false };
  }
  return { ok: false, error: "Expected true or false." };
}

function validateFieldValue(field, rawValue) {
  const schema = FIELD_SCHEMAS[field];
  if (!schema) {
    return { ok: false, error: "Unsupported inspector field." };
  }
  if (schema.type === "bool") {
    return parseBooleanValue(rawValue);
  }
  if (schema.type === "enum") {
    const valueText = String(rawValue ?? "").trim().toLowerCase();
    const normalized = schema.options
      .find((option) => String(option).toLowerCase() === valueText);
    if (!normalized) {
      return { ok: false, error: `Expected one of: ${schema.options.join(", ")}.` };
    }
    return { ok: true, value: normalized, clamped: false };
  }
  if (schema.type === "int") {
    const text = String(rawValue ?? "").trim();
    if (!/^-?\d+$/.test(text)) {
      return { ok: false, error: "Expected an integer." };
    }
    const value = Number(text);
    if (!Number.isFinite(value)) {
      return { ok: false, error: "Expected a finite integer." };
    }
    const min = Number.isFinite(schema.min) ? schema.min : Number.MIN_SAFE_INTEGER;
    const max = Number.isFinite(schema.max) ? schema.max : Number.MAX_SAFE_INTEGER;
    const clamped = clampInt(value, min, max);
    return { ok: true, value: clamped, clamped: clamped !== value };
  }
  if (schema.type === "number") {
    const text = String(rawValue ?? "").trim();
    if (!text.length) {
      return { ok: false, error: "Expected a numeric value." };
    }
    const value = Number(text);
    if (!Number.isFinite(value)) {
      return { ok: false, error: "Expected a finite number." };
    }
    const min = Number.isFinite(schema.min) ? schema.min : -Number.MAX_VALUE;
    const max = Number.isFinite(schema.max) ? schema.max : Number.MAX_VALUE;
    const clamped = clamp(value, min, max);
    return { ok: true, value: clamped, clamped: clamped !== value };
  }
  return { ok: false, error: "Unsupported field type." };
}

function resolveFieldMutation(selection, entity, field) {
  if (!selection || !entity) {
    return null;
  }
  const nodePartFieldMap = {
    "node.part.w": "w",
    "node.part.h": "h",
    "node.part.d": "d",
    "part.part.w": "w",
    "part.part.h": "h",
    "part.part.d": "d",
  };
  const partField = nodePartFieldMap[field];
  if (partField) {
    const nodeGene = getNodeGene(entity.nodeIndex);
    if (!nodeGene?.part || typeof nodeGene.part !== "object") {
      return null;
    }
    return (value) => {
      nodeGene.part[partField] = value;
    };
  }
  if (field === "node.part.mass" || field === "part.part.mass") {
    const nodeGene = getNodeGene(entity.nodeIndex);
    if (!nodeGene?.part || typeof nodeGene.part !== "object") {
      return null;
    }
    return (value) => {
      setPartMass(nodeGene.part, value);
    };
  }
  const nodeEffectorMap = {
    "node.brain.effectorX.gain": "effectorX",
    "node.brain.effectorY.gain": "effectorY",
    "node.brain.effectorZ.gain": "effectorZ",
  };
  const effectorKey = nodeEffectorMap[field];
  if (effectorKey) {
    const nodeGene = getNodeGene(entity.nodeIndex);
    const effector = nodeGene?.brain?.[effectorKey];
    if (!effector || typeof effector !== "object") {
      return null;
    }
    return (value) => {
      effector.gain = value;
    };
  }
  if (field.startsWith("part.edge.")) {
    const edgeGene = getEdgeGeneByRef(entity.incomingEdgeRef);
    if (!edgeGene || typeof edgeGene !== "object") {
      return null;
    }
    const key = field.slice("part.edge.".length);
    return (value) => {
      edgeGene[key] = value;
    };
  }
  if (field.startsWith("joint.edge.")) {
    const edgeGene = getEdgeGeneByRef(entity.edgeRef);
    if (!edgeGene || typeof edgeGene !== "object") {
      return null;
    }
    if (field === "joint.edge.stiffness") {
      return (value) => {
        setEdgeStiffness(edgeGene, value);
      };
    }
    if (field === "joint.edge.limitX") {
      return (value) => {
        setEdgeLimit(edgeGene, "X", value);
      };
    }
    if (field === "joint.edge.limitY") {
      return (value) => {
        setEdgeLimit(edgeGene, "Y", value);
      };
    }
    if (field === "joint.edge.limitZ") {
      return (value) => {
        setEdgeLimit(edgeGene, "Z", value);
      };
    }
    const key = field.slice("joint.edge.".length);
    return (value) => {
      edgeGene[key] = value;
    };
  }
  return null;
}

function applyFieldEdit(selection, field, rawValue) {
  const validSelection = selection && selection.kind && selection.id
    ? { kind: selection.kind, id: selection.id }
    : state.selection;
  if (!state.currentGenome || !validSelection) {
    setInspectorApplyStatus("No active selection for edit.", "warn");
    return false;
  }
  const model = state.visualModel;
  const entity = findSelectionEntity(model, validSelection);
  if (!entity) {
    setInspectorFieldError(validSelection, field, "Selection is stale.");
    renderInspector();
    return false;
  }
  const parsed = validateFieldValue(field, rawValue);
  if (!parsed.ok) {
    setInspectorFieldError(validSelection, field, parsed.error || "Invalid value.");
    renderInspector();
    return false;
  }
  const mutate = resolveFieldMutation(validSelection, entity, field);
  if (!mutate) {
    setInspectorFieldError(validSelection, field, "Field is unavailable for this selection.");
    renderInspector();
    return false;
  }

  mutate(parsed.value);
  clearInspectorFieldError(validSelection, field);
  genomeEditor.value = JSON.stringify(state.currentGenome, null, 2);
  rebuildVisualFromCurrentGenome({ fitCamera: false });
  if (validSelection) {
    setSelection(validSelection);
  }
  if (parsed.clamped) {
    setInspectorApplyStatus(`Updated ${field} (clamped).`, "warn");
    setStatus(`Updated ${field} with clamp.`, "warn");
  } else {
    setInspectorApplyStatus(`Updated ${field}.`, "ok");
    setStatus(`Updated ${field}.`, "ok");
  }
  return true;
}

function fieldClampHint(field) {
  const schema = FIELD_SCHEMAS[field];
  if (!schema) {
    return "";
  }
  if (schema.type === "number" || schema.type === "int") {
    const parts = [schema.type];
    if (Number.isFinite(schema.min) && Number.isFinite(schema.max)) {
      parts.push(`[${schema.min} .. ${schema.max}]`);
    }
    if (Number.isFinite(schema.step)) {
      parts.push(`step ${schema.step}`);
    }
    return parts.join(" ");
  }
  if (schema.type === "enum") {
    return `enum: ${schema.options.join(" | ")}`;
  }
  if (schema.type === "bool") {
    return "bool: true | false";
  }
  return "";
}

function fieldDescription(field) {
  const schema = FIELD_SCHEMAS[field];
  if (!schema || typeof schema.description !== "string") {
    return "";
  }
  return schema.description.trim();
}

function fieldTooltipText(field) {
  const rangeHint = fieldClampHint(field);
  const description = fieldDescription(field);
  if (rangeHint && description) {
    return `${rangeHint}\n${description}`;
  }
  return description || rangeHint;
}

function fieldClampNumberValue(schema, rawValue) {
  const min = Number.isFinite(schema.min) ? schema.min : -Number.MAX_VALUE;
  const max = Number.isFinite(schema.max) ? schema.max : Number.MAX_VALUE;
  const fallback = Number.isFinite(schema.min) ? schema.min : 0;
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) {
    return schema.type === "int"
      ? clampInt(fallback, min, max)
      : clamp(fallback, min, max);
  }
  return schema.type === "int"
    ? clampInt(parsed, min, max)
    : clamp(parsed, min, max);
}

function parseNumberDraftForSlider(schema, rawValue) {
  const text = String(rawValue ?? "").trim();
  if (!text.length) {
    return null;
  }
  if (schema.type === "int" && !/^-?\d+$/.test(text)) {
    return null;
  }
  const parsed = Number(text);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return fieldClampNumberValue(schema, parsed);
}

function editableFieldInput(selection, field, value) {
  const schema = FIELD_SCHEMAS[field];
  if (!schema) {
    return null;
  }
  const tooltipText = fieldTooltipText(field);
  if (schema.type === "enum") {
    const select = document.createElement("select");
    for (const optionValue of schema.options) {
      const option = document.createElement("option");
      option.value = String(optionValue);
      option.textContent = String(optionValue);
      select.appendChild(option);
    }
    select.value = String(value ?? schema.options[0] ?? "");
    if (tooltipText) {
      select.title = tooltipText;
    }
    select.addEventListener("change", () => {
      applyFieldEdit(selection, field, select.value);
    });
    return select;
  }
  if (schema.type === "bool") {
    const wrapper = document.createElement("label");
    wrapper.className = "inspectorCheckboxRow";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(value);
    const text = document.createElement("span");
    text.textContent = checkbox.checked ? "true" : "false";
    if (tooltipText) {
      wrapper.title = tooltipText;
      checkbox.title = tooltipText;
      text.title = tooltipText;
    }
    checkbox.addEventListener("change", () => {
      text.textContent = checkbox.checked ? "true" : "false";
      applyFieldEdit(selection, field, checkbox.checked);
    });
    wrapper.appendChild(checkbox);
    wrapper.appendChild(text);
    return wrapper;
  }
  if ((schema.type === "number" || schema.type === "int")
    && Number.isFinite(schema.min)
    && Number.isFinite(schema.max)) {
    const combo = document.createElement("div");
    combo.className = "inspectorNumericCombo";
    if (tooltipText) {
      combo.title = tooltipText;
    }

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "inspectorSlider";
    slider.min = String(schema.min);
    slider.max = String(schema.max);
    if (Number.isFinite(schema.step)) {
      slider.step = String(schema.step);
    }

    const numberInput = document.createElement("input");
    numberInput.type = "number";
    numberInput.className = "inspectorNumberInput";
    numberInput.placeholder = fieldClampHint(field);
    if (Number.isFinite(schema.step)) {
      numberInput.step = String(schema.step);
    }
    if (Number.isFinite(schema.min)) {
      numberInput.min = String(schema.min);
    }
    if (Number.isFinite(schema.max)) {
      numberInput.max = String(schema.max);
    }
    if (tooltipText) {
      slider.title = tooltipText;
      numberInput.title = tooltipText;
    }

    const initial = fieldClampNumberValue(schema, value);
    const initialText = String(initial);
    slider.value = initialText;
    numberInput.value = initialText;

    slider.addEventListener("input", () => {
      numberInput.value = slider.value;
    });
    slider.addEventListener("change", () => {
      numberInput.value = slider.value;
      applyFieldEdit(selection, field, slider.value);
    });

    numberInput.addEventListener("input", () => {
      const sliderValue = parseNumberDraftForSlider(schema, numberInput.value);
      if (sliderValue === null) {
        return;
      }
      slider.value = String(sliderValue);
    });

    const commitNumberInput = () => {
      applyFieldEdit(selection, field, numberInput.value);
    };
    numberInput.addEventListener("change", commitNumberInput);
    numberInput.addEventListener("blur", commitNumberInput);
    numberInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitNumberInput();
      }
    });

    combo.appendChild(slider);
    combo.appendChild(numberInput);
    return combo;
  }

  const input = document.createElement("input");
  input.type = "number";
  input.value = Number.isFinite(Number(value)) ? String(value) : "";
  input.placeholder = fieldClampHint(field);
  if (Number.isFinite(schema.step)) {
    input.step = String(schema.step);
  }
  if (Number.isFinite(schema.min)) {
    input.min = String(schema.min);
  }
  if (Number.isFinite(schema.max)) {
    input.max = String(schema.max);
  }
  if (tooltipText) {
    input.title = tooltipText;
  }
  const commit = () => {
    applyFieldEdit(selection, field, input.value);
  };
  input.addEventListener("change", commit);
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commit();
    }
  });
  return input;
}

function appendEditableField(container, selection, key, field, value) {
  const row = document.createElement("div");
  row.className = "fieldRow inspectorFieldRow";
  const keyEl = document.createElement("div");
  keyEl.className = "k inspectorFieldLabel";
  const keyText = document.createElement("div");
  keyText.className = "inspectorFieldName";
  keyText.textContent = key;
  const tooltipText = fieldTooltipText(field);
  if (tooltipText) {
    keyText.classList.add("hasHelp");
    keyText.title = tooltipText;
  }
  keyEl.appendChild(keyText);

  const valEl = document.createElement("div");
  valEl.className = "v inspectorFieldValue";
  const controlWrap = document.createElement("div");
  controlWrap.className = "inspectorFieldControl";
  if (tooltipText) {
    controlWrap.title = tooltipText;
  }
  const input = editableFieldInput(selection, field, value);
  if (input) {
    controlWrap.appendChild(input);
  } else {
    controlWrap.textContent = String(value ?? "");
  }
  const inlineError = inspectorFieldError(selection, field);
  if (inlineError) {
    row.classList.add("invalid");
    const errorEl = document.createElement("div");
    errorEl.className = "inspectorFieldError";
    errorEl.textContent = inlineError;
    controlWrap.appendChild(errorEl);
  }
  valEl.appendChild(controlWrap);
  row.appendChild(keyEl);
  row.appendChild(valEl);
  container.appendChild(row);
}

function renderNodeInspector(entity, selection) {
  const nodeGene = getNodeGene(entity.nodeIndex) || {};
  const partGene = nodeGene.part || {};
  const brain = nodeGene.brain || {};
  const effectorKeys = ["effectorX", "effectorY", "effectorZ"];
  const effectorCount = effectorKeys.filter((key) => brain[key] && typeof brain[key] === "object").length;
  appendField(inspectorContent, "node index", String(entity.nodeIndex));
  appendField(inspectorContent, "part instances", String(entity.partIds.length));
  appendField(inspectorContent, "edge genes", String(entity.edgeCount));
  appendEditableField(inspectorContent, selection, "part w", "node.part.w", toFinite(partGene.w, 0.7));
  appendEditableField(inspectorContent, selection, "part h", "node.part.h", toFinite(partGene.h, 1.0));
  appendEditableField(inspectorContent, selection, "part d", "node.part.d", toFinite(partGene.d, 0.7));
  appendEditableField(inspectorContent, selection, "part mass", "node.part.mass", getPartMass(partGene));
  appendField(inspectorContent, "part ids", entity.partIds.join(", ") || "none");
  appendField(inspectorContent, "brain neurons (read-only)", String(asArray(brain.neurons).length));
  appendField(inspectorContent, "brain effectors", String(effectorCount));
  if (brain.effectorX && typeof brain.effectorX === "object") {
    appendEditableField(inspectorContent, selection, "effectorX gain", "node.brain.effectorX.gain", toFinite(brain.effectorX.gain, 1.0));
  } else {
    appendField(inspectorContent, "effectorX gain", "n/a");
  }
  if (brain.effectorY && typeof brain.effectorY === "object") {
    appendEditableField(inspectorContent, selection, "effectorY gain", "node.brain.effectorY.gain", toFinite(brain.effectorY.gain, 1.0));
  } else {
    appendField(inspectorContent, "effectorY gain", "n/a");
  }
  if (brain.effectorZ && typeof brain.effectorZ === "object") {
    appendEditableField(inspectorContent, selection, "effectorZ gain", "node.brain.effectorZ.gain", toFinite(brain.effectorZ.gain, 1.0));
  } else {
    appendField(inspectorContent, "effectorZ gain", "n/a");
  }
}

function renderPartInspector(entity, selection) {
  const nodeGene = getNodeGene(entity.nodeIndex) || {};
  const partGene = nodeGene.part || {};
  appendField(inspectorContent, "part id", entity.id);
  appendField(inspectorContent, "expanded index", String(entity.expandedIndex));
  appendField(inspectorContent, "node index", String(entity.nodeIndex));
  appendField(inspectorContent, "parent", entity.parentPartId || "none (root)");
  appendField(inspectorContent, "depth", String(entity.depth || 0));
  appendField(inspectorContent, "expanded size w/h/d", `${fixed(entity.size[0], 3)} x ${fixed(entity.size[1], 3)} x ${fixed(entity.size[2], 3)}`);
  appendField(inspectorContent, "center", formatVec3(entity.center, 3));
  appendEditableField(inspectorContent, selection, "node part w", "part.part.w", toFinite(partGene.w, 0.7));
  appendEditableField(inspectorContent, selection, "node part h", "part.part.h", toFinite(partGene.h, 1.0));
  appendEditableField(inspectorContent, selection, "node part d", "part.part.d", toFinite(partGene.d, 0.7));
  appendEditableField(inspectorContent, selection, "node part mass", "part.part.mass", getPartMass(partGene));

  if (!entity.incomingEdgeRef) {
    appendField(inspectorContent, "incoming edge", "none (root)");
    return;
  }
  const edgeGene = getEdgeGeneByRef(entity.incomingEdgeRef);
  if (!edgeGene) {
    appendField(inspectorContent, "incoming edge", "unavailable");
    return;
  }
  appendField(inspectorContent, "incoming joint", String(edgeGene.jointType || "hinge"));
  appendEditableField(inspectorContent, selection, "incoming scale", "part.edge.scale", toFinite(edgeGene.scale, 1));
  appendEditableField(inspectorContent, selection, "anchor X", "part.edge.anchorX", toFinite(edgeGene.anchorX, 0));
  appendEditableField(inspectorContent, selection, "anchor Y", "part.edge.anchorY", toFinite(edgeGene.anchorY, 0));
  appendEditableField(inspectorContent, selection, "anchor Z", "part.edge.anchorZ", toFinite(edgeGene.anchorZ, 0));
  appendEditableField(inspectorContent, selection, "dir X", "part.edge.dirX", toFinite(edgeGene.dirX, 0));
  appendEditableField(inspectorContent, selection, "dir Y", "part.edge.dirY", toFinite(edgeGene.dirY, -1));
  appendEditableField(inspectorContent, selection, "dir Z", "part.edge.dirZ", toFinite(edgeGene.dirZ, 0));
  appendEditableField(inspectorContent, selection, "axis Y", "part.edge.axisY", toFinite(edgeGene.axisY, 0));
  appendEditableField(inspectorContent, selection, "axis Z", "part.edge.axisZ", toFinite(edgeGene.axisZ, 0));
}

function renderJointInspector(entity, selection) {
  appendField(inspectorContent, "joint id", entity.id);
  appendField(inspectorContent, "parent part", entity.parentPartId);
  appendField(inspectorContent, "child part", entity.childPartId);
  appendField(inspectorContent, "anchor world", formatVec3(entity.anchorWorld, 3));
  appendField(inspectorContent, "axis world", formatVec3(entity.axisWorld, 3));
  const edgeGene = getEdgeGeneByRef(entity.edgeRef);
  if (!edgeGene) {
    appendField(inspectorContent, "joint edge gene", "unavailable");
    return;
  }
  appendEditableField(inspectorContent, selection, "joint type", "joint.edge.jointType", String(edgeGene.jointType || "hinge"));
  appendEditableField(inspectorContent, selection, "limit X", "joint.edge.limitX", getEdgeLimit(edgeGene, "X"));
  appendEditableField(inspectorContent, selection, "limit Y", "joint.edge.limitY", getEdgeLimit(edgeGene, "Y"));
  appendEditableField(inspectorContent, selection, "limit Z", "joint.edge.limitZ", getEdgeLimit(edgeGene, "Z"));
  appendEditableField(inspectorContent, selection, "motor strength", "joint.edge.motorStrength", toFinite(edgeGene.motorStrength, 1.0));
  appendEditableField(inspectorContent, selection, "stiffness", "joint.edge.stiffness", getEdgeStiffness(edgeGene));
  appendEditableField(inspectorContent, selection, "recursive limit", "joint.edge.recursiveLimit", clampInt(edgeGene.recursiveLimit ?? 1, 1, 5));
  appendEditableField(inspectorContent, selection, "terminal only", "joint.edge.terminalOnly", Boolean(edgeGene.terminalOnly));
  appendEditableField(inspectorContent, selection, "reflect X", "joint.edge.reflectX", Boolean(edgeGene.reflectX));
}

function renderInspector() {
  inspectorContent.innerHTML = "";
  const model = state.visualModel;
  const selection = state.selection;

  if (!model) {
    selectionBreadcrumb.textContent = "Nothing selected.";
    appendEmpty(inspectorContent, "No model loaded.");
    setInspectorPanelError("");
    setInspectorApplyStatus("Load a genome to edit.", "warn");
    return;
  }

  if (!selection) {
    selectionBreadcrumb.textContent = "Nothing selected.";
    appendEmpty(inspectorContent, "Select a node, part, or joint in the viewport or outliner.");
    setInspectorPanelError("");
    setInspectorApplyStatus("Selection required for editing.", "warn");
    return;
  }

  const entity = findSelectionEntity(model, selection);
  if (!entity) {
    selectionBreadcrumb.textContent = "Nothing selected.";
    appendEmpty(inspectorContent, "Selection is stale after rebuild.");
    setInspectorPanelError("");
    setInspectorApplyStatus("Selection became stale after rebuild.", "warn");
    return;
  }

  selectionBreadcrumb.textContent = `Selected: ${selection.kind} / ${selection.id}`;
  setInspectorPanelError(inspectorFieldError(selection, state.inspectorError?.field || ""));

  if (selection.kind === "node") {
    renderNodeInspector(entity, selection);
    return;
  }
  if (selection.kind === "part") {
    renderPartInspector(entity, selection);
    return;
  }
  if (selection.kind === "joint") {
    renderJointInspector(entity, selection);
  }
}

function rebuildVisualFromEditor() {
  try {
    const genome = parseGenomeEditor();
    setGenomeState(genome);
    rebuildVisualFromCurrentGenome({ fitCamera: false });
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
    return [{ nodeIndex: 0, parentIndex: null, incomingEdge: null, incomingEdgeRef: null }];
  }
  const root = clampInt(graph?.root ?? 0, 0, nodes.length - 1);
  const maxParts = clampInt(graph?.maxParts ?? 32, 1, MAX_GRAPH_PARTS);
  const expanded = [{
    nodeIndex: root,
    parentIndex: null,
    incomingEdge: null,
    incomingEdgeRef: null,
    depth: 0,
  }];
  const queue = [{ expandedIndex: 0, nodeIndex: root, ancestry: [root], depth: 0 }];
  while (queue.length && expanded.length < maxParts) {
    const current = queue.shift();
    const node = nodes[current.nodeIndex];
    if (!node) {
      continue;
    }
    const edges = asArray(node.edges).slice(0, MAX_GRAPH_EDGES_PER_NODE);
    for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += 1) {
      const edge = edges[edgeIndex];
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
        incomingEdgeRef: { fromNodeIndex: current.nodeIndex, edgeIndex },
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
        incomingEdgeRef: { fromNodeIndex: root, edgeIndex: 0 },
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
    incomingEdgeRef: null,
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
    const incomingEdgeRef = expandedPart.incomingEdgeRef
      ? { ...expandedPart.incomingEdgeRef }
      : null;
    parts.push({
      id: childPartId,
      expandedIndex,
      depth: expandedPart.depth ?? 0,
      nodeId: nodeEntries[nodeIndex].id,
      nodeIndex,
      parentPartId: parent.id,
      parentPartIndex,
      incomingEdge,
      incomingEdgeRef,
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
      edgeRef: incomingEdgeRef,
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

  rebuild(genome, options = {}) {
    const shouldFitCamera = options.fitCamera !== false;
    this.clearGroup(this.segmentGroup);
    this.clearGroup(this.jointGroup);
    this.selectionObjects.clear();
    this.pickables = [];

    const model = buildVisualModel(genome);
    this.latestModel = model;
    if (!model.parts.length) {
      this.infoEl.textContent = "No graph nodes to render. Load a genome with graph nodes.";
      if (shouldFitCamera) {
        this.fitCamera();
      }
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
    if (shouldFitCamera) {
      this.fitCamera();
    }
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
if (toggleSidebarBtn && layoutRoot) {
  toggleSidebarBtn.addEventListener("click", () => {
    const nextCollapsed = !layoutRoot.classList.contains("sidebarCollapsed");
    setSidebarCollapsed(nextCollapsed);
  });
}
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
setSidebarCollapsed(false);
setEditorMode("visual");
rebuildVisualFromCurrentGenome();
renderOutliner();
renderInspector();
void refreshCreatureList();
void refreshModeStatus();
