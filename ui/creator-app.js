import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const AXIS_TILT_GAIN = 1.9;
const EDGE_OUTWARD_GROWTH_MIN_DOT = 0.22;
const MAX_GRAPH_PARTS = 72;
const MAX_GRAPH_EDGES_PER_NODE = 4;
const LOCAL_SENSOR_DIM = 14;
const MIN_LOCAL_NEURONS = 2;
const MAX_LOCAL_NEURONS = 4;
const MIN_GLOBAL_NEURONS = 2;
const MAX_GLOBAL_NEURONS = 4;
const DEFAULT_OUTLINER_PANEL_WIDTH = 260;
const DEFAULT_INSPECTOR_PANEL_WIDTH = 300;
const MIN_OUTLINER_PANEL_WIDTH = 200;
const MAX_OUTLINER_PANEL_WIDTH = 520;
const MIN_INSPECTOR_PANEL_WIDTH = 240;
const MAX_INSPECTOR_PANEL_WIDTH = 560;
const MIN_VIEWPORT_PANEL_WIDTH = 360;
const PANEL_RESIZER_WIDTH = 8;
const PANEL_WIDTHS_STORAGE_KEY = "creator.panelWidths.v1";
const VISUAL_TOOL_SELECT = "select";
const VISUAL_TOOL_MOVE_ANCHOR = "moveAnchor";
const VISUAL_TOOL_GROWTH_DIR = "growthDir";

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
  visualTool: VISUAL_TOOL_SELECT,
  selection: null,
  inspectorError: null,
};
let paneResizeSession = null;

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
  "node.brain.neuronCount": {
    type: "int",
    min: MIN_LOCAL_NEURONS,
    max: MAX_LOCAL_NEURONS,
    step: 1,
    description: "Number of local controller neurons for this node template.",
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
  "edge.to": {
    type: "int",
    min: 0,
    max: MAX_GRAPH_PARTS - 1,
    step: 1,
    description: "Target node template index this edge connects to.",
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
const visualWorkspace = document.getElementById("visualWorkspace");
const outlinerPanel = document.getElementById("outlinerPanel");
const inspectorPanel = document.getElementById("inspectorPanel");
const outlinerResizeHandle = document.getElementById("outlinerResizeHandle");
const inspectorResizeHandle = document.getElementById("inspectorResizeHandle");
const jsonPane = document.getElementById("jsonPane");
const showVisualBtn = document.getElementById("showVisualBtn");
const showJsonBtn = document.getElementById("showJsonBtn");
const rebuildVisualBtn = document.getElementById("rebuildVisualBtn");
const toolSelectBtn = document.getElementById("toolSelectBtn");
const toolMoveAnchorBtn = document.getElementById("toolMoveAnchorBtn");
const toolGrowthDirBtn = document.getElementById("toolGrowthDirBtn");
const addChildBtn = document.getElementById("addChildBtn");
const duplicateBranchBtn = document.getElementById("duplicateBranchBtn");
const deleteSelectedBtn = document.getElementById("deleteSelectedBtn");
const toolStatusEl = document.getElementById("toolStatus");
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

function clampBySchema(field, value, fallback = 0) {
  const schema = FIELD_SCHEMAS[field];
  if (!schema || (schema.type !== "number" && schema.type !== "int")) {
    return toFinite(value, fallback);
  }
  return fieldClampNumberValue(schema, value);
}

function safeUnitVector(vec, fallback = new THREE.Vector3(0, -1, 0)) {
  const next = vec?.clone?.() || new THREE.Vector3();
  if (next.lengthSq() <= 1e-8) {
    return fallback.clone();
  }
  next.normalize();
  return next;
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

function visualToolLabel(tool) {
  if (tool === VISUAL_TOOL_MOVE_ANCHOR) {
    return "move anchor";
  }
  if (tool === VISUAL_TOOL_GROWTH_DIR) {
    return "growth handle";
  }
  return "select";
}

function refreshVisualToolUi() {
  const canEdgeEdit = Boolean(getVisualEdgeContext());
  if (!canEdgeEdit && state.visualTool !== VISUAL_TOOL_SELECT) {
    state.visualTool = VISUAL_TOOL_SELECT;
    if (visualView) {
      visualView.setEditingTool(VISUAL_TOOL_SELECT);
    }
  }
  if (toolSelectBtn) {
    toolSelectBtn.classList.toggle("active", state.visualTool === VISUAL_TOOL_SELECT);
  }
  if (toolMoveAnchorBtn) {
    toolMoveAnchorBtn.classList.toggle("active", state.visualTool === VISUAL_TOOL_MOVE_ANCHOR);
    toolMoveAnchorBtn.disabled = !canEdgeEdit;
    toolMoveAnchorBtn.title = canEdgeEdit
      ? "Drag the anchor gizmo to move this branch attachment."
      : "Select a non-root part or a joint to edit anchor placement.";
  }
  if (toolGrowthDirBtn) {
    toolGrowthDirBtn.classList.toggle("active", state.visualTool === VISUAL_TOOL_GROWTH_DIR);
    toolGrowthDirBtn.disabled = !canEdgeEdit;
    toolGrowthDirBtn.title = canEdgeEdit
      ? "Drag the growth handle to adjust branch growth direction."
      : "Select a non-root part or a joint to edit growth direction.";
  }
  if (toolStatusEl) {
    toolStatusEl.textContent = `Tool: ${visualToolLabel(state.visualTool)}`;
  }
  if (addChildBtn) {
    const canAdd = Boolean(resolveAddChildParentPart());
    addChildBtn.disabled = !canAdd;
    addChildBtn.title = canAdd
      ? "Create a new child segment from the selected part/joint."
      : "Select a part, node, or joint to add a child segment.";
  }
  if (duplicateBranchBtn) {
    const canDuplicate = Boolean(resolveDuplicateTarget());
    duplicateBranchBtn.disabled = !canDuplicate;
    duplicateBranchBtn.title = canDuplicate
      ? "Duplicate selected branch as an independent template branch."
      : "Select an edge, non-root part/joint, or instantiated node to duplicate its branch.";
  }
  if (deleteSelectedBtn) {
    const canDelete = Boolean(resolveDeleteTarget());
    deleteSelectedBtn.disabled = !canDelete;
    deleteSelectedBtn.title = canDelete
      ? "Delete selected branch/template."
      : "Select an edge, non-root part/joint branch, or an unused node template to delete.";
  }
}

function setVisualTool(tool) {
  const next = tool === VISUAL_TOOL_MOVE_ANCHOR || tool === VISUAL_TOOL_GROWTH_DIR
    ? tool
    : VISUAL_TOOL_SELECT;
  const canEdgeEdit = Boolean(getVisualEdgeContext());
  if (next !== VISUAL_TOOL_SELECT && !canEdgeEdit) {
    state.visualTool = VISUAL_TOOL_SELECT;
    refreshVisualToolUi();
    if (visualView) {
      visualView.setEditingTool(VISUAL_TOOL_SELECT);
    }
    setStatus("Select a non-root part or a joint to use edge edit tools.", "warn");
    renderInspector();
    return;
  }
  state.visualTool = next;
  refreshVisualToolUi();
  if (visualView) {
    visualView.setEditingTool(next);
  }
  renderInspector();
}

function setSidebarCollapsed(collapsed) {
  const isCollapsed = Boolean(collapsed);
  if (!layoutRoot || !toggleSidebarBtn) {
    return;
  }
  layoutRoot.classList.toggle("sidebarCollapsed", isCollapsed);
  toggleSidebarBtn.textContent = isCollapsed ? "Show controls" : "Hide controls";
  toggleSidebarBtn.setAttribute("aria-pressed", isCollapsed ? "true" : "false");
  requestAnimationFrame(() => {
    const current = getCurrentPanelWidths();
    applyPanelWidths(current.outliner, current.inspector, false);
    if (visualView) {
      visualView.resize();
    }
  });
}

function parseStoredPanelWidths() {
  try {
    const raw = localStorage.getItem(PANEL_WIDTHS_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    const outliner = toFinite(parsed?.outliner, NaN);
    const inspector = toFinite(parsed?.inspector, NaN);
    if (!Number.isFinite(outliner) || !Number.isFinite(inspector)) {
      return null;
    }
    return { outliner, inspector };
  } catch {
    return null;
  }
}

function getCurrentPanelWidths() {
  const outliner = outlinerPanel?.getBoundingClientRect?.().width ?? DEFAULT_OUTLINER_PANEL_WIDTH;
  const inspector = inspectorPanel?.getBoundingClientRect?.().width ?? DEFAULT_INSPECTOR_PANEL_WIDTH;
  return {
    outliner: Number.isFinite(outliner) ? outliner : DEFAULT_OUTLINER_PANEL_WIDTH,
    inspector: Number.isFinite(inspector) ? inspector : DEFAULT_INSPECTOR_PANEL_WIDTH,
  };
}

function clampPanelWidths(outlinerWidth, inspectorWidth) {
  const workspaceWidth = visualWorkspace?.getBoundingClientRect?.().width ?? 0;
  const resizerTotal = PANEL_RESIZER_WIDTH * 2;
  let outliner = toFinite(outlinerWidth, DEFAULT_OUTLINER_PANEL_WIDTH);
  let inspector = toFinite(inspectorWidth, DEFAULT_INSPECTOR_PANEL_WIDTH);

  if (workspaceWidth > 0) {
    const maxOutlinerBySpace = Math.max(
      MIN_OUTLINER_PANEL_WIDTH,
      workspaceWidth - resizerTotal - MIN_VIEWPORT_PANEL_WIDTH - MIN_INSPECTOR_PANEL_WIDTH
    );
    outliner = clamp(
      outliner,
      MIN_OUTLINER_PANEL_WIDTH,
      Math.min(MAX_OUTLINER_PANEL_WIDTH, maxOutlinerBySpace)
    );
    const maxInspectorBySpace = Math.max(
      MIN_INSPECTOR_PANEL_WIDTH,
      workspaceWidth - resizerTotal - MIN_VIEWPORT_PANEL_WIDTH - outliner
    );
    inspector = clamp(
      inspector,
      MIN_INSPECTOR_PANEL_WIDTH,
      Math.min(MAX_INSPECTOR_PANEL_WIDTH, maxInspectorBySpace)
    );
    const maxOutlinerAfterInspector = Math.max(
      MIN_OUTLINER_PANEL_WIDTH,
      workspaceWidth - resizerTotal - MIN_VIEWPORT_PANEL_WIDTH - inspector
    );
    outliner = clamp(
      outliner,
      MIN_OUTLINER_PANEL_WIDTH,
      Math.min(MAX_OUTLINER_PANEL_WIDTH, maxOutlinerAfterInspector)
    );
  } else {
    outliner = clamp(outliner, MIN_OUTLINER_PANEL_WIDTH, MAX_OUTLINER_PANEL_WIDTH);
    inspector = clamp(inspector, MIN_INSPECTOR_PANEL_WIDTH, MAX_INSPECTOR_PANEL_WIDTH);
  }

  return { outliner, inspector };
}

function applyPanelWidths(outlinerWidth, inspectorWidth, persist = true) {
  if (!visualWorkspace) {
    return;
  }
  const clamped = clampPanelWidths(outlinerWidth, inspectorWidth);
  visualWorkspace.style.setProperty("--outliner-width", `${Math.round(clamped.outliner)}px`);
  visualWorkspace.style.setProperty("--inspector-width", `${Math.round(clamped.inspector)}px`);
  if (persist) {
    try {
      localStorage.setItem(PANEL_WIDTHS_STORAGE_KEY, JSON.stringify(clamped));
    } catch {
      // Ignore storage failures.
    }
  }
}

function beginPanelResize(kind, event) {
  if (!visualWorkspace || !outlinerPanel || !inspectorPanel) {
    return;
  }
  if (window.matchMedia("(max-width: 1020px)").matches) {
    return;
  }
  const startWidths = getCurrentPanelWidths();
  paneResizeSession = {
    kind,
    startX: event.clientX,
    startOutliner: startWidths.outliner,
    startInspector: startWidths.inspector,
  };
  document.body.classList.add("resizingPanes");
  if (kind === "outliner") {
    outlinerResizeHandle?.classList.add("active");
  } else {
    inspectorResizeHandle?.classList.add("active");
  }
  window.addEventListener("pointermove", onPanelResizeMove);
  window.addEventListener("pointerup", endPanelResize);
  window.addEventListener("pointercancel", endPanelResize);
  event.preventDefault();
}

function onPanelResizeMove(event) {
  if (!paneResizeSession) {
    return;
  }
  const deltaX = event.clientX - paneResizeSession.startX;
  let outliner = paneResizeSession.startOutliner;
  let inspector = paneResizeSession.startInspector;
  if (paneResizeSession.kind === "outliner") {
    outliner = paneResizeSession.startOutliner + deltaX;
  } else {
    inspector = paneResizeSession.startInspector - deltaX;
  }
  applyPanelWidths(outliner, inspector, false);
  if (visualView) {
    visualView.resize();
  }
}

function endPanelResize() {
  if (!paneResizeSession) {
    return;
  }
  paneResizeSession = null;
  document.body.classList.remove("resizingPanes");
  outlinerResizeHandle?.classList.remove("active");
  inspectorResizeHandle?.classList.remove("active");
  window.removeEventListener("pointermove", onPanelResizeMove);
  window.removeEventListener("pointerup", endPanelResize);
  window.removeEventListener("pointercancel", endPanelResize);
  const current = getCurrentPanelWidths();
  applyPanelWidths(current.outliner, current.inspector, true);
  if (visualView) {
    visualView.resize();
  }
}

function initializePanelWidths() {
  if (!visualWorkspace) {
    return;
  }
  const stored = parseStoredPanelWidths();
  const defaults = stored || {
    outliner: DEFAULT_OUTLINER_PANEL_WIDTH,
    inspector: DEFAULT_INSPECTOR_PANEL_WIDTH,
  };
  applyPanelWidths(defaults.outliner, defaults.inspector, false);
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
  if (selection.kind === "edge") {
    return model.edges.find((item) => item.id === selection.id) || null;
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
  refreshVisualEditBindings();
  refreshVisualToolUi();
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

function syncGenomeEditorFromState() {
  if (!state.currentGenome) {
    genomeEditor.value = "";
    return;
  }
  genomeEditor.value = JSON.stringify(state.currentGenome, null, 2);
}

function refreshVisualEditBindings() {
  if (!visualView) {
    return;
  }
  visualView.setEditingTool(state.visualTool);
  const edgeContext = getVisualEdgeContext();
  if (!edgeContext) {
    visualView.setEditingContext(null);
    return;
  }
  const childDir = safeUnitVector(
    edgeContext.childPart.center.clone().sub(edgeContext.frame.anchorWorld),
    edgeContext.frame.growthWorld.clone()
  );
  const growthDir = safeUnitVector(edgeContext.frame.growthWorld, childDir.clone());
  const handleDistance = Math.max(
    0.28,
    toFinite(edgeContext.childPart.size?.[1], 0.7) * 0.9
  );
  const growthHandleWorld = edgeContext.frame.anchorWorld
    .clone()
    .addScaledVector(growthDir, handleDistance);
  visualView.setEditingContext({
    anchorWorld: edgeContext.frame.anchorWorld.clone(),
    growthWorld: growthHandleWorld,
    childDirWorld: childDir,
    childSize: edgeContext.childPart.size.slice(),
  });
}

function applyVisualAnchorMutation(anchorWorld) {
  const context = getVisualEdgeContext();
  if (!context) {
    return false;
  }
  const edgeGene = context.edgeGene;
  const parent = context.parentPart;
  const localOffset = anchorWorld
    .clone()
    .sub(parent.center)
    .applyQuaternion(parent.quaternion.clone().invert());

  const halfX = Math.max(parent.size[0] * 0.5, 1e-5);
  const halfY = Math.max(parent.size[1] * 0.5, 1e-5);
  const halfZ = Math.max(parent.size[2] * 0.5, 1e-5);

  edgeGene.anchorX = clampBySchema("part.edge.anchorX", localOffset.x / halfX, 0);
  edgeGene.anchorY = clampBySchema("part.edge.anchorY", localOffset.y / halfY, -0.8);
  edgeGene.anchorZ = clampBySchema("part.edge.anchorZ", localOffset.z / halfZ, 0);
  return true;
}

function applyVisualGrowthMutation(handleWorld) {
  const context = getVisualEdgeContext();
  if (!context) {
    return false;
  }
  const edgeGene = context.edgeGene;
  const parent = context.parentPart;
  const reflectSign = Boolean(edgeGene.reflectX) ? -1 : 1;
  const parentInv = parent.quaternion.clone().invert();
  const localVector = handleWorld
    .clone()
    .sub(context.frame.anchorWorld)
    .applyQuaternion(parentInv);
  const baseLocal = safeUnitVector(
    localVector,
    new THREE.Vector3(
      toFinite(edgeGene.dirX, 0) * reflectSign,
      toFinite(edgeGene.dirY, -1),
      toFinite(edgeGene.dirZ, 0) * reflectSign
    )
  );
  const anchorHint = new THREE.Vector3(
    toFinite(edgeGene.anchorX, 0) * reflectSign,
    toFinite(edgeGene.anchorY, -0.8),
    toFinite(edgeGene.anchorZ, 0)
  );
  const localGrowth = outwardBiasedGrowthDir(baseLocal, anchorHint);
  edgeGene.dirX = clampBySchema("part.edge.dirX", localGrowth.x * reflectSign, 0);
  edgeGene.dirY = clampBySchema("part.edge.dirY", localGrowth.y, -1);
  edgeGene.dirZ = clampBySchema("part.edge.dirZ", localGrowth.z * reflectSign, 0);
  return true;
}

function applyVisualToolMutation(kind, worldPoint) {
  if (!state.currentGenome || !worldPoint || !visualView) {
    return false;
  }
  const world = worldPoint.clone ? worldPoint.clone() : new THREE.Vector3(worldPoint.x, worldPoint.y, worldPoint.z);
  const changed = kind === "anchor"
    ? applyVisualAnchorMutation(world)
    : applyVisualGrowthMutation(world);
  if (!changed) {
    return false;
  }
  syncGenomeEditorFromState();
  rebuildVisualFromCurrentGenome({ fitCamera: false });
  refreshVisualEditBindings();
  return true;
}

function resolveAddChildParentPart(selection = state.selection, model = state.visualModel) {
  if (!selection || !model) {
    return null;
  }
  if (selection.kind === "part") {
    return model.parts.find((part) => part.id === selection.id) || null;
  }
  if (selection.kind === "joint") {
    const joint = model.joints.find((item) => item.id === selection.id) || null;
    if (!joint) {
      return null;
    }
    return model.parts[joint.childPartIndex] || null;
  }
  if (selection.kind === "node") {
    const node = model.nodes.find((item) => item.id === selection.id) || null;
    if (!node || !node.partIds.length) {
      return null;
    }
    return model.parts.find((part) => part.id === node.partIds[0]) || null;
  }
  return null;
}

function defaultChildNodeGene(parentNodeGene) {
  const parentPartGene = parentNodeGene?.part || {};
  const baseW = clampBySchema("node.part.w", toFinite(parentPartGene.w, 0.6) * 0.78, 0.6);
  const baseH = clampBySchema("node.part.h", toFinite(parentPartGene.h, 0.9) * 0.72, 0.9);
  const baseD = clampBySchema("node.part.d", toFinite(parentPartGene.d, 0.6) * 0.78, 0.6);
  const baseMass = clampBySchema("node.part.mass", toFinite(getPartMass(parentPartGene), 0.8) * 0.7, 0.8);
  const node = {
    part: {
      w: baseW,
      h: baseH,
      d: baseD,
      mass: baseMass,
    },
    edges: [],
  };
  if (parentNodeGene?.brain && typeof parentNodeGene.brain === "object") {
    node.brain = deepCloneJson(parentNodeGene.brain);
  }
  return node;
}

function defaultChildEdgeGene(parentNodeGene, toNodeIndex) {
  const existingEdges = asArray(parentNodeGene?.edges);
  const slot = existingEdges.length;
  const anchorBand = [-0.42, 0, 0.42];
  const anchorX = anchorBand[slot % anchorBand.length];
  const anchorZ = slot >= anchorBand.length
    ? ((Math.floor(slot / anchorBand.length) % 2 === 0) ? 0.35 : -0.35)
    : 0;
  const anchorY = -0.76;
  const growthLocal = outwardBiasedGrowthDir(
    new THREE.Vector3(anchorX * 0.45, -1.0, anchorZ * 0.45),
    new THREE.Vector3(anchorX, anchorY, anchorZ)
  );
  return {
    to: toNodeIndex,
    anchorX: clampBySchema("part.edge.anchorX", anchorX, 0),
    anchorY: clampBySchema("part.edge.anchorY", anchorY, -0.8),
    anchorZ: clampBySchema("part.edge.anchorZ", anchorZ, 0),
    axisY: clampBySchema("part.edge.axisY", 0, 0),
    axisZ: clampBySchema("part.edge.axisZ", 0, 0),
    dirX: clampBySchema("part.edge.dirX", growthLocal.x, 0),
    dirY: clampBySchema("part.edge.dirY", growthLocal.y, -1),
    dirZ: clampBySchema("part.edge.dirZ", growthLocal.z, 0),
    scale: clampBySchema("part.edge.scale", 0.92, 1),
    reflectX: false,
    recursiveLimit: 1,
    terminalOnly: false,
    jointType: "hinge",
    limitX: clampBySchema("joint.edge.limitX", 1.2, 1.2),
    limitY: clampBySchema("joint.edge.limitY", 0.75, 0.75),
    limitZ: clampBySchema("joint.edge.limitZ", 0.75, 0.75),
    motorStrength: clampBySchema("joint.edge.motorStrength", 1.05, 1),
    jointStiffness: clampBySchema("joint.edge.stiffness", 42, 42),
  };
}

function addChildFromSelection() {
  if (!state.currentGenome || !state.visualModel) {
    setStatus("Load a genome before adding children.", "warn");
    return false;
  }
  const graph = state.currentGenome.graph;
  if (!graph || !Array.isArray(graph.nodes)) {
    setStatus("Genome graph is missing.", "err");
    return false;
  }
  const parentPart = resolveAddChildParentPart();
  if (!parentPart) {
    setStatus("Select a part, node, or joint to add a child branch.", "warn");
    return false;
  }
  const parentNodeGene = getNodeGene(parentPart.nodeIndex);
  if (!parentNodeGene) {
    setStatus("Selected parent node is unavailable.", "err");
    return false;
  }
  parentNodeGene.edges = asArray(parentNodeGene.edges);
  if (parentNodeGene.edges.length >= MAX_GRAPH_EDGES_PER_NODE) {
    setStatus(`Parent already has ${MAX_GRAPH_EDGES_PER_NODE} edges (max).`, "warn");
    return false;
  }
  if (graph.nodes.length >= MAX_GRAPH_PARTS) {
    setStatus(`Cannot add node: reached ${MAX_GRAPH_PARTS} node templates.`, "warn");
    return false;
  }

  const newNodeIndex = graph.nodes.length;
  const newNode = defaultChildNodeGene(parentNodeGene);
  graph.nodes.push(newNode);
  parentNodeGene.edges.push(defaultChildEdgeGene(parentNodeGene, newNodeIndex));
  graph.maxParts = clampInt(
    Math.max(toFinite(graph.maxParts, 0), newNodeIndex + 2),
    1,
    MAX_GRAPH_PARTS
  );

  syncGenomeEditorFromState();
  rebuildVisualFromCurrentGenome({ fitCamera: false });
  const newNodeId = `node-${newNodeIndex}`;
  setSelection({ kind: "node", id: newNodeId }, "app");
  refreshVisualEditBindings();
  setStatus(`Added child node ${newNodeId} from ${parentPart.id}.`, "ok");
  setInspectorApplyStatus("Added child segment from current selection.", "ok");
  return true;
}

function resolveDuplicateTarget(selection = state.selection, model = state.visualModel) {
  if (!selection || !model) {
    return null;
  }
  if (selection.kind === "edge") {
    const edge = model.edges.find((item) => item.id === selection.id) || null;
    if (edge?.edgeRef) {
      return {
        edgeRef: edge.edgeRef,
        label: edge.id,
      };
    }
    return null;
  }
  if (selection.kind === "part") {
    const part = model.parts.find((item) => item.id === selection.id) || null;
    if (part?.incomingEdgeRef) {
      return {
        edgeRef: part.incomingEdgeRef,
        label: part.id,
      };
    }
    return null;
  }
  if (selection.kind === "joint") {
    const joint = model.joints.find((item) => item.id === selection.id) || null;
    if (joint?.edgeRef) {
      return {
        edgeRef: joint.edgeRef,
        label: joint.id,
      };
    }
    return null;
  }
  if (selection.kind === "node") {
    const node = model.nodes.find((item) => item.id === selection.id) || null;
    if (!node || !node.partIds.length) {
      return null;
    }
    for (const partId of node.partIds) {
      const part = model.parts.find((item) => item.id === partId) || null;
      if (part?.incomingEdgeRef) {
        return {
          edgeRef: part.incomingEdgeRef,
          label: `${node.id} via ${part.id}`,
        };
      }
    }
  }
  return null;
}

function collectReachableNodeIndices(graph, startIndex, maxCount) {
  const nodes = asArray(graph?.nodes);
  if (!nodes.length || !Number.isInteger(startIndex) || startIndex < 0 || startIndex >= nodes.length || maxCount <= 0) {
    return [];
  }
  const visited = new Set();
  const queue = [startIndex];
  const ordered = [];
  while (queue.length && ordered.length < maxCount) {
    const nodeIndex = queue.shift();
    if (!Number.isInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= nodes.length || visited.has(nodeIndex)) {
      continue;
    }
    visited.add(nodeIndex);
    ordered.push(nodeIndex);
    const node = nodes[nodeIndex];
    const edges = asArray(node?.edges).slice(0, MAX_GRAPH_EDGES_PER_NODE);
    for (const edge of edges) {
      const to = Math.trunc(toFinite(edge?.to, -1));
      if (to >= 0 && to < nodes.length && !visited.has(to)) {
        queue.push(to);
      }
    }
  }
  return ordered;
}

function duplicateSelectedBranch() {
  if (!state.currentGenome || !state.visualModel) {
    setStatus("Load a genome before duplicating branches.", "warn");
    return false;
  }
  const target = resolveDuplicateTarget();
  if (!target) {
    setStatus(
      "Select a non-root part, joint, or instantiated node to duplicate that branch.",
      "warn"
    );
    return false;
  }
  const graph = state.currentGenome.graph;
  if (!graph || !Array.isArray(graph.nodes)) {
    setStatus("Genome graph is missing.", "err");
    return false;
  }

  const edgeRef = target.edgeRef || {};
  if (!Number.isInteger(edgeRef.fromNodeIndex) || !Number.isInteger(edgeRef.edgeIndex)) {
    setStatus("Duplicate source edge is invalid.", "err");
    return false;
  }
  const parentNode = graph.nodes[edgeRef.fromNodeIndex];
  if (!parentNode) {
    setStatus("Duplicate source parent node is unavailable.", "warn");
    return false;
  }
  const parentEdges = asArray(parentNode.edges);
  if (!parentEdges.length || edgeRef.edgeIndex < 0 || edgeRef.edgeIndex >= parentEdges.length) {
    setStatus("Selected branch is stale after rebuild. Re-select and try again.", "warn");
    return false;
  }
  if (parentEdges.length >= MAX_GRAPH_EDGES_PER_NODE) {
    setStatus(`Parent already has ${MAX_GRAPH_EDGES_PER_NODE} edges (max).`, "warn");
    return false;
  }
  if (graph.nodes.length >= MAX_GRAPH_PARTS) {
    setStatus(`Cannot duplicate branch: reached ${MAX_GRAPH_PARTS} node templates.`, "warn");
    return false;
  }

  const sourceEdge = parentEdges[edgeRef.edgeIndex];
  const sourceRootIndex = Math.trunc(toFinite(sourceEdge?.to, -1));
  if (sourceRootIndex < 0 || sourceRootIndex >= graph.nodes.length) {
    setStatus("Selected branch target node is invalid.", "warn");
    return false;
  }

  const availableSlots = Math.max(0, MAX_GRAPH_PARTS - graph.nodes.length);
  const reachable = collectReachableNodeIndices(graph, sourceRootIndex, availableSlots + 1);
  if (!reachable.length) {
    setStatus("Unable to collect branch templates for duplication.", "warn");
    return false;
  }

  const nodeMap = new Map();
  for (const originalNodeIndex of reachable) {
    if (graph.nodes.length >= MAX_GRAPH_PARTS) {
      break;
    }
    const sourceNode = graph.nodes[originalNodeIndex];
    if (!sourceNode) {
      continue;
    }
    const clonedNode = deepCloneJson(sourceNode);
    clonedNode.edges = asArray(clonedNode.edges).slice(0, MAX_GRAPH_EDGES_PER_NODE);
    const newNodeIndex = graph.nodes.length;
    graph.nodes.push(clonedNode);
    nodeMap.set(originalNodeIndex, newNodeIndex);
  }
  if (!nodeMap.has(sourceRootIndex)) {
    setStatus("Duplicate failed: source node was not cloned.", "err");
    return false;
  }

  for (const [, clonedNodeIndex] of nodeMap.entries()) {
    const clonedNode = graph.nodes[clonedNodeIndex];
    if (!clonedNode) {
      continue;
    }
    const remappedEdges = [];
    for (const edge of asArray(clonedNode.edges).slice(0, MAX_GRAPH_EDGES_PER_NODE)) {
      const remappedEdge = deepCloneJson(edge);
      const to = Math.trunc(toFinite(remappedEdge?.to, 0));
      remappedEdge.to = nodeMap.has(to)
        ? nodeMap.get(to)
        : clampInt(to, 0, Math.max(0, graph.nodes.length - 1));
      remappedEdges.push(remappedEdge);
    }
    clonedNode.edges = remappedEdges;
  }

  const duplicatedEdge = deepCloneJson(sourceEdge);
  duplicatedEdge.to = nodeMap.get(sourceRootIndex);
  parentEdges.push(duplicatedEdge);
  parentNode.edges = parentEdges;
  graph.maxParts = clampInt(
    Math.max(toFinite(graph.maxParts, 0), graph.nodes.length + 1),
    1,
    MAX_GRAPH_PARTS
  );

  const newEdgeIndex = parentEdges.length - 1;
  const newRootNodeIndex = nodeMap.get(sourceRootIndex);
  syncGenomeEditorFromState();
  rebuildVisualFromCurrentGenome({ fitCamera: false });
  setSelection({ kind: "node", id: `node-${newRootNodeIndex}` }, "app");
  refreshVisualEditBindings();
  setStatus(
    `Duplicated branch '${target.label}' (${nodeMap.size} template${nodeMap.size === 1 ? "" : "s"}).`,
    "ok"
  );
  setInspectorApplyStatus(`Duplicated branch with independent templates.`, "ok");

  const duplicatedPart = state.visualModel?.parts?.find((part) => {
    const ref = part?.incomingEdgeRef;
    return ref
      && ref.fromNodeIndex === edgeRef.fromNodeIndex
      && ref.edgeIndex === newEdgeIndex;
  });
  if (duplicatedPart) {
    setSelection({ kind: "part", id: duplicatedPart.id }, "app");
  }
  return true;
}

function resolveDeleteTarget(selection = state.selection, model = state.visualModel) {
  if (!selection || !model) {
    return null;
  }
  if (selection.kind === "edge") {
    const edge = model.edges.find((item) => item.id === selection.id) || null;
    if (edge?.edgeRef) {
      return {
        type: "edge",
        edgeRef: edge.edgeRef,
        label: edge.id,
      };
    }
    return null;
  }
  if (selection.kind === "part") {
    const part = model.parts.find((item) => item.id === selection.id) || null;
    if (part?.incomingEdgeRef) {
      return {
        type: "edge",
        edgeRef: part.incomingEdgeRef,
        label: part.id,
      };
    }
    return null;
  }
  if (selection.kind === "joint") {
    const joint = model.joints.find((item) => item.id === selection.id) || null;
    if (joint?.edgeRef) {
      return {
        type: "edge",
        edgeRef: joint.edgeRef,
        label: joint.id,
      };
    }
    return null;
  }
  if (selection.kind === "node") {
    const node = model.nodes.find((item) => item.id === selection.id) || null;
    if (!node || node.partIds.length > 0) {
      return null;
    }
    return {
      type: "unused-node",
      nodeIndex: node.nodeIndex,
      label: node.id,
    };
  }
  return null;
}

function deleteSelectedEntity() {
  if (!state.currentGenome || !state.visualModel) {
    setStatus("Load a genome before deleting.", "warn");
    return false;
  }
  const target = resolveDeleteTarget();
  if (!target) {
    setStatus(
      "Delete is available for edges, non-root part/joint branches, and unused node templates.",
      "warn"
    );
    return false;
  }
  const graph = state.currentGenome.graph;
  if (!graph || !Array.isArray(graph.nodes)) {
    setStatus("Genome graph is missing.", "err");
    return false;
  }

  if (target.type === "edge") {
    const edgeRef = target.edgeRef || {};
    const parentNode = graph.nodes[edgeRef.fromNodeIndex];
    const edges = asArray(parentNode?.edges);
    if (!parentNode || !edges.length || !Number.isInteger(edgeRef.edgeIndex) || edgeRef.edgeIndex < 0 || edgeRef.edgeIndex >= edges.length) {
      setStatus("Selected branch is stale after rebuild. Re-select and try again.", "warn");
      return false;
    }
    edges.splice(edgeRef.edgeIndex, 1);
    parentNode.edges = edges;
    syncGenomeEditorFromState();
    rebuildVisualFromCurrentGenome({ fitCamera: false });
    setSelection(null, "app");
    setStatus(`Deleted branch '${target.label}'.`, "ok");
    setInspectorApplyStatus("Deleted selected branch.", "ok");
    return true;
  }

  if (target.type === "unused-node") {
    const removeIndex = target.nodeIndex;
    if (!Number.isInteger(removeIndex) || removeIndex < 0 || removeIndex >= graph.nodes.length) {
      setStatus("Selected unused node is stale after rebuild. Re-select and try again.", "warn");
      return false;
    }
    if (graph.nodes.length <= 1) {
      setStatus("Cannot delete the final node template.", "warn");
      return false;
    }
    graph.nodes.splice(removeIndex, 1);
    for (const node of graph.nodes) {
      const nextEdges = [];
      for (const edge of asArray(node.edges)) {
        const to = Math.trunc(toFinite(edge?.to, -1));
        if (to === removeIndex) {
          continue;
        }
        const cloned = { ...edge };
        if (to > removeIndex) {
          cloned.to = to - 1;
        } else {
          cloned.to = to;
        }
        nextEdges.push(cloned);
      }
      node.edges = nextEdges;
    }
    const rootNow = clampInt(toFinite(graph.root, 0), 0, Math.max(0, graph.nodes.length - 1));
    if (rootNow > removeIndex) {
      graph.root = rootNow - 1;
    } else {
      graph.root = rootNow;
    }
    syncGenomeEditorFromState();
    rebuildVisualFromCurrentGenome({ fitCamera: false });
    setSelection(null, "app");
    setStatus(`Deleted unused node template '${target.label}'.`, "ok");
    setInspectorApplyStatus("Deleted unused node template.", "ok");
    return true;
  }

  return false;
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

  const ICONS = {
    node: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>`,
    part: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>`,
    joint: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><line x1="3" y1="12" x2="9" y2="12"></line><line x1="15" y1="12" x2="21" y2="12"></line></svg>`,
    edge: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>`
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
    button.dataset.kind = kind;
    button.dataset.id = id;
    button.title = details || "";

    let indentHtml = "";
    for (let i = 0; i < depth; i++) {
        indentHtml += `<div class="tree-line ${i === depth - 1 ? 'leaf' : ''}"></div>`;
    }

    button.innerHTML = `
      ${indentHtml}
      <div class="outlinerItem__icon">${ICONS[kind] || ""}</div>
      <span class="outlinerItem__label">${label}</span>
      <span class="outlinerItem__id">${id}</span>
    `;

    if (selectedToken === selectionToken({ kind, id })) {
      button.classList.add("selected");
    }
    button.addEventListener("click", () => {
      setSelection({ kind, id }, "outliner");
    });
    outlinerList.appendChild(button);
  };

  const nodeByIndex = new Map(model.nodes.map((node) => [node.nodeIndex, node]));
  const edgesByNode = new Map();
  for (const edge of asArray(model.edges)) {
    if (!edgesByNode.has(edge.fromNodeIndex)) {
      edgesByNode.set(edge.fromNodeIndex, []);
    }
    edgesByNode.get(edge.fromNodeIndex).push(edge);
  }
  for (const list of edgesByNode.values()) {
    list.sort((a, b) => a.edgeIndex - b.edgeIndex);
  }
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

  addSection(`Creature Hierarchy (${model.parts.length} parts, ${model.joints.length} joints, ${model.edges.length} edge genes)`);
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
      const nodeEdges = edgesByNode.get(node.nodeIndex) || [];
      for (const edge of nodeEdges) {
        addItem({
          kind: "edge",
          id: edge.id,
          label: `edge ${edge.edgeIndex}: node ${edge.fromNodeIndex} -> node ${edge.toNodeIndex}`,
          details: `${edge.jointType} | ${edge.instanceCount} instance${edge.instanceCount === 1 ? "" : "s"}`,
          depth: depth + 2,
          searchText: `edge ${edge.edgeIndex} from ${edge.fromNodeIndex} to ${edge.toNodeIndex} ${edge.jointType}`,
        });
      }
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
      const nodeEdges = edgesByNode.get(node.nodeIndex) || [];
      for (const edge of nodeEdges) {
        addItem({
          kind: "edge",
          id: edge.id,
          label: `edge ${edge.edgeIndex}: node ${edge.fromNodeIndex} -> node ${edge.toNodeIndex}`,
          details: `${edge.jointType} | ${edge.instanceCount} instance${edge.instanceCount === 1 ? "" : "s"}`,
          depth: 2,
          searchText: `unused edge ${edge.edgeIndex} from ${edge.fromNodeIndex} to ${edge.toNodeIndex} ${edge.jointType}`,
        });
      }
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

function edgeRefKey(edgeRef) {
  if (!edgeRef || !Number.isInteger(edgeRef.fromNodeIndex) || !Number.isInteger(edgeRef.edgeIndex)) {
    return "";
  }
  return `${edgeRef.fromNodeIndex}:${edgeRef.edgeIndex}`;
}

function edgeRefsEqual(a, b) {
  return edgeRefKey(a) !== "" && edgeRefKey(a) === edgeRefKey(b);
}

function defaultGlobalBrainGene() {
  const neuronCount = MIN_GLOBAL_NEURONS;
  const neurons = [];
  for (let i = 0; i < neuronCount; i += 1) {
    neurons.push({
      activation: "tanh",
      inputWeights: new Array(8).fill(0),
      recurrentWeights: new Array(neuronCount).fill(0),
      globalWeights: [],
      bias: 0,
      leak: 0.6,
    });
  }
  return { neurons };
}

function normalizeNeuronGene(neuron, recurrentCount, globalCount) {
  const next = (neuron && typeof neuron === "object") ? neuron : {};
  if (!next.activation) {
    next.activation = "tanh";
  }
  next.inputWeights = asArray(next.inputWeights).slice(0, LOCAL_SENSOR_DIM);
  while (next.inputWeights.length < LOCAL_SENSOR_DIM) {
    next.inputWeights.push(0);
  }
  next.recurrentWeights = asArray(next.recurrentWeights).slice(0, recurrentCount);
  while (next.recurrentWeights.length < recurrentCount) {
    next.recurrentWeights.push(0);
  }
  next.globalWeights = asArray(next.globalWeights).slice(0, globalCount);
  while (next.globalWeights.length < globalCount) {
    next.globalWeights.push(0);
  }
  next.bias = toFinite(next.bias, 0);
  next.leak = clamp(toFinite(next.leak, 0.6), 0.05, 1.0);
  return next;
}

function normalizeEffectorGene(effector, localCount, globalCount) {
  const next = (effector && typeof effector === "object") ? effector : {};
  next.localWeights = asArray(next.localWeights).slice(0, localCount);
  while (next.localWeights.length < localCount) {
    next.localWeights.push(0);
  }
  next.globalWeights = asArray(next.globalWeights).slice(0, globalCount);
  while (next.globalWeights.length < globalCount) {
    next.globalWeights.push(0);
  }
  next.bias = toFinite(next.bias, 0);
  next.gain = clamp(toFinite(next.gain, 1), 0.2, 2.0);
  return next;
}

function resizeLocalBrainGene(nodeGene, targetCount) {
  if (!nodeGene || typeof nodeGene !== "object") {
    return;
  }
  if (!nodeGene.brain || typeof nodeGene.brain !== "object") {
    nodeGene.brain = {};
  }
  const graph = state.currentGenome?.graph || {};
  const globalBrain = graph.globalBrain && typeof graph.globalBrain === "object"
    ? graph.globalBrain
    : defaultGlobalBrainGene();
  if (!Array.isArray(graph.globalBrain?.neurons)) {
    graph.globalBrain = globalBrain;
  }
  const globalCount = clampInt(asArray(globalBrain.neurons).length, MIN_GLOBAL_NEURONS, MAX_GLOBAL_NEURONS);
  const localCount = clampInt(targetCount, MIN_LOCAL_NEURONS, MAX_LOCAL_NEURONS);
  const brain = nodeGene.brain;
  brain.neurons = asArray(brain.neurons);
  while (brain.neurons.length < localCount) {
    brain.neurons.push({});
  }
  brain.neurons = brain.neurons.slice(0, localCount).map((neuron) => (
    normalizeNeuronGene(neuron, localCount, globalCount)
  ));
  brain.effectorX = normalizeEffectorGene(brain.effectorX, localCount, globalCount);
  brain.effectorY = normalizeEffectorGene(brain.effectorY, localCount, globalCount);
  brain.effectorZ = normalizeEffectorGene(brain.effectorZ, localCount, globalCount);
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
  const entityEdgeRef = entity?.edgeRef || entity?.incomingEdgeRef || null;
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
  if (field === "node.brain.neuronCount") {
    const nodeGene = getNodeGene(entity.nodeIndex);
    if (!nodeGene) {
      return null;
    }
    return (value) => {
      resizeLocalBrainGene(nodeGene, value);
    };
  }
  if (field === "edge.to") {
    const edgeGene = getEdgeGeneByRef(entityEdgeRef);
    if (!edgeGene) {
      return null;
    }
    return (value) => {
      const nodeCount = graphNodesFromState().length;
      edgeGene.to = clampInt(value, 0, Math.max(0, nodeCount - 1));
    };
  }
  if (field.startsWith("part.edge.")) {
    const edgeGene = getEdgeGeneByRef(entityEdgeRef);
    if (!edgeGene || typeof edgeGene !== "object") {
      return null;
    }
    const key = field.slice("part.edge.".length);
    return (value) => {
      edgeGene[key] = value;
    };
  }
  if (field.startsWith("joint.edge.")) {
    const edgeGene = getEdgeGeneByRef(entityEdgeRef);
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
  appendEditableField(
    inspectorContent,
    selection,
    "local neurons",
    "node.brain.neuronCount",
    clampInt(asArray(brain.neurons).length || MIN_LOCAL_NEURONS, MIN_LOCAL_NEURONS, MAX_LOCAL_NEURONS)
  );
  appendField(inspectorContent, "brain template scope", "Shared by all part instances of this node template.");
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

function renderEdgeInspector(entity, selection) {
  const edgeRef = entity?.edgeRef || null;
  const edgeGene = getEdgeGeneByRef(edgeRef);
  appendField(inspectorContent, "edge id", entity.id || "edge");
  appendField(inspectorContent, "from node", String(entity.fromNodeIndex));
  appendField(inspectorContent, "edge index", String(entity.edgeIndex));
  appendField(inspectorContent, "instances", String(entity.instanceCount || 0));
  appendField(
    inspectorContent,
    "instanced joints",
    asArray(entity.instanceJointIds).join(", ") || "none"
  );
  appendField(
    inspectorContent,
    "instanced parts",
    asArray(entity.instancePartIds).join(", ") || "none"
  );
  if (!edgeGene) {
    appendField(inspectorContent, "edge gene", "unavailable");
    return;
  }
  appendEditableField(inspectorContent, selection, "to node", "edge.to", clampInt(toFinite(edgeGene.to, 0), 0, MAX_GRAPH_PARTS - 1));
  appendEditableField(inspectorContent, selection, "joint type", "joint.edge.jointType", String(edgeGene.jointType || "hinge"));
  appendEditableField(inspectorContent, selection, "scale", "part.edge.scale", toFinite(edgeGene.scale, 1));
  appendEditableField(inspectorContent, selection, "anchor X", "part.edge.anchorX", toFinite(edgeGene.anchorX, 0));
  appendEditableField(inspectorContent, selection, "anchor Y", "part.edge.anchorY", toFinite(edgeGene.anchorY, 0));
  appendEditableField(inspectorContent, selection, "anchor Z", "part.edge.anchorZ", toFinite(edgeGene.anchorZ, 0));
  appendEditableField(inspectorContent, selection, "dir X", "part.edge.dirX", toFinite(edgeGene.dirX, 0));
  appendEditableField(inspectorContent, selection, "dir Y", "part.edge.dirY", toFinite(edgeGene.dirY, -1));
  appendEditableField(inspectorContent, selection, "dir Z", "part.edge.dirZ", toFinite(edgeGene.dirZ, 0));
  appendEditableField(inspectorContent, selection, "axis Y", "part.edge.axisY", toFinite(edgeGene.axisY, 0));
  appendEditableField(inspectorContent, selection, "axis Z", "part.edge.axisZ", toFinite(edgeGene.axisZ, 0));
  appendEditableField(inspectorContent, selection, "limit X", "joint.edge.limitX", getEdgeLimit(edgeGene, "X"));
  appendEditableField(inspectorContent, selection, "limit Y", "joint.edge.limitY", getEdgeLimit(edgeGene, "Y"));
  appendEditableField(inspectorContent, selection, "limit Z", "joint.edge.limitZ", getEdgeLimit(edgeGene, "Z"));
  appendEditableField(inspectorContent, selection, "motor strength", "joint.edge.motorStrength", toFinite(edgeGene.motorStrength, 1));
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
    appendEmpty(inspectorContent, "Select a node, edge, part, or joint in the viewport or outliner.");
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
  const edgeContext = getVisualEdgeContext(selection, model);
  appendField(inspectorContent, "viewport tool", visualToolLabel(state.visualTool));
  if (edgeContext) {
    appendField(
      inspectorContent,
      "visual handles",
      "Anchor and growth handles are available in Move anchor / Growth handle tools."
    );
  } else {
    appendField(
      inspectorContent,
      "visual handles",
      "Select a non-root part/joint (or an instantiated edge) to use anchor/growth drag handles."
    );
  }

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
    return;
  }
  if (selection.kind === "edge") {
    renderEdgeInspector(entity, selection);
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
    return { nodes: [], parts: [], joints: [], edges: [], nodeCount: nodes.length };
  }
  const nodeEntries = nodes.map((node, nodeIndex) => ({
    id: `node-${nodeIndex}`,
    nodeIndex,
    edgeCount: asArray(node?.edges).length,
    part: node?.part || {},
    brain: node?.brain || null,
    partIds: [],
  }));
  const edgeEntries = [];
  for (let fromNodeIndex = 0; fromNodeIndex < nodes.length; fromNodeIndex += 1) {
    const edges = asArray(nodes[fromNodeIndex]?.edges);
    for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += 1) {
      const edge = edges[edgeIndex] || {};
      edgeEntries.push({
        id: `edge-${fromNodeIndex}-${edgeIndex}`,
        fromNodeIndex,
        edgeIndex,
        edgeRef: { fromNodeIndex, edgeIndex },
        toNodeIndex: clampInt(toFinite(edge.to, 0), 0, Math.max(0, nodes.length - 1)),
        jointType: String(edge.jointType || "hinge"),
        instancePartIds: [],
        instanceJointIds: [],
        instanceCount: 0,
      });
    }
  }
  const edgeEntryByRef = new Map(edgeEntries.map((entry) => [edgeRefKey(entry.edgeRef), entry]));
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
    const edgeEntry = edgeEntryByRef.get(edgeRefKey(incomingEdgeRef));
    if (edgeEntry) {
      edgeEntry.instancePartIds.push(childPartId);
      edgeEntry.instanceCount += 1;
    }

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
    if (edgeEntry) {
      edgeEntry.instanceJointIds.push(jointId);
    }
  }

  return {
    nodes: nodeEntries,
    parts,
    joints,
    edges: edgeEntries,
    nodeCount: nodes.length,
  };
}

function edgeFrameFromParent(parentPart, edgeGene) {
  if (!parentPart || !edgeGene) {
    return null;
  }
  const reflectSign = Boolean(edgeGene.reflectX) ? -1 : 1;
  const anchorX = toFinite(edgeGene.anchorX, 0);
  const anchorY = toFinite(edgeGene.anchorY, -0.8);
  const anchorZ = toFinite(edgeGene.anchorZ, 0);
  const anchorHint = new THREE.Vector3(anchorX * reflectSign, anchorY, anchorZ);
  const pivotLocal = new THREE.Vector3(
    anchorX * parentPart.size[0] * 0.5,
    anchorY * parentPart.size[1] * 0.5,
    anchorZ * parentPart.size[2] * 0.5
  );
  const anchorWorld = parentPart.center
    .clone()
    .add(pivotLocal.applyQuaternion(parentPart.quaternion.clone()));
  const localGrowth = outwardBiasedGrowthDir(
    new THREE.Vector3(
      toFinite(edgeGene.dirX, 0) * reflectSign,
      toFinite(edgeGene.dirY, -1),
      toFinite(edgeGene.dirZ, 0) * reflectSign
    ),
    anchorHint
  );
  const growthWorld = localGrowth.clone().applyQuaternion(parentPart.quaternion.clone());
  return {
    anchorWorld,
    growthWorld,
    localGrowth,
    reflectSign,
  };
}

function getVisualEdgeContext(selection = state.selection, model = state.visualModel) {
  if (!selection || !model) {
    return null;
  }
  let childPart = null;
  if (selection.kind === "part") {
    childPart = model.parts.find((item) => item.id === selection.id) || null;
  } else if (selection.kind === "joint") {
    const joint = model.joints.find((item) => item.id === selection.id) || null;
    if (joint) {
      childPart = model.parts[joint.childPartIndex] || null;
    }
  } else if (selection.kind === "edge") {
    const edge = model.edges.find((item) => item.id === selection.id) || null;
    if (edge?.edgeRef) {
      childPart = model.parts.find((part) => edgeRefsEqual(part.incomingEdgeRef, edge.edgeRef)) || null;
    }
  }
  if (!childPart || !childPart.incomingEdgeRef || !Number.isInteger(childPart.parentPartIndex)) {
    return null;
  }
  const parentPart = model.parts[childPart.parentPartIndex] || null;
  if (!parentPart) {
    return null;
  }
  const edgeGene = getEdgeGeneByRef(childPart.incomingEdgeRef);
  if (!edgeGene) {
    return null;
  }
  const frame = edgeFrameFromParent(parentPart, edgeGene);
  if (!frame) {
    return null;
  }
  return {
    edgeRef: childPart.incomingEdgeRef,
    edgeGene,
    childPart,
    parentPart,
    frame,
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
    this.toolPickables = [];
    this.pointerDown = null;
    this.raycaster = new THREE.Raycaster();
    this.pointerNdc = new THREE.Vector2(0, 0);
    this.editingTool = VISUAL_TOOL_SELECT;
    this.editingContext = null;
    this.onToolEdit = null;
    this.activeDrag = null;
    this.dragPlane = new THREE.Plane();

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
    this.toolGroup = new THREE.Group();
    this.modelGroup.add(this.segmentGroup);
    this.modelGroup.add(this.jointGroup);
    this.modelGroup.add(this.toolGroup);
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

  setEditingTool(tool) {
    this.editingTool = tool === VISUAL_TOOL_MOVE_ANCHOR || tool === VISUAL_TOOL_GROWTH_DIR
      ? tool
      : VISUAL_TOOL_SELECT;
    this.rebuildEditingHandles();
  }

  setEditingContext(context) {
    if (!context || !context.anchorWorld || !context.growthWorld) {
      this.editingContext = null;
      this.rebuildEditingHandles();
      return;
    }
    this.editingContext = {
      anchorWorld: context.anchorWorld.clone(),
      growthWorld: context.growthWorld.clone(),
      childDirWorld: context.childDirWorld?.clone?.() || context.growthWorld.clone().normalize(),
      childSize: Array.isArray(context.childSize) ? context.childSize.slice(0, 3) : [0.7, 0.9, 0.7],
    };
    this.rebuildEditingHandles();
  }

  setToolEditCallback(callback) {
    this.onToolEdit = typeof callback === "function" ? callback : null;
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

  registerSelectable(selection, object3d, pickable = true, role = "") {
    const key = this.makeSelectionKey(selection.kind, selection.id);
    object3d.userData.selection = { kind: selection.kind, id: selection.id };
    object3d.userData.selectionKey = key;
    if (role) {
      object3d.userData.selectionRole = role;
    }
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

  buildJointBasis(axisWorld, hintWorld) {
    const x = axisWorld?.clone?.() || new THREE.Vector3(1, 0, 0);
    if (x.lengthSq() <= 1e-8) {
      x.set(1, 0, 0);
    } else {
      x.normalize();
    }

    let y = hintWorld?.clone?.() || new THREE.Vector3(0, 1, 0);
    y.addScaledVector(x, -y.dot(x));
    if (y.lengthSq() <= 1e-8) {
      const fallback = Math.abs(x.y) < 0.9
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(0, 0, 1);
      y = fallback.addScaledVector(x, -fallback.dot(x));
    }
    if (y.lengthSq() <= 1e-8) {
      y = new THREE.Vector3().crossVectors(x, new THREE.Vector3(1, 0, 0));
    }
    if (y.lengthSq() <= 1e-8) {
      y = new THREE.Vector3().crossVectors(x, new THREE.Vector3(0, 0, 1));
    }
    if (y.lengthSq() <= 1e-8) {
      y = new THREE.Vector3(0, 1, 0);
    } else {
      y.normalize();
    }

    const z = new THREE.Vector3().crossVectors(x, y);
    if (z.lengthSq() <= 1e-8) {
      z.set(0, 0, 1);
    } else {
      z.normalize();
    }
    const yOrtho = new THREE.Vector3().crossVectors(z, x);
    if (yOrtho.lengthSq() > 1e-8) {
      yOrtho.normalize();
      return { x, y: yOrtho, z };
    }
    return { x, y, z };
  }

  buildArcPoints(center, axisA, axisB, radius, startAngle, endAngle, segments = 28) {
    const safeRadius = Math.max(0, toFinite(radius, 0));
    if (safeRadius <= 1e-6) {
      return [];
    }
    const a = axisA?.clone?.();
    const b = axisB?.clone?.();
    if (!a || !b || a.lengthSq() <= 1e-8 || b.lengthSq() <= 1e-8) {
      return [];
    }
    a.normalize();
    b.normalize();

    const steps = Math.max(8, Math.trunc(toFinite(segments, 28)));
    const points = [];
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      const angle = THREE.MathUtils.lerp(startAngle, endAngle, t);
      const radial = a.clone().multiplyScalar(Math.cos(angle)).addScaledVector(b, Math.sin(angle));
      points.push(center.clone().addScaledVector(radial, safeRadius));
    }
    return points;
  }

  addJointArc(selection, options = {}) {
    const center = options.center;
    if (!center) {
      return;
    }
    const startAngle = toFinite(options.startAngle, 0);
    const endAngle = toFinite(options.endAngle, 0);
    if (Math.abs(endAngle - startAngle) <= 1e-4) {
      return;
    }
    const points = this.buildArcPoints(
      center,
      options.axisA,
      options.axisB,
      options.radius,
      startAngle,
      endAngle,
      options.segments
    );
    if (points.length < 2) {
      return;
    }
    const color = toFinite(options.color, 0xffffff);
    const opacity = clamp(toFinite(options.opacity, 0.9), 0, 1);
    const renderOrder = Math.trunc(toFinite(options.renderOrder, 21));
    const arc = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        depthTest: false,
      })
    );
    arc.renderOrder = renderOrder;
    this.registerSelectable(selection, arc, true, "joint-wire");
    this.jointGroup.add(arc);

    if (!options.addBoundaryRays) {
      return;
    }
    const spokes = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([
        center.clone(),
        points[0].clone(),
        center.clone(),
        points[points.length - 1].clone(),
      ]),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: clamp(opacity + 0.05, 0, 1),
        depthTest: false,
      })
    );
    spokes.renderOrder = renderOrder;
    this.registerSelectable(selection, spokes, true, "joint-wire");
    this.jointGroup.add(spokes);
  }

  registerDragHandle(kind, object3d, pickable = true) {
    object3d.userData.dragKind = kind;
    object3d.userData.isDragHandle = true;
    if (pickable) {
      this.pickables.push(object3d);
      this.toolPickables.push(object3d);
    }
  }

  rebuildEditingHandles() {
    if (this.toolPickables.length) {
      const stale = new Set(this.toolPickables);
      this.pickables = this.pickables.filter((object3d) => !stale.has(object3d));
      this.toolPickables = [];
    }
    this.clearGroup(this.toolGroup);
    if (!this.editingContext) {
      return;
    }
    const gizmoSelection = this.currentSelection && this.currentSelection.kind && this.currentSelection.id
      ? { kind: this.currentSelection.kind, id: this.currentSelection.id }
      : null;
    const ctx = this.editingContext;
    const anchor = ctx.anchorWorld;
    const growth = ctx.growthWorld;
    const childDir = safeUnitVector(ctx.childDirWorld, growth.clone().sub(anchor));
    const growthDir = safeUnitVector(growth.clone().sub(anchor), childDir.clone());
    const childExtent = Math.max(
      toFinite(ctx.childSize?.[0], 0.6),
      toFinite(ctx.childSize?.[1], 0.8),
      toFinite(ctx.childSize?.[2], 0.6)
    );
    const guideEnd = anchor.clone().addScaledVector(childDir, Math.max(0.28, childExtent * 0.5));
    const growthGuide = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([anchor.clone(), growth.clone()]),
      new THREE.LineBasicMaterial({
        color: 0x6ec3b2,
        transparent: true,
        opacity: 0.82,
        depthTest: false,
      })
    );
    growthGuide.renderOrder = 24;
    if (gizmoSelection) {
      this.registerSelectable(gizmoSelection, growthGuide, false, "gizmo-wire");
    }
    this.toolGroup.add(growthGuide);

    const childHint = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([anchor.clone(), guideEnd]),
      new THREE.LineDashedMaterial({
        color: 0xb3d98a,
        dashSize: 0.08,
        gapSize: 0.045,
        transparent: true,
        opacity: 0.7,
        depthTest: false,
      })
    );
    childHint.computeLineDistances();
    childHint.renderOrder = 24;
    if (gizmoSelection) {
      this.registerSelectable(gizmoSelection, childHint, false, "gizmo-wire");
    }
    this.toolGroup.add(childHint);

    const anchorActive = this.editingTool === VISUAL_TOOL_MOVE_ANCHOR;
    const growthActive = this.editingTool === VISUAL_TOOL_GROWTH_DIR;

    const anchorHandle = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 20, 14),
      new THREE.MeshStandardMaterial({
        color: anchorActive ? 0xffd98f : 0xd7c89a,
        emissive: anchorActive ? 0x3d2b0f : 0x1f1a12,
        roughness: 0.35,
        metalness: 0.08,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      })
    );
    anchorHandle.position.copy(anchor);
    anchorHandle.renderOrder = 25;
    if (gizmoSelection) {
      this.registerSelectable(gizmoSelection, anchorHandle, false, "gizmo-geometry");
    }
    this.registerDragHandle("anchor", anchorHandle, anchorActive);
    this.toolGroup.add(anchorHandle);

    const growthHandle = new THREE.Mesh(
      new THREE.ConeGeometry(0.06, 0.19, 18),
      new THREE.MeshStandardMaterial({
        color: growthActive ? 0x83d4ff : 0x86b7c8,
        emissive: growthActive ? 0x113448 : 0x0d1e27,
        roughness: 0.34,
        metalness: 0.12,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      })
    );
    growthHandle.position.copy(growth);
    growthHandle.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), growthDir.clone());
    growthHandle.renderOrder = 25;
    if (gizmoSelection) {
      this.registerSelectable(gizmoSelection, growthHandle, false, "gizmo-geometry");
    }
    this.registerDragHandle("growth", growthHandle, growthActive);
    this.toolGroup.add(growthHandle);

    this.applySelectionHighlight();
  }

  dragKindAllowed(kind) {
    return (kind === "anchor" && this.editingTool === VISUAL_TOOL_MOVE_ANCHOR)
      || (kind === "growth" && this.editingTool === VISUAL_TOOL_GROWTH_DIR);
  }

  dragPlanePointFromEvent(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) {
      return null;
    }
    this.pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const point = new THREE.Vector3();
    const ok = this.raycaster.ray.intersectPlane(this.dragPlane, point);
    return ok ? point : null;
  }

  getPointerHits(event) {
    if (!this.pickables.length) {
      return [];
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) {
      return [];
    }
    this.pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    return this.raycaster.intersectObjects(this.pickables, false);
  }

  beginDrag(kind, event, originWorld) {
    if (!this.dragKindAllowed(kind) || !originWorld) {
      return false;
    }
    const normal = this.camera.getWorldDirection(new THREE.Vector3()).normalize();
    if (normal.lengthSq() <= 1e-8) {
      normal.set(0, 0, -1);
    }
    this.dragPlane.setFromNormalAndCoplanarPoint(normal, originWorld);
    const hit = this.dragPlanePointFromEvent(event) || originWorld.clone();
    this.activeDrag = {
      kind,
      originWorld: originWorld.clone(),
      offset: originWorld.clone().sub(hit),
    };
    this.controls.enabled = false;
    return true;
  }

  updateDrag(event) {
    if (!this.activeDrag || !this.onToolEdit) {
      return false;
    }
    const hit = this.dragPlanePointFromEvent(event);
    if (!hit) {
      return false;
    }
    const target = hit.clone().add(this.activeDrag.offset);
    return this.onToolEdit(this.activeDrag.kind, target);
  }

  endDrag() {
    if (!this.activeDrag) {
      return;
    }
    this.activeDrag = null;
    this.controls.enabled = true;
  }

  installPointerHandlers() {
    const element = this.renderer.domElement;
    element.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      const hits = this.getPointerHits(event);
      const dragHit = hits.find((entry) => this.dragKindAllowed(entry.object?.userData?.dragKind));
      const dragKind = dragHit?.object?.userData?.dragKind;
      if (dragKind && this.editingContext) {
        const origin = dragKind === "anchor"
          ? this.editingContext.anchorWorld
          : this.editingContext.growthWorld;
        if (this.beginDrag(dragKind, event, origin)) {
          element.setPointerCapture?.(event.pointerId);
          return;
        }
      }
      this.pointerDown = { x: event.clientX, y: event.clientY };
    });
    element.addEventListener("pointermove", (event) => {
      if (!this.activeDrag) {
        return;
      }
      this.updateDrag(event);
    });
    element.addEventListener("pointerup", (event) => {
      if (event.button !== 0) {
        return;
      }
      if (this.activeDrag) {
        this.updateDrag(event);
        this.endDrag();
        element.releasePointerCapture?.(event.pointerId);
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
      if (this.activeDrag) {
        this.endDrag();
      }
    });
    element.addEventListener("pointercancel", () => {
      this.pointerDown = null;
      if (this.activeDrag) {
        this.endDrag();
      }
    });
  }

  pickFromPointerEvent(event, options = {}) {
    const hits = this.getPointerHits(event);
    if (!hits.length) {
      return null;
    }
    const hit = hits[0].object;
    if (options.silent) {
      return hit;
    }
    const selection = hit?.userData?.selection;
    if (!selection || !this.onSelect) {
      return hit;
    }
    this.onSelect(selection, "viewport");
    return hit;
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
    if (selection.kind === "edge") {
      const edge = this.latestModel?.edges?.find((item) => item.id === selection.id);
      if (!edge) {
        return [];
      }
      const keys = [];
      for (const jointId of asArray(edge.instanceJointIds)) {
        keys.push(this.makeSelectionKey("joint", jointId));
      }
      for (const partId of asArray(edge.instancePartIds)) {
        keys.push(this.makeSelectionKey("part", partId));
      }
      return keys;
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
    const tint = selection.kind === "joint"
      ? 0xffdb8a
      : (selection.kind === "edge" ? 0x9fc9e0 : 0x9fdba4);
    const wireHighlight = 0xfff57a;
    const gizmoHighlight = 0xffffff;
    for (const key of highlightKeys) {
      const objects = this.selectionObjects.get(key) || [];
      for (const object3d of objects) {
        const material = object3d.material;
        if (material && !Array.isArray(material)) {
          const role = String(object3d.userData.selectionRole || "");
          if (role === "part-wire") {
            if (material.color) {
              material.color.setHex(wireHighlight);
            }
            if (material.transparent) {
              material.opacity = 1.0;
            }
            continue;
          }
          if (role === "gizmo-wire" || role === "gizmo-geometry") {
            if (material.color) {
              material.color.setHex(gizmoHighlight);
            }
            if (material.emissive) {
              material.emissive.setHex(0x2a2a2a);
            }
            if (material.transparent) {
              material.opacity = 1.0;
            }
            continue;
          }
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
    this.clearGroup(this.toolGroup);
    this.selectionObjects.clear();
    this.pickables = [];
    this.toolPickables = [];

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
      const partSelection = { kind: "part", id: part.id };
      this.registerSelectable(partSelection, mesh, true, "part-fill");
      const edgeLines = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color: 0x0a1210,
          transparent: true,
          opacity: 0.6,
          depthTest: false,
        })
      );
      edgeLines.renderOrder = 3;
      this.registerSelectable(partSelection, edgeLines, false, "part-wire");
      mesh.add(edgeLines);
      this.segmentGroup.add(mesh);
    }

    const hingeJointColor = 0xf3bc67;
    const ballJointColor = 0x72ccff;
    const axisColor = 0x61d0c3;
    const limitColorX = 0xffd089;
    const limitColorY = 0x8dd3a1;
    const limitColorZ = 0x8cc9ff;

    for (const joint of model.joints) {
      const jointSelection = { kind: "joint", id: joint.id };
      const childPart = model.parts[joint.childPartIndex] || null;
      const hint = childPart
        ? childPart.center.clone().sub(joint.anchorWorld)
        : new THREE.Vector3(0, -1, 0);
      const basis = this.buildJointBasis(joint.axisWorld, hint);

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
      this.registerSelectable(jointSelection, stem, true);
      this.jointGroup.add(stem);

      const limitX = clamp(Math.abs(getEdgeLimit(joint.edge, "X")), 0, Math.PI * 0.98);
      const limitY = clamp(Math.abs(getEdgeLimit(joint.edge, "Y")), 0, Math.PI * 0.98);
      const limitZ = clamp(Math.abs(getEdgeLimit(joint.edge, "Z")), 0, Math.PI * 0.98);

      if (joint.jointType === "ball") {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.07, 16, 12),
          new THREE.MeshBasicMaterial({
            color: ballJointColor,
            transparent: true,
            opacity: 0.9,
            depthTest: false,
          })
        );
        marker.position.copy(joint.anchorWorld);
        marker.renderOrder = 20;
        this.registerSelectable(jointSelection, marker, true);
        this.jointGroup.add(marker);

        // Faint orthogonal circles establish a gimbal-like frame.
        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.y,
          axisB: basis.z,
          radius: 0.13,
          startAngle: -Math.PI,
          endAngle: Math.PI,
          color: limitColorX,
          opacity: 0.24,
          renderOrder: 20,
          segments: 34,
        });
        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.z,
          axisB: basis.x,
          radius: 0.13,
          startAngle: -Math.PI,
          endAngle: Math.PI,
          color: limitColorY,
          opacity: 0.21,
          renderOrder: 20,
          segments: 34,
        });
        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.x,
          axisB: basis.y,
          radius: 0.13,
          startAngle: -Math.PI,
          endAngle: Math.PI,
          color: limitColorZ,
          opacity: 0.21,
          renderOrder: 20,
          segments: 34,
        });

        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.y,
          axisB: basis.z,
          radius: 0.19,
          startAngle: -limitX,
          endAngle: limitX,
          color: limitColorX,
          opacity: 0.9,
          renderOrder: 22,
        });
        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.z,
          axisB: basis.x,
          radius: 0.23,
          startAngle: -limitY,
          endAngle: limitY,
          color: limitColorY,
          opacity: 0.88,
          renderOrder: 22,
        });
        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.x,
          axisB: basis.y,
          radius: 0.27,
          startAngle: -limitZ,
          endAngle: limitZ,
          color: limitColorZ,
          opacity: 0.88,
          renderOrder: 22,
        });
      } else {
        const axle = new THREE.Mesh(
          new THREE.CylinderGeometry(0.03, 0.03, 0.24, 16),
          new THREE.MeshBasicMaterial({
            color: hingeJointColor,
            transparent: true,
            opacity: 0.94,
            depthTest: false,
          })
        );
        axle.position.copy(joint.anchorWorld);
        axle.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), basis.x.clone());
        axle.renderOrder = 20;
        this.registerSelectable(jointSelection, axle, true);
        this.jointGroup.add(axle);

        const axisSpan = basis.x.clone().multiplyScalar(0.32);
        const axisGeo = new THREE.BufferGeometry().setFromPoints([
          joint.anchorWorld.clone().sub(axisSpan),
          joint.anchorWorld.clone().add(axisSpan),
        ]);
        const axis = new THREE.Line(
          axisGeo,
          new THREE.LineBasicMaterial({
            color: axisColor,
            transparent: true,
            opacity: 0.95,
            depthTest: false,
          })
        );
        axis.renderOrder = 21;
        this.registerSelectable(jointSelection, axis, true);
        this.jointGroup.add(axis);

        this.addJointArc(jointSelection, {
          center: joint.anchorWorld,
          axisA: basis.y,
          axisB: basis.z,
          radius: 0.245,
          startAngle: -limitX,
          endAngle: limitX,
          color: limitColorX,
          opacity: 0.92,
          renderOrder: 22,
          addBoundaryRays: true,
        });
      }
    }

    this.infoEl.textContent = `nodes: ${model.nodeCount} | edge genes: ${model.edges.length} | expanded parts: ${model.parts.length} | joints: ${model.joints.length}
Selection: click viewport object or outliner row.
Stage-4 tools: select non-root part/joint, then use Move anchor or Growth handle.`;
    this.rebuildEditingHandles();
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
  visualView.setToolEditCallback((kind, worldPoint) => {
    const ok = applyVisualToolMutation(kind, worldPoint);
    if (ok) {
      setInspectorApplyStatus(
        kind === "anchor"
          ? "Updated anchor from viewport handle."
          : "Updated growth direction from viewport handle.",
        "ok"
      );
    }
    return ok;
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
if (toolSelectBtn) {
  toolSelectBtn.addEventListener("click", () => {
    setVisualTool(VISUAL_TOOL_SELECT);
  });
}
if (toolMoveAnchorBtn) {
  toolMoveAnchorBtn.addEventListener("click", () => {
    setVisualTool(VISUAL_TOOL_MOVE_ANCHOR);
  });
}
if (toolGrowthDirBtn) {
  toolGrowthDirBtn.addEventListener("click", () => {
    setVisualTool(VISUAL_TOOL_GROWTH_DIR);
  });
}
if (addChildBtn) {
  addChildBtn.addEventListener("click", () => {
    addChildFromSelection();
  });
}
if (duplicateBranchBtn) {
  duplicateBranchBtn.addEventListener("click", () => {
    duplicateSelectedBranch();
  });
}
if (deleteSelectedBtn) {
  deleteSelectedBtn.addEventListener("click", () => {
    deleteSelectedEntity();
  });
}
if (outlinerResizeHandle) {
  outlinerResizeHandle.addEventListener("pointerdown", (event) => {
    beginPanelResize("outliner", event);
  });
  outlinerResizeHandle.addEventListener("dblclick", () => {
    applyPanelWidths(DEFAULT_OUTLINER_PANEL_WIDTH, getCurrentPanelWidths().inspector, true);
    if (visualView) {
      visualView.resize();
    }
  });
}
if (inspectorResizeHandle) {
  inspectorResizeHandle.addEventListener("pointerdown", (event) => {
    beginPanelResize("inspector", event);
  });
  inspectorResizeHandle.addEventListener("dblclick", () => {
    applyPanelWidths(getCurrentPanelWidths().outliner, DEFAULT_INSPECTOR_PANEL_WIDTH, true);
    if (visualView) {
      visualView.resize();
    }
  });
}
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
window.addEventListener("resize", () => {
  const current = getCurrentPanelWidths();
  applyPanelWidths(current.outliner, current.inspector, false);
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
initializePanelWidths();
setSidebarCollapsed(false);
setEditorMode("visual");
setVisualTool(VISUAL_TOOL_SELECT);
rebuildVisualFromCurrentGenome();
renderOutliner();
renderInspector();
void refreshCreatureList();
void refreshModeStatus();
