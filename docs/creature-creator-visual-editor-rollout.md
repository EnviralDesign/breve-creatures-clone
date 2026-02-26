# Creature Creator Visual Editor Rollout

This document defines a staged rollout from the current JSON-first creator to a full visual editor with synchronized text mode.

## Goal

Build a creature authoring tool where:

- visual editing is primary (3D viewport + outliner + attribute editor),
- JSON editing remains available (text mode),
- both modes edit the same underlying source in real time,
- authored creatures can be loaded from current/best genomes for fast validation.

## Product Direction (Non-Negotiables)

- Keep the creator as a dedicated app surface (`/creator.html`).
- Keep the left sidebar as "creator controls" (save/load/mode/app-level actions).
- Add an explicit top-level mode switch: `Visual` vs `JSON`.
- Use Three.js in the creator viewport (same rendering stack direction as main evolution UI).
- Treat outliner + viewport selection as first-class, not optional.
- Keep attribute editing as fallback for non-visual parameters.

## Data Model Principle

Single source of truth:

- one in-memory `genome` object,
- visual mode reads/writes it,
- JSON mode reads/writes it,
- outliner reads/writes it,
- attribute editor reads/writes it.

No independent visual-only state that can drift from genome state.

## Rollout Stages

## Stage 1 - Read-Only Visual Parity

Focus:

- Prove that visual rendering matches genome JSON structure.
- No visual editing yet.

Work:

- Add `Visual | JSON` mode switch.
- Render full creature from current genome in Three.js.
- Render body segments and joint markers (read-only).
- Add "refresh/rebuild from genome" action to validate JSON edits.
- Keep existing JSON editor fully functional.

Exit criteria:

- Editing JSON and rebuilding viewport consistently shows correct geometry/joint placement.
- Load `current` and `best` genomes and get stable visual parity.

## Stage 2 - Selection Model + Outliner

Focus:

- Introduce coherent selection architecture before editing actions.

Work:

- Add hierarchical outliner (root -> nodes/parts -> joints).
- Bidirectional selection sync:
  - click viewport object -> selects outliner item,
  - click outliner item -> highlights viewport object.
- Add inspect-only attribute panel for selected item.

Exit criteria:

- Every selectable visual element is represented in outliner.
- Selection is stable, synchronized, and debuggable.

## Stage 3 - Attribute Editing (Form-First)

Focus:

- Enable safe editing through structured controls before gizmo interactions.

Work:

- Enable attribute panel edits for selected element:
  - part dims/mass,
  - joint anchors/axes/directions/limits/type,
  - controller/effector parameters (form fields),
  - lock profile controls.
- Immediate mutation of in-memory genome.
- Real-time refresh of affected viewport entities.
- JSON pane updates from same source when opened.

Exit criteria:

- Users can fully author a creature without touching raw JSON.
- Switching modes does not lose state or introduce diffs beyond intended edits.

## Stage 4 - Visual Editing Interactions

Focus:

- Promote high-value fields from form edits to direct 3D manipulation.

Work:

- Add viewport authoring interactions:
  - create child segment from selected part/joint anchor,
  - move anchor with gizmo,
  - adjust growth direction with handle,
  - optional limit visualization/handles.
- Keep attribute panel as exact-value override.

Exit criteria:

- Common structure-edit flows are faster in visual mode than JSON mode.
- Generated genome remains valid and simulation-safe after manipulations.

## Stage 5 - Neural Authoring UX + Validation Workflows

Focus:

- Make controller authoring practical without forcing raw JSON for all neural fields.

Work:

- Add neural inspector/editor:
  - local/global neuron counts,
  - activations/leak/bias,
  - effector weights/gains.
- Keep advanced JSON fallback for power users.
- Add validation helpers:
  - schema validation,
  - graph/joint sanity checks,
  - simulation preflight hints.

Exit criteria:

- Typical controller edits can be made visually/forms-first.
- Invalid creature states are surfaced early with actionable errors.

## Stage 6 - Usability Hardening + Authoring Throughput

Focus:

- Make creator efficient for long iterative sessions.

Work:

- Undo/redo stack.
- Dirty-state tracking and change summary.
- Keyboard shortcuts and quick actions.
- Better presets/templates and cloning workflows.
- Performance optimization for large graphs.

Exit criteria:

- Reliable daily-driver authoring flow.
- Low-friction iterative loop: load -> edit -> apply -> test -> save.

## Cross-Stage Guardrails

- Keep JSON mode available in every stage.
- Prefer incremental vertical slices over big-bang rewrites.
- Avoid silent auto-fixes that mutate authored intent.
- Keep backend format authoritative; creator is an editor, not an alternate schema.

## Immediate Next Step

Start Stage 1 now:

- add mode switch,
- implement read-only Three.js reconstruction from genome,
- render visible joint markers,
- wire manual rebuild from JSON to verify parity.
