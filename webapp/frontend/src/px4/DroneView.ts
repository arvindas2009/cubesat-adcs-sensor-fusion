// Attitude scene of the PX4 replay (docs/specs/px4_replay.md §3.3): a quadcopter modelled in FRD body
// coordinates (x forward, y right, z down), drawn at the origin. Ours is solid, the EKF2 reference is a
// translucent amber wireframe ghost (hidden while its quaternion is null). Both are oriented with
// bodyNedToWorld (math/ned.ts); north is world +X, east world +Z, up world +Y.
import * as THREE from 'three';
import { SCREEN } from '../theme/palette';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { bodyNedToWorld } from '../math/ned';
import { AXIS_COLORS, COLORS } from '../adcs/theme';
import { VIEW_LEGENDS } from '../ui/glyphs';
import { renderLegend } from '../ui/legend';
import { createViewport, disposeObject, labelElement, observeSize } from '../adcs/viewport';
import { finiteRow } from './series';
import { ARM, ROTOR_R, buildQuadcopter } from '../models/quadcopter';
import { loadRoleModel } from '../models/gltf';
import type { Row } from './types';

const GRID_Y = -0.95;

function hex(c: string): number {
  return new THREE.Color(c).getHex();
}

/** Ours: the detailed solid quadcopter plus a small body triad (x red, y green, z blue) in FRD coordinates. */
function buildOurs(): { root: THREE.Group; model: THREE.Group } {
  const root = new THREE.Group();
  const model = buildQuadcopter({ solid: true });
  root.add(model);
  const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  dirs.forEach((d, i) => root.add(new THREE.ArrowHelper(d, new THREE.Vector3(), 0.95, hex(AXIS_COLORS[i]), 0.22, 0.12)));
  return { root, model };
}

export class DroneView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly labels = new CSS2DRenderer();
  private readonly controls: OrbitControls;
  private readonly oursParts = buildOurs();
  private readonly ours = this.oursParts.root;
  private readonly ghost = buildQuadcopter({ solid: false, ghostColor: hex(COLORS.ekf2) });
  private dirty = true;

  constructor(container: HTMLElement) {
    const vp = createViewport(container, 'Drone attitude view: our estimate and the EKF2 reference, with the body axes');
    this.renderer = vp.renderer;
    this.labels.domElement.className = 'label-layer';
    vp.body.append(this.labels.domElement);

    this.scene.add(new THREE.AmbientLight(0xb8c6e8, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(3, 8, 5);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8fb0ff, 0.7);
    fill.position.set(-5, -2, -4);
    this.scene.add(fill);

    // Local horizontal with north (world +X) and east (world +Z) marked.
    const grid = new THREE.GridHelper(8, 16, new THREE.Color(SCREEN.gridMajor), new THREE.Color(SCREEN.gridMinor));
    grid.position.y = GRID_Y;
    this.scene.add(grid);
    for (const [text, pos] of [
      ['N', new THREE.Vector3(4.35, GRID_Y, 0)],
      ['E', new THREE.Vector3(0, GRID_Y, 4.35)],
    ] as const) {
      const o = new CSS2DObject(labelElement(text, COLORS.nadir, 'strong'));
      o.position.copy(pos);
      this.scene.add(o);
    }

    this.ours.visible = false;
    this.ghost.visible = false;
    this.ghost.scale.setScalar(1.04);
    this.scene.add(this.ours, this.ghost);
    // An open-licence model listed in public/models/manifest.json replaces the procedural one (ours only).
    void loadRoleModel('quadcopter', 2 * (ARM + ROTOR_R)).then((m) => {
      if (!m) return;
      disposeObject(this.oursParts.model);
      this.ours.add(m);
      this.dirty = true;
    });

    this.camera.position.set(-4.2, 2.8, -4.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0);
    this.controls.enableDamping = false;
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 30;
    this.controls.addEventListener('change', () => (this.dirty = true));

    vp.overlay.append(this.buildLegend());
    observeSize(vp.body, (w, h) => {
      this.renderer.setSize(w, h, false);
      this.labels.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.dirty = true;
    });
  }

  private buildLegend(): HTMLElement {
    const lg = renderLegend(VIEW_LEGENDS.drone(COLORS), { surface: 'screen' });
    lg.el.classList.add('vp-legend');
    return lg.el;
  }

  /** Orient both drones from wire quaternions [w, x, y, z] (body FRD to NED); null / non-finite hides that drone. */
  update(qOurs: Row | null | undefined, qEkf2: Row | null | undefined): void {
    const a = finiteRow(qOurs);
    const b = finiteRow(qEkf2);
    this.ours.visible = a !== null;
    if (a) this.ours.quaternion.copy(bodyNedToWorld(a));
    this.ghost.visible = b !== null;
    if (b) this.ghost.quaternion.copy(bodyNedToWorld(b));
    this.dirty = true;
  }

  /** No replay loaded: hide both drones. */
  clear(): void {
    this.ours.visible = false;
    this.ghost.visible = false;
    this.dirty = true;
  }

  invalidate(): void {
    this.dirty = true;
  }

  render(): void {
    if (!this.dirty) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.dirty = false;
  }

  dispose(): void {
    disposeObject(this.ours);
    disposeObject(this.ghost);
    this.controls.dispose();
    this.renderer.dispose();
  }
}
