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
import { createViewport, disposeObject, labelElement, observeSize } from '../adcs/viewport';
import { finiteRow } from './series';
import type { Row } from './types';

const ARM = 1.15; // centre to rotor centre [model units]
const ROTOR_R = 0.5;
const GRID_Y = -0.95;

function hex(c: string): number {
  return new THREE.Color(c).getHex();
}

/** The quadcopter in FRD coordinates. `solid` = ours; otherwise the translucent amber wireframe ghost. */
function buildDrone(solid: boolean): THREE.Group {
  const g = new THREE.Group();
  const bodyGeo = new THREE.BoxGeometry(0.62, 0.42, 0.16); // x forward, y right, z down
  const armGeo = new THREE.BoxGeometry(ARM * 2, 0.07, 0.05);
  const rotorGeo = new THREE.CylinderGeometry(ROTOR_R, ROTOR_R, 0.02, 32);
  rotorGeo.rotateX(Math.PI / 2); // cylinder axis y -> z (the rotor axis in FRD)
  const noseGeo = new THREE.ConeGeometry(0.1, 0.34, 16);
  noseGeo.rotateZ(-Math.PI / 2); // cone axis +y -> +x

  const ghostMat = new THREE.LineBasicMaterial({ color: hex(COLORS.ekf2), transparent: true, opacity: 0.75, depthWrite: false });
  // Ours: shaded meshes. Ghost: edge outlines only (a wireframe would draw every rotor spoke and hide ours).
  const part = (geo: THREE.BufferGeometry, color: number, opacity = 1): THREE.Object3D =>
    solid
      ? new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.25, transparent: opacity < 1, opacity }))
      : new THREE.LineSegments(new THREE.EdgesGeometry(geo, 20), ghostMat);

  g.add(part(bodyGeo, 0xd2d9e6));
  for (const a of [Math.PI / 4, -Math.PI / 4]) {
    const arm = part(armGeo, 0x8d98ad);
    arm.rotation.z = a; // two bars across the centre form the X frame
    g.add(arm);
  }
  for (const sx of [1, -1]) {
    for (const sy of [1, -1]) {
      const rotor = part(rotorGeo, 0xaeb8cc, 0.55);
      rotor.position.set((sx * ARM) / Math.SQRT2, (sy * ARM) / Math.SQRT2, -0.07); // above the arms (up = -z)
      g.add(rotor);
    }
  }
  const nose = part(noseGeo, 0xff5a5a);
  nose.position.set(0.47, 0, 0);
  g.add(nose);

  if (solid) {
    // Small body triad (x red, y green, z blue) in FRD coordinates.
    const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
    dirs.forEach((d, i) => g.add(new THREE.ArrowHelper(d, new THREE.Vector3(), 0.95, hex(AXIS_COLORS[i]), 0.22, 0.12)));
  }
  return g;
}

export class DroneView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly labels = new CSS2DRenderer();
  private readonly controls: OrbitControls;
  private readonly ours = buildDrone(true);
  private readonly ghost = buildDrone(false);
  private dirty = true;

  constructor(container: HTMLElement) {
    const vp = createViewport(container);
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
    const items: [string, string][] = [
      ['ours', COLORS.nadir],
      ['EKF2 (reference)', COLORS.ekf2],
      ['body x', COLORS.axisX],
      ['body y', COLORS.axisY],
      ['body z', COLORS.axisZ],
    ];
    const el = document.createElement('div');
    el.className = 'vp-legend';
    for (const [name, color] of items) {
      const s = document.createElement('span');
      s.innerHTML = `<i style="background:${color}"></i>${name}`;
      el.append(s);
    }
    return el;
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
