// Display-frame scene (docs/specs/frontend.md §2): the 3U body drawn to scale in the local frame
// (zenith up, nadir down), the estimate as a cyan ghost, nadir / Sun / B arrows and the wheel bars.
// 1 world unit = 10 cm. Frame matrices go through math/frames.ts.
import * as THREE from 'three';
import { SCREEN } from '../theme/palette';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { displayMatrixToWorld, zUpToWorld } from '../math/frames';
import type { Frame, Info, Vec3 } from '../types';
import { AXIS_COLORS, COLORS } from './theme';
import { createViewport, labelElement, observeSize } from './viewport';

const UNIT = 10; // world units per metre (1 unit = 10 cm)
const GRID_Y = -2.5;
const NADIR_LEN = 3.9;
const SUN_LEN = 4.5;
const B_LEN = 3.3;
const WHEEL_H = 2.6; // bar height at 100 % of the maximum wheel speed
const DEFAULT_SIZE: Vec3 = [0.1, 0.1, 0.34];
const RAD_S_TO_RPM = 60 / (2 * Math.PI);

function hex(c: string): number {
  return new THREE.Color(c).getHex();
}

type FaceKind = 'panel' | 'top' | 'bottom';

/** Canvas texture for one face of the box: solar-cell grid, axis label, and the +z / -z end styles. */
function faceTexture(label: string, wUnits: number, hUnits: number, kind: FaceKind, color: string): THREE.CanvasTexture {
  const px = 150;
  const w = Math.max(32, Math.round(wUnits * px));
  const h = Math.max(32, Math.round(hUnits * px));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d');
  if (g) {
    if (kind === 'panel') {
      g.fillStyle = '#17315c';
      g.fillRect(0, 0, w, h);
      g.strokeStyle = 'rgba(160,190,235,0.28)';
      g.lineWidth = 2;
      const longer = Math.max(w, h);
      const cells = Math.max(1, Math.round(longer / px / 1.133)); // ~one cell per 1U (11.3 cm)
      for (let i = 1; i < cells; i++) {
        g.beginPath();
        if (w >= h) {
          g.moveTo((i * w) / cells, 0);
          g.lineTo((i * w) / cells, h);
        } else {
          g.moveTo(0, (i * h) / cells);
          g.lineTo(w, (i * h) / cells);
        }
        g.stroke();
      }
    } else if (kind === 'top') {
      g.fillStyle = '#c9d3e6';
      g.fillRect(0, 0, w, h);
      g.strokeStyle = color;
      g.lineWidth = Math.max(4, Math.min(w, h) * 0.05);
      g.beginPath();
      g.arc(w / 2, h / 2, Math.min(w, h) * 0.38, 0, 2 * Math.PI);
      g.stroke();
    } else {
      g.fillStyle = '#0d1018';
      g.fillRect(0, 0, w, h);
      g.strokeStyle = '#c98d12';
      g.lineWidth = Math.max(5, Math.min(w, h) * 0.07);
      for (let k = -h; k < w + h; k += Math.min(w, h) * 0.3) {
        g.beginPath();
        g.moveTo(k, 0);
        g.lineTo(k + h, h);
        g.stroke();
      }
    }
    const fs = Math.round(Math.min(w, h) * (kind === 'panel' ? 0.34 : 0.4));
    g.font = `700 ${fs}px system-ui, sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    if (kind === 'bottom') {
      g.fillStyle = 'rgba(8,10,16,0.85)';
      g.fillRect(w / 2 - fs * 1.1, h / 2 - fs * 0.75, fs * 2.2, fs * 1.5);
    }
    g.fillStyle = color;
    g.fillText(label, w / 2, h / 2);
    g.strokeStyle = 'rgba(255,255,255,0.35)';
    g.lineWidth = 3;
    g.strokeRect(1.5, 1.5, w - 3, h - 3);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function dashedAxis(dir: THREE.Vector3, len: number, color: string): THREE.Line {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), dir.clone().multiplyScalar(len)]);
  const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color: hex(color), dashSize: 0.16, gapSize: 0.1, transparent: true, opacity: 0.9 }));
  l.computeLineDistances();
  return l;
}

interface Bar {
  mesh: THREE.Mesh;
  mat: THREE.MeshStandardMaterial;
  label: HTMLDivElement;
}

export class SatView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly labels = new CSS2DRenderer();
  private readonly controls: OrbitControls;
  private readonly body = new THREE.Group();
  private readonly ghost = new THREE.Group();
  private size: Vec3 = DEFAULT_SIZE;
  private maxRpm = 1;
  private readonly nadirArrow: THREE.ArrowHelper;
  private readonly sunArrow: THREE.ArrowHelper;
  private readonly bArrow: THREE.ArrowHelper;
  private readonly sunLabel: HTMLDivElement;
  private readonly sunLabelObj: CSS2DObject;
  private readonly bars: Bar[] = [];
  private lit = true;
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

    // Local horizontal.
    const grid = new THREE.GridHelper(14, 14, new THREE.Color(SCREEN.gridMajor), new THREE.Color(SCREEN.gridMinor));
    grid.position.y = GRID_Y;
    this.scene.add(grid);

    // Body and ghost are body-fixed groups driven by a matrix (columns = body axes in world coordinates).
    this.body.matrixAutoUpdate = false;
    this.ghost.matrixAutoUpdate = false;
    this.ghost.visible = false;
    this.scene.add(this.body, this.ghost);
    this.buildBody();

    // Reference arrows in the display frame.
    this.nadirArrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(), NADIR_LEN, hex(COLORS.nadir), 0.4, 0.2);
    this.sunArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), SUN_LEN, hex(COLORS.sun), 0.4, 0.2);
    this.bArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), B_LEN, hex(COLORS.field), 0.4, 0.2);
    this.scene.add(this.nadirArrow, this.sunArrow, this.bArrow);
    this.addLabel(this.scene, 'NADIR', COLORS.nadir, new THREE.Vector3(0, -NADIR_LEN - 0.3, 0));
    this.sunLabel = labelElement('SUN', COLORS.sun, 'strong');
    this.sunLabelObj = new CSS2DObject(this.sunLabel);
    this.scene.add(this.sunLabelObj);
    const bLabel = labelElement('B', COLORS.field, 'strong');
    this.bLabelObj = new CSS2DObject(bLabel);
    this.scene.add(this.bLabelObj);

    this.buildWheels();

    this.camera.position.set(6.5, 4.2, 8.5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -0.4, 0);
    this.controls.enableDamping = false;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 40;
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

  private readonly bLabelObj: CSS2DObject;

  private addLabel(parent: THREE.Object3D, text: string, color: string, pos: THREE.Vector3, strong = true): CSS2DObject {
    const o = new CSS2DObject(labelElement(text, color, strong ? 'strong' : ''));
    o.position.copy(pos);
    parent.add(o);
    return o;
  }

  private buildLegend(): HTMLElement {
    const items: [string, string][] = [
      ['body x', COLORS.axisX],
      ['body y', COLORS.axisY],
      ['body z', COLORS.axisZ],
      ['estimate', COLORS.estimate],
      ['nadir', COLORS.nadir],
      ['Sun', COLORS.sun],
      ['B field', COLORS.field],
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

  /** (Re)build the box, ghost and axes for the given body size (m). */
  private buildBody(): void {
    for (const grp of [this.body, this.ghost]) {
      for (const ch of [...grp.children]) {
        grp.remove(ch);
      }
    }
    const [sx, sy, sz] = this.size.map((m) => m * UNIT) as Vec3;

    const faces: THREE.MeshStandardMaterial[] = [
      ['+X', sz, sy, 'panel', COLORS.axisX],
      ['−X', sz, sy, 'panel', COLORS.axisX],
      ['+Y', sx, sz, 'panel', COLORS.axisY],
      ['−Y', sx, sz, 'panel', COLORS.axisY],
      ['+Z', sx, sy, 'top', COLORS.axisZ],
      ['−Z', sx, sy, 'bottom', COLORS.axisZ],
    ].map(
      ([label, w, h, kind, color]) =>
        new THREE.MeshStandardMaterial({
          map: faceTexture(label as string, w as number, h as number, kind as FaceKind, color as string),
          roughness: 0.65,
          metalness: 0.15,
        }),
    );
    const geom = new THREE.BoxGeometry(sx, sy, sz);
    this.body.add(new THREE.Mesh(geom, faces));
    this.body.add(new THREE.LineSegments(new THREE.EdgesGeometry(geom), new THREE.LineBasicMaterial({ color: 0xcfdcf5, transparent: true, opacity: 0.8 })));

    const half = [sx / 2, sy / 2, sz / 2];
    const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
    const names = ['x', 'y', 'z'];
    dirs.forEach((d, i) => {
      const len = half[i] + 1.05;
      this.body.add(new THREE.ArrowHelper(d, new THREE.Vector3(), len, hex(AXIS_COLORS[i]), 0.38, 0.2));
      this.addLabel(this.body, names[i], AXIS_COLORS[i], d.clone().multiplyScalar(len + 0.22));
    });

    // Ghost: translucent cyan box, dashed axes.
    const gs = 1.05;
    const gg = new THREE.BoxGeometry(sx * gs, sy * gs, sz * gs);
    this.ghost.add(new THREE.Mesh(gg, new THREE.MeshBasicMaterial({ color: hex(COLORS.estimate), transparent: true, opacity: 0.16, depthWrite: false })));
    this.ghost.add(new THREE.LineSegments(new THREE.EdgesGeometry(gg), new THREE.LineBasicMaterial({ color: hex(COLORS.estimate), transparent: true, opacity: 0.9 })));
    dirs.forEach((d, i) => this.ghost.add(dashedAxis(d, half[i] + 1.05, COLORS.estimate)));
    this.addLabel(this.ghost, 'est', COLORS.estimate, new THREE.Vector3(0, 0, half[2] * gs + 0.35), false);
    this.dirty = true;
  }

  private buildWheels(): void {
    const names = ['x', 'y', 'z'];
    const x = -6.1; // left edge of the grid, away from the body
    const z0 = 1.0;
    const dz = 1.1;
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(0.7, WHEEL_H, 3.5)),
      new THREE.LineBasicMaterial({ color: new THREE.Color(SCREEN.gridMajor).getHex(), transparent: true, opacity: 0.6 }),
    );
    frame.position.set(x, GRID_Y + WHEEL_H / 2, z0 + dz);
    this.scene.add(frame);
    this.addLabel(this.scene, 'WHEELS [rpm]', COLORS.nadir, new THREE.Vector3(x, GRID_Y + WHEEL_H + 0.35, z0 + dz), false);
    for (let i = 0; i < 3; i++) {
      const z = z0 + dz * i;
      const mat = new THREE.MeshStandardMaterial({ color: hex(COLORS.wheelPos), roughness: 0.6 });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1, 0.5), mat);
      mesh.position.set(x, GRID_Y, z);
      mesh.scale.y = 0.001;
      this.scene.add(mesh);
      const label = labelElement(`${names[i]} —`, AXIS_COLORS[i]);
      const obj = new CSS2DObject(label);
      obj.position.set(x, GRID_Y - 0.3, z);
      this.scene.add(obj);
      this.bars.push({ mesh, mat, label });
    }
  }

  /** New session (or reset): size the body and the wheel scale from `info`. */
  setInfo(info: Info): void {
    const s = info.body_size_m;
    if (s[0] !== this.size[0] || s[1] !== this.size[1] || s[2] !== this.size[2]) {
      this.size = s;
      this.buildBody();
    }
    this.maxRpm = Math.max(1e-9, info.wheel_max_speed_rad_s * RAD_S_TO_RPM);
    this.dirty = true;
  }

  update(f: Frame): void {
    const d = f.display;
    this.body.matrix.copy(displayMatrixToWorld(d.C));
    this.body.matrixWorldNeedsUpdate = true;
    if (d.C_est) {
      this.ghost.matrix.copy(displayMatrixToWorld(d.C_est));
      this.ghost.matrixWorldNeedsUpdate = true;
      this.ghost.visible = true;
    } else {
      this.ghost.visible = false;
    }

    const sun = zUpToWorld(d.sun).normalize();
    this.sunArrow.setDirection(sun);
    this.sunLabelObj.position.copy(sun).multiplyScalar(SUN_LEN + 0.35);
    if (f.lit !== this.lit) {
      this.lit = f.lit;
      const col = f.lit ? COLORS.sun : COLORS.sunOff;
      this.sunArrow.setColor(hex(col));
      this.sunLabel.textContent = f.lit ? 'SUN' : 'ECLIPSE';
      this.sunLabel.style.color = col;
    }
    const b = zUpToWorld(d.b).normalize();
    this.bArrow.setDirection(b);
    this.bLabelObj.position.copy(b).multiplyScalar(B_LEN + 0.3);

    const rpm = f.metrics.wheel_speed_rpm;
    for (let i = 0; i < 3; i++) {
      const bar = this.bars[i];
      const h = Math.min(1, Math.abs(rpm[i]) / this.maxRpm) * WHEEL_H;
      bar.mesh.scale.y = Math.max(0.001, h);
      bar.mesh.position.y = GRID_Y + Math.max(0.001, h) / 2;
      bar.mat.color.set(rpm[i] >= 0 ? COLORS.wheelPos : COLORS.wheelNeg);
      const sign = rpm[i] >= 0 ? '+' : '−';
      const txt = `${['x', 'y', 'z'][i]} ${sign}${Math.abs(rpm[i]).toFixed(0)}`;
      if (bar.label.textContent !== txt) bar.label.textContent = txt;
    }
    this.dirty = true;
  }

  render(): void {
    if (!this.dirty) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.dirty = false;
  }
}
