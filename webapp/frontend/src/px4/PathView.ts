// 3D NED path scene of the PX4 replay (docs/specs/px4_replay.md §3.3). Ours is a solid teal line, the EKF2
// reference a dashed amber line; 1 world unit = 1 m via nedToWorld. The whole path is drawn faint and the
// part up to the current time bright (the same geometry with a draw range). Null EKF2 samples split the
// line, so nothing is drawn through a gap. The camera fits both paths when a replay is loaded.
import * as THREE from 'three';
import { SCREEN } from '../theme/palette';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { nedToWorld } from '../math/ned';
import { COLORS } from '../adcs/theme';
import { VIEW_LEGENDS } from '../ui/glyphs';
import { renderLegend } from '../ui/legend';
import { createViewport, disposeObject, labelElement, observeSize } from '../adcs/viewport';
import { finiteRow, pathSegments } from './series';
import type { PathSegment } from './series';
import type { Replay, Row } from './types';

const MIN_EXTENT = 2; // m: a hover log may span only a metre or two

function hex(c: string): number {
  return new THREE.Color(c).getHex();
}

/** Smallest of 1, 2, 5 x 10^n that is >= x (x > 0). */
function niceStep(x: number): number {
  const e = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 5, 10]) if (m * e >= x) return m * e;
  return 10 * e;
}

interface Track {
  rows: readonly Row[];
  segments: PathSegment[];
  bright: THREE.Line[];
  marker: THREE.Mesh;
}

export class PathView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly labels = new CSS2DRenderer();
  private readonly controls: OrbitControls;
  private content: THREE.Group | null = null;
  private tracks: Track[] = [];
  private dirty = true;

  constructor(container: HTMLElement) {
    const vp = createViewport(container, 'Path view: the flight path of our estimate and of the EKF2 reference, in north-east-down metres');
    this.renderer = vp.renderer;
    this.labels.domElement.className = 'label-layer';
    vp.body.append(this.labels.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    this.camera.position.set(-8, 6, -10);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
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
    const lg = renderLegend(VIEW_LEGENDS.path(COLORS), { surface: 'screen' });
    lg.el.classList.add('vp-legend');
    return lg.el;
  }

  /** Release the geometry of the previous replay. */
  private clearContent(): void {
    if (this.content) disposeObject(this.content);
    this.content = null;
    this.tracks = [];
  }

  clear(): void {
    this.clearContent();
    this.dirty = true;
  }

  /** Build the lines, markers and ground grid for a replay and fit the camera to both paths. */
  setReplay(rep: Replay): void {
    this.clearContent();
    const group = new THREE.Group();
    this.content = group;
    this.scene.add(group);

    const box = new THREE.Box3();
    for (const rows of [rep.ours.p_ned_m, rep.ekf2.p_ned_m]) {
      for (const r of rows) {
        const p = finiteRow(r);
        if (p) box.expandByPoint(nedToWorld(p));
      }
    }
    if (box.isEmpty()) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const extent = Math.max(MIN_EXTENT, size.x, size.y, size.z);

    const first = finiteRow(rep.ours.p_ned_m.find((r) => finiteRow(r) !== null) ?? null);
    const groundY = first ? nedToWorld(first).y : box.min.y;

    // Ground grid at the height of the first sample, with N (+X) and E (+Z) marked.
    const step = niceStep(extent / 8);
    const half = step * (Math.ceil(extent / 2 / step) + 1);
    const grid = new THREE.GridHelper(half * 2, Math.round((half * 2) / step), new THREE.Color(SCREEN.gridMajor), new THREE.Color(SCREEN.gridMinor));
    grid.position.set(Math.round(center.x / step) * step, groundY, Math.round(center.z / step) * step);
    group.add(grid);
    for (const [text, pos] of [
      ['N', new THREE.Vector3(grid.position.x + half * 1.04, groundY, grid.position.z)],
      ['E', new THREE.Vector3(grid.position.x, groundY, grid.position.z + half * 1.04)],
    ] as const) {
      const o = new CSS2DObject(labelElement(text, COLORS.nadir, 'strong'));
      o.position.copy(pos);
      group.add(o);
    }

    const markerR = extent * 0.014;
    this.tracks = [
      this.buildTrack(group, rep.ours.p_ned_m, COLORS.ours, false, extent, markerR),
      this.buildTrack(group, rep.ekf2.p_ned_m, COLORS.ekf2, true, extent, markerR),
    ];

    if (first) {
      const start = new THREE.Mesh(new THREE.OctahedronGeometry(markerR * 1.4), new THREE.MeshBasicMaterial({ color: hex(COLORS.nadir) }));
      start.position.copy(nedToWorld(first));
      group.add(start);
    }

    // Fit the camera: look at the centre of the bounding box from a 3/4 view.
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const dist = (extent * 0.75) / Math.tan(fov / 2) + extent * 0.5;
    this.camera.near = Math.max(0.01, dist / 500);
    this.camera.far = dist * 50;
    this.camera.position.copy(center).add(new THREE.Vector3(-0.8, 0.65, -1).normalize().multiplyScalar(dist));
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.minDistance = extent * 0.05;
    this.controls.maxDistance = dist * 10;
    this.controls.update();
    this.dirty = true;
  }

  private buildTrack(group: THREE.Group, rows: readonly Row[], color: string, dashed: boolean, extent: number, markerR: number): Track {
    const segments = pathSegments(rows);
    const bright: THREE.Line[] = [];
    for (const seg of segments) {
      const n = seg.end - seg.start;
      const pos = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const p = finiteRow(rows[seg.start + i]);
        const w = nedToWorld(p ?? [0, 0, 0]);
        pos[3 * i] = w.x;
        pos[3 * i + 1] = w.y;
        pos[3 * i + 2] = w.z;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const mk = (g: THREE.BufferGeometry, opacity: number): THREE.Line => {
        const mat = dashed
          ? new THREE.LineDashedMaterial({ color: hex(color), transparent: true, opacity, dashSize: extent / 120, gapSize: extent / 240 })
          : new THREE.LineBasicMaterial({ color: hex(color), transparent: true, opacity });
        return new THREE.Line(g, mat);
      };
      const faint = mk(geo, 0.28);
      if (dashed) faint.computeLineDistances(); // adds the lineDistance attribute to `geo`
      // The bright line has its own geometry (its own draw range) but shares the attribute buffers.
      const litGeo = new THREE.BufferGeometry();
      for (const name of Object.keys(geo.attributes)) litGeo.setAttribute(name, geo.getAttribute(name));
      litGeo.setDrawRange(0, 0);
      const lit = mk(litGeo, 1);
      group.add(faint, lit);
      bright.push(lit);
    }
    const marker = new THREE.Mesh(new THREE.SphereGeometry(markerR, 16, 12), new THREE.MeshBasicMaterial({ color: hex(color) }));
    marker.visible = false;
    group.add(marker);
    return { rows, segments, bright, marker };
  }

  /** Show the path up to grid sample `i` brightly and put the markers at that sample. */
  update(i: number): void {
    for (const tr of this.tracks) {
      tr.segments.forEach((seg, k) => {
        const count = Math.min(seg.end - seg.start, Math.max(0, i + 1 - seg.start));
        tr.bright[k].geometry.setDrawRange(0, count);
      });
      const p = finiteRow(tr.rows[i]);
      tr.marker.visible = p !== null;
      if (p) tr.marker.position.copy(nedToWorld(p));
    }
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
    this.clearContent();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
