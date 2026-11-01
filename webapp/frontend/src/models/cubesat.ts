// Detailed 3U CubeSat built from primitives (docs/specs/ui_dashboard.md §4). Everything is placed in the
// body frame used by SatView: x, y across the 10 cm faces, z along the 34 cm long axis, so the attitude
// mapping (math/frames.ts) is unchanged. Units are the caller's (SatView: 1 unit = 10 cm). The bus can be
// given the caller's per-face materials (SatView's labelled faces); everything else is decoration.
// DOM-free unless a document exists (cell textures), so the vitest node environment can build it.
import * as THREE from 'three';

export type Vec3 = [number, number, number];

export interface CubeSatOptions {
  /** Bus size [sx, sy, sz] in model units (SatView: 1, 1, 3.4). */
  size: Vec3;
  /** Six materials for the bus box (+X, -X, +Y, -Y, +Z, -Z); a cell texture is used when omitted. */
  bodyMaterials?: THREE.Material[];
  /** Draw the deployable wings (default true). */
  wings?: boolean;
}

export interface CubeSatModel {
  readonly group: THREE.Group;
  /** The bus box mesh alone (its bounding box equals `size`). */
  readonly bus: THREE.Mesh;
  /** The two deployed wings, +Y first. */
  readonly wings: THREE.Group[];
  /** Half extents of the whole model (bus + hardware) per body axis. */
  readonly halfExtent: Vec3;
}

const RAIL = 0.085; // rail cross-section, as a fraction of the 1U face (8.5 mm on a 10 cm face)
const WING_GAP = 0.08;

let cellTex: THREE.Texture | null = null;

/** A generic solar-cell texture (one 2x4 cell string per 1U); null without a DOM. */
function cellTexture(): THREE.Texture | null {
  if (typeof document === 'undefined') return null;
  if (cellTex) return cellTex;
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 512;
  const g = c.getContext('2d');
  if (!g) return null;
  g.fillStyle = '#c3cad6'; // substrate
  g.fillRect(0, 0, 256, 512);
  const cw = 256 / 2;
  const ch = 512 / 4;
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 4; j++) {
      const x = i * cw + 6;
      const y = j * ch + 6;
      const grad = g.createLinearGradient(x, y, x + cw, y + ch);
      grad.addColorStop(0, '#1b2f5e');
      grad.addColorStop(1, '#10204a');
      g.fillStyle = grad;
      g.beginPath();
      g.moveTo(x + 10, y);
      g.lineTo(x + cw - 12, y);
      g.lineTo(x + cw - 12, y + ch - 22);
      g.lineTo(x + cw - 22, y + ch - 12);
      g.lineTo(x, y + ch - 12);
      g.lineTo(x, y + 10);
      g.closePath();
      g.fill();
      g.strokeStyle = 'rgba(190,205,235,0.35)'; // busbars and fingers
      g.lineWidth = 1;
      for (let k = 1; k < 10; k++) {
        g.beginPath();
        g.moveTo(x + (k * (cw - 12)) / 10, y);
        g.lineTo(x + (k * (cw - 12)) / 10, y + ch - 12);
        g.stroke();
      }
    }
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  cellTex = t;
  return t;
}

function std(color: number, roughness: number, metalness: number, extra: Partial<THREE.MeshStandardMaterialParameters> = {}): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });
}

function box(w: number, h: number, d: number, mat: THREE.Material, x = 0, y = 0, z = 0): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  return m;
}

/** Cylinder with its axis along `axis` ('x' | 'y' | 'z'). */
function cyl(r: number, len: number, mat: THREE.Material, axis: 'x' | 'y' | 'z', x = 0, y = 0, z = 0, seg = 24): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(r, r, len, seg);
  if (axis === 'x') geo.rotateZ(-Math.PI / 2);
  if (axis === 'z') geo.rotateX(Math.PI / 2);
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  return m;
}

/** Build the CubeSat. The returned group is in the body frame with the bus centred at the origin. */
export function buildCubeSat(opts: CubeSatOptions): CubeSatModel {
  const [sx, sy, sz] = opts.size;
  const u = Math.min(sx, sy); // 1U face width in model units
  const group = new THREE.Group();
  group.name = 'cubesat';

  const tex = cellTexture();
  const cellMat = std(0x1d3366, 0.35, 0.3, tex ? { map: tex, color: 0xffffff } : {});
  const aluminium = std(0xb9c0c8, 0.32, 0.85);
  const anodised = std(0x2b2f36, 0.45, 0.6);
  const gold = std(0xd4a640, 0.3, 0.9);
  const glass = std(0x0c1424, 0.05, 0.2, { emissive: 0x0b2350, emissiveIntensity: 0.35 });
  const ceramic = std(0xe6dcc4, 0.7, 0.05);
  const tape = std(0xe8e4d6, 0.4, 0.5);

  // ---- bus (the box the attitude is read from)
  const bus = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), opts.bodyMaterials ?? cellMat);
  bus.name = 'bus';
  group.add(bus);

  // ---- structure: 4 corner rails with feet, plus a rib at every U boundary
  const r = RAIL * u;
  for (const ix of [1, -1]) {
    for (const iy of [1, -1]) {
      const rail = box(r, r, sz + 0.07 * u, aluminium, ix * (sx / 2 - r / 2 + 0.02 * u), iy * (sy / 2 - r / 2 + 0.02 * u), 0);
      rail.name = 'rail';
      group.add(rail);
    }
  }
  const nU = Math.max(1, Math.round(sz / (1.133 * u)));
  for (let k = 1; k < nU; k++) {
    const z = -sz / 2 + (k * sz) / nU;
    group.add(box(sx + 0.03 * u, sy + 0.03 * u, 0.035 * u, aluminium, 0, 0, z));
  }

  // ---- deployable wings on +Y and -Y, hinged on the -X edge, deployed flat along +-Y (cells face +X)
  const wings: THREE.Group[] = [];
  if (opts.wings !== false) {
    for (const s of [1, -1]) {
      const w = new THREE.Group();
      w.name = 'wing';
      const span = 1.0 * u;
      const len = sz * 0.94;
      const panel = box(0.03 * u, span, len, cellMat, 0, s * (sy / 2 + WING_GAP * u + span / 2), 0);
      w.add(panel);
      // back side (bare substrate) and frame edge
      w.add(box(0.012 * u, span, len, anodised, -0.022 * u, s * (sy / 2 + WING_GAP * u + span / 2), 0));
      for (const zf of [-0.32, 0.32]) {
        // hinge brackets
        w.add(box(0.05 * u, (WING_GAP + 0.06) * u, 0.08 * u, aluminium, 0, s * (sy / 2 + (WING_GAP * u) / 2), zf * sz));
      }
      w.position.x = -sx / 2 + 0.04 * u;
      group.add(w);
      wings.push(w);
    }
  }

  // ---- +Z end: GPS patch, UHF deployer and four tape-spring antennas
  const zTop = sz / 2;
  group.add(box(0.36 * u, 0.36 * u, 0.04 * u, ceramic, 0.18 * u, 0.18 * u, zTop + 0.02 * u));
  group.add(box(0.2 * u, 0.2 * u, 0.012 * u, gold, 0.18 * u, 0.18 * u, zTop + 0.046 * u));
  group.add(box(0.9 * u, 0.9 * u, 0.025 * u, anodised, 0, 0, zTop + 0.0125 * u));
  const whip = 0.85 * u;
  const dirs: [number, number][] = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
  ];
  for (const [dx, dy] of dirs) {
    const a = cyl(0.008 * u, whip, tape, dx !== 0 ? 'x' : 'y', 0, 0, zTop + 0.03 * u, 8);
    a.position.x = dx * (0.42 * u + whip / 2);
    a.position.y = dy * (0.42 * u + whip / 2);
    a.name = 'antenna';
    group.add(a);
  }

  // ---- -Z end: nadir camera barrel with lens
  const zBot = -sz / 2;
  group.add(cyl(0.24 * u, 0.16 * u, anodised, 'z', 0, 0, zBot - 0.08 * u, 32));
  group.add(cyl(0.17 * u, 0.02 * u, glass, 'z', 0, 0, zBot - 0.17 * u, 32));

  // ---- +X face: star-tracker baffle near the top
  const st = new THREE.Group();
  st.add(cyl(0.16 * u, 0.3 * u, anodised, 'x', 0, 0, 0, 28));
  st.add(cyl(0.11 * u, 0.02 * u, glass, 'x', 0.155 * u, 0, 0, 28));
  st.position.set(sx / 2 + 0.15 * u, -0.18 * u, zTop - 0.42 * u);
  group.add(st);

  // ---- coarse sun sensors: one gold window per long face, one on each end
  const css = 0.13 * u;
  const t = 0.02 * u;
  group.add(box(t, css, css, gold, sx / 2 + t / 2, 0.22 * u, zTop - 0.9 * u));
  group.add(box(t, css, css, gold, -sx / 2 - t / 2, 0.22 * u, zTop - 0.9 * u));
  group.add(box(css, t, css, gold, 0.22 * u, sy / 2 + t / 2, zTop - 0.9 * u));
  group.add(box(css, t, css, gold, 0.22 * u, -sy / 2 - t / 2, zTop - 0.9 * u));
  group.add(box(css, css, t, gold, -0.3 * u, -0.3 * u, zTop + t / 2));
  group.add(box(css, css, t, gold, 0.3 * u, 0.3 * u, zBot - t / 2));

  // ---- remove-before-flight tag and access port on -X
  group.add(box(t, 0.12 * u, 0.3 * u, std(0xc8202a, 0.6, 0.05), -sx / 2 - t / 2, -0.3 * u, zBot + 0.4 * u));

  const bb = new THREE.Box3().setFromObject(group);
  const halfExtent: Vec3 = [
    Math.max(Math.abs(bb.min.x), Math.abs(bb.max.x)),
    Math.max(Math.abs(bb.min.y), Math.abs(bb.max.y)),
    Math.max(Math.abs(bb.min.z), Math.abs(bb.max.z)),
  ];
  return { group, bus, wings, halfExtent };
}
