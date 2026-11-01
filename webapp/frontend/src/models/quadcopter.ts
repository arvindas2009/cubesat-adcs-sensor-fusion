// Detailed quadcopter in FRD body coordinates (x forward, y right, z down), docs/specs/ui_dashboard.md §4.
// `solid` = our estimate (shaded, with the extra detail parts); otherwise the EKF2 ghost: edge outlines of
// the main parts only, so it stays readable. DOM-free (the vitest node environment builds it).
import * as THREE from 'three';

export const ARM = 1.15; // centre to rotor centre [model units]
export const ROTOR_R = 0.5;

/** FRD positions of the four rotor centres (x forward, y right, z down; rotors sit above the arms). */
export function rotorPositions(): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  for (const sx of [1, -1]) {
    for (const sy of [1, -1]) out.push(new THREE.Vector3((sx * ARM) / Math.SQRT2, (sy * ARM) / Math.SQRT2, -0.07));
  }
  return out;
}

export interface QuadOptions {
  solid: boolean;
  /** Ghost outline colour (hex). */
  ghostColor?: number;
}

function at<T extends THREE.Object3D>(o: T, x: number, y: number, z: number): T {
  o.position.set(x, y, z);
  return o;
}

export function buildQuadcopter(opts: QuadOptions): THREE.Group {
  const g = new THREE.Group();
  g.name = opts.solid ? 'quad' : 'quad-ghost';
  const ghostMat = new THREE.LineBasicMaterial({ color: opts.ghostColor ?? 0xffb347, transparent: true, opacity: 0.75, depthWrite: false });
  const part = (geo: THREE.BufferGeometry, mat: THREE.Material): THREE.Object3D =>
    opts.solid ? new THREE.Mesh(geo, mat) : new THREE.LineSegments(new THREE.EdgesGeometry(geo, 20), ghostMat);
  const std = (color: number, roughness: number, metalness: number, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });

  const shell = std(0xdfe4ec, 0.45, 0.15);
  const carbon = std(0x24272d, 0.5, 0.35);
  const motor = std(0x9aa3ae, 0.3, 0.85);
  const prop = std(0x1a1d22, 0.55, 0.1);
  const sweep = std(0xaeb8cc, 0.6, 0.1, { transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide });

  // ---- centre body: lower plate + rounded upper shell (up = -z)
  g.add(part(new THREE.BoxGeometry(0.66, 0.44, 0.08), carbon));
  const top = new THREE.SphereGeometry(0.3, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2);
  top.rotateX(-Math.PI / 2); // dome opens towards -z (up in FRD)
  top.scale(1.15, 0.78, 0.45);
  const dome = part(top, shell);
  dome.position.z = -0.04;
  g.add(dome);

  // ---- X frame: two carbon tubes across the centre
  for (const a of [Math.PI / 4, -Math.PI / 4]) {
    const tube = new THREE.CylinderGeometry(0.035, 0.035, ARM * 2, 12);
    tube.rotateZ(Math.PI / 2); // axis y -> x
    const arm = part(tube, carbon);
    arm.rotation.z = a;
    g.add(arm);
  }

  // ---- motors, props and prop-sweep discs
  rotorPositions().forEach((p, i) => {
    const bell = new THREE.CylinderGeometry(0.1, 0.11, 0.12, 20);
    bell.rotateX(Math.PI / 2); // axis y -> z
    const m = part(bell, motor);
    m.position.set(p.x, p.y, p.z + 0.02);
    m.name = 'motor';
    g.add(m);
    if (!opts.solid) return;
    const blade = new THREE.BoxGeometry(ROTOR_R * 1.9, 0.06, 0.012);
    const b = new THREE.Mesh(blade, prop);
    b.position.set(p.x, p.y, p.z - 0.05);
    b.rotation.z = (i * Math.PI) / 3 + 0.4;
    g.add(b);
    const disc = new THREE.Mesh(new THREE.CircleGeometry(ROTOR_R, 40), sweep);
    disc.position.set(p.x, p.y, p.z - 0.05);
    g.add(disc);
    const hub = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), motor);
    hub.position.set(p.x, p.y, p.z - 0.07);
    g.add(hub);
  });

  // ---- nose marker (forward = +x), kept on both so the heading is unmistakable
  const nose = new THREE.ConeGeometry(0.08, 0.26, 16);
  nose.rotateZ(-Math.PI / 2); // axis +y -> +x
  const n = part(nose, std(0xff5a5a, 0.5, 0.1));
  n.position.set(0.45, 0, 0);
  n.name = 'nose';
  g.add(n);

  if (opts.solid) {
    // battery strapped on top, GPS mast with puck, front camera, landing legs (down = +z)
    g.add(at(new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.2, 0.09), std(0x3a4150, 0.6, 0.2)), -0.08, 0, -0.17));
    const mast = new THREE.CylinderGeometry(0.012, 0.012, 0.32, 8);
    mast.rotateX(Math.PI / 2);
    g.add(at(new THREE.Mesh(mast, carbon), -0.22, 0, -0.32));
    const puck = new THREE.CylinderGeometry(0.09, 0.09, 0.035, 24);
    puck.rotateX(Math.PI / 2);
    g.add(at(new THREE.Mesh(puck, std(0x2e3440, 0.5, 0.2)), -0.22, 0, -0.49));
    const cam = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.1, 0.08), carbon);
    cam.position.set(0.3, 0, 0.07);
    g.add(cam);
    const lens = new THREE.CylinderGeometry(0.03, 0.03, 0.02, 16);
    lens.rotateZ(-Math.PI / 2);
    g.add(at(new THREE.Mesh(lens, std(0x0c1424, 0.05, 0.3, { emissive: 0x0b2350, emissiveIntensity: 0.4 })), 0.345, 0, 0.07));
    for (const sy of [1, -1]) {
      for (const sx of [1, -1]) {
        const leg = new THREE.CylinderGeometry(0.018, 0.018, 0.34, 8);
        leg.rotateX(Math.PI / 2);
        const l = new THREE.Mesh(leg, carbon);
        l.position.set(sx * 0.2, sy * 0.24, 0.2);
        l.rotation.x = sy * -0.25; // splayed outwards
        l.name = 'leg';
        g.add(l);
      }
      const skid = new THREE.CylinderGeometry(0.02, 0.02, 0.62, 10);
      skid.rotateZ(Math.PI / 2);
      g.add(at(new THREE.Mesh(skid, carbon), 0, sy * 0.285, 0.36));
    }
  }
  return g;
}
