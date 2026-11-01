// Optional open-licence model swap-in (docs/specs/ui_dashboard.md §4). public/models/manifest.json lists
// the .glb files and their licences; with no entry for a role the procedural model is kept.
// glTF models are +Y up, +Z forward, +X left. FRD body axes are x forward, y right, z down, so
//   glTF +Z -> FRD +x,  glTF +Y -> FRD -z,  glTF +X -> FRD -y   (a proper rotation, det = +1).
import * as THREE from 'three';

export interface ModelEntry {
  role: string;
  file: string;
  license: string;
  source: string;
  author: string;
}

export const ALLOWED_LICENSES = ['CC0-1.0', 'CC-BY-4.0'];

/** Columns are the images of the glTF basis vectors in FRD body coordinates. */
export const GLTF_TO_FRD = new THREE.Matrix4().set(
  0, 0, 1, 0,
  -1, 0, 0, 0,
  0, -1, 0, 0,
  0, 0, 0, 1,
);

/** Valid manifest entries only (known licence, all fields present). */
export function validEntries(manifest: unknown): ModelEntry[] {
  const list = (manifest as { models?: unknown })?.models;
  if (!Array.isArray(list)) return [];
  return list.filter(
    (e): e is ModelEntry =>
      !!e &&
      typeof e.role === 'string' &&
      typeof e.file === 'string' &&
      /^[\w.-]+\.glb$/.test(e.file) &&
      typeof e.source === 'string' &&
      typeof e.author === 'string' &&
      ALLOWED_LICENSES.includes(e.license),
  );
}

/**
 * Load the model listed for `role`, rotated into FRD and scaled so its largest horizontal size is `span`.
 * Resolves null when there is no entry or loading fails (the caller keeps its procedural model).
 */
export async function loadRoleModel(role: string, span: number): Promise<THREE.Group | null> {
  try {
    const base = `${import.meta.env.BASE_URL}models/`;
    const res = await fetch(`${base}manifest.json`, { cache: 'no-cache' });
    if (!res.ok) return null;
    const entry = validEntries(await res.json()).find((e) => e.role === role);
    if (!entry) return null;
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
    const gltf = await new GLTFLoader().loadAsync(`${base}${entry.file}`);
    const model = gltf.scene;
    const bb = new THREE.Box3().setFromObject(model);
    const size = bb.getSize(new THREE.Vector3());
    const centre = bb.getCenter(new THREE.Vector3());
    model.position.sub(centre);
    const inner = new THREE.Group();
    inner.add(model);
    inner.scale.setScalar(span / Math.max(1e-9, size.x, size.z));
    const wrap = new THREE.Group();
    wrap.name = `gltf:${role}`;
    wrap.applyMatrix4(GLTF_TO_FRD);
    wrap.add(inner);
    const outer = new THREE.Group();
    outer.add(wrap);
    return outer;
  } catch {
    return null;
  }
}
