// Shared plumbing for the two WebGL viewports: a canvas that fills its panel, resize handling,
// dirty-flag rendering and a small overlay toolbar. Browser-only (not imported by the tests).
import * as THREE from 'three';
import { SCREEN } from '../theme/palette';

export interface Viewport {
  readonly body: HTMLElement;
  readonly renderer: THREE.WebGLRenderer;
  readonly overlay: HTMLElement;
}

/** Create the canvas area (below the panel's <h2>) and a WebGL renderer sized to it. */
export function createViewport(container: HTMLElement): Viewport {
  const body = document.createElement('div');
  body.className = 'vp-body';
  const overlay = document.createElement('div');
  overlay.className = 'vp-overlay';
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(new THREE.Color(SCREEN.clear), 1); // dark plum "screen" in both modes
  renderer.domElement.className = 'vp-canvas';
  body.append(renderer.domElement, overlay);
  container.append(body);
  return { body, renderer, overlay };
}

/** Observe the viewport body; `onSize(w, h)` is called with CSS pixels whenever it changes. */
export function observeSize(el: HTMLElement, onSize: (w: number, h: number) => void): ResizeObserver {
  const ro = new ResizeObserver(() => {
    const w = Math.max(1, Math.floor(el.clientWidth));
    const h = Math.max(1, Math.floor(el.clientHeight));
    onSize(w, h);
  });
  ro.observe(el);
  return ro;
}

/** Small text sprite-free label helper for CSS2D objects. */
export function labelElement(text: string, color: string, extraClass = ''): HTMLDivElement {
  const d = document.createElement('div');
  d.className = `label3d ${extraClass}`.trim();
  d.textContent = text;
  d.style.color = color;
  return d;
}

/** Free the GPU resources (geometries, materials, textures) of everything under `root`, and detach it. */
export function disposeObject(root: THREE.Object3D): void {
  root.traverse((o) => {
    // CSS2D labels: `removeFromParent` only fires 'removed' on the root, so detach their DOM nodes here.
    const label = (o as { element?: unknown }).element;
    if (label instanceof HTMLElement) label.remove();
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    for (const mm of Array.isArray(mat) ? mat : mat ? [mat] : []) {
      (mm as THREE.MeshBasicMaterial).map?.dispose();
      mm.dispose();
    }
  });
  root.removeFromParent();
}
