// Dashboard hero scene (docs/specs/ui_dashboard.md §3): the detailed CubeSat drifting over the Earth's
// limb. Decorative only (no data). Renders only while on screen and the dashboard is shown; a single
// still frame under prefers-reduced-motion. Browser-only.
import * as THREE from 'three';
import { buildCubeSat } from '../models/cubesat';
import { createViewport, observeSize } from '../adcs/viewport';

const TEXTURE_URL = `${import.meta.env.BASE_URL}textures/earth_bluemarble_3840.jpg`;
const EARTH_R = 60;

function stars(n: number): THREE.Points {
  let a = 0x9e3779b9;
  const rnd = (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = 2 * rnd() - 1;
    const phi = 2 * Math.PI * rnd();
    const s = Math.sqrt(1 - u * u);
    pos.set([400 * s * Math.cos(phi), 400 * u, 400 * s * Math.sin(phi)], 3 * i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color: 0xc8d2e6, size: 1.1, sizeAttenuation: false, transparent: true, opacity: 0.8 }));
}

export class HeroView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(32, 1, 0.1, 1000);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly sat: THREE.Group;
  private readonly earth: THREE.Mesh;
  private readonly reduced: boolean;
  private onScreen = true;
  private shown = false;
  private raf = 0;
  private last = 0;
  private t = 0;

  constructor(container: HTMLElement) {
    const vp = createViewport(container, 'A 3U CubeSat with deployed solar wings drifting above the Earth');
    this.renderer = vp.renderer;
    this.renderer.setClearColor(0x05070b, 1);
    this.reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.scene.add(new THREE.AmbientLight(0x8fa4cc, 0.55));
    const sun = new THREE.DirectionalLight(0xfff4e0, 3.0);
    sun.position.set(-30, 18, 25);
    this.scene.add(sun);
    const rim = new THREE.DirectionalLight(0x6fa8ff, 1.0);
    rim.position.set(20, -10, -20);
    this.scene.add(rim);
    this.scene.add(stars(1400));

    const earthMat = new THREE.MeshStandardMaterial({ color: 0x9fb4d8, roughness: 0.9, metalness: 0 });
    new THREE.TextureLoader().load(TEXTURE_URL, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      earthMat.map = tex;
      earthMat.color.set(0xffffff);
      earthMat.needsUpdate = true;
      this.renderOnce();
    });
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R, 96, 64), earthMat);
    this.earth.position.set(6, -EARTH_R - 5.5, -28);
    this.earth.rotation.set(0.35, 2.2, 0.15);
    this.scene.add(this.earth);
    const atmo = new THREE.Mesh(
      new THREE.SphereGeometry(EARTH_R * 1.025, 96, 64),
      new THREE.MeshBasicMaterial({ color: 0x4f8dff, transparent: true, opacity: 0.22, side: THREE.BackSide, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    atmo.position.copy(this.earth.position);
    this.scene.add(atmo);

    this.sat = buildCubeSat({ size: [1, 1, 3.4] }).group;
    this.sat.rotation.set(0.5, -0.6, 0.3);
    this.scene.add(this.sat);

    this.camera.position.set(0, 1.2, 13);
    this.camera.lookAt(0, -0.6, 0);

    observeSize(vp.body, (w, h) => {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      // keep the satellite framed on narrow (portrait) screens
      this.camera.position.z = w / h < 1 ? 13 / Math.max(0.45, w / h) : 13;
      this.camera.updateProjectionMatrix();
      this.renderOnce();
    });
    if (typeof IntersectionObserver === 'function') {
      new IntersectionObserver((es) => {
        this.onScreen = es.some((e) => e.isIntersecting);
        this.kick();
      }).observe(vp.body);
    }
    document.addEventListener('visibilitychange', () => this.kick());
  }

  /** The dashboard route is (not) shown. */
  setVisible(v: boolean): void {
    this.shown = v;
    this.kick();
  }

  private running(): boolean {
    return this.shown && this.onScreen && !document.hidden && !this.reduced;
  }

  private kick(): void {
    if (this.shown) this.renderOnce();
    if (this.running() && !this.raf) {
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.frame);
    }
  }

  private readonly frame = (now: number): void => {
    this.raf = 0;
    if (!this.running()) return;
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.t += dt;
    this.sat.rotation.y += dt * 0.12;
    this.sat.rotation.x = 0.5 + 0.12 * Math.sin(this.t * 0.25);
    this.sat.position.y = 0.15 * Math.sin(this.t * 0.4);
    this.earth.rotation.y += dt * 0.004;
    this.renderOnce();
    this.raf = requestAnimationFrame(this.frame);
  };

  private renderOnce(): void {
    this.renderer.render(this.scene, this.camera);
  }
}
