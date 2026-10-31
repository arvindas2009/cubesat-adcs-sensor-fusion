// ECI scene (docs/specs/frontend.md §2): textured Earth turning with GMST, orbit ring and trail, the
// satellite marker with a body-axes triad, Sun arrow and a directional light along the Sun vector.
// 1 world unit = 1000 km. Every coordinate conversion goes through math/frames.ts.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { bodyInEciToWorld, earthRotationY, eciToWorld, zUpToWorld } from '../math/frames';
import type { Frame, Info } from '../types';
import { AXIS_COLORS, COLORS } from './theme';
import { ORBIT_VIEW_COLORS, VIEW_LEGENDS } from '../ui/glyphs';
import { renderLegend } from '../ui/legend';
import { createViewport, observeSize } from './viewport';

const SCALE = 1e-6; // m -> world units (1 unit = 1000 km)
const EARTH_RADIUS = 6.378137; // WGS-84 equatorial radius in world units
const MARKER_LENGTH = 0.24; // exaggerated body length (the real 0.34 m is invisible at this scale)
const TRAIL_MAX = 500;
const TRAIL_MIN_STEP = 0.03; // world units between trail points (30 km)
const TEXTURE_URL = `${import.meta.env.BASE_URL}textures/earth_bluemarble_3840.jpg`;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeStars(): THREE.Points {
  const rnd = mulberry32(12345);
  const n = 1800;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = 2 * rnd() - 1;
    const phi = 2 * Math.PI * rnd();
    const s = Math.sqrt(1 - u * u);
    const r = 900;
    pos[3 * i] = r * s * Math.cos(phi);
    pos[3 * i + 1] = r * u;
    pos[3 * i + 2] = r * s * Math.sin(phi);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.PointsMaterial({ color: 0x9aa7c7, size: 1.3, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false });
  const p = new THREE.Points(g, m);
  p.frustumCulled = false;
  return p;
}

export class OrbitView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.02, 3000);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly earth: THREE.Mesh;
  private readonly sunLight = new THREE.DirectionalLight(0xfff4e0, 3.2);
  private readonly sunArrow: THREE.ArrowHelper;
  private readonly sat = new THREE.Group();
  private readonly satBox: THREE.Mesh;
  private readonly satMaterial: THREE.MeshBasicMaterial;
  private readonly triad: THREE.ArrowHelper[] = [];
  private ring: THREE.Line | null = null;
  private readonly trail: THREE.Line;
  private readonly trailPos = new Float32Array(TRAIL_MAX * 3);
  private trailCount = 0;
  private readonly followBtn: HTMLButtonElement;
  private follow = true;
  private placed = false; // camera placed relative to the satellite yet?
  private lit = true;
  private dirty = true;
  private readonly satWorld = new THREE.Vector3();
  private lastT = -Infinity;

  constructor(container: HTMLElement) {
    const vp = createViewport(container, 'Orbit view: the Earth, the first orbit and recent trail of the satellite, and the direction of the Sun');
    this.renderer = vp.renderer;

    // Earth: plain shaded sphere first; the texture replaces the colour when it has loaded.
    const earthMat = new THREE.MeshStandardMaterial({ color: 0x1d4672, roughness: 1, metalness: 0 });
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS, 96, 48), earthMat);
    this.scene.add(this.earth);
    // Thin atmosphere shell (back faces only) so the night side keeps a silhouette against space.
    this.scene.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(EARTH_RADIUS * 1.025, 64, 32),
        new THREE.MeshBasicMaterial({ color: 0x4a8dff, transparent: true, opacity: 0.16, side: THREE.BackSide, depthWrite: false }),
      ),
    );
    new THREE.TextureLoader().load(
      TEXTURE_URL,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
        earthMat.map = tex;
        earthMat.color.set(0xffffff);
        earthMat.needsUpdate = true;
        this.dirty = true;
      },
      undefined,
      () => {
        /* keep the fallback sphere; never block on the texture */
      },
    );

    // Equator and polar axis (ECI-fixed, so they stay put while the surface turns).
    const eq: THREE.Vector3[] = [];
    for (let i = 0; i < 180; i++) {
      const a = (2 * Math.PI * i) / 180;
      eq.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)).multiplyScalar(EARTH_RADIUS * 1.003));
    }
    this.scene.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(eq), new THREE.LineBasicMaterial({ color: 0xbcd0f0, transparent: true, opacity: 0.25 })));
    this.scene.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -EARTH_RADIUS * 1.35, 0), new THREE.Vector3(0, EARTH_RADIUS * 1.35, 0)]),
        new THREE.LineBasicMaterial({ color: 0xbcd0f0, transparent: true, opacity: 0.35 }),
      ),
    );

    // Light: a directional light along the Sun vector gives a real terminator; ambient stays low.
    this.scene.add(this.sunLight, this.sunLight.target);
    this.scene.add(new THREE.AmbientLight(0x8fa4d0, 0.9));
    this.sunLight.position.set(100, 0, 0);

    this.sunArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 9.2, new THREE.Color(COLORS.sun).getHex(), 0.45, 0.2);
    this.scene.add(this.sunArrow);
    this.scene.add(makeStars());

    // Satellite marker (body-fixed group) with a small triad.
    this.satMaterial = new THREE.MeshBasicMaterial({ color: new THREE.Color(ORBIT_VIEW_COLORS.satellite), transparent: true });
    this.satBox = new THREE.Mesh(new THREE.BoxGeometry(MARKER_LENGTH * 0.3, MARKER_LENGTH * 0.3, MARKER_LENGTH), this.satMaterial);
    this.sat.add(this.satBox);
    const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
    dirs.forEach((d, i) => {
      const a = new THREE.ArrowHelper(d, new THREE.Vector3(), 0.5, new THREE.Color(AXIS_COLORS[i]).getHex(), 0.14, 0.08);
      (a.line.material as THREE.LineBasicMaterial).transparent = true;
      (a.cone.material as THREE.MeshBasicMaterial).transparent = true;
      this.triad.push(a);
      this.sat.add(a);
    });
    this.sat.visible = false;
    this.scene.add(this.sat);

    // Trail of recent positions.
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(this.trailPos, 3));
    tg.setDrawRange(0, 0);
    this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: new THREE.Color(ORBIT_VIEW_COLORS.trail), transparent: true, opacity: 0.85 }));
    this.trail.frustumCulled = false;
    this.scene.add(this.trail);

    this.camera.position.set(0, 9, 19);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 120;
    this.controls.addEventListener('change', () => (this.dirty = true));

    this.followBtn = document.createElement('button');
    this.followBtn.type = 'button';
    this.followBtn.className = 'vp-btn';
    this.followBtn.addEventListener('click', () => this.setFollow(!this.follow));
    vp.overlay.append(this.followBtn);
    const legend = renderLegend(VIEW_LEGENDS.orbit(COLORS), { surface: 'screen' });
    legend.el.classList.add('vp-legend');
    vp.overlay.append(legend.el);
    this.refreshFollowButton();

    observeSize(vp.body, (w, h) => {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.dirty = true;
    });
  }

  /** New session (or reset): draw the first-orbit ring and clear the trail. */
  setInfo(info: Info): void {
    if (this.ring) {
      this.scene.remove(this.ring);
      this.ring.geometry.dispose();
    }
    const pts = info.orbit_track_eci_m.map((r) => eciToWorld(r, SCALE));
    this.ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: new THREE.Color(ORBIT_VIEW_COLORS.ring), transparent: true, opacity: 0.55 }));
    this.scene.add(this.ring);
    this.clearTrail();
    this.dirty = true;
  }

  clearTrail(): void {
    this.trailCount = 0;
    this.trail.geometry.setDrawRange(0, 0);
    this.lastT = -Infinity;
    this.placed = false;
    this.dirty = true;
  }

  setFollow(on: boolean): void {
    this.follow = on;
    if (on) {
      this.placed = false; // re-place the camera next to the satellite on the next frame
    } else {
      this.controls.target.set(0, 0, 0);
      this.camera.position.set(0, 9, 19);
    }
    this.refreshFollowButton();
    this.dirty = true;
  }

  private refreshFollowButton(): void {
    this.followBtn.textContent = this.follow ? 'Camera: following satellite' : 'Camera: free (Earth)';
    this.followBtn.setAttribute('aria-pressed', String(this.follow));
  }

  update(f: Frame): void {
    const e = f.eci;
    this.earth.rotation.y = earthRotationY(e.gmst_rad);

    const sun = zUpToWorld(e.sun).normalize();
    this.sunLight.position.copy(sun).multiplyScalar(100);
    this.sunArrow.setDirection(sun);

    this.satWorld.copy(eciToWorld(e.r_m, SCALE));
    this.sat.position.copy(this.satWorld);
    this.sat.quaternion.copy(bodyInEciToWorld(e.q));
    this.sat.visible = true;

    if (f.lit !== this.lit) {
      this.lit = f.lit;
      this.satMaterial.color.set(f.lit ? ORBIT_VIEW_COLORS.satellite : ORBIT_VIEW_COLORS.satelliteDim);
      this.satMaterial.opacity = f.lit ? 1 : 0.8;
      for (const a of this.triad) {
        (a.line.material as THREE.LineBasicMaterial).opacity = f.lit ? 1 : 0.35;
        (a.cone.material as THREE.MeshBasicMaterial).opacity = f.lit ? 1 : 0.35;
      }
    }

    // Trail: restart if time went backwards (reset); skip points closer than the minimum step.
    if (f.t < this.lastT) this.trailCount = 0;
    this.lastT = f.t;
    const n = this.trailCount;
    const last = n > 0 ? new THREE.Vector3(this.trailPos[3 * n - 3], this.trailPos[3 * n - 2], this.trailPos[3 * n - 1]) : null;
    if (!last || last.distanceTo(this.satWorld) >= TRAIL_MIN_STEP) {
      if (n >= TRAIL_MAX) {
        this.trailPos.copyWithin(0, 3, 3 * TRAIL_MAX);
        this.trailCount = TRAIL_MAX - 1;
      }
      const i = this.trailCount++;
      this.trailPos[3 * i] = this.satWorld.x;
      this.trailPos[3 * i + 1] = this.satWorld.y;
      this.trailPos[3 * i + 2] = this.satWorld.z;
      const attr = this.trail.geometry.getAttribute('position') as THREE.BufferAttribute;
      attr.needsUpdate = true;
      this.trail.geometry.setDrawRange(0, this.trailCount);
    }

    if (this.follow) {
      if (!this.placed) {
        const out = this.satWorld.clone().normalize().multiplyScalar(7).add(new THREE.Vector3(0, 3, 0));
        this.camera.position.copy(this.satWorld).add(out);
        this.placed = true;
      } else {
        this.camera.position.add(this.satWorld.clone().sub(this.controls.target));
      }
      this.controls.target.copy(this.satWorld);
    }
    this.dirty = true;
  }

  /** Redraw on the next render() (after the view was hidden or covered while its size or content changed). */
  invalidate(): void {
    this.dirty = true;
  }

  render(): void {
    if (!this.dirty) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.dirty = false;
  }
}
