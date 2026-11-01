// Landing hero scene (docs/specs/ui_dashboard.md §6): the Earth with a glowing atmosphere, dipole
// magnetic-field lines carrying travelling pulses, an inclined orbit ring with the detailed CubeSat,
// its trail and nadir beam, the Moon, a distant ringed planet, the Sun glow and parallax star layers.
// Decorative, not data: the orbit and field are drawn schematically, not to scale.
// Interaction: the camera eases toward the pointer (critically damped springs); dragging spins the
// Earth system 1:1 and a release keeps the flick's momentum, decaying like a scroll. Renders only while
// shown and on screen; one still frame under prefers-reduced-motion. Browser-only.
import * as THREE from 'three';
import { buildCubeSat } from '../models/cubesat';
import { createViewport, observeSize } from '../adcs/viewport';
import { Spring, velocityOf } from './spring';

const TEXTURE_URL = `${import.meta.env.BASE_URL}textures/earth_bluemarble_3840.jpg`;
const R = 5; // Earth radius [scene units]
const ORBIT_R = 6.7; // exaggerated so the orbit reads at this scale
const INCLINATION = (51.6 * Math.PI) / 180; // the simulated orbit (adcs/config.py)
const TRAIL = 90;

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function starLayer(n: number, radius: number, size: number, seed: number, color: number): THREE.Points {
  const rnd = rng(seed);
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = 2 * rnd() - 1;
    const phi = 2 * Math.PI * rnd();
    const s = Math.sqrt(1 - u * u);
    pos.set([radius * s * Math.cos(phi), radius * u, radius * s * Math.sin(phi)], 3 * i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color, size, sizeAttenuation: false, transparent: true, opacity: 0.85, depthWrite: false }));
}

/** Soft radial glow texture for sprites (Sun, sat beacon). */
function glowTexture(inner: string, outer: string): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, inner);
  grad.addColorStop(0.25, inner);
  grad.addColorStop(1, outer);
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Deep-space backdrop: navy at the top, a faint blue haze towards the bottom. */
function skyTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = 4;
  c.height = 512;
  const g = c.getContext('2d')!;
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, '#030611');
  grad.addColorStop(0.55, '#060d22');
  grad.addColorStop(1, '#0d1f4a');
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 512);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Fresnel rim glow on a back-faced shell: the atmosphere. */
function atmosphere(radius: number, color: THREE.Color, power: number, strength: number): THREE.Mesh {
  const mat = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: color }, uPower: { value: power }, uStrength: { value: strength } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uPower; uniform float uStrength; varying vec3 vN; varying vec3 vV;
      void main(){ float f = pow(1.0 - abs(dot(vN, vV)), uPower) * uStrength; gl_FragColor = vec4(uColor * f, f); }`,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
  });
  return new THREE.Mesh(new THREE.SphereGeometry(radius, 96, 64), mat);
}

/**
 * Dipole field lines r = L sin²θ in `meridians` magnetic meridian planes. Each line carries a `t`
 * attribute (0..1 along it) so the shader can run bright pulses along the field: the "current".
 */
function fieldLines(shells: number[], meridians: number): { obj: THREE.LineSegments; mat: THREE.ShaderMaterial } {
  const pos: number[] = [];
  const ts: number[] = [];
  const seeds: number[] = [];
  const steps = 120;
  for (const L of shells) {
    const theta0 = Math.asin(Math.sqrt(Math.min(1, (R * 1.01) / L))); // where the line meets the surface
    for (let m = 0; m < meridians; m++) {
      const phi = (m / meridians) * Math.PI * 2 + L * 0.37;
      const seed = (m * 0.137 + L * 0.71) % 1;
      let prev: THREE.Vector3 | null = null;
      let prevT = 0;
      for (let k = 0; k <= steps; k++) {
        const th = theta0 + ((Math.PI - 2 * theta0) * k) / steps;
        const r = L * Math.sin(th) ** 2;
        const p = new THREE.Vector3(r * Math.sin(th) * Math.cos(phi), r * Math.cos(th), r * Math.sin(th) * Math.sin(phi));
        const t = k / steps;
        if (prev) {
          pos.push(prev.x, prev.y, prev.z, p.x, p.y, p.z);
          ts.push(prevT, t);
          seeds.push(seed, seed);
        }
        prev = p;
        prevT = t;
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aT', new THREE.Float32BufferAttribute(ts, 1));
  g.setAttribute('aSeed', new THREE.Float32BufferAttribute(seeds, 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uBase: { value: new THREE.Color(0x2f6bff) }, uHot: { value: new THREE.Color(0x7fe9ff) } },
    vertexShader: `attribute float aT; attribute float aSeed; varying float vT; varying float vSeed;
      void main(){ vT = aT; vSeed = aSeed; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `uniform float uTime; uniform vec3 uBase; uniform vec3 uHot; varying float vT; varying float vSeed;
      void main(){
        float x = fract(vT - uTime * 0.12 - vSeed);
        float pulse = smoothstep(0.0, 0.05, x) * (1.0 - smoothstep(0.05, 0.16, x));
        float ends = smoothstep(0.0, 0.08, vT) * smoothstep(0.0, 0.08, 1.0 - vT);
        float a = (0.12 + 0.85 * pulse) * ends;
        gl_FragColor = vec4(mix(uBase, uHot, pulse) * a, a);
      }`,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
  });
  return { obj: new THREE.LineSegments(g, mat), mat };
}

export class HeroView {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(36, 1, 0.1, 2000);
  private readonly renderer: THREE.WebGLRenderer;
  private readonly system = new THREE.Group(); // Earth, field, orbit: spun by dragging
  private readonly earth: THREE.Mesh;
  private readonly clouds: THREE.Mesh;
  private readonly field: THREE.ShaderMaterial;
  private readonly orbitPlane = new THREE.Group();
  private readonly sat: THREE.Group;
  private readonly beam: THREE.Mesh;
  private readonly trail: THREE.Line;
  private readonly trailPos: Float32Array;
  private readonly moon: THREE.Mesh;
  private readonly planet = new THREE.Group();
  private readonly starsNear: THREE.Points;
  private readonly starsFar: THREE.Points;
  private readonly reduced: boolean;
  // pointer parallax (critically damped) and drag spin with momentum
  private readonly px = new Spring(0, { damping: 1, response: 0.9 });
  private readonly py = new Spring(0, { damping: 1, response: 0.9 });
  private spin = 0;
  private spinVel = 0; // rad/s
  private dragging = false;
  private dragHist: { x: number; t: number }[] = [];
  private lastDragX = 0;
  private scrollK = 0; // 0 at the top, 1 when the hero has scrolled away
  private layout = { x: 3.4, scale: 1 };
  private onScreen = true;
  private shown = false;
  private raf = 0;
  private last = 0;
  private t = 0;
  private orbitPhase = 0.6;

  constructor(container: HTMLElement) {
    const vp = createViewport(container, 'The Earth with its magnetic field lines and a CubeSat on an inclined orbit, the Moon and a distant planet');
    this.renderer = vp.renderer;
    this.renderer.setClearColor(0x040816, 1);
    this.scene.background = skyTexture();
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ---- light: one Sun from the upper left, a faint blue fill on the night side
    const sunDir = new THREE.Vector3(-0.7, 0.45, 0.9).normalize(); // lights the visible face, a terminator on the right
    const sun = new THREE.DirectionalLight(0xfff3e0, 3.2);
    sun.position.copy(sunDir).multiplyScalar(100);
    this.scene.add(sun, new THREE.AmbientLight(0x3a5aa8, 0.5));
    const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture('rgba(255,244,214,1)', 'rgba(255,190,120,0)'), blending: THREE.AdditiveBlending, depthWrite: false }));
    sunGlow.position.copy(sunDir).multiplyScalar(400);
    sunGlow.scale.setScalar(70);
    this.scene.add(sunGlow);

    // ---- stars: two depths for parallax
    this.starsFar = starLayer(1600, 800, 1.0, 7, 0x9fb4e6);
    this.starsNear = starLayer(420, 500, 1.8, 11, 0xdbe6ff);
    this.scene.add(this.starsFar, this.starsNear);

    // ---- Earth, clouds-ish sheen and atmosphere
    const earthMat = new THREE.MeshStandardMaterial({ color: 0x2b4f9c, roughness: 0.85, metalness: 0 });
    new THREE.TextureLoader().load(TEXTURE_URL, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 8;
      earthMat.map = tex;
      earthMat.color.set(0xffffff);
      earthMat.needsUpdate = true;
      this.renderOnce();
    });
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(R, 128, 96), earthMat);
    this.clouds = new THREE.Mesh(
      new THREE.SphereGeometry(R * 1.008, 96, 64),
      new THREE.MeshStandardMaterial({ color: 0x9cc4ff, transparent: true, opacity: 0.06, roughness: 1, depthWrite: false }),
    );
    this.system.add(this.earth, this.clouds);
    this.system.add(atmosphere(R * 1.05, new THREE.Color(0x4f9dff), 3.2, 0.9)); // a soft rim, not a ring
    this.system.add(atmosphere(R * 1.18, new THREE.Color(0x2f5bff), 5.0, 0.35));

    // ---- magnetic field (tilted ~10° like the real dipole)
    const f = fieldLines([R * 1.7, R * 2.3, R * 3.1], 10);
    this.field = f.mat;
    f.obj.rotation.z = (10 * Math.PI) / 180;
    this.system.add(f.obj);

    // ---- orbit ring + CubeSat + trail + nadir beam (in the inclined orbit plane)
    this.orbitPlane.rotation.x = Math.PI / 2 - INCLINATION; // ring tilted towards the viewer
    this.orbitPlane.rotation.z = 0.25;
    const ringPts: THREE.Vector3[] = [];
    for (let i = 0; i <= 256; i++) {
      const a = (i / 256) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(ORBIT_R * Math.cos(a), 0, ORBIT_R * Math.sin(a)));
    }
    const ring = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: 0x6fb6ff, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    this.orbitPlane.add(ring);

    this.sat = buildCubeSat({ size: [1, 1, 3.4] }).group;
    this.sat.scale.setScalar(0.16);
    this.orbitPlane.add(this.sat);
    const beacon = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture('rgba(140,230,255,1)', 'rgba(60,140,255,0)'), blending: THREE.AdditiveBlending, depthWrite: false }));
    beacon.scale.setScalar(1.1);
    this.sat.add(beacon);
    beacon.scale.setScalar(1.1 / 0.16);

    const beamGeo = new THREE.ConeGeometry(0.55, ORBIT_R - R, 32, 1, true);
    beamGeo.translate(0, -(ORBIT_R - R) / 2, 0);
    this.beam = new THREE.Mesh(
      beamGeo,
      new THREE.MeshBasicMaterial({ color: 0x55d6ff, transparent: true, opacity: 0.1, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }),
    );
    this.orbitPlane.add(this.beam);

    this.trailPos = new Float32Array(TRAIL * 3);
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(this.trailPos, 3));
    const alpha = new Float32Array(TRAIL).map((_, i) => i / (TRAIL - 1));
    tg.setAttribute('aA', new THREE.BufferAttribute(alpha, 1));
    this.trail = new THREE.Line(
      tg,
      new THREE.ShaderMaterial({
        vertexShader: 'attribute float aA; varying float vA; void main(){ vA = aA; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
        fragmentShader: 'varying float vA; void main(){ float a = vA*vA*0.9; gl_FragColor = vec4(vec3(0.5,0.9,1.0)*a, a); }',
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    this.orbitPlane.add(this.trail);
    this.system.add(this.orbitPlane);
    this.system.rotation.x = 0.18;
    this.scene.add(this.system);

    // ---- the Moon and a distant ringed planet
    this.moon = new THREE.Mesh(new THREE.SphereGeometry(1.1, 48, 32), new THREE.MeshStandardMaterial({ color: 0xc9ccd6, roughness: 0.95 }));
    this.moon.position.set(1, 15, -36); // high in the empty sky between the title and the Earth
    this.scene.add(this.moon);
    const gas = new THREE.Mesh(new THREE.SphereGeometry(3.2, 64, 48), new THREE.MeshStandardMaterial({ color: 0x5d7fd6, roughness: 0.7, emissive: 0x0b1a44, emissiveIntensity: 0.6 }));
    const ringGeo = new THREE.RingGeometry(4.2, 6.6, 96);
    const ringMesh = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0x9fbaff, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
    ringMesh.rotation.x = Math.PI / 2.4;
    this.planet.add(gas, ringMesh);
    this.planet.position.set(40, 16, -120);
    this.planet.rotation.z = 0.35;
    this.scene.add(this.planet);

    observeSize(vp.body, (w, h) => {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      // wide: Earth on the right of the copy; narrow: centred, pushed down and smaller
      const wide = w / h > 1.15;
      this.layout = wide ? { x: 4.6, scale: 1 } : { x: 0, scale: 0.72 };
      this.camera.updateProjectionMatrix();
      this.renderOnce();
    });

    this.bindPointer(vp.body);
    if (typeof IntersectionObserver === 'function') {
      new IntersectionObserver((es) => {
        this.onScreen = es.some((e) => e.isIntersecting);
        this.kick();
      }).observe(vp.body);
    }
    document.addEventListener('visibilitychange', () => this.kick());
    this.place(0);
  }

  private bindPointer(el: HTMLElement): void {
    el.style.touchAction = 'pan-y'; // vertical swipes still scroll the page; horizontal ones spin the Earth
    window.addEventListener('pointermove', (e) => {
      if (!this.shown) return;
      this.px.target = (e.clientX / window.innerWidth) * 2 - 1;
      this.py.target = (e.clientY / window.innerHeight) * 2 - 1;
      this.kick();
    });
    el.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      el.setPointerCapture(e.pointerId);
      this.lastDragX = e.clientX;
      this.dragHist = [{ x: e.clientX, t: e.timeStamp }];
      this.spinVel = 0; // grabbing stops the spin at once: the hand holds it
      el.classList.add('grabbing');
    });
    el.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastDragX;
      this.lastDragX = e.clientX;
      this.spin += dx * 0.006; // 1:1 tracking
      this.dragHist.push({ x: e.clientX, t: e.timeStamp });
      if (this.dragHist.length > 8) this.dragHist.shift();
      this.renderOnce();
    });
    const end = (e: PointerEvent): void => {
      if (!this.dragging) return;
      this.dragging = false;
      el.classList.remove('grabbing');
      this.spinVel = velocityOf(this.dragHist) * 0.006; // hand the release velocity to the momentum
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
      this.kick();
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
  }

  /** The landing route is (not) shown. */
  setVisible(v: boolean): void {
    this.shown = v;
    this.kick();
  }

  /** 0 = hero at the top of the page, 1 = scrolled a full hero height away. */
  setScroll(k: number): void {
    this.scrollK = Math.min(1, Math.max(0, k));
    if (this.reduced) this.renderOnce();
  }

  private running(): boolean {
    return this.shown && this.onScreen && !document.hidden && !this.reduced;
  }

  private kick(): void {
    if (this.shown && !this.raf) this.renderOnce();
    if (this.running() && !this.raf) {
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.frame);
    }
  }

  private place(dt: number): void {
    this.system.updateMatrixWorld();
    this.t += dt;
    // momentum: exponential decay, the scroll-deceleration feel (0.998 per ms)
    if (!this.dragging && this.spinVel !== 0) {
      this.spin += this.spinVel * dt;
      this.spinVel *= Math.pow(0.998, dt * 1000);
      if (Math.abs(this.spinVel) < 1e-3) this.spinVel = 0;
    }
    this.system.rotation.y = this.spin + this.t * 0.03;
    this.earth.rotation.y = this.t * 0.02;
    this.clouds.rotation.y = this.t * 0.028;
    this.field.uniforms.uTime.value = this.t;

    // satellite along the ring, nose along-track, beam to the centre
    this.orbitPhase += dt * 0.16;
    const a = this.orbitPhase;
    const p = new THREE.Vector3(ORBIT_R * Math.cos(a), 0, ORBIT_R * Math.sin(a));
    this.sat.position.copy(p);
    this.sat.lookAt(new THREE.Vector3(-Math.sin(a), 0, Math.cos(a)).add(p).applyMatrix4(this.orbitPlane.matrixWorld)); // long axis along-track
    this.beam.position.copy(p);
    this.beam.lookAt(this.orbitPlane.localToWorld(new THREE.Vector3()));
    this.beam.rotateX(-Math.PI / 2);
    for (let i = 0; i < TRAIL; i++) {
      const b = a - (TRAIL - 1 - i) * 0.012;
      this.trailPos.set([ORBIT_R * Math.cos(b), 0, ORBIT_R * Math.sin(b)], i * 3);
    }
    this.trail.geometry.attributes.position.needsUpdate = true;

    this.moon.rotation.y = this.t * 0.01;
    this.planet.rotation.y = this.t * 0.02;

    // camera: pointer parallax + scroll dolly
    this.px.step(dt);
    this.py.step(dt);
    const s = this.layout.scale;
    // narrow: the Earth rises from the bottom like a horizon, below the copy
    this.system.position.set(this.layout.x, this.layout.x === 0 ? -6.6 : -0.3, 0);
    this.system.scale.setScalar(s);
    this.camera.position.set(this.px.value * 1.6, 1.0 - this.py.value * 0.9 + this.scrollK * 2.5, 24 - this.scrollK * 4);
    this.camera.lookAt(this.layout.x * 0.55, -0.4 + this.scrollK * 1.5, 0);
    this.starsNear.rotation.y = this.px.value * 0.02;
    this.starsFar.rotation.y = this.px.value * 0.008;
  }

  private readonly frame = (now: number): void => {
    this.raf = 0;
    if (!this.running()) return;
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.place(dt);
    this.renderOnce();
    this.raf = requestAnimationFrame(this.frame);
  };

  private renderOnce(): void {
    this.system.updateMatrixWorld();
    this.renderer.render(this.scene, this.camera);
  }
}
