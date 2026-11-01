// Landing-page background (docs/specs/ui_dashboard.md §6): a slow drifting network of nodes joined by
// faint links, brighter near the pointer, with sparks that run along a link now and then: a calm
// "electric" texture. One fixed 2D canvas behind the content; paused when the route is hidden or the
// tab is in the background; a single still frame under prefers-reduced-motion. Browser-only.
import { currentTheme, onThemeChange } from '../theme';

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  ox: number; // pointer push offset
  oy: number;
}

interface Spark {
  a: number;
  b: number;
  t: number;
}

const LINK = 140; // px
const POINTER_R = 180;

export class ElectricField {
  private readonly canvas: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D | null;
  private nodes: Node[] = [];
  private sparks: Spark[] = [];
  private w = 0;
  private h = 0;
  private dpr = 1;
  private mx = -1e4;
  private my = -1e4;
  private shown = false;
  private raf = 0;
  private last = 0;
  private readonly reduced: boolean;
  private colors = { node: '', link: '', hot: '' };

  constructor(host: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'dash-bg';
    this.canvas.setAttribute('aria-hidden', 'true');
    host.prepend(this.canvas);
    this.g = this.canvas.getContext('2d');
    this.reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.syncColors();
    onThemeChange(() => {
      this.syncColors();
      this.draw(0);
    });
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('pointermove', (e) => {
      this.mx = e.clientX;
      this.my = e.clientY;
    });
    document.addEventListener('visibilitychange', () => this.kick());
    this.resize();
  }

  private syncColors(): void {
    this.colors =
      currentTheme() === 'dark'
        ? { node: 'rgba(140,190,255,0.55)', link: '90,150,255', hot: '120,230,255' }
        : { node: 'rgba(47,98,224,0.35)', link: '47,98,224', hot: '10,143,196' };
  }

  private resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
    const n = Math.min(130, Math.round((this.w * this.h) / 13000));
    let seed = 1234567;
    const rnd = (): number => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    this.nodes = Array.from({ length: n }, () => ({
      x: rnd() * this.w,
      y: rnd() * this.h,
      vx: (rnd() - 0.5) * 12,
      vy: (rnd() - 0.5) * 12,
      ox: 0,
      oy: 0,
    }));
    this.draw(0);
  }

  setVisible(v: boolean): void {
    this.shown = v;
    this.kick();
  }

  private kick(): void {
    if (this.shown && !this.reduced && !document.hidden && !this.raf) {
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.frame);
    }
  }

  private readonly frame = (now: number): void => {
    this.raf = 0;
    if (!this.shown || document.hidden) return;
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.draw(dt);
    this.raf = requestAnimationFrame(this.frame);
  };

  private draw(dt: number): void {
    const g = this.g;
    if (!g) return;
    const { w, h } = this;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    for (const n of this.nodes) {
      n.x += n.vx * dt;
      n.y += n.vy * dt;
      if (n.x < -20) n.x = w + 20;
      if (n.x > w + 20) n.x = -20;
      if (n.y < -20) n.y = h + 20;
      if (n.y > h + 20) n.y = -20;
      // the pointer gently pushes nodes aside; the offset relaxes back (critically damped feel)
      const dx = n.x - this.mx;
      const dy = n.y - this.my;
      const d = Math.hypot(dx, dy);
      const push = d < POINTER_R ? (1 - d / POINTER_R) * 26 : 0;
      const tx = d > 0 ? (dx / d) * push : 0;
      const ty = d > 0 ? (dy / d) * push : 0;
      const k = 1 - Math.exp(-dt * 6);
      n.ox += (tx - n.ox) * k;
      n.oy += (ty - n.oy) * k;
    }

    const P = this.nodes.map((n) => [n.x + n.ox, n.y + n.oy] as const);
    const links: [number, number][] = [];
    g.lineWidth = 1;
    for (let i = 0; i < P.length; i++) {
      for (let j = i + 1; j < P.length; j++) {
        const d = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]);
        if (d > LINK) continue;
        links.push([i, j]);
        const near = Math.max(0, 1 - Math.hypot((P[i][0] + P[j][0]) / 2 - this.mx, (P[i][1] + P[j][1]) / 2 - this.my) / POINTER_R);
        const a = (1 - d / LINK) * (0.14 + 0.5 * near);
        g.strokeStyle = `rgba(${near > 0.05 ? this.colors.hot : this.colors.link},${a.toFixed(3)})`;
        g.beginPath();
        g.moveTo(P[i][0], P[i][1]);
        g.lineTo(P[j][0], P[j][1]);
        g.stroke();
      }
    }
    g.fillStyle = this.colors.node;
    for (const [x, y] of P) {
      g.beginPath();
      g.arc(x, y, 1.6, 0, Math.PI * 2);
      g.fill();
    }

    // sparks: a bright bead running along a link
    if (dt > 0 && links.length && this.sparks.length < 6 && Math.random() < dt * 2.2) {
      const [a, b] = links[Math.floor(Math.random() * links.length)];
      this.sparks.push({ a, b, t: 0 });
    }
    this.sparks = this.sparks.filter((s) => (s.t += dt * 1.4) < 1 && s.a < P.length && s.b < P.length);
    for (const s of this.sparks) {
      const x = P[s.a][0] + (P[s.b][0] - P[s.a][0]) * s.t;
      const y = P[s.a][1] + (P[s.b][1] - P[s.a][1]) * s.t;
      const grad = g.createRadialGradient(x, y, 0, x, y, 10);
      grad.addColorStop(0, `rgba(${this.colors.hot},0.9)`);
      grad.addColorStop(1, `rgba(${this.colors.hot},0)`);
      g.fillStyle = grad;
      g.beginPath();
      g.arc(x, y, 10, 0, Math.PI * 2);
      g.fill();
    }
  }
}
