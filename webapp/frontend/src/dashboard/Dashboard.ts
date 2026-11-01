// Landing page (docs/specs/ui_dashboard.md §6): a full-bleed interactive space hero with the project
// name and a brief introduction, the project in two vehicles, the sense-estimate-control-actuate loop
// (interactive), a draggable mission timeline, the two instruments, sourced numbers, the stack and a
// footer, over a calm "electric" node-field background. Plain links drive the routing (main.ts listens
// to hashchange). Browser-only.
import { getPresets } from '../api';
import { getLogs } from '../px4/api';
import { FACTS } from './facts';
import { LOOP, PHASES, STACK } from './content';
import { ElectricField } from './ElectricField';
import { HeroView } from './HeroView';
import { Scrubber } from './Scrubber';

const ARROW =
  '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path d="M3 8h9M8.5 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const ICONS: Record<string, string> = {
  sense:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="2.5"/><path d="M7.5 7.5a6.4 6.4 0 0 0 0 9M16.5 7.5a6.4 6.4 0 0 1 0 9M4.6 4.6a10.5 10.5 0 0 0 0 14.8M19.4 4.6a10.5 10.5 0 0 1 0 14.8"/></svg>',
  estimate:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 17c3-9 6-9 9 0s6 9 9 0" /><circle cx="12" cy="17" r="2"/><path d="M12 4v5M9.5 6.5h5"/></svg>',
  control:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3"/><path d="M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4"/></svg>',
  actuate:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17M3.5 12h17" opacity=".45"/><path d="M12 12l5.5-5.5" /><circle cx="12" cy="12" r="1.8"/></svg>',
};

function words(line: string): string {
  return line
    .split(' ')
    .map((w, i) => `<span class="w" style="--i:${i}">${w}</span>`)
    .join(' ');
}

const MARKUP = `
  <div class="dash">
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-view" id="hero-view"></div>
      <div class="hero-fade" aria-hidden="true"></div>
      <div class="hero-copy">
        <span class="chip-glass"><span class="pulse-dot" aria-hidden="true"></span>CubeSat ADCS · PX4 sensor fusion</span>
        <h1 class="hero-title" id="hero-title">Attitude <span class="grad">Lab</span></h1>
        <p class="hero-tag">Know where it points. Then point it right.</p>
        <p class="hero-intro">An open-source estimation and control lab. One error-state Kalman filter and quaternion core flies two vehicles: a simulated 3U CubeSat that has to find and hold its attitude in orbit, and a real drone whose PX4 flight logs check the filter against the autopilot.</p>
        <div class="hero-ctas">
          <a class="btn btn-primary" href="#adcs">Launch the ADCS sim ${ARROW}</a>
          <a class="btn btn-glass" href="#px4">Replay a PX4 flight</a>
        </div>
        <ul class="hero-telemetry" aria-label="Mission at a glance">
          <li><span>Bus</span><b>3U · 4 kg</b></li>
          <li><span>Orbit</span><b>400 km · 51.6°</b></li>
          <li><span>Sensors</span><b>1 Hz</b></li>
          <li><span>Flight logs</span><b>3 PX4</b></li>
        </ul>
      </div>
      <p class="hero-hint" aria-hidden="true"><span class="drag-icon"></span>Drag to spin the Earth</p>
      <a class="hero-scroll" href="#about" aria-label="Scroll to the introduction"><span></span></a>
    </section>

    <section class="block about" id="about" aria-labelledby="about-title">
      <div class="about-copy">
        <span class="eyebrow"><span class="dot" aria-hidden="true"></span>The project</span>
        <h2 class="h2 reveal" id="about-title">${words('One estimation core.')}<br>${words('Two vehicles.')}</h2>
        <p class="lede">Attitude determination asks <em>which way am I facing?</em> from sensors that are noisy, biased and sometimes blind (a Sun sensor in eclipse sees nothing). Attitude control then has to act on that answer. Both halves share one tested core: scalar-first Hamilton quaternions and an error-state EKF.</p>
      </div>
      <div class="about-cards">
        <article class="glass mini reveal-card">
          <span class="tag">Project A</span>
          <h3>Drone sensor fusion</h3>
          <p>A 15-state ES-EKF fuses IMU, GPS, barometer and magnetometer from real PX4 logs, and is compared with the autopilot's own EKF2.</p>
        </article>
        <article class="glass mini reveal-card">
          <span class="tag">Project B</span>
          <h3>CubeSat ADCS</h3>
          <p>Orbit, environment and disturbance models, a MEKF on magnetometer, gyro and Sun sensors, and B-dot plus pointing control, in a closed loop.</p>
        </article>
      </div>
    </section>

    <section class="block loop" aria-labelledby="loop-title">
      <div class="block-head">
        <span class="eyebrow"><span class="dot" aria-hidden="true"></span>How it works</span>
        <h2 class="h2" id="loop-title">The loop, closed.</h2>
        <p class="note">Pick a stage. Every part named here is in the simulator.</p>
      </div>
      <div class="loop-rail" role="tablist" aria-label="Loop stages">
        <svg class="loop-wire" viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true"><path d="M60 20 H940"/><path class="current" d="M60 20 H940"/></svg>
        ${LOOP.map(
          (s, i) => `
          <button class="loop-node" role="tab" id="loop-tab-${s.id}" aria-controls="loop-panel" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-i="${i}">
            <span class="loop-icon">${ICONS[s.id]}</span>
            <span class="loop-label"><small>0${i + 1}</small>${s.title}</span>
          </button>`,
        ).join('')}
      </div>
      <div class="glass loop-panel" id="loop-panel" role="tabpanel" aria-live="polite"></div>
    </section>

    <section class="block mission" aria-labelledby="mission-title">
      <div class="block-head">
        <span class="eyebrow"><span class="dot" aria-hidden="true"></span>A mission, start to finish</span>
        <h2 class="h2" id="mission-title">From tumble to pointing.</h2>
        <p class="note">Drag the satellite along its timeline, or flick it.</p>
      </div>
      <div class="glass scrub" id="scrub">
        <div class="scrub-labels">
          ${PHASES.map((p, i) => `<button class="scrub-label" data-i="${i}" type="button">${p.title}</button>`).join('')}
        </div>
        <div class="scrub-track">
          <div class="scrub-rail"></div>
          <div class="scrub-fill"></div>
          ${PHASES.map((_, i) => `<span class="scrub-stop" style="left:${(i / (PHASES.length - 1)) * 100}%"></span>`).join('')}
          <div class="scrub-knob" role="slider" tabindex="0" aria-label="Mission phase" aria-valuemin="1" aria-valuemax="${PHASES.length}" aria-valuenow="1">
            <svg viewBox="0 0 32 32" aria-hidden="true"><rect x="12" y="7" width="8" height="18" rx="2"/><rect x="2" y="11" width="9" height="10" rx="1.5"/><rect x="21" y="11" width="9" height="10" rx="1.5"/></svg>
          </div>
        </div>
        <div class="scrub-detail" aria-live="polite">
          <p class="scrub-metric" id="scrub-metric"></p>
          <div><h3 id="scrub-title"></h3><p id="scrub-text"></p></div>
        </div>
      </div>
    </section>

    <section class="block" aria-labelledby="inst-title">
      <div class="block-head">
        <span class="eyebrow"><span class="dot" aria-hidden="true"></span>Try it</span>
        <h2 class="h2" id="inst-title">Two instruments, live.</h2>
      </div>
      <div class="cards">
        <a class="glass card card-adcs" href="#adcs">
          <div class="card-art" aria-hidden="true">
            <svg viewBox="0 0 200 110" preserveAspectRatio="xMidYMid slice"><defs><radialGradient id="ga" cx="70%" cy="125%" r="95%"><stop offset="0" stop-color="#3d7bff" stop-opacity=".7"/><stop offset=".55" stop-color="#0b1a3a" stop-opacity=".5"/><stop offset="1" stop-color="#050a18" stop-opacity="0"/></radialGradient></defs><rect width="200" height="110" fill="url(#ga)"/><ellipse cx="140" cy="205" rx="190" ry="110" fill="none" stroke="#7fd4ff" stroke-opacity=".45"/><ellipse class="art-orbit" cx="100" cy="60" rx="80" ry="20" fill="none" stroke="#cfe0ff" stroke-opacity=".35" stroke-dasharray="2 4" transform="rotate(-14 100 60)"/><g class="art-sat" transform="translate(120 42) rotate(-20)"><rect x="-4" y="-10" width="8" height="20" rx="2" fill="#c9d3e6"/><rect x="-21" y="-6" width="15" height="12" rx="2" fill="#1d3366" stroke="#7fd4ff" stroke-width=".6"/><rect x="6" y="-6" width="15" height="12" rx="2" fill="#1d3366" stroke="#7fd4ff" stroke-width=".6"/></g></svg>
          </div>
          <div class="card-body">
            <span class="tag">Project B · Live simulation</span>
            <h3 class="card-title">CubeSat ADCS simulator</h3>
            <p>Detumble with B-dot, hand over to nadir pointing, and watch the MEKF estimate track the truth through eclipse. Kick it, pause it, step it.</p>
            <div class="card-foot">
              <span class="status" id="dash-status-adcs" data-state="pending">Checking server…</span>
              <span class="btn btn-primary btn-sm">Open simulator ${ARROW}</span>
            </div>
          </div>
        </a>
        <a class="glass card card-px4" href="#px4">
          <div class="card-art" aria-hidden="true">
            <svg viewBox="0 0 200 110" preserveAspectRatio="xMidYMid slice"><defs><radialGradient id="gb" cx="25%" cy="115%" r="95%"><stop offset="0" stop-color="#2fd3ff" stop-opacity=".45"/><stop offset=".6" stop-color="#0a2240" stop-opacity=".45"/><stop offset="1" stop-color="#050a18" stop-opacity="0"/></radialGradient></defs><rect width="200" height="110" fill="url(#gb)"/><path class="art-path" d="M14 92 C40 36, 70 106, 96 58 S150 16, 186 36" fill="none" stroke="#3ddc97" stroke-width="1.8" stroke-linecap="round"/><path d="M14 94 C42 40, 70 108, 98 62 S152 20, 186 40" fill="none" stroke="#ffb347" stroke-width="1" stroke-dasharray="3 3" stroke-opacity=".8"/><g transform="translate(150 30)"><path d="M-12-12L12 12M-12 12L12-12" stroke="#e6ecff" stroke-width="2" stroke-linecap="round"/><g fill="none" stroke="#9fdcff" stroke-opacity=".7"><circle cx="-12" cy="-12" r="6"/><circle cx="12" cy="12" r="6"/><circle cx="-12" cy="12" r="6"/><circle cx="12" cy="-12" r="6"/></g><rect x="-4" y="-4" width="8" height="8" rx="2.5" fill="#e6ecff"/></g></svg>
          </div>
          <div class="card-body">
            <span class="tag">Project A · Real flight logs</span>
            <h3 class="card-title">PX4 flight replay</h3>
            <p>The 15-state error-state EKF on real PX4 drone logs, replayed side by side with the autopilot's own EKF2 estimate.</p>
            <div class="card-foot">
              <span class="status" id="dash-status-px4" data-state="pending">Checking logs…</span>
              <span class="btn btn-primary btn-sm">Open replay ${ARROW}</span>
            </div>
          </div>
        </a>
      </div>
    </section>

    <section class="block facts" aria-labelledby="facts-title">
      <div class="block-head">
        <span class="eyebrow"><span class="dot" aria-hidden="true"></span>Measured, not claimed</span>
        <h2 class="h2" id="facts-title">What the numbers say.</h2>
        <p class="note">Every figure is copied from the project log (<code>PROGRESS.md</code>). Simulation results are labelled as such; flight results compare against PX4's EKF2, which is a reference, not truth.</p>
      </div>
      <div class="facts-grid">
        ${FACTS.map(
          (f, i) => `
          <article class="glass fact reveal-card">
            <span class="fact-n">${String(i + 1).padStart(2, '0')}</span>
            <p class="fact-value"><span>${f.value}${f.unit === '°' ? '°' : ''}</span>${f.unit && f.unit !== '°' ? `<small>${f.unit}</small>` : ''}</p>
            <h3>${f.label}</h3>
            <p class="fact-detail">${f.detail}</p>
            <p class="fact-src">${f.source}</p>
          </article>`,
        ).join('')}
      </div>
    </section>

    <section class="block stack" aria-labelledby="stack-title">
      <h2 class="eyebrow" id="stack-title"><span class="dot" aria-hidden="true"></span>Built with</h2>
      <ul class="stack-list">${STACK.map((s) => `<li>${s}</li>`).join('')}</ul>
    </section>

    <div class="band" aria-hidden="true"><div class="marquee"><span>SENSE · ESTIMATE · CONTROL · ACTUATE · VERIFY ·&nbsp;</span><span>SENSE · ESTIMATE · CONTROL · ACTUATE · VERIFY ·&nbsp;</span></div></div>

    <footer class="glass foot">
      <div>
        <p class="foot-brand">Attitude <span class="grad">Lab</span></p>
        <p class="note">Visual aid only. Reported numbers come from <code>adcs/report/</code> and <code>sensor_fusion/report/</code>.</p>
      </div>
      <nav class="foot-links" aria-label="Instruments">
        <a href="#adcs">ADCS simulator</a>
        <a href="#px4">PX4 replay</a>
        <a href="#home" data-top>Back to top</a>
      </nav>
    </footer>
  </div>
`;

export class Dashboard {
  private hero: HeroView | null = null;
  private readonly field: ElectricField;
  private statusLoaded = false;

  constructor(private readonly page: HTMLElement) {
    page.innerHTML = MARKUP;
    this.field = new ElectricField(page);
    try {
      this.hero = new HeroView(page.querySelector<HTMLElement>('#hero-view')!);
    } catch {
      page.querySelector('.hero')?.classList.add('no-gl'); // WebGL unavailable: the CSS sky remains
    }
    this.bindReveal();
    this.bindLoop();
    this.bindMission();

    // the hero dollies as it scrolls away (the page scrolls itself; narrow layouts scroll the window)
    const onScroll = (): void => {
      const top = page.scrollTop || window.scrollY;
      const h = page.querySelector<HTMLElement>('.hero')?.offsetHeight || 1;
      this.hero?.setScroll(top / h);
    };
    page.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });

    page.querySelector('a[data-top]')?.addEventListener('click', (e) => {
      e.preventDefault();
      page.scrollTo({ top: 0, behavior: 'smooth' });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    page.querySelector('.hero-scroll')?.addEventListener('click', (e) => {
      e.preventDefault(); // #about would change the route; scroll instead
      page.querySelector('#about')?.scrollIntoView({ behavior: 'smooth' });
    });
  }

  private bindReveal(): void {
    const items = this.page.querySelectorAll<HTMLElement>('.reveal, .reveal-card');
    if (typeof IntersectionObserver !== 'function') {
      items.forEach((el) => el.classList.add('in'));
      return;
    }
    const io = new IntersectionObserver(
      (es) =>
        es.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        }),
      { threshold: 0.15 },
    );
    items.forEach((el) => io.observe(el));
  }

  private bindLoop(): void {
    const panel = this.page.querySelector<HTMLElement>('#loop-panel')!;
    const nodes = [...this.page.querySelectorAll<HTMLButtonElement>('.loop-node')];
    const rail = this.page.querySelector<HTMLElement>('.loop-rail')!;
    const show = (i: number, focus = false): void => {
      const s = LOOP[i];
      nodes.forEach((n, k) => {
        n.setAttribute('aria-selected', String(k === i));
        n.tabIndex = k === i ? 0 : -1;
      });
      rail.style.setProperty('--sel', String(i));
      panel.setAttribute('aria-labelledby', `loop-tab-${s.id}`);
      panel.innerHTML = `
        <div class="loop-panel-head"><span class="loop-icon big">${ICONS[s.id]}</span><div><h3>${s.title}</h3><p>${s.summary}</p></div></div>
        <ul class="loop-items">${s.items.map((it) => `<li><b>${it.name}</b><span>${it.detail}</span></li>`).join('')}</ul>`;
      panel.classList.remove('swap');
      void panel.offsetWidth; // restart the cross-fade
      panel.classList.add('swap');
      if (focus) nodes[i].focus();
    };
    nodes.forEach((n, i) => {
      n.addEventListener('pointerdown', () => show(i)); // respond on press, not release
      n.addEventListener('keydown', (e) => {
        const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (d) {
          e.preventDefault();
          show((i + d + nodes.length) % nodes.length, true);
        } else if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          show(i);
        }
      });
    });
    show(0);
  }

  private bindMission(): void {
    const root = this.page.querySelector<HTMLElement>('#scrub')!;
    const metric = root.querySelector<HTMLElement>('#scrub-metric')!;
    const title = root.querySelector<HTMLElement>('#scrub-title')!;
    const text = root.querySelector<HTMLElement>('#scrub-text')!;
    const labels = [...root.querySelectorAll<HTMLButtonElement>('.scrub-label')];
    const scrub = new Scrubber(root, PHASES.length, (i) => {
      const p = PHASES[i];
      metric.textContent = p.metric;
      title.textContent = p.title;
      text.textContent = p.detail;
      labels.forEach((l, k) => l.classList.toggle('on', k === i));
      root.dataset.phase = p.id;
    });
    labels.forEach((l, i) => l.addEventListener('click', () => scrub.goTo(i, 1)));
  }

  setVisible(v: boolean): void {
    this.hero?.setVisible(v);
    this.field.setVisible(v);
    if (v && !this.statusLoaded) {
      this.statusLoaded = true;
      void this.loadStatus();
    }
  }

  private setStatus(id: string, state: 'ok' | 'off', text: string): void {
    const el = this.page.querySelector<HTMLElement>(`#${id}`);
    if (!el) return;
    el.dataset.state = state;
    el.textContent = text;
  }

  private async loadStatus(): Promise<void> {
    const [presets, logs] = await Promise.allSettled([getPresets(), getLogs()]);
    if (presets.status === 'fulfilled') {
      const n = Object.keys(presets.value.params_schema?.properties ?? {}).length;
      this.setStatus('dash-status-adcs', 'ok', `Server online · ${n} tunable parameters`);
    } else {
      this.setStatus('dash-status-adcs', 'off', 'Server offline · start python -m webapp');
    }
    if (logs.status === 'fulfilled') {
      const n = logs.value.logs.length;
      this.setStatus('dash-status-px4', 'ok', `${n} flight log${n === 1 ? '' : 's'} available`);
    } else {
      this.setStatus('dash-status-px4', 'off', 'Logs unavailable');
    }
  }
}
