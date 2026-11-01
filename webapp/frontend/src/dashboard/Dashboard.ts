// Landing dashboard (docs/specs/ui_dashboard.md §3): editorial hero with a live 3D CubeSat, kinetic
// headline, the two instruments as cards that route to #adcs / #px4, sourced headline numbers and a
// footer band. Plain links drive the routing (main.ts listens to hashchange). Browser-only.
import { getPresets } from '../api';
import { getLogs } from '../px4/api';
import { FACTS } from './facts';
import { HeroView } from './HeroView';

const ARROW =
  '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path d="M3 8h9M8.5 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function words(line: string): string {
  return line
    .split(' ')
    .map((w, i) => `<span class="w" style="--i:${i}">${w}</span>`)
    .join(' ');
}

const MARKUP = `
  <div class="dash">
    <section class="dash-hero" aria-labelledby="dash-title">
      <div class="dash-hero-view" id="hero-view"></div>
      <div class="dash-hero-top">
        <span class="dash-eyebrow"><span class="dot" aria-hidden="true"></span>Open-source estimation &amp; control lab</span>
      </div>
      <div class="dash-hero-bottom">
        <h1 class="dash-hero-title" id="dash-title">Attitude, determined.</h1>
        <dl class="dash-hero-spec">
          <div><dt>Bus</dt><dd>3U · 4 kg</dd></div>
          <div><dt>Orbit</dt><dd>400 km · 51.6°</dd></div>
          <div><dt>Filter</dt><dd>MEKF · ES-EKF</dd></div>
        </dl>
      </div>
    </section>

    <section class="dash-intro">
      <p class="dash-lede">A satellite can only point where it thinks it is pointing. This lab closes that loop, from raw sensors to a pointing controller, and checks itself against real flight data.</p>
    </section>

    <section class="dash-kinetic" aria-label="What the lab does">
      <p class="dash-big reveal">${words('KNOW WHERE')}<br>${words('IT POINTS.')}</p>
      <div class="dash-kinetic-right">
        <p class="dash-big reveal">${words('THEN POINT')}<br>${words('IT RIGHT.')}</p>
        <p class="dash-big dash-big-outline reveal">${words('VERIFIED.')}</p>
      </div>
    </section>

    <section class="dash-cards" aria-label="Instruments">
      <a class="dash-card dash-card-adcs reveal" href="#adcs">
        <div class="dash-card-art" aria-hidden="true">
          <svg viewBox="0 0 200 120" preserveAspectRatio="xMidYMid slice"><defs><radialGradient id="g1" cx="70%" cy="120%" r="90%"><stop offset="0" stop-color="#2d6bff" stop-opacity=".55"/><stop offset=".55" stop-color="#0b1a3a" stop-opacity=".4"/><stop offset="1" stop-color="#05070b" stop-opacity="0"/></radialGradient></defs><rect width="200" height="120" fill="url(#g1)"/><ellipse cx="140" cy="210" rx="190" ry="110" fill="none" stroke="#7fb0ff" stroke-opacity=".35"/><ellipse cx="100" cy="64" rx="78" ry="20" fill="none" stroke="#e6e9ed" stroke-opacity=".25" stroke-dasharray="2 3" transform="rotate(-14 100 64)"/><g transform="translate(118 46) rotate(-20)"><rect x="-4" y="-10" width="8" height="20" rx="1" fill="#c9d3e6"/><rect x="-20" y="-6" width="14" height="12" rx="1" fill="#1d3366" stroke="#7fb0ff" stroke-width=".5"/><rect x="6" y="-6" width="14" height="12" rx="1" fill="#1d3366" stroke="#7fb0ff" stroke-width=".5"/></g></svg>
        </div>
        <div class="dash-card-body">
          <span class="dash-tag">Project B · Live simulation</span>
          <h2 class="dash-card-title">CubeSat ADCS simulator</h2>
          <p class="dash-card-text">Detumble with B-dot, hand over to nadir pointing, and watch the MEKF estimate track the truth through eclipse. Kick it, pause it, step it.</p>
          <div class="dash-card-foot">
            <span class="dash-status" id="dash-status-adcs" data-state="pending">Checking server…</span>
            <span class="dash-cta">Open simulator ${ARROW}</span>
          </div>
        </div>
      </a>
      <a class="dash-card dash-card-px4 reveal" href="#px4">
        <div class="dash-card-art" aria-hidden="true">
          <svg viewBox="0 0 200 120" preserveAspectRatio="xMidYMid slice"><defs><radialGradient id="g2" cx="25%" cy="110%" r="90%"><stop offset="0" stop-color="#ff8a1f" stop-opacity=".5"/><stop offset=".6" stop-color="#3a1a05" stop-opacity=".35"/><stop offset="1" stop-color="#05070b" stop-opacity="0"/></radialGradient></defs><rect width="200" height="120" fill="url(#g2)"/><path d="M14 96 C40 40, 70 110, 96 62 S150 20, 186 40" fill="none" stroke="#3ddc97" stroke-width="1.6" stroke-opacity=".85"/><path d="M14 98 C42 44, 70 112, 98 66 S152 24, 186 44" fill="none" stroke="#ffb347" stroke-width="1" stroke-dasharray="3 3" stroke-opacity=".8"/><g transform="translate(150 34)"><path d="M-12-12L12 12M-12 12L12-12" stroke="#e6e9ed" stroke-width="2"/><g fill="none" stroke="#e6e9ed" stroke-opacity=".6"><circle cx="-12" cy="-12" r="6"/><circle cx="12" cy="12" r="6"/><circle cx="-12" cy="12" r="6"/><circle cx="12" cy="-12" r="6"/></g><rect x="-4" y="-4" width="8" height="8" rx="2" fill="#e6e9ed"/></g></svg>
        </div>
        <div class="dash-card-body">
          <span class="dash-tag">Project A · Real flight logs</span>
          <h2 class="dash-card-title">PX4 flight replay</h2>
          <p class="dash-card-text">A 15-state error-state EKF run on real PX4 drone logs, replayed side by side with the autopilot's own EKF2 estimate.</p>
          <div class="dash-card-foot">
            <span class="dash-status" id="dash-status-px4" data-state="pending">Checking logs…</span>
            <span class="dash-cta">Open replay ${ARROW}</span>
          </div>
        </div>
      </a>
    </section>

    <section class="dash-facts" aria-labelledby="dash-facts-title">
      <div class="dash-facts-head">
        <span class="dash-eyebrow"><span class="dot" aria-hidden="true"></span>Measured, not claimed</span>
        <h2 class="dash-h2" id="dash-facts-title">What the numbers say.</h2>
        <p class="dash-note">Every figure is copied from the project log (<code>PROGRESS.md</code>). Simulation results are labelled as such; flight results compare against PX4's EKF2, which is a reference, not truth.</p>
      </div>
      <div class="dash-facts-grid">
        ${FACTS.map(
          (f, i) => `
          <article class="dash-fact reveal">
            <span class="dash-fact-n">${String(i + 1).padStart(2, '0')}</span>
            <p class="dash-fact-value"><span>${f.value}${f.unit === '°' ? '°' : ''}</span>${f.unit && f.unit !== '°' ? `<small>${f.unit}</small>` : ''}</p>
            <h3 class="dash-fact-label">${f.label}</h3>
            <p class="dash-fact-detail">${f.detail}</p>
            <p class="dash-fact-src">${f.source}</p>
          </article>`,
        ).join('')}
      </div>
    </section>

    <section class="dash-band" aria-hidden="true">
      <div class="dash-marquee"><span>SENSE · ESTIMATE · CONTROL · VERIFY · SENSE · ESTIMATE · CONTROL · VERIFY ·&nbsp;</span><span>SENSE · ESTIMATE · CONTROL · VERIFY · SENSE · ESTIMATE · CONTROL · VERIFY ·&nbsp;</span></div>
    </section>

    <footer class="dash-foot">
      <div>
        <p class="dash-foot-brand">Attitude Lab</p>
        <p class="dash-note">Visual aid only. Reported numbers come from <code>adcs/report/</code> and <code>sensor_fusion/report/</code>.</p>
      </div>
      <nav class="dash-foot-links" aria-label="Instruments">
        <a href="#adcs">ADCS simulator</a>
        <a href="#px4">PX4 replay</a>
        <a href="#home">Back to top</a>
      </nav>
    </footer>
  </div>
`;

export class Dashboard {
  private hero: HeroView | null = null;
  private statusLoaded = false;

  constructor(private readonly page: HTMLElement) {
    page.innerHTML = MARKUP;
    try {
      this.hero = new HeroView(page.querySelector<HTMLElement>('#hero-view')!);
    } catch {
      page.querySelector('.dash-hero')?.classList.add('no-gl'); // WebGL unavailable: keep the gradient backdrop
    }
    const items = page.querySelectorAll<HTMLElement>('.reveal');
    if (typeof IntersectionObserver === 'function') {
      const io = new IntersectionObserver(
        (es) =>
          es.forEach((e) => {
            if (e.isIntersecting) {
              e.target.classList.add('in');
              io.unobserve(e.target);
            }
          }),
        { threshold: 0.2 },
      );
      items.forEach((el) => io.observe(el));
    } else {
      items.forEach((el) => el.classList.add('in'));
    }
    // "Back to top" while already on #home: scroll instead of a no-op hash change.
    page.querySelector('a[href="#home"]')?.addEventListener('click', (e) => {
      e.preventDefault();
      page.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  setVisible(v: boolean): void {
    this.hero?.setVisible(v);
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
