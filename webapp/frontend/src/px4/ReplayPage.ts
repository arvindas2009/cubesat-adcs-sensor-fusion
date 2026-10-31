// Page controller of the PX4 replay tab (docs/specs/px4_replay.md §3.3). Created once, lazily on the first
// visit. It owns the DOM of #page-px4, the two 3D views, the panels, the playback clock and the render loop
// (which only runs while the tab is visible). The replay is one JSON document from the server; the browser
// plays and scrubs it locally and does no physics.
import { ApiError } from '../api';
import { DroneView } from './DroneView';
import { PathView } from './PathView';
import { ReplayPanels } from './ReplayPanels';
import { PlaybackClock } from './clock';
import { getLogs, loadReplay } from './api';
import { metaChips } from './meta';
import { nextReset, prevReset } from './series';
import type { Replay } from './types';

/** Permanent caveat (docs/specs/px4_replay.md §3.3); `replay.caveat` replaces the text once a replay is loaded. */
export const CAVEAT = 'EKF2 is a reference, not truth';
export const FOOTER_TEXT = 'Replay — visual aid only; reported numbers come from sensor_fusion/report/. EKF2 is a reference, not truth.';

const RETRY_MS = 2000;
const MAX_DT_S = 0.25; // a throttled background tab must not make the replay jump

const SHELL = `
  <div class="px4-head">
    <span class="caveat-badge" id="px4-caveat" role="note"></span>
    <div class="px4-meta" id="px4-meta"></div>
  </div>
  <div class="px4-banner" id="px4-banner" role="alert"></div>
  <main class="layout px4-layout">
    <section class="px4-views">
      <div class="panel viewport" id="px4-drone"><div class="card-head"><h2>Drone attitude</h2><span class="card-sub">body FRD → NED; ours solid, EKF2 outline</span></div></div>
      <div class="panel viewport" id="px4-path"><div class="card-head"><h2>Path</h2><span class="card-sub">NED, 1 unit = 1 m</span></div></div>
    </section>
    <aside class="px4-side">
      <div class="block block-sun">
        <div class="panel" id="px4-log"><div class="card-head"><h2>Log</h2></div><div class="panel-body"></div></div>
        <div class="panel" id="px4-transport"><div class="card-head"><h2>Transport</h2></div><div class="panel-body"></div></div>
      </div>
      <div class="block block-peri">
        <div class="panel" id="px4-readouts"><div class="card-head"><h2>Readouts</h2></div><div class="panel-body"></div></div>
      </div>
    </aside>
    <section class="px4-bottom block block-mint">
      <div class="panel" id="px4-timeline"></div>
      <div class="px4-plots">
        <div class="panel" id="px4-plot-1"></div>
        <div class="panel" id="px4-plot-2"></div>
        <div class="panel" id="px4-plot-3"></div>
        <div class="panel" id="px4-plot-4"></div>
      </div>
    </section>
  </main>
`;

function must<T extends HTMLElement>(root: ParentNode, sel: string): T {
  const e = root.querySelector<T>(sel);
  if (!e) throw new Error(`missing element ${sel}`);
  return e;
}

function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message === 'Failed to fetch' ? 'Cannot reach the server.' : e.message;
  return String(e);
}

/** Keys that Space should leave alone (a focused control handles it itself). */
function spaceIsForControl(target: EventTarget | null): boolean {
  const t = target as HTMLElement | null;
  if (!t || !t.tagName) return false;
  if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.tagName === 'BUTTON') return true;
  return t.tagName === 'INPUT' && (t as HTMLInputElement).type !== 'range';
}

export class ReplayPage {
  private readonly panels: ReplayPanels;
  private drone: DroneView | null = null;
  private path: PathView | null = null;
  private readonly caveat: HTMLElement;
  private readonly meta: HTMLElement;
  private readonly banner: HTMLElement;
  private rep: Replay | null = null;
  private clock: PlaybackClock | null = null;
  private speed = 1;
  private visible = false;
  private raf = 0;
  private lastNow = 0;
  private shownT = Number.NaN;
  private shownPlaying = false;
  private loading: string | null = null;
  private logsTimer = 0;

  constructor(root: HTMLElement) {
    root.innerHTML = SHELL;
    this.caveat = must(root, '#px4-caveat');
    this.meta = must(root, '#px4-meta');
    this.banner = must(root, '#px4-banner');
    this.caveat.textContent = CAVEAT;
    this.setMeta(['No log loaded']);

    // WebGL may be unavailable (blocked GPU, headless): keep the panels and plots working and say so.
    try {
      this.drone = new DroneView(must(root, '#px4-drone'));
      this.path = new PathView(must(root, '#px4-path'));
    } catch (e) {
      this.drone = null;
      this.path = null;
      this.setBanner(`3D views are unavailable (${e instanceof Error ? e.message : String(e)}); panels and plots still work.`);
    }

    this.panels = new ReplayPanels(
      {
        log: must(root, '#px4-log'),
        transport: must(root, '#px4-transport'),
        readouts: must(root, '#px4-readouts'),
        timeline: must(root, '#px4-timeline'),
        plots: [must(root, '#px4-plot-1'), must(root, '#px4-plot-2'), must(root, '#px4-plot-3'), must(root, '#px4-plot-4')],
      },
      {
        onLoad: (role) => void this.load(role),
        onTogglePlay: () => this.togglePlay(),
        onSpeed: (v) => {
          this.speed = v;
          if (this.clock) this.clock.speed = v;
        },
        onSeek: (t) => this.seek(t),
        onPrevReset: () => this.jumpReset(false),
        onNextReset: () => this.jumpReset(true),
      },
    );

    window.addEventListener('keydown', (e) => {
      if (!this.visible || e.code !== 'Space' || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      if (spaceIsForControl(e.target)) return;
      e.preventDefault();
      this.togglePlay();
    });

    void this.loadLogs();
  }

  // ---------------------------------------------------------------- visibility and loop

  /** Show or hide the page: the render loop runs only while visible, and the 3D views re-fit on showing. */
  setVisible(on: boolean): void {
    if (on === this.visible) return;
    this.visible = on;
    if (on) {
      this.lastNow = 0;
      this.shownT = Number.NaN; // force a refresh of every readout
      this.drone?.invalidate();
      this.path?.invalidate();
      if (!this.raf) this.raf = requestAnimationFrame(this.loop);
    } else if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
  }

  private readonly loop = (now: number): void => {
    this.raf = 0;
    if (!this.visible) return;
    const dt = this.lastNow ? Math.min(MAX_DT_S, Math.max(0, (now - this.lastNow) / 1000)) : 0;
    this.lastNow = now;
    const clock = this.clock;
    if (clock && this.rep) {
      clock.tick(dt);
      if (clock.t !== this.shownT || clock.playing !== this.shownPlaying) this.show(this.rep, clock);
    }
    this.drone?.render();
    this.path?.render();
    this.raf = requestAnimationFrame(this.loop);
  };

  /** Push the current clock time into the panels and the 3D views. */
  private show(rep: Replay, clock: PlaybackClock): void {
    const i = this.panels.onTime(rep, clock);
    this.drone?.update(rep.ours.q[i], rep.ekf2.q[i]);
    this.path?.update(i);
    this.shownT = clock.t;
    this.shownPlaying = clock.playing;
  }

  // ---------------------------------------------------------------- transport

  private togglePlay(): void {
    if (!this.clock || !this.rep) return;
    this.clock.toggle();
    this.show(this.rep, this.clock);
  }

  private seek(t: number): void {
    if (!this.clock || !this.rep) return;
    this.clock.seek(t);
    this.show(this.rep, this.clock);
  }

  private jumpReset(forward: boolean): void {
    if (!this.clock || !this.rep) return;
    const r = forward ? nextReset(this.rep, this.clock.t) : prevReset(this.rep, this.clock.t);
    if (r !== null) this.seek(r);
  }

  // ---------------------------------------------------------------- data flow

  /** The header pills: one small pill per fact instead of one middle-dot string. */
  private setMeta(chips: string[]): void {
    this.meta.replaceChildren(
      ...chips.map((text) => {
        const e = document.createElement('span');
        e.className = 'pill num';
        e.textContent = text;
        return e;
      }),
    );
  }

  private setBanner(message: string | null): void {
    this.banner.textContent = message ?? '';
    this.banner.classList.toggle('visible', Boolean(message));
  }

  private async loadLogs(): Promise<void> {
    window.clearTimeout(this.logsTimer);
    try {
      this.panels.setLogs(await getLogs());
    } catch (e) {
      this.panels.setLogsFailed(`${errorText(e)} Retrying…`);
      this.logsTimer = window.setTimeout(() => void this.loadLogs(), RETRY_MS);
    }
  }

  private async load(role: string): Promise<void> {
    if (this.loading) return;
    this.loading = role;
    this.setBanner(null);
    this.panels.setBusy(role);
    try {
      const rep = await loadReplay(role);
      this.setReplay(rep);
      this.panels.setActiveLog(role);
    } catch (e) {
      this.setBanner(errorText(e));
    } finally {
      this.loading = null;
      this.panels.setBusy(null);
      void this.loadLogs(); // the chip changes from "ready" to "cached"
    }
  }

  /** Install a freshly loaded replay: rebuild plots and the path, dispose what the previous one held. */
  private setReplay(rep: Replay): void {
    const n = Array.isArray(rep?.t) ? rep.t.length : 0;
    if (rep?.schema_version !== 1 || n < 2) throw new Error('The server sent a replay in an unexpected format.');
    this.rep = rep;
    const clock = new PlaybackClock(rep.t[0], rep.t[n - 1]);
    clock.speed = this.speed;
    this.clock = clock;
    this.caveat.textContent = rep.caveat || CAVEAT;
    const dur = rep.log.duration_s ?? rep.t[n - 1] - rep.t[0];
    this.setMeta(metaChips({ role: rep.log.role, id: rep.log.id, durationS: dur, rateHz: rep.rate_hz, resets: rep.events.ekf2_resets_s.length }));
    this.path?.setReplay(rep);
    this.panels.setReplay(rep, clock);
    this.shownT = Number.NaN;
    this.show(rep, clock);
  }
}
