// App bootstrap for the ADCS page: renders the shell, creates the two 3D views and the panels, and
// wires the session flow (presets -> POST -> WebSocket). The page is a visual aid only: every number
// shown comes from server frames (docs/specs/webapp_protocol.md).
import '@fontsource/barlow-semi-condensed/latin-400.css';
import '@fontsource/barlow-semi-condensed/latin-500.css';
import '@fontsource/barlow-semi-condensed/latin-600.css';
import '@fontsource-variable/jetbrains-mono/wght.css';
import './style.css';
import { ApiError, createSession, deleteSession, getPresets, wsUrl } from './api';
import { OrbitView } from './adcs/OrbitView';
import { Panels } from './adcs/Panels';
import { SatView } from './adcs/SatView';
import { History } from './adcs/history';
import { FrameLatch, ModeTrack, Throttle, closeBanner } from './adcs/logic';
import { ReplayPage, CAVEAT as PX4_CAVEAT } from './px4/ReplayPage';
import { Sheet, sheetMarkup } from './ui/Sheet';
import { nextTab } from './ui/a11y';
import { closeChartDialog } from './ui/ChartDialog';
import { SessionSocket } from './ws';
import { currentTheme, initTheme, onThemeChange, otherTheme, toggleTheme } from './theme';
import type { Frame, FrameEvents, Hello, SessionParams } from './types';

export const FOOTER_TEXT = 'Simulation — visual aid only; reported numbers come from adcs/report/.';

type Tab = 'adcs' | 'px4';

const TABS: Tab[] = ['adcs', 'px4'];

/** Document title of each tab (the bar brand is "Attitude Lab" on both). */
const DOC_TITLE: Record<Tab, string> = {
  adcs: 'Attitude lab: CubeSat ADCS live simulator',
  px4: 'Attitude lab: PX4 flight replay',
};

/** `#px4` opens the replay tab; anything else (including no hash) the ADCS tab. */
function tabFromHash(hash: string): Tab {
  return hash.replace(/^#/, '') === 'px4' ? 'px4' : 'adcs';
}

const HISTORY_CAPACITY = 20000;
const CHART_INTERVAL_MS = 125; // plots and timeline refresh at about 8 Hz

const ICON_SUN =
  '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
const ICON_MOON =
  '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path d="M13.5 9.6A5.8 5.8 0 0 1 6.4 2.5a5.8 5.8 0 1 0 7.1 7.1z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>';

/** Top bar (spec ui_console.md §2) and the two tab panels. The PX4 caveat is here, so it shows before ReplayPage exists. */
const SHELL = `
  <header class="app-bar">
    <h1 class="brand" id="title"><span class="brand-mark" aria-hidden="true">■</span><span class="brand-text">Attitude Lab</span></h1>
    <nav class="tabs" role="tablist" aria-label="Instrument">
      <button class="tab" id="tab-adcs" role="tab" aria-selected="true" aria-controls="page-adcs" tabindex="0">ADCS</button>
      <button class="tab" id="tab-px4" role="tab" aria-selected="false" aria-controls="page-px4" tabindex="-1">PX4 replay</button>
    </nav>
    <div class="bar-status" id="bar-status">
      <div class="bar-slot" id="bar-status-adcs"></div>
      <div class="bar-slot" id="bar-status-px4" hidden>
        <span class="caveat" id="px4-caveat" role="note">${PX4_CAVEAT}</span>
        <span class="bar-meta" id="px4-meta"></span>
      </div>
    </div>
    <button class="theme-toggle" id="theme-toggle" type="button"></button>
  </header>
  <main id="stage-root">
    <section class="page" id="page-adcs" role="tabpanel" aria-labelledby="tab-adcs">
      <div id="banner" role="alert"></div>
      <div class="stage" id="adcs-stage">
        <div class="stage-views" id="views">
          <div class="panel viewport" id="orbit-view"><div class="card-head"><h2>Orbit</h2><span class="card-sub">Earth-centred inertial frame</span></div></div>
          <div class="panel viewport" id="sat-view"><div class="card-head"><h2>Attitude</h2><span class="card-sub">local frame, zenith up</span></div></div>
        </div>
        <div class="stage-dim" aria-hidden="true"></div>
        <aside class="stage-side" id="side" data-own-scroll>
          <div class="panel" id="form"><div class="card-head"><h2>Session</h2></div><div class="panel-body"></div></div>
          <div class="panel" id="transport"><div class="card-head"><h2>Transport</h2></div><div class="panel-body"></div></div>
          <div class="panel" id="kick"><div class="card-head"><h2>Kick</h2></div><div class="panel-body"></div></div>
          <div class="panel" id="readouts"><div class="card-head"><h2>Readouts</h2></div><div class="panel-body"></div></div>
        </aside>
        ${sheetMarkup({
          id: 'sheet-adcs',
          timelineId: 'timeline',
          plotsId: 'plots',
          plotIds: ['plot-1', 'plot-2', 'plot-3', 'plot-4'],
          footerId: 'footer',
          footerHtml: FOOTER_TEXT.replace('adcs/report/', '<code>adcs/report/</code>'),
        })}
      </div>
    </section>
    <section class="page" id="page-px4" role="tabpanel" aria-labelledby="tab-px4" hidden></section>
  </main>
  <div id="toasts" aria-live="polite"></div>
`;

function must<T extends HTMLElement>(root: ParentNode, sel: string): T {
  const e = root.querySelector<T>(sel);
  if (!e) throw new Error(`missing element ${sel}`);
  return e;
}

function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.status === 429 ? `${e.message} (too many sessions in use)` : e.message;
  if (e instanceof Error) return e.message === 'Failed to fetch' ? 'Cannot reach the server.' : e.message;
  return String(e);
}

function mount(root: HTMLElement): void {
  initTheme(); // the inline script in index.html already set data-theme; this re-reads the same sources
  root.innerHTML = SHELL;

  // ---- theme toggle: the button shows the mode it switches to ----
  const themeBtn = must<HTMLButtonElement>(root, '#theme-toggle');
  function syncThemeButton(): void {
    const next = otherTheme(currentTheme());
    themeBtn.innerHTML = next === 'dark' ? ICON_MOON : ICON_SUN; // the icon shows the mode it switches to
    themeBtn.setAttribute('aria-label', `Switch to ${next} mode`);
    themeBtn.title = `Switch to ${next} mode`;
  }
  syncThemeButton();
  themeBtn.addEventListener('click', () => {
    toggleTheme();
    syncThemeButton();
  });
  onThemeChange(syncThemeButton);

  // ---- tabs: #page-adcs / #page-px4, switched by the buttons, the arrow keys and the URL hash (#adcs / #px4) ----
  const pageAdcs = must(root, '#page-adcs');
  const pagePx4 = must(root, '#page-px4');
  const slotAdcs = must(root, '#bar-status-adcs');
  const slotPx4 = must(root, '#bar-status-px4');
  const tabButtons = TABS.map((t) => must<HTMLButtonElement>(root, `#tab-${t}`));
  let adcsVisible = true; // the ADCS session keeps running while hidden; only rendering is skipped
  let replay: ReplayPage | null = null;
  let chartsForceOnShow = false;

  function selectTab(tab: Tab, writeHash: boolean): void {
    closeChartDialog(); // an expanded plot belongs to the tab it was opened from
    document.title = DOC_TITLE[tab];
    root.dataset.tab = tab;
    TABS.forEach((t, i) => {
      const on = t === tab;
      tabButtons[i].setAttribute('aria-selected', String(on));
      tabButtons[i].tabIndex = on ? 0 : -1; // roving tabindex
    });
    pageAdcs.hidden = tab !== 'adcs';
    pagePx4.hidden = tab !== 'px4';
    slotAdcs.hidden = tab !== 'adcs';
    slotPx4.hidden = tab !== 'px4';
    adcsVisible = tab === 'adcs';
    if (adcsVisible) {
      chartsForceOnShow = true;
      panels.syncCharts(); // uPlot caches its rect, which was 0 while the page was hidden
    }
    if (tab === 'px4' && !replay) replay = new ReplayPage(pagePx4, slotPx4); // lazily, once, after the page is visible
    replay?.setVisible(tab === 'px4');
    if (writeHash && tabFromHash(location.hash) !== tab) location.hash = tab;
  }
  tabButtons.forEach((b, i) => b.addEventListener('click', () => selectTab(TABS[i], true)));
  must(root, '.tabs').addEventListener('keydown', (e) => {
    const ke = e as KeyboardEvent;
    const cur = tabButtons.findIndex((b) => b === document.activeElement || b.getAttribute('aria-selected') === 'true');
    const to = nextTab(ke.key, Math.max(0, cur), TABS.length);
    if (to === null) return;
    ke.preventDefault();
    selectTab(TABS[to], true);
    tabButtons[to].focus();
  });
  window.addEventListener('hashchange', () => selectTab(tabFromHash(location.hash), false));

  const history = new History(HISTORY_CAPACITY);
  const modes = new ModeTrack();
  const latch = new FrameLatch();
  const chartThrottle = new Throttle(CHART_INTERVAL_MS);
  let chartsDirty = false;
  let forceCharts = false;
  let latestEvents: FrameEvents | null = null;
  let latestT = 0;

  let sock: SessionSocket | null = null;
  let sessionId: string | null = null;

  // WebGL may be unavailable (blocked GPU, headless): keep the panels working and say so.
  let orbit: OrbitView | null = null;
  let sat: SatView | null = null;
  let glError: string | null = null;
  try {
    orbit = new OrbitView(must(root, '#orbit-view'));
    sat = new SatView(must(root, '#sat-view'));
  } catch (e) {
    glError = `3D views are unavailable (${e instanceof Error ? e.message : String(e)}); panels and plots still work.`;
    orbit = null;
    sat = null;
  }

  function closeSocket(): void {
    sock?.close();
    sock = null;
  }

  const panels: Panels = new Panels(
    {
      form: must(root, '#form'),
      transport: must(root, '#transport'),
      kick: must(root, '#kick'),
      readouts: must(root, '#readouts'),
      timeline: must(root, '#timeline'),
      plots: [must(root, '#plot-1'), must(root, '#plot-2'), must(root, '#plot-3'), must(root, '#plot-4')],
      banner: must(root, '#banner'),
      toasts: must(root, '#toasts'),
      status: slotAdcs,
    },
    {
      onStart: (params) => void startSession(params),
      onPlay: () => sock?.play(),
      onPause: () => sock?.pause(),
      onSpeed: (v) => sock?.speed(v),
      onStep: (n) => sock?.step(n),
      onReset: () => sock?.reset(),
      onKick: (cmd) => sock?.kick(cmd.axis, cmd.deg_s),
    },
  );
  if (glError) panels.setBanner(glError);

  // The rising plot sheet: wheel / keys / drag, parallax, inert and 3D render pausing (spec ui_console.md §3).
  const sheet = new Sheet(must(root, '#adcs-stage'), {
    isActive: () => adcsVisible,
    onSettle: () => panels.syncCharts(),
    onResumeRender: () => {
      orbit?.invalidate();
      sat?.invalidate();
    },
  });

  // ---- data flow ----

  function applyHello(h: Hello): void {
    history.clear();
    modes.clear();
    latestEvents = null;
    latestT = 0;
    orbit?.setInfo(h.info);
    sat?.setInfo(h.info);
    panels.onHello(h.info, h.status);
    panels.clearCharts();
    chartsDirty = true;
    forceCharts = true;
  }

  function onFrame(f: Frame): void {
    history.push(f); // every frame goes into the history ...
    modes.push(f.t, f.mode);
    latestEvents = f.events;
    latestT = f.t;
    latch.offer(f); // ... but only the newest one is drawn per animation frame
    chartsDirty = true;
    if (!f.playing || f.done) forceCharts = true; // paused/done: show the final state immediately
  }

  function connect(id: string): void {
    let first = true; // a new session starts playing; a later hello (Reset) keeps the paused state
    const s: SessionSocket = new SessionSocket(wsUrl(id, window.location), {
      onHello: (h) => {
        if (sock !== s) return;
        panels.setPreparing(false);
        panels.setBanner(glError);
        applyHello(h);
        if (first) s.play();
        first = false;
      },
      onFrame: (f) => {
        if (sock === s) onFrame(f);
      },
      onError: (e) => {
        if (sock === s) panels.toast(e.message);
      },
      onClose: (code) => {
        if (sock !== s) return;
        panels.setPreparing(false);
        panels.onClosed();
        panels.setBanner(closeBanner(code)); // 4001 replaced, 4404 unknown session, anything else: lost
      },
    });
    sock = s;
  }

  async function startSession(params: SessionParams): Promise<void> {
    panels.setBanner(glError);
    panels.setPreparing(true);
    const old = sessionId;
    closeSocket();
    sessionId = null;
    if (old) deleteSession(old).catch(() => undefined);
    try {
      const res = await createSession(params);
      sessionId = res.id;
      panels.setPreparing(true, 'Connecting…');
      connect(res.id);
    } catch (e) {
      panels.setPreparing(false);
      const msg = errorText(e);
      panels.setFormError(msg);
      panels.toast(msg);
    }
  }

  async function loadPresets(): Promise<void> {
    try {
      panels.setPresets(await getPresets());
    } catch (e) {
      panels.setPresetsFailed(`${errorText(e)} Retrying…`);
      window.setTimeout(() => void loadPresets(), 2000);
    }
  }
  void loadPresets();

  // ---- render loop: newest frame only, charts throttled ----

  function loop(now: number): void {
    // A hidden ADCS tab keeps receiving frames into the history (and keeps the newest one latched) but
    // skips all drawing; on showing the tab the newest frame and the charts are drawn at once.
    if (adcsVisible) {
      if (chartsForceOnShow) {
        chartsForceOnShow = false;
        chartsDirty = true;
        forceCharts = true;
      }
      const f = latch.take();
      if (f) {
        orbit?.update(f);
        sat?.update(f);
        panels.onFrame(f);
      }
      if (chartsDirty && chartThrottle.shouldRun(now, forceCharts)) {
        panels.refreshCharts(history.columns(), latestT, modes.segments(), latestEvents);
        chartsDirty = false;
        forceCharts = false;
      }
      if (!sheet.renderPaused) {
        orbit?.render(); // skipped while the raised sheet covers the views; update() above keeps state current
        sat?.render();
      }
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  selectTab(tabFromHash(location.hash), false);

  // Free the server-side session when the tab goes away.
  window.addEventListener('pagehide', () => {
    if (sessionId) {
      void fetch(`/api/adcs/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE', keepalive: true }).catch(() => undefined);
    }
  });
}

const root = document.getElementById('app');
if (root) mount(root);
