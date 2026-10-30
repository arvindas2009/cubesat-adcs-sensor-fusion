// App bootstrap for the ADCS page: renders the shell, creates the two 3D views and the panels, and
// wires the session flow (presets -> POST -> WebSocket). The page is a visual aid only: every number
// shown comes from server frames (docs/specs/webapp_protocol.md).
import './style.css';
import { ApiError, createSession, deleteSession, getPresets, wsUrl } from './api';
import { OrbitView } from './adcs/OrbitView';
import { Panels } from './adcs/Panels';
import { SatView } from './adcs/SatView';
import { History } from './adcs/history';
import { FrameLatch, ModeTrack, Throttle, closeBanner } from './adcs/logic';
import { ReplayPage, FOOTER_TEXT as PX4_FOOTER_TEXT } from './px4/ReplayPage';
import { SessionSocket } from './ws';
import type { Frame, FrameEvents, Hello, SessionParams } from './types';

export const FOOTER_TEXT = 'Simulation — visual aid only; reported numbers come from adcs/report/.';

type Tab = 'adcs' | 'px4';

/** Header title, document title and footer (HTML) of each tab. */
const TAB_TEXT: Record<Tab, { h1: string; doc: string; footer: string }> = {
  adcs: {
    h1: 'CubeSat ADCS <span class="h1-sub">live simulator</span>',
    doc: 'CubeSat ADCS live simulator',
    footer: FOOTER_TEXT.replace('adcs/report/', '<code>adcs/report/</code>'),
  },
  px4: {
    h1: 'PX4 flight <span class="h1-sub">replay</span>',
    doc: 'PX4 flight replay',
    footer: PX4_FOOTER_TEXT.replace('sensor_fusion/report/', '<code>sensor_fusion/report/</code>'),
  },
};

/** `#px4` opens the replay tab; anything else (including no hash) the ADCS tab. */
function tabFromHash(hash: string): Tab {
  return hash.replace(/^#/, '') === 'px4' ? 'px4' : 'adcs';
}

const HISTORY_CAPACITY = 20000;
const CHART_INTERVAL_MS = 125; // plots and timeline refresh at about 8 Hz

const SHELL = `
  <header class="app-header">
    <h1 id="title"></h1>
    <nav class="tabs" role="tablist">
      <button class="tab" id="tab-adcs" role="tab" aria-selected="true" aria-controls="page-adcs">ADCS</button>
      <button class="tab" id="tab-px4" role="tab" aria-selected="false" aria-controls="page-px4">PX4 replay</button>
    </nav>
  </header>
  <div id="banner" role="alert"></div>
  <main class="layout" id="page-adcs">
    <section id="views">
      <div class="panel viewport" id="orbit-view"><h2>Orbit (ECI)</h2></div>
      <div class="panel viewport" id="sat-view"><h2>Attitude (local frame, zenith up)</h2></div>
    </section>
    <aside id="side">
      <div class="panel" id="form"><h2>Session</h2><div class="panel-body"></div></div>
      <div class="panel" id="transport"><h2>Transport</h2><div class="panel-body"></div></div>
      <div class="panel" id="kick"><h2>Kick</h2><div class="panel-body"></div></div>
      <div class="panel" id="readouts"><h2>Readouts</h2><div class="panel-body"></div></div>
    </aside>
    <section id="bottom">
      <div class="panel" id="timeline"></div>
      <div id="plots">
        <div class="panel" id="plot-1"></div>
        <div class="panel" id="plot-2"></div>
        <div class="panel" id="plot-3"></div>
        <div class="panel" id="plot-4"></div>
      </div>
    </section>
  </main>
  <div class="page-px4" id="page-px4" hidden></div>
  <div id="toasts" aria-live="polite"></div>
  <footer class="app-footer" id="footer"></footer>
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
  root.innerHTML = SHELL;

  // ---- tabs: #page-adcs / #page-px4, switched by the buttons and the URL hash (#adcs / #px4) ----
  const pageAdcs = must(root, '#page-adcs');
  const pagePx4 = must(root, '#page-px4');
  let adcsVisible = true; // the ADCS session keeps running while hidden; only rendering is skipped
  let replay: ReplayPage | null = null;
  let chartsForceOnShow = false;

  function selectTab(tab: Tab, writeHash: boolean): void {
    const text = TAB_TEXT[tab];
    must(root, '#title').innerHTML = text.h1;
    document.title = text.doc;
    must(root, '#footer').innerHTML = text.footer;
    root.dataset.tab = tab;
    for (const t of ['adcs', 'px4'] as Tab[]) must(root, `#tab-${t}`).setAttribute('aria-selected', String(t === tab));
    pageAdcs.hidden = tab !== 'adcs';
    pagePx4.hidden = tab !== 'px4';
    adcsVisible = tab === 'adcs';
    if (adcsVisible) chartsForceOnShow = true;
    if (tab === 'px4' && !replay) replay = new ReplayPage(pagePx4); // lazily, once, after the page is visible
    replay?.setVisible(tab === 'px4');
    if (writeHash && tabFromHash(location.hash) !== tab) location.hash = tab;
  }
  must(root, '#tab-adcs').addEventListener('click', () => selectTab('adcs', true));
  must(root, '#tab-px4').addEventListener('click', () => selectTab('px4', true));
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
    const s: SessionSocket = new SessionSocket(wsUrl(id, window.location), {
      onHello: (h) => {
        if (sock !== s) return;
        panels.setPreparing(false);
        panels.setBanner(glError);
        applyHello(h);
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
      orbit?.render();
      sat?.render();
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
