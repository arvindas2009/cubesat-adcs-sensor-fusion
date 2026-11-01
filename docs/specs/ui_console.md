# Spec: browser simulator UI "Flight-deck console" (light + dark)

This spec supersedes `docs/specs/ui_theme.md` ("Colour blocks"). It is the contract for `web-implementer`, written for three runs:
- **A**: tokens, shell, sheet.
- **B**: legends, charts, dialog.
- **C**: readouts, help, a11y, docs.

## 0. Scope and invariants

**What must not change:**
- the data flow;
- the WebSocket protocol;
- server endpoints;
- the colour *roles* of `frontend.md` §2 and `px4_replay.md` §3: body axes x/y/z, estimate, Sun, B field, nadir, DETUMBLE/POINTING, eclipse, kick, ours vs EKF2, NIS sensors and danger.

The page is a visual aid only. The "EKF2 is a reference, not truth" caveat stays permanently visible on the replay tab. No data shown today may be removed; it may be regrouped or relabelled.

**What the user asked for:**
- compact top bar;
- plots that rise over the 3D views as a sheet with parallax, instead of a page scroll;
- an expand button per plot;
- one consistent legend system, replacing the canvas square swatches;
- easier to understand and accessible without losing data;
- a less playful, more "angry/techy" look in the chosen "Flight-deck amber" colours.

**Confirmed wheel rule:** a plain scroll anywhere moves the sheet. The 3D camera zooms with Ctrl/⌘+scroll, trackpad pinch or middle-drag.

## 1. Tokens, fonts and look

### 1.1 UI tokens

These live in `palette.ts` as `UI: Record<Theme, UiTokens>`, which replaces `BRAND`. They are mirrored as CSS variables. All values in the mirrored CSS blocks are literal 6-digit hex.

| CSS var | Dark | Light | Use and constraint |
|---|---|---|---|
| `--page` | `#0D0F12` | `#E9ECEF` | page |
| `--bar` | `#0A0C0F` | `#F7F8FA` | top bar |
| `--panel` (= `PLOT_CARD`) | `#15191E` | `#F7F8FA` | panels, plot cards, sheet |
| `--inset` | `#0F1216` | `#FFFFFF` | inputs, timeline band |
| `--edge` | `#2A3038` | `#C5CBD3` | 1 px panel edges, rules |
| `--edge-input` | `#6B7480` | `#6B7480` | control borders (≥ 3:1 on panel) |
| `--bracket` | `#5A6472` | `#8A94A1` | panel corner brackets |
| `--text` | `#E6E9ED` | `#14181D` | ≥ 4.5:1 on page and panel |
| `--dim` | `#9AA4B2` | `#4A5563` | ≥ 4.5:1 on page and panel |
| `--action` | `#FF8A1F` | `#D9600B` | primary fills, active-tab underline, slider thumb. Light is never text |
| `--action-text` | `#FF8A1F` | `#A84800` | active tab label, link-like text (≥ 4.5:1) |
| `--on-action` | `#0D0F12` | `#0D0F12` | text on action fills (never white) |
| `--action-hover` | `#FFA04A` | `#E8742A` | hover fill (ink ≥ 4.5:1) |
| `--data` / `--neutral` | `#3FD0E0` / `#9AA4B2` | `#0B7C8A` / `#4A5563` | data accent |
| `--alarm` | `#FF3B3B` | `#C81E1E` | errors, danger (≥ 4.5:1 as text) |
| `--ok` / `--ok-text` | `#46D36B` / `#46D36B` | `#1E8A3E` / `#176F31` | glyph and fill / text |
| `--focus` | `#FF8A1F` | `#D9600B` | 2 px outline, 2 px offset (≥ 3:1 on page and panel) |
| `--fill-detumble` | `#FFB020` | `#FFB020` | ink-text status chips (same in both modes) |
| `--fill-pointing` | `#46D36B` | `#46D36B` | ink-text status chip |
| `--fill-sunlit` | `#FFD23F` | `#FFD23F` | ink-text status chip |
| `--fill-dump` | `#B39BFF` | `#B39BFF` | ink-text status chip |
| `--fill-eclipse` | `#05070A` | `#05070A` | status chip, with `#E6E9ED` text |

**3D screens** are the same in both modes. `SCREEN = { clear: '#080A0D', gridMajor: '#1F2730', gridMinor: '#141A21', text: '#E6E9ED', textDim: '#9AA4B2' }`. As before, `--screen` is defined only in the light/root block.

### 1.2 Data colours

The roles are unchanged. Every value must be ≥ 3:1 on the plot card of its mode.

| Role | Dark | Light |
|---|---|---|
| axisX / Y / Z | `#FF5A5A` `#4FD96B` `#4A8DFF` | `#D63B3B` `#1A8F43` `#2F6BE0` |
| estimate | `#3FD0E0` | `#0B7C8A` |
| sun | `#FFD23F` | `#A87B00` |
| field | `#B39BFF` | `#6B4FD8` |
| nadir | `#D7DCE3` | `#4A5563` |
| detumble | `#FFB020` | `#B86500` |
| pointing | `#46D36B` | `#1E8A3E` |
| kick | `#FF5FC4` | `#D4198F` |
| danger | `#FF3B3B` | `#C81E1E` |
| ours / ekf2 | `#3DDC97` / `#FFB347` | `#0B8F6E` / `#C0680A` |
| sigma3 | `#7FB9C2` | `#4F7F88` |
| nisGpsPos / Vel / Baro / Mag | `#4A8DFF` `#B39BFF` `#D7DCE3` `#FF5C8A` | `#2F6BE0` `#6B4FD8` `#4A5563` `#D0305F` |

- `NIS_FALLBACK` uses the new estimate colours.
- `adcs/theme.ts` `COLORS` equals the dark table, with `sunOff` `#6B7480`. It must stay ≥ 3:1 on `SCREEN.clear`.
- 3D label text-shadow is `#080A0D`.

### 1.3 Chart chrome

| Field | Dark | Light |
|---|---|---|
| card | `#15191E` | `#F7F8FA` |
| text | `#E6E9ED` | `#14181D` |
| dim | `#9AA4B2` | `#4A5563` |
| grid | `#232A32` | `#E3E6EA` |
| tick | `#3A434E` | `#C5CBD3` |
| band | `#0F1216` | `#FFFFFF` |
| bandEdge | `#2A3038` | `#C5CBD3` |
| eclipse | `#05070A` | `#3A4350` |
| eclipseEdge | `#5A6472` | `#14181D` |
| excluded | `#262D36` | `#DDE1E6` |
| cursor | `#FFFFFF` | `#0D0F12` |
| marker | `#E6E9ED` | `#14181D` |

### 1.4 Fonts

- **UI:** `@fontsource/barlow-semi-condensed`, weights 400/500/600, latin subset.
- **Numerals:** `@fontsource-variable/jetbrains-mono` for readout values, axis ticks, legend values, times and meta chips.
- Remove `@fontsource-variable/schibsted-grotesk`.
- `palette.ts` exports `FONT_UI = '"Barlow Semi Condensed", system-ui, sans-serif'` and `FONT_MONO = '"JetBrains Mono Variable", ui-monospace, Consolas, monospace'`. They are mirrored as `--font-ui` / `--font-mono`, and the canvas code uses them.
- If Barlow reads too narrow in the browser check, the agreed fallback is IBM Plex Sans Condensed + IBM Plex Mono.

### 1.5 Type scale and spacing

- **Text sizes:** minimum 12 px.

  | Element | Style |
  |---|---|
  | Brand | 13/700, uppercase, 0.12em |
  | Tabs | 13/600, uppercase, 0.06em |
  | Panel titles and group headings | 12/600, uppercase, 0.08em, `--dim` |
  | Body and controls | 13/400–500, line-height 1.35 |
  | Readout values | mono 14/500, units 12 `--dim` |
  | Legends | 12 |
  | Axis ticks | mono 12 (13 in the dialog) |
  | Dialog title / help | 18/600 / 14 at line-height 1.5 |

- Uppercase is only for short labels, never sentences.
- **Spacing:** 4 px grid (4/8/12/16/24); panel gap 8; panel padding 10×12; controls 28 px tall. *Amended by `ui_dashboard.md` §5: gap 12, controls 34 px.*
- **Radius and edges:** 4 px on panels, 3 px on controls and chips. No pills, no drop shadows, 1 px edges. *Amended by `ui_dashboard.md` §5: radius tokens 8 / 12 / 20 px, pill tabs, a faint shadow on light-mode panels, no corner brackets.*
- **Corner brackets:**
  - Every `.panel` and 3D viewport gets a `::before` (`position:absolute; inset:0; pointer-events:none`).
  - It draws eight `linear-gradient(var(--bracket) 0 0)` layers: a 10×1.5 and a 1.5×10 at each corner.
  - On `:focus-within` they turn `--action`.
  - Viewport brackets sit above the canvas.

## 2. Bar and tabs

The bar is 40 px tall on one line, with `--bar` background and a 1 px bottom edge.

**Contents:**
- brand "■ ATTITUDE LAB";
- the tabs;
- a status slot, flex 1, with `#bar-status-adcs` and `#bar-status-px4` and one of them hidden;
- a square 28 px theme toggle showing the mode it switches to.

The toggle logic, the `localStorage` key and the inline first-paint script in `index.html` are unchanged.

**Removals:** `.hdr-art`, the shapes and `@keyframes settle`.

**Tabs:**
- Uppercase text buttons. The active one has a 2 px `--action` underline and `--action-text` colour.
- Roving `tabindex`. ←/→/Home/End move and activate, through the pure `nextTab(key, i, n)` in `src/ui/a11y.ts`.
- The pages are `role=tabpanel` with `aria-labelledby`.

**ADCS status slot:**
- A session-state chip: no session / running / paused / complete.
- Then mono `t 1234 s`, `orbit 0.143` and `49×`.
- A small `StatusBar` is updated from `Panels.onFrame`, `onHello` and `onClosed`.
- The state chip in the Transport card head is removed.

**PX4 status slot:**
- The caveat text "EKF2 is a reference, not truth" is written in the main SHELL, so it shows before the lazily created `ReplayPage` exists. It is styled as a caution: 3 px `--action` left bar, `--panel` background, uppercase 12/600, `role=note`, `flex-shrink:0`.
- After it come the `metaChips()` pills, with `overflow:hidden`. A container query hides the rate chip below 520 px of slot width and the resets chip below 440 px.
- The full meta text goes in the slot's `title`.
- `ReplayPage` receives the slot element, and the `.px4-head` row is deleted.

**Toasts** sit at the top right, `top: 48px`.

## 3. Stage and rising sheet

### 3.1 DOM and CSS

```
#app  (display:grid; grid-template-rows: 40px minmax(0,1fr); height:100dvh; overflow:hidden)
 header.app-bar
 main#stage-root
   section.page#page-adcs[role=tabpanel]   (grid rows: auto | 1fr)
     #banner[role=alert]                   (in flow; shrinks the stage)
     .stage  (grid cols minmax(0,62fr) minmax(340px,38fr); gap 8; padding 8; overflow:hidden; position:relative)
       .stage-views#views   (col 1; rows 55fr/45fr; padding-bottom: calc(var(--peek) + 8px))  + .stage-dim[aria-hidden]
       aside.stage-side#side[data-own-scroll]  (col 2; overflow-y:auto; overscroll-behavior:contain)
       section.sheet#sheet-adcs[aria-label="Timeline and plots"]  (col 1, same row as views; z 2; full stage height;
                                                                   grid rows auto auto 1fr)
          .sheet-head (28 px: grip, "TIMELINE & PLOTS", hint "scroll ↓ · PgDn", toggle button
                       [aria-expanded][aria-controls=sheet-adcs-body] "▲ Show plots" / "▼ Hide plots")
          .panel#timeline (DOM legend row + canvas band)
          .sheet-body#sheet-adcs-body → #plots (2×2, rows 1fr) + footer.app-footer
```

- The PX4 page uses the same structure. `ReplayPage`'s inner `<main class="layout">` becomes a div.
- The sheet covers the **views column only**. Side controls and readouts stay usable while it is raised.
- `--peek: 104px` covers the head plus the full timeline.
- Remove `--top-h`, the views' `min-height` and the 4-across plot rule ≥ 1700 px.

### 3.2 Motion

The sheet always keeps its full layout size; only `transform` changes, so **charts never resize during motion**.

| | Lowered | Raised |
|---|---|---|
| Sheet translation | `travel = H − peek` | 0 |

Per animation frame, the JS writes only:
- `sheet.style.transform = translate3d(0,(1−p)·travel,0)`;
- `views.style.transform = translate3d(0,−p·0.3·travel,0)` (the parallax);
- `dim.style.opacity = 0.55·p`.

**Settled states:**

| | Raised | Lowered |
|---|---|---|
| `views.inert` | true | false |
| `sheetBody.inert` | false | true |
| `stage.dataset.sheet` | `raised` | `lowered` |
| 3D rendering | skipped (`update()` continues) | resumes; views get `invalidate()` |

- `OrbitView` and `SatView` gain an `invalidate()`.
- Before setting `inert`, move focus out of the subtree that becomes inert, to the sheet toggle.
- On settle, call `syncRect(true)` on every live chart, because uPlot caches its rect.

### 3.3 Pure model `src/ui/sheetModel.ts` (DOM-free, unit-tested)

```ts
export type SheetState = 'lowered' | 'raised';
export interface SheetConfig { travelPx: number; durationMs: number /*260*/; idleMs: number /*140*/;
  commitFrac: number /*0.08*/; discretePx: number /*40*/; reducedMotion: boolean }
export class SheetModel {
  constructor(cfg: SheetConfig, initial?: SheetState);
  wheel(deltaPx: number, now: number): void;            // + = scroll down = raise
  dragStart(now: number): void; dragMove(dyPx: number, now: number): void; dragEnd(now: number): void;
  toggle(now: number): void; set(s: SheetState, now: number): void;
  setTravel(px: number): void; setReducedMotion(b: boolean): void;
  tick(now: number): { p: number; state: SheetState; moving: boolean; settledNow: boolean };
}
export function normalizeWheel(deltaY: number, deltaMode: 0 | 1 | 2, pagePx: number): number; // line ×16, page ×pagePx
export type WheelRoute = 'sheet' | 'camera' | 'native';
export function routeWheel(c: { enabled: boolean; dialogOpen: boolean; overCanvas: boolean;
  zoomModifier: boolean; overOwnScroller: boolean }): WheelRoute;
export type KeyRoute = 'raise' | 'lower' | null;
export function routeKey(c: { key: string; enabled: boolean; dialogOpen: boolean;
  target: 'text' | 'select' | 'range' | 'slider' | 'other'; state: SheetState }): KeyRoute;
export function easeOutCubic(x: number): number;
```

**Continuous mode** (default):
- Wheel and drag move `p` 1:1 over `travelPx`, clamped to [0, 1].
- After `idleMs` without input it snaps. If `|p − p_gestureStart| ≥ commitFrac` it animates to the end in the gesture direction; otherwise it returns. The animation uses `easeOutCubic` over `durationMs` and ends exactly at 0 or 1.
- `settledNow` is true exactly once per settle.

**Reduced motion:**
- The first ≥ `discretePx` of a gesture flips the state instantly; the rest of that gesture is ignored until idle.
- Parallax factor 0 (CSS also sets `.stage-views{transform:none!important}`), and the dim switches without a fade.

### 3.4 Input (`src/ui/Sheet.ts`)

**Wheel.** One `window` wheel listener `{capture:true, passive:false}` uses `routeWheel`:

| Case | Route | Effect |
|---|---|---|
| Plain wheel anywhere on the stage, bar or sheet, including over 3D canvases and uPlot | `sheet` | `preventDefault()` + `stopPropagation()`, so OrbitControls never sees it |
| Ctrl/⌘ + wheel or pinch (`ctrlKey`) over a `.vp-canvas` | `camera` | untouched; OrbitControls zooms |
| Ctrl + wheel elsewhere | `native` | browser zoom |
| Over an overflowing `[data-own-scroll]` | `native` | its own scroll |
| Dialog open, or layout disabled | `native` | |

- A plain wheel over a canvas also shows a 1.5 s hint inside that viewport: "Scroll moves the plot sheet · Ctrl + scroll or pinch to zoom". It shows at most once per 10 s.

**Keys.**
- PageDown raises and PageUp lowers. They are ignored when the target is a text or number input, select, range, `role=slider`, or inside the dialog.
- Space stays the PX4 play toggle.

**Pointer.**
- Drag on `.sheet-head` (`touch-action:none`, pointer capture) drives `dragMove`/`dragEnd`.
- The toggle button calls `toggle()`.
- Touch elsewhere stays native.

`travelPx` comes from a ResizeObserver on `.stage`.

### 3.5 Fallback

Under `@media (max-width:1099px), (max-height:599px)` (tracked with `matchMedia`):
- `#app` returns to `height:auto` with normal page scroll.
- The stage stacks views (420/360 px), then the side column, then the sheet.
- No transforms, no `inert`, sheet head hidden, controller disabled.
- Plots one per row at 240 px.
- On a mode change, reset `p = 0` and clear `inert`.

## 4. One legend system

### 4.1 `src/ui/glyphs.ts` (pure)

```ts
export type Glyph =
  | { kind: 'line'; dash?: boolean }        // series, paths, 3σ, thresholds
  | { kind: 'band' }                        // eclipse, excluded, mode-strip segment
  | { kind: 'tick' }                        // vertical event line: switch, EKF2 reset, cursor
  | { kind: 'marker'; shape: 'diamond' | 'triangle' | 'circle' | 'square'; hollow?: boolean }
  | { kind: 'arrow' }                       // 3D arrows (axes, Sun, B, nadir)
  | { kind: 'box'; outline?: boolean };     // solid body / ghost or wireframe
export interface LegendItem { id: string; label: string; color: string; glyph: Glyph; help?: TermId }
export const GLYPH_W = 16, GLYPH_H = 10;
export function glyphSvg(g: Glyph, color: string): string;   // always viewBox "0 0 16 10"; dashed → stroke-dasharray
export const ADCS_EVENT_SHAPE = { tInit: 'diamond', fallback: 'triangle', kick: 'circle' } as const;
export function drawMarker(ctx: MinimalCtx, shape, x: number, y: number, size: number, color: string, hollow?: boolean): void;
export function adcsTimelineItems(d: DataColors, c: ChartChrome): LegendItem[]; // eclipse band, detumble band, pointing band,
                                                     // TRIAD init (diamond), switch (tick), fallback (triangle), kick (circle), cursor (tick)
export function px4TimelineItems(d: DataColors, c: ChartChrome): LegendItem[];  // excluded band, EKF2 reset tick, cursor tick
export function seriesGlyph(s: { dash?: number[]; points?: boolean; hollow?: boolean }): Glyph;
export const VIEW_LEGENDS: { sat; drone; path; orbit }; // each (colors) => LegendItem[]
```

- Band glyph: 16×6 rect. Tick: 1.5 px vertical line. Markers: 8 px.
- `MinimalCtx` is a structural type (beginPath/moveTo/lineTo/arc/closePath/fill/stroke/fillStyle/strokeStyle/lineWidth), so node can import the module.

### 4.2 `src/ui/legend.ts` (DOM)

```ts
renderLegend(items, { surface: 'panel' | 'screen'; live?: boolean; toggle?: (id: string, on: boolean) => void })
  → { el; setValue(id: string, text: string): void; setColors(items: LegendItem[]): void }
```

- Markup: `<ul class="legend">` of `<li>`, or `<button aria-pressed>` items when `toggle` is given.
- Same typography everywhere: 12 px, 16×10 glyph, 6 px gap, 14 px between items; values in mono.
- `surface:'screen'` adds a `rgba(8,10,13,.78)` backing with a 1 px `#2A3038` edge.

### 4.3 Migration

| Today | New |
|---|---|
| ADCS `Timeline.draw` canvas swatches (`adcs/Panels.ts` ~L556-572) | DOM legend row above the canvas; canvas `bandTop` 18 → 4; markers drawn with `drawMarker` |
| PX4 `TimelineBar.draw` swatches (`px4/ReplayPanels.ts` ~L388-400) | Same treatment; the in-band "excluded from report statistics" label stays |
| uPlot built-in legends | `legend.show:false` + bound DOM legend (`bindLegend`, §5). Items toggle series with `u.setSeries(i, {show})`, in cards and in the dialog |
| NIS legend + `hideLegend` hack + subtitle | One filled circle per sensor, "rejected" (hollow circle), "95 % band" (dashed line) |
| `SatView` legend | arrows: body x/y/z, nadir, Sun, B; dashed box outline: estimate |
| `DroneView` legend | box: ours; outlined box: EKF2 (reference); arrows: body x/y/z |
| `PathView` legend | line: ours; dashed line: EKF2 (reference); diamond: start |
| OrbitView (new) | Sun direction (arrow), first orbit (line), trail (line), satellite, dim in eclipse (square) |

- On a theme change, chart and timeline legends call `setColors`. The 3D legends use the theme-independent `COLORS`.
- Toggling timeline legend items is out of scope.

## 5. Charts, expand dialog and help texts

### 5.1 Refactor

| File | Contents |
|---|---|
| `src/ui/chart.ts` (DOM) | `axisSet(variant)` (fonts, sizes, `chromeColors()`), `bindLegend(u, legend, fmt)` (the `setLegend` hook writes the value at `u.cursor.idx`, or the latest sample when not hovering), `interface LiveChart { rebuild(); redraw(); syncRect(); destroy() }` |
| `src/ui/zoom.ts` (pure) | `class XZoom { active; min; max; select(a, b, dataLo, dataHi, minSpan); reset(); range(def: [number, number]): [number, number] }` |
| `src/adcs/charts.ts` | Moves `SeriesSpec`/`PlotSpec`/`PLOTS`/`tickLabel`/`LOG_FLOOR` and the threshold draw hook out of `Panels.ts` (~L676-874). `class AdcsChart implements LiveChart`: `constructor(host, spec, { variant: 'card' \| 'large', info: () => Info \| null })`, `setInfo`, `setData(cols)`, `resetZoom()`, `onZoom(cb)` |
| `src/px4/charts.ts` | Moves `PlotModel`/`MODELS`/`PLOT_HEADS`/`lineSeries`/`pointSeries`/`nisColor` and the reset/band draw hook out of `ReplayPanels.ts` (~L462-758). `class ReplayChart implements LiveChart`: `constructor(host, idx, onSeek, { variant })`, `setReplay(rep)`, `setTime(t)`, `resetZoom()`, `onZoom(cb)`. Model data is computed once per replay, series styles per theme |

`Panels` and `ReplayPanels` keep `live: Set<…>`, holding the 4 cards plus the open dialog chart. Every loop over plots uses it:
- `refreshCharts` and `clearCharts`;
- `onHello → setInfo`;
- `onTime → setTime`;
- `setReplay`;
- `onThemeChange → rebuild`;
- `whenFontsReady → redraw`;
- sheet settle → `syncRect`.

Public methods: `Panels.expand(i)` and `ReplayPanels.expand(i)`.

### 5.2 Card and large variants

| | Card | Large (in the dialog) |
|---|---|---|
| Cursor sync | sync key `'adcs'` / `'px4'` | none |
| Drag | none | `cursor.drag = {x:true, y:false, dist:6, setScale:true}` |
| Axis text | 12 px | 13 px, wider axes |
| x-scale `range` | ADCS `timeRange`; PX4 `[t0, t1]` | `(u, min, max) => zoom.active ? [min, max] : default` (uPlot passes zoom selections through `range`) |
| Zoom bookkeeping | n/a | `hooks.setSelect` → `zoom.select(...)`, then `u.setSelect({left:0, width:0, top:0, height:0}, false)` |
| Reset | n/a | double-click (`cursor.bind.dblclick`) or the dialog's Reset → `resetZoom()` |
| ADCS live data | `setData(cols, true)` | `setData(cols, !zoom.active)` |
| PX4 pointer | click/drag seeks (as now), `.plot-now` overlay | pointerdown remembers x; pointerup with < 4 px movement seeks; a drag zooms; `.plot-now` overlay kept |

### 5.3 `src/ui/ChartDialog.ts`

- One native `<dialog class="chart-dialog" aria-labelledby aria-describedby>`, opened with `showModal()`. This gives the top layer, an inert background and Esc via `cancel`.
- Size: 94vw × 90vh (max 1600 × 1000).
- **Header:** title, "Reset zoom" (disabled while not zoomed), Close ✕.
- **Body:** the legend row, then the chart host (flex 1). An aside "How to read this", 300 px wide, sits beside the chart (below it under 1000 px).
- **API:**

  ```ts
  open(o: { title: string; unit?: string; help: ChartHelp; returnFocus: HTMLElement;
            mount(chartHost: HTMLElement, legendHost: HTMLElement): { chart: LiveChart & { resetZoom(): void };
                                                                       onZoom(cb: (zoomed: boolean) => void): void };
            onClose(): void }): void
  close(): void
  ```

- **Focus:** on open it goes to the `h2` (`tabindex=-1`), so Space still reaches the PX4 play toggle (`spaceIsForControl` lets H2/DIV through).
- **Closing:** Esc, ✕, or a backdrop click (the target is the `<dialog>` element itself). Closing destroys the chart (uPlot `destroy`, observers disconnected, listeners removed), removes it from `live`, and returns focus to the expand button.
- **Forced close:** on a tab change (`selectTab`) and before a PX4 `setReplay`.
- **Expand button:** in each plot card head; square, ⤢ glyph, `aria-haspopup="dialog"`, `aria-label="Expand plot: <title>"`.

### 5.4 `CHART_HELP` (`src/ui/glossary.ts`, pure)

`{ title, paragraphs[] }` for each plot; each paragraph is at most 300 characters.

**ADCS**
1. **Nadir pointing error:**
   - The angle between where the satellite points and straight down (nadir). "True" is the simulator's ground truth; "estimated" is what the onboard filter believes. Lower is better.
   - The y-axis is logarithmic: each grid decade is ×10, so 10° → 1° takes as much height as 1° → 0.1°.
2. **Knowledge error and 3σ:**
   - Knowledge error is the angle between the true and the estimated attitude. 3σ is three times the filter's own uncertainty.
   - If the filter's uncertainty is honest, the solid line stays under the dashed one nearly all the time. Where it rises above, the filter is over-confident.
3. **Body rate:** the spin rate |ω| (true). Two dashed thresholds come from the session:
   - **Detumble threshold:** POINTING is allowed once the measured rate has stayed below it for a hold time (plus other conditions).
   - **Safe rate:** above it for a hold time, the controller falls back to DETUMBLE.
   - The mode logic uses gyro minus estimated bias, so it can differ slightly from this true line.
4. **Wheel speeds:**
   - Reaction-wheel speed per body axis; the sign is the spin direction.
   - Near ± max speed a wheel cannot absorb more momentum, and the magnetorquers then "dump" it.

**PX4**
1. **Attitude difference:**
   - Tilt is the combined roll/pitch disagreement with EKF2; yaw is the signed heading difference (ours − EKF2).
   - EKF2 is PX4's onboard filter: a reference, not truth. Magenta ticks mark EKF2 resets.
2. **Position difference:** the horizontal distance and the vertical difference (positive = ours lower) between our estimate and EKF2's.
3. **NIS:**
   - How surprising each sensor update was, given the filter's own uncertainty.
   - For a consistent filter it follows a chi-square distribution whose degrees of freedom (dof) are the number of measured values, so about 95 % of points fall between that dof's dashed lines.
   - Many points above: the filter is too confident, or the sensor is noisier than assumed. Many below: the filter is too cautious.
   - Hollow points are rejected updates. The log scale is used because NIS spans decades.
4. **Height above start:** ours (solid) vs EKF2 (dashed), relative to the first sample. It is not altitude above sea level.

## 6. Readouts and glossary

### 6.1 Help tips

**`src/ui/glossary.ts`** (pure):

```ts
export type TermId = 'triad'|'sigma3'|'knowledge'|'nadir'|'estNadir'|'dumping'|'detumble'|'pointing'|'eclipse'|'rate'|
  'wheelUse'|'dipole'|'nis'|'nisBand'|'ekf2'|'tilt'|'excluded'|'ekf2Reset'|'logScale'|'kick';
export const GLOSSARY: Record<TermId, { term: string; text: string }>;   // plain language, ≤ 300 chars, no "altitude" except genuine height
```

**`src/ui/help.ts`** (DOM): `helpTip(term)` returns two elements:
- the button `<button class="help" type="button" aria-label="Explain: TRIAD" popovertarget="tip-triad" aria-expanded>`, 18 px square with a 12 px "?";
- the tip `<div id popover="auto" role="note">`.

The Popover API gives Esc and light dismiss. Hover also opens the tip. Placement comes from the pure `placeTip(anchor, tip, viewport)` in `a11y.ts`: below the anchor, flipped or clamped at the edges.

### 6.2 ADCS readouts

This rewrites `Panels.ts` `Readouts`, keeping every current value, `fmt*` formatter and text cache. Each group is a `<section>` with an uppercase heading and a `<dl>` grid, and units sit dim after the value.

| Group | Contents |
|---|---|
| **Status** | chips: control mode DETUMBLE/POINTING (help: detumble/pointing), Sunlit/Eclipse (help), Momentum dumping on/off (help) |
| **Time** | Sim time `s` · Orbits completed · TRIAD start `s` (help) |
| **Pointing** | Nadir pointing error `°` (help) · Estimated nadir error `°` (help) |
| **Attitude knowledge** | Knowledge error `°` (help) · Filter 3σ bound `°` (help) |
| **Motion** | Body rate \|ω\| `°/s` (help) · Estimated rate ω̂ x, y, z `°/s` |
| **Actuators** | Wheel speeds x, y, z `rpm` · Wheel use `% of max` (help) · Magnetorquer dipole x, y, z `A·m²` (help) |
| **Orbit** | Eclipse share of run `%` (help) |

### 6.3 PX4 readouts

| Group | Contents |
|---|---|
| **Attitude and height** | Table with columns "ours" (line glyph) and "EKF2 (reference)" (dashed glyph + help `ekf2`); rows roll / pitch / yaw `°`, height above start `m` |
| **Difference, ours − EKF2** | tilt `°` (help) · yaw `°` (signed) · horizontal `m` · vertical `m` (+ = ours lower) |
| **Filter confidence** | 3σ attitude, largest axis `°` (help) |
| **Time** | Log time `s` · Elapsed `s` · Position `%` |

## 7. Accessibility

- **PX4 timeline:** `role="slider" tabindex=0` with `aria-valuemin`, `aria-valuemax`, `aria-valuenow` and `aria-valuetext` ("12.3 s of 280.0 s"). ←/→ move ±1 s (Shift ±10 s); Home/End jump to the ends. All moves go through the existing `onSeek`.
- **ADCS timeline:** `role="img"` with `aria-label` from the pure `timelineSummary(info, segments, events, t)`, updated at most once per second.
- **Plot cards:** `<figure aria-labelledby aria-describedby>` with a visually hidden summary from the pure `chartSummary(title, labels, latest)` (null → "—"), updated at most every 2 s.
- **3D views:** `.vp-body` is `role="img"` with a static `aria-label`; the canvas is `aria-hidden`.
- **Focus and text:** `:focus-visible` is 2 px `--focus` with a 2 px offset everywhere, including the viewport buttons. No text under 12 px.

## 8. Tests (vitest, node environment; new pure modules must not touch DOM globals at import time)

All existing tests stay green. `tests/theme.test.ts` is adapted, not deleted.

**`theme.test.ts`**
- **CSS sync:** every `UI` token is mirrored as a CSS var in both blocks, replacing the BRAND, sunflower, ink and card-mint checks. `FONT_UI` / `FONT_MONO` equal `--font-ui` / `--font-mono`.
- **Contrast ≥ 4.5:**
  - `--text`, `--dim`, `--action-text`, `--alarm` and `--ok-text` on `--page` and `--panel`, in both modes;
  - `--on-action` on `--action` and `--action-hover`;
  - ink on every `--fill-*` (eclipse fill with its light text).
- **Contrast ≥ 3:** `--focus` and `--edge-input` on page and panel; `COLORS` on `SCREEN.clear`; eclipse or eclipseEdge on band.
- **Kept checks:** data ≥ 3 on the plot card, chrome text/dim ≥ 4.5, grid < 1.6 and below every series.

**New test files**

| File | Covers |
|---|---|
| `sheet.test.ts` | follow + snap (commit and return); inertia clamped; reduced-motion discrete toggle (one per gesture); drag; `toggle`/`set`; `tick` reaches exactly 0 or 1 with `settledNow` once; `normalizeWheel` modes; the full `routeWheel` truth table; `routeKey` |
| `legend.test.ts` | `glyphSvg` viewBox and dash; band/tick/marker geometry; ADCS timeline items cover all eight drawn kinds and use `ADCS_EVENT_SHAPE`; `seriesGlyph` mapping; legend colours are valid hex |
| `zoom.test.ts` | `select` clamps to the data and keeps the minimum span; `reset`; `range` returns the default when inactive |
| `glossary.test.ts` | every `TermId` and all 8 `CHART_HELP` entries are non-empty; paragraphs ≤ 300 chars; no "altitude" except "not altitude above sea level" |
| `a11y.test.ts` | `nextTab` (arrows wrap, Home/End); `timelineSummary` and `chartSummary` (null → "—"); `placeTip` flip and clamp |

**Also:** `npm run build` (`tsc --noEmit` + vite) clean; `pytest webapp` unaffected.

## 9. Browser checks (main session)

**Matrix:** 1536×730, 1920×1080, 1099×800 (fallback) and a 590 px tall window (fallback); dark and light; both tabs; an ADCS session playing and a PX4 replay loaded.

**Bar**
- 40 px, one line.
- The caveat is visible at every width ≥ 1100 px.
- The status slot ticks live.

**Sheet**
- One wheel notch over a 3D view raises the sheet, with visible parallax and dim. Scrolling up lowers it.
- The peek shows the head and the full timeline.
- The handle, drag and PgDn/PgUp work. PgDn/PgUp are ignored on a focused select or slider.
- Ctrl+wheel and pinch zoom the camera, and the hint appears on a plain wheel over a view.
- An overflowing side column scrolls itself.
- No chart resize during motion.
- Cursor sync and the PX4 seek / now-line stay aligned after raising and lowering.
- Reduced-motion emulation gives an instant switch.

**Dialog**
- Opens with focus on the title. Esc, ✕ and the backdrop close it, and focus returns to the expand button. Tab stays inside.
- ADCS keeps streaming, and a drag-zoom holds while data grows. Double-click or Reset restores the range.
- PX4: Space plays and pauses, the now-line moves, a click seeks and a drag zooms.
- After 10 open/close cycles there are exactly 4 `.uplot` elements.

**Everything else**
- **Legends:** glyphs match the drawn marks everywhere and have the same size.
- **Readouts:** grouped, with units. Every "?" opens by keyboard and closes with Esc.
- **Tabs:** arrow keys switch tabs; the `#px4` deep link works.
- **Theme:** toggling during playback loses no state.
- **Console:** no errors.

## 10. npm changes

- Add `@fontsource/barlow-semi-condensed` and `@fontsource-variable/jetbrains-mono`.
- Remove `@fontsource-variable/schibsted-grotesk`.
- Nothing else.
