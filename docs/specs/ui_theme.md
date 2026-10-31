# Spec: browser simulator visual theme "Colour blocks" (light + dark)

Restyles `webapp/frontend` (both tabs). Behaviour, data, wire protocol and the colour **roles** of `docs/specs/frontend.md` §2 do not change. The page is still a visual aid only. The user asked for the vibe of a bold colour-block landing page (saturated sunflower / mint / plum / periwinkle / pink sections, big rounded shapes, pill buttons, clean grotesk type), in both light and dark mode. Do not copy any other site's logo, wordmark or copy.

## 1. Tokens

Palette (named, used everywhere through CSS variables; `theme.ts` mirrors them):

| Name | Hex | Role |
|---|---|---|
| Plum | `#3A2236` | 3D "screens" (both modes), dark-mode surfaces family, ink-ish accents |
| Sunflower | `#FFC531` | control-column block (Session / Log, Transport, Kick) |
| Mint | `#3DDC97` | plots band block; POINTING |
| Periwinkle | `#7C80EE` | readouts block |
| Bubblegum | `#EBA3D6` | header shapes, active-tab/selected accents |
| Ink | `#1C1220` | text and primary pill buttons (light mode) |
| Paper | `#F3F3F6` | light-mode page background (cool off-white, not cream) |

* **Light mode:** page `Paper`; each region is a solid block in its colour; cards inside a block are a pale tint of that block (≈ 18 % block over white, e.g. sunflower card `#FFF2CC`, mint card `#DDF8EB`, periwinkle card `#E7E8FD`); text `Ink`.
* **Dark mode:** page `#160E17`; blocks become deep tints of their colour (≈ 16 % block over `#160E17`); cards are `#251826` with a 3 px top edge in the block's saturated colour; text `#F2EAF0`, dim `#B9A8B6`.
* **3D viewports stay dark "screens" in both modes** (Plum family: scene clear colour `#24152A`, grid lines `#4A3350` / `#33223A`), like a dark code panel set into a light page. Earth, axes, Sun, B, estimate, ours/EKF2 colours keep their roles.
* Data colours (plots, markers) need contrast ≥ 3:1 on both the light plot card and the dark card; define a light and a dark value per role where needed (e.g. ours `#0E9F7A` light / `#3DDC97` dark; EKF2 `#D9822B` light / `#FFB347` dark; Sun `#C99400` / `#FFD23F`; field `#6B4FD8` / `#B39BFF`; axes x/y/z `#D63B3B` / `#1F9D4C` / `#2F6BE0` light, current values dark; kick/reset magenta `#D4198F` / `#FF5FC4`; DETUMBLE `#E08A00` / `#FFB020`; POINTING `#12A866` / `#3DDC97`).
* **Type:** one family, **Schibsted Grotesk** (variable, OFL) via `@fontsource-variable/schibsted-grotesk` (npm, bundled locally — no CDN). Fallback `system-ui`. Numbers in readouts, scrub time, axis ticks: same family with `font-variant-numeric: tabular-nums` (verify the font provides `tnum`; if it does not, use `@fontsource-variable/jetbrains-mono` only for numerals in readouts and axis ticks and say so). Scale (px): header title 30 / 800 / −0.02em; region title 18 / 700; card title 15 / 650; body 14 / 400; small 12.5. Sentence case everywhere: **no** uppercase-with-tracking labels (remove the current `text-transform: uppercase` panel headings).
* **Radius hierarchy:** region blocks 28 px; cards and 3D screens 18 px; controls, chips, badges, buttons: full pill. **No drop shadows**; separation comes from colour. Focus ring: 3 px outline in Periwinkle (light) / Bubblegum (dark), offset 2 px, on every interactive element.

## 2. Layout and components

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◖shapes◗  Attitude lab            [ ADCS | PX4 replay ]   [◐ Dark]       │  header (compact, ~72 px)
├───────────────────────────────────────────┬──────────────────────────────┤
│ ┌─────────── plum screen ───────────────┐ │ ███ sunflower block ████████ │
│ │  Orbit / Drone attitude               │ │  [card] Session / Log        │
│ └───────────────────────────────────────┘ │  [card] Transport            │
│ ┌─────────── plum screen ───────────────┐ │  [card] Kick (ADCS only)     │
│ │  Attitude / Path                      │ │ ███ periwinkle block ███████ │
│ └───────────────────────────────────────┘ │  [card] Readouts             │
├───────────────────────────────────────────┴──────────────────────────────┤
│ ███ mint block: timeline card + 2×2 plot cards ███████████████████████   │
└──────────────────────────────────────────────────────────────────────────┘
```

* Keep the existing grid proportions (views ≈ 62 %, side column ≈ 38 %, bottom band full width); left-aligned text inside cards; no horizontal page scroll down to 1280 px wide; below 1100 px the side column stacks under the views.
* At a 1536 × 730 CSS-px viewport (a 1080p laptop at 125 % scaling — the user's likely setup), the header, both 3D screens and the side column must be visible without scrolling; the plot band may scroll into view. At 1920 × 1080 everything including the plots fits.
* **Header (the one bold moment):** title "Attitude lab" (left), no subtitle; the tab switcher carries the instrument names. Three decorative rounded shapes in Sunflower, Bubblegum and Mint peek in from the header edges (absolutely positioned, `aria-hidden`, partly clipped, like floating pills/blocks); they never overlap the title, tabs or toggle. No animation except one 400 ms settle of the shapes on first load, disabled under `prefers-reduced-motion`.
* **Tabs:** a segmented pill (Ink track in light / `#2E1F2F` in dark) with the active segment filled Sunflower with Ink text (both modes); inactive segments Paper-coloured text.
* **Theme toggle:** a pill button "Dark" / "Light" (shows the mode it switches to), right of the tabs. Default = `prefers-color-scheme`; the choice is stored in `localStorage` under `theme` (all access in try/catch) and applied as `data-theme="light|dark"` on `<html>` before first paint (inline script in `index.html` is fine) to avoid a flash.
* **Buttons:** primary = Ink pill with Paper text (light), Sunflower pill with Ink text (dark); secondary = transparent pill with 1.5 px Ink (light) / `#F2EAF0` (dark) outline; disabled = 40 % opacity, `cursor: not-allowed`. Hover darkens/lightens fill by ~8 %; press scales 0.98.
* **Inputs/selects/sliders:** pill-shaped, white (light) / `#1D131E` (dark) fill, 1.5 px outline; range slider track in the block colour, thumb Ink (light) / Sunflower (dark).
* **Chips and badges:** mode badge DETUMBLE / POINTING as filled pills in the role colour with Ink text; Sunlit/Eclipse pill; log status chips (`cached`, `ready`, `not downloaded`) as pills. The PX4 caveat badge "EKF2 is a reference, not truth" becomes a Bubblegum pill with Ink text, always visible in the replay header row.
* Replace the PX4 meta string joined with middle dots (`hover · f4e81624 · 280 s · …`) by separate small pills: role, short id, duration, grid rate, reset count. Similarly the ADCS speed text `set … · actual …` becomes "set 50× / actual 49×" or two labelled values.
* Card titles: drop parenthetical jargon from the visible title and move it into a smaller second line or `title` tooltip where it is still needed (e.g. "Drone attitude" + small line "body FRD → NED; ours solid, EKF2 outline").
* Toasts and banners: pill/rounded cards, banner in Bubblegum (light) / deep pink tint (dark) with Ink/light text; errors keep the danger colour as a left bar.

## 3. Runtime theming

* `src/theme/` (or extend `adcs/theme.ts`): `currentTheme()`, `setTheme(t)`, `onThemeChange(cb)`, and `dataColors()` returning the role colours for the active mode (from §1). `COLORS` may remain for the 3D screens (theme-independent) but the plots, timeline canvases and DOM must read the active palette.
* On theme change: CSS updates by variables; **uPlot charts are rebuilt** with the new axis/grid/series colours (ADCS: refill from `History`; PX4: from the loaded replay) and the timeline canvases redrawn; 3D scenes need no change. No reload, no lost session state.
* Chart styling: axis text dim colour, grid lines a faint tint of the plot card's block colour, ticks hidden, 2 px series lines, legend in sentence case.

## 4. Tests and checks

* Existing vitest (107) stay green; add pure tests for the theme module: stored-choice parsing (bad/missing value → system default), `dataColors('light'|'dark')` returns every role, and a WCAG contrast check (≥ 3:1) of each data colour against its plot card colour in both modes.
* `npm run build` clean; `pytest webapp` unaffected.
* Browser check (main session): both tabs in both modes at 1536 × 730 and 1920 × 1080; theme toggle while an ADCS session plays (no lost frames/state) and with a replay loaded; keyboard focus visible; no console errors.
