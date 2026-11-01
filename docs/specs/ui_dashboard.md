# Spec: dashboard landing page, soft UI and detailed models

Amends `docs/specs/ui_console.md` (the "Flight-deck console"). Requested by the user: an editorial landing dashboard (large condensed type, generous white space, rounded image cards) that routes into the two instruments; light and dark mode; a more comfortable in-app UI with rounder corners; more detailed 3D models (hybrid: procedural CubeSat, optional open-licence glTF quadcopter).

## 0. Invariants

These must not change:
- the data flow;
- the WebSocket protocol;
- the server endpoints;
- the colour roles in `frontend.md` §2 and `px4_replay.md` §3;
- the permanent "EKF2 is a reference, not truth" caveat on the replay route;
- the body frames. CubeSat models are built in the SatView body frame. The quadcopter is built in FRD.

Every number shown on the dashboard must come from `PROGRESS.md`.

## 1. Look

The design is original; no third-party brand, name or logo is copied. The display face is Anton (OFL, `@fontsource/anton`), used for headlines only. UI text and numerals keep Barlow Semi Condensed and JetBrains Mono.

## 2. Routes

Routing is hash-only (`src/router.ts`, `routeFromHash`):

| Hash | Page |
|---|---|
| `#home`, empty or unknown | dashboard |
| `#adcs` | live ADCS simulator |
| `#px4` | PX4 flight replay |

The top bar has the brand (a link to `#home`), three pill tabs (Dashboard / ADCS / PX4 replay, roving tabindex, arrow keys) and the theme toggle.

The ADCS page (3D views, panels, sheet, render loop) is built on the **first visit to `#adcs`**. The PX4 page keeps its lazy creation. The dashboard opens no session.

## 3. Dashboard (`src/dashboard/`)

1. **Hero.** A rounded panel (`--r-lg`) holding a live Three.js scene (`HeroView`): the detailed CubeSat drifting over the Earth's limb (the Blue Marble texture already in `public/textures`). On top sit an eyebrow, the title "Attitude, determined." and a spec strip (3U · 4 kg, 400 km · 51.6°, MEKF · ES-EKF; from the Group 5 config).
   - It renders only while the route is shown, on screen and the document is visible.
   - Under `prefers-reduced-motion` it shows one still frame.
   - Without WebGL the gradient backdrop remains.
2. **Lede and kinetic headline.** Two columns of display type. The words rise in on scroll, are staggered, and are off under reduced motion.
3. **Two instrument cards.** Each card is a link to `#adcs` or `#px4`, with SVG art, a tag, a title, a description, a live status line and a pill CTA.
   - ADCS status: presets reachable → "Server online · N tunable parameters".
   - PX4 status: the number of logs from `GET /api/px4/logs`.
4. **Facts.** Four tiles from `facts.ts`. Each fact carries `source` (a `PROGRESS.md` group), and a test asserts that its value appears verbatim in `PROGRESS.md`.
5. **Band and footer.** An outlined marquee (off under reduced motion), then a footer with links.

## 4. Models (`src/models/`)

- **`cubesat.ts`: `buildCubeSat({ size, bodyMaterials?, wings? })`.**
  - The bus box is exactly `size`. SatView passes its labelled face materials.
  - Added hardware: 4 aluminium corner rails with feet; ribs at the U boundaries; 2 deployable wings (hinged on the −X edge, deployed along ±Y, cells facing +X, no longer than the bus); a GPS patch and UHF deployer with 4 tape-spring antennas on +Z; a nadir camera barrel on −Z; a star-tracker baffle on +X; coarse-sun-sensor windows on all faces; and a remove-before-flight tag.
  - It returns `halfExtent`, which SatView uses to keep the body-axis arrows clear of the hardware.
- **`quadcopter.ts`: `buildQuadcopter({ solid })`**, in FRD.
  - Main parts: carbon base plate and shell dome, X arms, motor bells and the red nose at +x.
  - The solid model also gets props, translucent prop-sweep discs, hubs, a battery, a GPS mast and puck, a front camera, and splayed landing legs with skids (+z = down).
  - The EKF2 ghost draws edges of the main parts only.
- **`gltf.ts`: an optional swap-in.**
  - `public/models/manifest.json` lists `.glb` files with `role`, `file`, `license` (CC0-1.0 or CC-BY-4.0 only), `source` and `author`. With no entry, the procedural model is used. The repository ships none, because the session's network policy blocked model hosts.
  - glTF (+Y up, +Z forward) maps to FRD by `GLTF_TO_FRD` (glTF +Z → +x, +Y → −z, +X → −y; det +1).
  - The model is scaled to the procedural span.

## 5. Soft UI tokens (`style.css`)

| Token | Value |
|---|---|
| `--r-xs` | 4 px |
| `--r-sm` | 8 px (controls, chips) |
| `--r-md` | 12 px (panels, banners, toasts) |
| `--r-lg` | 20 px (sheet top, dashboard cards) |
| `--r-pill` | 999 px |
| `--bar-h` | 56 px |
| `--gap` | 12 px |

- Controls are 34 px tall (small ones 26–28 px). Base text is 14 px (12 → 13, 13 → 14).
- Light-mode panels get a faint `--lift` shadow; dark mode keeps 1 px edges only.
- The square corner brackets are removed, because they fight rounded corners.

## 6. Acceptance

- `npm run build` and `npm test` are green. The tests (`tests/dashboard.test.ts`) cover:
  - routes;
  - facts against `PROGRESS.md`;
  - CubeSat bus size, wing placement and extents;
  - quadcopter nose, motors and legs in FRD;
  - the glTF→FRD rotation;
  - manifest licence filtering;
  - radius tokens, with no hard-coded 1–4 px radius left.
- Headless check at 1440×900 and 390×844 in both modes: no horizontal scroll, the hero renders, the cards route to `#adcs`/`#px4`, and no page errors.
