// Colour roles (docs/specs/frontend.md §2). Mirrors the CSS variables in style.css; keep in sync.
export const COLORS = {
  bg: '#070b14',
  panel: '#0c1220',
  raised: '#131b2e',
  border: '#1f2a42',
  text: '#d5dcea',
  dim: '#7d8aa5',
  axisX: '#ff5a5a',
  axisY: '#4fd96b',
  axisZ: '#4a8dff',
  estimate: '#2fd6e6',
  sun: '#ffd23f',
  sunOff: '#6b7385',
  field: '#a77bff',
  nadir: '#e6ebf5',
  detumble: '#ffb020',
  pointing: '#35d07f',
  eclipse: '#02040a',
  kick: '#ff3fb4',
  danger: '#ff5c6c',
  wheelPos: '#2bd9b4',
  wheelNeg: '#ff8a3d',
} as const;

export const AXIS_COLORS: readonly string[] = [COLORS.axisX, COLORS.axisY, COLORS.axisZ];

/** DETUMBLE (mode 0) amber, POINTING (mode 1) green. */
export function modeColor(mode: number): string {
  return mode === 1 ? COLORS.pointing : COLORS.detumble;
}

export function modeName(mode: number): string {
  return mode === 1 ? 'POINTING' : 'DETUMBLE';
}
