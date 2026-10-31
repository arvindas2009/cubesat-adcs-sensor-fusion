// Colour roles (docs/specs/frontend.md §2, docs/specs/ui_theme.md §1).
// `COLORS` holds the roles as they look on the 3D "screens", which are dark plum in both modes, so these
// are the dark-mode values of src/theme/palette.ts. Everything that sits on a light or dark card (plots,
// timeline, DOM) must read the active palette instead: `dataColors()` / `chromeColors()` from '../theme'.
import { currentTheme, dataColors } from '../theme';
import { SCREEN } from '../theme/palette';

const D = dataColors('dark');

export const COLORS = {
  text: SCREEN.text,
  dim: SCREEN.textDim,
  axisX: D.axisX,
  axisY: D.axisY,
  axisZ: D.axisZ,
  estimate: D.estimate,
  sun: D.sun,
  sunOff: '#7E6C7C',
  field: D.field,
  nadir: D.nadir,
  detumble: D.detumble,
  pointing: D.pointing,
  kick: D.kick,
  danger: D.danger,
  wheelPos: '#2BD9B4',
  wheelNeg: '#FF8A3D',
  // PX4 replay: our filter vs the EKF2 reference.
  ours: D.ours,
  ekf2: D.ekf2,
} as const;

export const AXIS_COLORS: readonly string[] = [COLORS.axisX, COLORS.axisY, COLORS.axisZ];

/** DETUMBLE (mode 0) amber, POINTING (mode 1) green, in the active mode's data colours (for plots and timeline). */
export function modeColor(mode: number): string {
  const c = dataColors(currentTheme());
  return mode === 1 ? c.pointing : c.detumble;
}

export function modeName(mode: number): string {
  return mode === 1 ? 'POINTING' : 'DETUMBLE';
}
