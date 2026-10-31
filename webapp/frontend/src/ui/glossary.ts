// Pure help texts (docs/specs/ui_console.md §5.4, §6.1). Plain language, no numbers that claim results.
// Run B holds `TermId` (the legend items refer to it) and `CHART_HELP`, the "How to read this" text of the
// expand dialog; the `GLOSSARY` of the readout help tips is added with the readout work.

export type TermId =
  | 'triad'
  | 'sigma3'
  | 'knowledge'
  | 'nadir'
  | 'estNadir'
  | 'dumping'
  | 'detumble'
  | 'pointing'
  | 'eclipse'
  | 'rate'
  | 'wheelUse'
  | 'dipole'
  | 'nis'
  | 'nisBand'
  | 'ekf2'
  | 'tilt'
  | 'excluded'
  | 'ekf2Reset'
  | 'logScale'
  | 'kick';

export interface ChartHelp {
  title: string;
  /** Each paragraph is at most 300 characters. */
  paragraphs: string[];
}

export type ChartId =
  | 'adcs.nadir'
  | 'adcs.knowledge'
  | 'adcs.rate'
  | 'adcs.wheels'
  | 'px4.attitude'
  | 'px4.position'
  | 'px4.nis'
  | 'px4.height';

/** Plot order of each page (the index of a plot card is the index into these lists). */
export const ADCS_CHART_IDS: readonly ChartId[] = ['adcs.nadir', 'adcs.knowledge', 'adcs.rate', 'adcs.wheels'];
export const PX4_CHART_IDS: readonly ChartId[] = ['px4.attitude', 'px4.position', 'px4.nis', 'px4.height'];

export const CHART_HELP: Record<ChartId, ChartHelp> = {
  'adcs.nadir': {
    title: 'Nadir pointing error',
    paragraphs: [
      'The angle between where the satellite points and straight down (nadir). "True" is the simulator\'s ground truth; "estimated" is what the onboard filter believes. Lower is better.',
      'The y-axis is logarithmic: each grid decade is ×10, so 10° → 1° takes as much height as 1° → 0.1°.',
    ],
  },
  'adcs.knowledge': {
    title: 'Knowledge error and 3σ',
    paragraphs: [
      'Knowledge error is the angle between the true and the estimated attitude. 3σ is three times the filter\'s own uncertainty.',
      'If the filter\'s uncertainty is honest, the solid line stays under the dashed one nearly all the time. Where it rises above, the filter is over-confident.',
    ],
  },
  'adcs.rate': {
    title: 'Body rate',
    paragraphs: [
      'The spin rate |ω| (true). Two dashed thresholds come from the session.',
      'Detumble threshold: POINTING is allowed once the measured rate has stayed below it for a hold time (plus other conditions).',
      'Safe rate: above it for a hold time, the controller falls back to DETUMBLE.',
      'The mode logic uses gyro minus estimated bias, so it can differ slightly from this true line.',
    ],
  },
  'adcs.wheels': {
    title: 'Wheel speeds',
    paragraphs: [
      'Reaction-wheel speed per body axis; the sign is the spin direction.',
      'Near ± max speed a wheel cannot absorb more momentum, and the magnetorquers then "dump" it.',
    ],
  },
  'px4.attitude': {
    title: 'Attitude difference',
    paragraphs: [
      'Tilt is the combined roll/pitch disagreement with EKF2; yaw is the signed heading difference (ours − EKF2).',
      'EKF2 is PX4\'s onboard filter: a reference, not truth. Magenta ticks mark EKF2 resets.',
    ],
  },
  'px4.position': {
    title: 'Position difference',
    paragraphs: [
      'The horizontal distance and the vertical difference (positive = ours lower) between our estimate and EKF2\'s.',
    ],
  },
  'px4.nis': {
    title: 'NIS',
    paragraphs: [
      'How surprising each sensor update was, given the filter\'s own uncertainty.',
      'For a consistent filter it follows a chi-square distribution whose degrees of freedom (dof) are the number of measured values, so about 95 % of points fall between that dof\'s dashed lines.',
      'Many points above: the filter is too confident, or the sensor is noisier than assumed. Many below: the filter is too cautious.',
      'Hollow points are rejected updates. The log scale is used because NIS spans decades.',
    ],
  },
  'px4.height': {
    title: 'Height above start',
    paragraphs: [
      'Ours (solid) vs EKF2 (dashed), relative to the first sample. It is not altitude above sea level.',
    ],
  },
};

/**
 * Plain-language help of the readout and legend terms (§6.1). Each text is at most 300 characters; it explains
 * what the quantity is and how to read it, and never claims a result. "Altitude" is not used for attitude.
 */
export const GLOSSARY: Record<TermId, { term: string; text: string }> = {
  triad: {
    term: 'TRIAD',
    text: 'TRIAD gives a first attitude fix from two direction measurements, the Sun and the magnetic field. The filter starts from it. "TRIAD start" is the simulation time when that happened.',
  },
  sigma3: {
    term: 'Filter 3σ bound',
    text: 'The filter\'s own uncertainty (one standard deviation, σ) multiplied by three, as an angle. If the filter is honest, the true attitude error stays inside this bound almost all the time.',
  },
  knowledge: {
    term: 'Knowledge error',
    text: 'The angle between the filter\'s estimated attitude and the true attitude from the simulator. It tells how well the satellite knows where it points. Lower is better.',
  },
  nadir: {
    term: 'Nadir pointing error',
    text: 'Nadir is straight down, toward the Earth\'s centre. This is the angle between where the satellite points and nadir, using the true attitude from the simulator. Lower is better.',
  },
  estNadir: {
    term: 'Estimated nadir error',
    text: 'The same angle, but computed from the filter\'s estimated attitude: what the satellite believes about its pointing. Compare it with the true nadir pointing error.',
  },
  dumping: {
    term: 'Momentum dumping',
    text: 'Reaction wheels store angular momentum. When a wheel nears its maximum speed, the magnetorquers push against Earth\'s magnetic field to "dump" that momentum. "On" means this is happening now.',
  },
  detumble: {
    term: 'DETUMBLE',
    text: 'DETUMBLE is the control mode that slows the spin the satellite starts with. POINTING is allowed once the measured rate has stayed below the detumble threshold for a hold time.',
  },
  pointing: {
    term: 'POINTING',
    text: 'POINTING is the control mode in which the wheels hold the satellite pointed at nadir. If the rate grows past the safe rate for a hold time, the controller falls back to DETUMBLE.',
  },
  eclipse: {
    term: 'Eclipse',
    text: 'Eclipse means the satellite is in Earth\'s shadow, so the Sun is not visible to it. "Sunlit" is the opposite. The eclipse share is the part of the whole run spent in shadow.',
  },
  rate: {
    term: 'Body rate |ω|',
    text: 'How fast the satellite is spinning: the length of the angular-velocity vector, in degrees per second. The value shown is the true one from the simulator; the estimated rate comes from the filter.',
  },
  wheelUse: {
    term: 'Wheel use',
    text: 'Wheel speed as a share of the wheel\'s maximum speed. Near 100 % a wheel cannot absorb more momentum, and the magnetorquers then dump it.',
  },
  dipole: {
    term: 'Magnetorquer dipole',
    text: 'The magnetic dipole commanded to the magnetorquer coils, per body axis, in ampere-square-metres. A larger value means a stronger push against Earth\'s magnetic field.',
  },
  nis: {
    term: 'NIS',
    text: 'Normalised innovation squared: how surprising a sensor update was, given the filter\'s own uncertainty. For a consistent filter it follows a chi-square distribution.',
  },
  nisBand: {
    term: '95 % band',
    text: 'The dashed lines enclose the central 95 % of the chi-square distribution for that sensor\'s number of measured values (dof). About 95 % of the points should fall between them.',
  },
  ekf2: {
    term: 'EKF2',
    text: 'EKF2 is PX4\'s own onboard attitude and position filter. It is shown as a reference to compare with, not as the true state.',
  },
  tilt: {
    term: 'Tilt difference',
    text: 'The combined roll and pitch disagreement between our estimate and EKF2, as one angle. Yaw (heading) is shown separately because it behaves differently.',
  },
  excluded: {
    term: 'Excluded from report statistics',
    text: 'The first seconds of a log are left out of the reported statistics, because the filters are still settling. The shaded band on the timeline shows that part.',
  },
  ekf2Reset: {
    term: 'EKF2 reset',
    text: 'A moment when EKF2 reset part of its own state. Right after one, the difference between EKF2 and our filter can be larger. The timeline and the plots mark each one.',
  },
  logScale: {
    term: 'Logarithmic axis',
    text: 'On a logarithmic axis each grid line is ten times the previous one, so very small and very large values both stay readable. Equal distances mean equal ratios, not equal differences.',
  },
  kick: {
    term: 'Kick',
    text: 'A kick is a sudden change in spin rate that you apply to one body axis, to test how the controller recovers. Each one is marked on the timeline.',
  },
};
