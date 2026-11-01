// Landing-page copy that states facts about the system. Every `evidence` string is copied from
// PROGRESS.md and tests/dashboard.test.ts checks it is still there verbatim, so the page cannot drift
// from the project log.

export interface LoopStage {
  id: 'sense' | 'estimate' | 'control' | 'actuate';
  title: string;
  summary: string;
  items: { name: string; detail: string }[];
  evidence: string[];
}

export const LOOP: readonly LoopStage[] = [
  {
    id: 'sense',
    title: 'Sense',
    summary: 'Noisy, biased measurements of the field, the Sun and the body rate, sampled at 1 Hz on the CubeSat.',
    items: [
      { name: 'Magnetometer', detail: 'PNI RM3100, 15 nT per axis' },
      { name: 'Rate gyro', detail: 'ADXRS453, with bias and angle random walk' },
      { name: 'Sun sensors', detail: 'six cosine cells, Tensor Tech CSS-10' },
      { name: 'Drone (PX4)', detail: 'IMU, GPS, barometer and magnetometer from real logs' },
    ],
    evidence: ['PNI RM3100', '15 nT/axis', 'ADXRS453', 'Tensor Tech CSS-10'],
  },
  {
    id: 'estimate',
    title: 'Estimate',
    summary: 'Error-state Kalman filters on scalar-first quaternions turn those measurements into an attitude you can trust, with honest uncertainty.',
    items: [
      { name: 'MEKF', detail: 'attitude error + gyro bias, TRIAD start' },
      { name: 'ES-EKF', detail: '15 states on the PX4 flight logs' },
      { name: 'Consistency', detail: 'NEES and NIS checked against χ² bounds' },
    ],
    evidence: ['MEKF', 'TRIAD', '15-state ES-EKF', 'NEES'],
  },
  {
    id: 'control',
    title: 'Control',
    summary: 'Kill the tumble, hand over to nadir pointing, and keep the wheels away from saturation.',
    items: [
      { name: 'B-dot', detail: 'detumble with the magnetorquers' },
      { name: 'Quaternion PD', detail: 'nadir pointing, ωn 0.1 rad/s, ζ 0.7' },
      { name: 'Momentum dumping', detail: 'cross-product law on the field' },
    ],
    evidence: ['B-dot', 'wn 0.1 rad/s, ζ 0.7', 'cross-product dumping'],
  },
  {
    id: 'actuate',
    title: 'Actuate',
    summary: 'Torque goes out through reaction wheels and magnetorquer coils, then the orbit and the disturbances push back.',
    items: [
      { name: 'Reaction wheels', detail: 'three, one per body axis' },
      { name: 'Magnetorquers', detail: '0.2 A·m² dipole' },
      { name: 'Disturbances', detail: 'aero, gravity gradient, residual dipole' },
    ],
    evidence: ['0.2 A·m²', 'gravity gradient', 'residual dipole'],
  },
];

export interface Phase {
  id: string;
  title: string;
  metric: string;
  detail: string;
  evidence: string[];
}

/** Mission phases on the draggable timeline, in flight order. */
export const PHASES: readonly Phase[] = [
  {
    id: 'release',
    title: 'Release',
    metric: '10 °/s tumble',
    detail: 'Deployed from the dispenser and tumbling. No attitude estimate yet, only the magnetometer.',
    evidence: ['10 °/s'],
  },
  {
    id: 'detumble',
    title: 'Detumble',
    metric: '1.22 orbits',
    detail: 'B-dot drains the spin with the magnetorquers. Median over the design Monte Carlo campaign.',
    evidence: ['1.22'],
  },
  {
    id: 'acquire',
    title: 'Acquire',
    metric: 'TRIAD + MEKF',
    detail: 'At the first valid Sun sample TRIAD starts the MEKF (with a 20° field–Sun angle guard), and the filter takes over.',
    evidence: ['TRIAD', 'first valid sun sample'],
  },
  {
    id: 'point',
    title: 'Point',
    metric: '0.99° RMS',
    detail: 'Nadir pointing on the estimate with realistic sensors. Through eclipse the knowledge error grows, then recovers once the Sun is back.',
    evidence: ['0.99°', 'Eclipse'],
  },
];

export const STACK: readonly string[] = ['Python', 'NumPy', 'SciPy', 'sgp4', 'ppigrf', 'pyulog', 'FastAPI', 'TypeScript', 'Three.js', 'uPlot', 'Vite'];
