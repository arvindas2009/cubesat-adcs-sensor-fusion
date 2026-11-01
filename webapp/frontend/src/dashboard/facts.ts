// Headline numbers on the dashboard. Each value is copied from PROGRESS.md (simulation or PX4-log results
// recorded by the build groups); tests/dashboard.test.ts checks every `value` still appears there verbatim.
export interface Fact {
  value: string;
  unit: string;
  label: string;
  detail: string;
  /** PROGRESS.md section the number comes from. */
  source: string;
}

export const FACTS: readonly Fact[] = [
  {
    value: '0.99',
    unit: '°',
    label: 'Nadir pointing RMS',
    detail: 'Closed loop on the estimate, realistic sensors and the design filter (simulation, 5 seeds).',
    source: 'PROGRESS.md · Group 8',
  },
  {
    value: '99/100',
    unit: '',
    label: 'Monte Carlo runs meet every requirement',
    detail: 'Design campaign, 2–10 °/s initial tumble, ±10 % inertia error. Run 54 misses pointing at 5.19°.',
    source: 'PROGRESS.md · Group 8',
  },
  {
    value: '1.22',
    unit: 'orbits',
    label: 'Median detumble time',
    detail: 'B-dot on the magnetometer, design Monte Carlo campaign (100 runs).',
    source: 'PROGRESS.md · Group 8',
  },
  {
    value: '0.37',
    unit: '°',
    label: 'Tilt RMS vs PX4 EKF2',
    detail: '15-state ES-EKF on the aggressive real flight log. EKF2 is a reference, not truth.',
    source: 'PROGRESS.md · Group 4',
  },
];
