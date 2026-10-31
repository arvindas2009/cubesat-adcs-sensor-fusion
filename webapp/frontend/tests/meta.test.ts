import { describe, expect, it } from 'vitest';
import { metaChips } from '../src/px4/meta';

describe('metaChips', () => {
  it('gives one chip per fact (no middle-dot string)', () => {
    const chips = metaChips({ role: 'hover', id: 'f4e81624-7ccc-4859-bb45-d02fa0246b77', durationS: 280.4, rateHz: 50, resets: 3 });
    expect(chips).toEqual(['hover', 'f4e81624', '280 s', '50 Hz grid', '3 EKF2 resets']);
    for (const c of chips) expect(c).not.toContain('·');
  });

  it('singular reset and missing rate', () => {
    const chips = metaChips({ role: 'x', id: 'abc', durationS: 12, rateHz: 0, resets: 1 });
    expect(chips[3]).toBe('— Hz grid');
    expect(chips[4]).toBe('1 EKF2 reset');
    expect(chips[1]).toBe('abc');
  });
});
