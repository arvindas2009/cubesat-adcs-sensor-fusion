import { describe, expect, it } from 'vitest';
import { DASH, fmtAngle, fmtClock, fmtFixed, fmtPercent, fmtSpeed, fmtVec3 } from '../src/adcs/format';

describe('fmtFixed', () => {
  it('formats, pads and signs', () => {
    expect(fmtFixed(3.14159, 2)).toBe('3.14');
    expect(fmtFixed(3.14159, 2, 8)).toBe('    3.14');
    expect(fmtFixed(3.14159, 2, 0, true)).toBe('+3.14');
    expect(fmtFixed(-3.14159, 2, 0, true)).toBe('-3.14');
  });
  it('prints null, undefined and non-finite as a dash, never as 0', () => {
    for (const x of [null, undefined, NaN, Infinity]) expect(fmtFixed(x, 2)).toBe(DASH);
  });
  it('drops the sign of negative zero', () => {
    expect(fmtFixed(-0.0001, 2)).toBe('0.00');
    expect(fmtFixed(-0.0001, 2, 0, true)).toBe('+0.00');
  });
});

describe('fmtAngle', () => {
  it('keeps about four significant digits', () => {
    expect(fmtAngle(0.12345)).toBe('0.1235');
    expect(fmtAngle(1.23456)).toBe('1.235');
    expect(fmtAngle(12.3456)).toBe('12.35');
    expect(fmtAngle(123.456)).toBe('123.5');
    expect(fmtAngle(null)).toBe(DASH);
  });
});

describe('fmtVec3', () => {
  it('aligns components with fixed width', () => {
    expect(fmtVec3([1, -22.5, 3], 1, 6)).toBe('  +1.0   -22.5    +3.0');
  });
  it('prints three dashes for null', () => {
    expect(fmtVec3(null, 1, 4)).toBe(`   ${DASH}     ${DASH}     ${DASH}`);
  });
});

describe('fmtClock', () => {
  it('formats HH:MM:SS', () => {
    expect(fmtClock(0)).toBe('00:00:00');
    expect(fmtClock(3661.9)).toBe('01:01:01');
    expect(fmtClock(36000)).toBe('10:00:00');
    expect(fmtClock(null)).toBe(DASH);
    expect(fmtClock(-1)).toBe(DASH);
  });
});

describe('fmtSpeed / fmtPercent', () => {
  it('formats speed multipliers', () => {
    expect(fmtSpeed(500)).toBe('500×');
    expect(fmtSpeed(1)).toBe('1×');
    expect(fmtSpeed(1.5)).toBe('1.5×');
    expect(fmtSpeed(12.34)).toBe('12.3×');
    expect(fmtSpeed(null)).toBe(DASH);
  });
  it('formats percentages', () => {
    expect(fmtPercent(0.352)).toBe('35 %');
    expect(fmtPercent(0.352, 1)).toBe('35.2 %');
    expect(fmtPercent(null)).toBe(DASH);
  });
});
