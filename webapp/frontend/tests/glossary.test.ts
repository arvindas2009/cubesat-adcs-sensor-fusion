// Pure tests of the help texts (docs/specs/ui_console.md §8): the CHART_HELP entries of the expand dialog and
// the GLOSSARY of the readout / legend help tips.
import { describe, expect, it } from 'vitest';
import { ADCS_CHART_IDS, CHART_HELP, GLOSSARY, PX4_CHART_IDS, type TermId } from '../src/ui/glossary';

/** Every TermId of the spec (§6.1); the compiler checks that GLOSSARY has exactly these keys. */
const TERMS: TermId[] = [
  'triad',
  'sigma3',
  'knowledge',
  'nadir',
  'estNadir',
  'dumping',
  'detumble',
  'pointing',
  'eclipse',
  'rate',
  'wheelUse',
  'dipole',
  'nis',
  'nisBand',
  'ekf2',
  'tilt',
  'excluded',
  'ekf2Reset',
  'logScale',
  'kick',
];

describe('GLOSSARY', () => {
  it('has an entry for every TermId and nothing else', () => {
    expect(Object.keys(GLOSSARY).sort()).toEqual([...TERMS].sort());
  });

  it('every entry has a term and a plain-language text of at most 300 characters', () => {
    for (const id of TERMS) {
      const e = GLOSSARY[id];
      expect(e.term.trim().length, id).toBeGreaterThan(0);
      expect(e.text.trim().length, id).toBeGreaterThan(20);
      expect(e.text.length, `${id}: ${e.text}`).toBeLessThanOrEqual(300);
      expect(e.text.endsWith('.'), id).toBe(true);
    }
  });

  it('"altitude" appears only as "not altitude above sea level"', () => {
    for (const id of TERMS) {
      const all = `${GLOSSARY[id].term} ${GLOSSARY[id].text}`.replace(/not altitude above sea level/gi, '');
      expect(all.toLowerCase(), id).not.toContain('altitude');
    }
  });

  it('terms are unique, so two tips never share a heading', () => {
    expect(new Set(TERMS.map((id) => GLOSSARY[id].term)).size).toBe(TERMS.length);
  });

  it('the mode, shadow and reference terms say what they must', () => {
    expect(GLOSSARY.detumble.text).toContain('DETUMBLE');
    expect(GLOSSARY.pointing.text).toContain('nadir');
    expect(GLOSSARY.eclipse.text).toContain('shadow');
    expect(GLOSSARY.ekf2.text).toContain('not as the true state');
    expect(GLOSSARY.nisBand.text).toContain('95 %');
    expect(GLOSSARY.nadir.text).toContain('straight down');
  });
});

describe('CHART_HELP', () => {
  const ids = Object.keys(CHART_HELP) as (keyof typeof CHART_HELP)[];

  it('has one entry per plot of both pages (8)', () => {
    expect(ids).toHaveLength(8);
    expect([...ADCS_CHART_IDS, ...PX4_CHART_IDS].sort()).toEqual([...ids].sort());
    expect(ADCS_CHART_IDS).toHaveLength(4);
    expect(PX4_CHART_IDS).toHaveLength(4);
  });

  it('every entry has a title and non-empty paragraphs of at most 300 characters', () => {
    for (const id of ids) {
      const h = CHART_HELP[id];
      expect(h.title.trim().length, id).toBeGreaterThan(0);
      expect(h.paragraphs.length, id).toBeGreaterThan(0);
      for (const p of h.paragraphs) {
        expect(p.trim().length, id).toBeGreaterThan(0);
        expect(p.length, `${id}: ${p}`).toBeLessThanOrEqual(300);
      }
    }
  });

  it('"altitude" appears only as "not altitude above sea level"', () => {
    for (const id of ids) {
      for (const p of CHART_HELP[id].paragraphs) {
        const stripped = p.replace(/not altitude above sea level/gi, '');
        expect(stripped.toLowerCase(), `${id}: ${p}`).not.toContain('altitude');
      }
    }
    expect(CHART_HELP['px4.height'].paragraphs.join(' ')).toContain('not altitude above sea level');
  });

  it('the plots the spec describes say what the spec says', () => {
    const text = (id: keyof typeof CHART_HELP): string => CHART_HELP[id].paragraphs.join(' ');
    expect(text('adcs.nadir')).toContain('logarithmic');
    expect(text('adcs.knowledge')).toContain('over-confident');
    expect(text('adcs.rate')).toContain('Detumble threshold');
    expect(text('adcs.rate')).toContain('Safe rate');
    expect(text('adcs.wheels')).toContain('dump');
    expect(text('px4.attitude')).toContain('a reference, not truth');
    expect(text('px4.nis')).toContain('chi-square');
    expect(text('px4.nis')).toContain('Hollow points are rejected');
  });
});
