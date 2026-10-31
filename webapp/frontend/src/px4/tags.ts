// Pure layout of the small "dof N" tags at the right edge of the NIS plot: tags whose lines are closer than
// `minGap` px on the log axis are merged into one ("dof 1, 3") so they never overlap.

export interface BandTag {
  /** y of the band's upper line, px from the top of the plot area. */
  y: number;
  dof: number;
}

export interface PlacedTag {
  y: number;
  text: string;
}

export function mergeBandTags(tags: BandTag[], minGap = 12): PlacedTag[] {
  const sorted = tags.filter((t) => Number.isFinite(t.y)).sort((a, b) => a.y - b.y || a.dof - b.dof);
  const groups: { y: number; last: number; dofs: number[] }[] = [];
  for (const t of sorted) {
    const g = groups[groups.length - 1];
    if (g && t.y - g.last < minGap) {
      g.dofs.push(t.dof);
      g.last = t.y;
    } else groups.push({ y: t.y, last: t.y, dofs: [t.dof] });
  }
  return groups.map((g) => ({ y: g.y, text: `dof ${[...new Set(g.dofs)].sort((a, b) => a - b).join(', ')}` }));
}
