// Pure x-zoom bookkeeping of the large (dialog) charts (docs/specs/ui_console.md §5.2). The chart's
// x-scale `range` callback asks `range(default)`; a drag selection is fed to `select`.

export class XZoom {
  active = false;
  min = 0;
  max = 0;

  /**
   * Zoom to the selection [a, b] (either order). The result is clamped to the data span [dataLo, dataHi]
   * and never narrower than `minSpan` (it widens around the selection centre, shifting to stay in the data).
   */
  select(a: number, b: number, dataLo: number, dataHi: number, minSpan: number): void {
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(dataLo) || !Number.isFinite(dataHi)) return;
    const full = dataHi - dataLo;
    if (!(full > 0)) return;
    const span = Math.min(Math.max(minSpan, 0), full);
    let lo = Math.max(dataLo, Math.min(a, b));
    let hi = Math.min(dataHi, Math.max(a, b));
    if (hi - lo < span) {
      const mid = (lo + hi) / 2;
      lo = mid - span / 2;
      hi = mid + span / 2;
      if (lo < dataLo) {
        hi += dataLo - lo;
        lo = dataLo;
      }
      if (hi > dataHi) {
        lo -= hi - dataHi;
        hi = dataHi;
      }
      lo = Math.max(lo, dataLo);
    }
    this.min = lo;
    this.max = hi;
    this.active = true;
  }

  reset(): void {
    this.active = false;
  }

  /** The x range to use: the zoomed window when active, else the chart's default. */
  range(def: [number, number]): [number, number] {
    return this.active ? [this.min, this.max] : def;
  }
}
