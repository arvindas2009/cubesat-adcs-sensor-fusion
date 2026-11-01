// Hash routes (docs/specs/ui_dashboard.md §2): #home (or no hash) = dashboard, #adcs = live ADCS
// simulator, #px4 = PX4 flight replay. Pure, so it is unit-tested.
export type Route = 'home' | 'adcs' | 'px4';

export const ROUTES: readonly Route[] = ['home', 'adcs', 'px4'];

export function routeFromHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, '').toLowerCase();
  return h === 'adcs' || h === 'px4' ? h : 'home';
}
