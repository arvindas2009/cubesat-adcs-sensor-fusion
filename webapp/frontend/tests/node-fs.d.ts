// Minimal typing for the one Node API the tests use (reading src/style.css); avoids adding @types/node.
declare module 'node:fs' {
  export function readFileSync(path: string | URL, encoding: 'utf8'): string;
}
