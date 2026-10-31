// Help tips (docs/specs/ui_console.md §6.1): a small "?" button that opens a plain-language explanation of a
// term. Uses the Popover API (popover="auto"): Esc and light dismiss come from the browser, the button is
// keyboard-operable, and hovering with a mouse opens the tip as a convenience. Placement is `placeTip`.
import { placeTip } from './a11y';
import { GLOSSARY, type TermId } from './glossary';

const used = new Map<string, number>();

/** A unique element id for the tip of a term (a term may have several help buttons on the page). */
function tipId(term: TermId): string {
  const n = (used.get(term) ?? 0) + 1;
  used.set(term, n);
  return n === 1 ? `tip-${term}` : `tip-${term}-${n}`;
}

function isOpen(tip: HTMLElement): boolean {
  try {
    return tip.matches(':popover-open');
  } catch {
    return false;
  }
}

const HOVER_OPEN_MS = 250;
const HOVER_CLOSE_MS = 160;

/**
 * The "?" button and its tip. Append both to the DOM next to the label they explain:
 * `const [btn, tip] = helpTip('triad'); label.append(btn, tip);`
 */
export function helpTip(term: TermId): [HTMLButtonElement, HTMLElement] {
  const entry = GLOSSARY[term];
  const id = tipId(term);

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'help';
  button.textContent = '?';
  button.setAttribute('aria-label', `Explain: ${entry.term}`);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('popovertarget', id);

  const tip = document.createElement('div');
  tip.id = id;
  tip.className = 'help-tip';
  tip.setAttribute('popover', 'auto');
  tip.setAttribute('role', 'note');
  const head = document.createElement('strong');
  head.className = 'help-term';
  head.textContent = entry.term;
  const text = document.createElement('p');
  text.textContent = entry.text;
  tip.append(head, text);

  let hoverOpened = false;
  let openTimer = 0;
  let closeTimer = 0;

  const place = (): void => {
    const a = button.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    const p = placeTip(a, { width: t.width, height: t.height }, { width: window.innerWidth, height: window.innerHeight });
    tip.style.left = `${p.left}px`;
    tip.style.top = `${p.top}px`;
  };

  tip.addEventListener('toggle', (e) => {
    const open = (e as ToggleEvent).newState === 'open';
    button.setAttribute('aria-expanded', String(open));
    if (open) place();
    else hoverOpened = false;
  });

  const scheduleClose = (): void => {
    window.clearTimeout(openTimer);
    window.clearTimeout(closeTimer);
    closeTimer = window.setTimeout(() => {
      if (hoverOpened && isOpen(tip) && !tip.matches(':hover') && !button.matches(':hover')) tip.hidePopover();
    }, HOVER_CLOSE_MS);
  };

  // Hover convenience (mouse only; touch and keyboard use the button).
  button.addEventListener('pointerenter', (e) => {
    if (e.pointerType !== 'mouse') return;
    window.clearTimeout(closeTimer);
    window.clearTimeout(openTimer);
    openTimer = window.setTimeout(() => {
      if (!isOpen(tip)) {
        hoverOpened = true;
        tip.showPopover();
      }
    }, HOVER_OPEN_MS);
  });
  button.addEventListener('pointerleave', scheduleClose);
  tip.addEventListener('pointerenter', () => window.clearTimeout(closeTimer));
  tip.addEventListener('pointerleave', scheduleClose);
  // A click on a tip that the hover opened keeps it open (the built-in toggle would close it again).
  button.addEventListener('click', (e) => {
    if (hoverOpened && isOpen(tip)) {
      e.preventDefault();
      hoverOpened = false;
    }
  });

  return [button, tip];
}

/** Append a help button and its tip to `parent`. */
export function appendHelp(parent: HTMLElement, term: TermId): void {
  const [button, tip] = helpTip(term);
  parent.append(button, tip);
}
