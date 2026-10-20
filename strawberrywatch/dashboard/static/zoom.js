// The zoom buttons' arithmetic, kept out of app.js so it can be tested without
// a browser.
// Adapted from uPlot demos/zoom-wheel.html (MIT). A view is {min, max} on the
// x scale, always kept inside the data.

// Stop at two readings across, the least worth showing
export const MIN_SPAN = 2 * 900;

export const RANGES = [
  ['Last 6 hours', 6],
  ['Last 24 hours', 24],
  ['Last 2 days', 48],
];

// Slides a view back inside the data, keeping its width when it fits
export function clamp(min, max, full) {
  const span = max - min;
  if (span >= full.max - full.min) return { min: full.min, max: full.max };
  if (min < full.min) return { min: full.min, max: full.min + span };
  if (max > full.max) return { min: full.max - span, max: full.max };
  return { min, max };
}

// The last so many hours, ending at the newest reading
export function ending(hours, full) {
  return clamp(full.max - hours * 3600, full.max, full);
}

// Half or twice the span, around the middle of what's showing
export function zoom(view, factor, full) {
  const middle = (view.min + view.max) / 2;
  const half = Math.max(MIN_SPAN, (view.max - view.min) * factor) / 2;
  return clamp(middle - half, middle + half, full);
}

// Zoomed means any view short of the full data, give or take a second
export function zoomed(view, full) {
  return Math.abs(view.min - full.min) > 1 || Math.abs(view.max - full.max) > 1;
}
