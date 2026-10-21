// Menu, dark mode, charts, the Settings map and the settings draft. Chart data
// comes from <script type="application/json"> blocks filled with tojson, and
// this file only writes textContent, not HTML.

import { RANGES, ending, zoom, zoomed } from './zoom.js';

// The menu button and dark mode, as the website's static/js/site.js does them

const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const open = navLinks.classList.toggle('show');
    navToggle.setAttribute('aria-expanded', String(open));
  });
}

const darkButton = document.getElementById('toggleDarkMode');
const darkIcon = document.getElementById('dark-mode-icon');
const darkLabel = document.getElementById('mode-label');

function showMode(dark) {
  if (darkIcon) darkIcon.textContent = dark ? '☀️' : '🌙';
  if (darkLabel) darkLabel.textContent = dark ? 'Light Mode' : 'Dark Mode';
  if (darkButton) darkButton.setAttribute('aria-pressed', String(dark));
}

showMode(document.body.classList.contains('dark-mode'));
if (darkButton) {
  darkButton.addEventListener('click', () => {
    const dark = !document.body.classList.contains('dark-mode');
    document.body.classList.toggle('dark-mode', dark);
    document.documentElement.classList.toggle('dark', dark);
    try {
      localStorage.setItem('darkMode', String(dark));
    } catch (e) {
      // Storage can be blocked. Then the mode only lasts for this page
    }
    showMode(dark);
    redrawAll();
  });
}

// The print page is always light, since it's going on paper
const printing = document.body.dataset.swPrint === 'yes';
if (printing) {
  document.body.classList.remove('dark-mode');
  document.documentElement.classList.remove('dark');
}

// Charts. Colours are read from the stylesheet's tokens at draw time, so
// switching mode only needs a redraw.

function colours() {
  const style = getComputedStyle(document.body);
  const get = (name) => style.getPropertyValue(name).trim();
  return {
    ink: get('--chart-ink'),
    muted: get('--chart-muted'),
    grid: get('--chart-grid'),
    rain: get('--chart-rain'),
    rainFill: get('--chart-rain-fill'),
    band: get('--chart-band'),
    flag: get('--flag'),
    font: '13px Sen, system-ui, sans-serif',
  };
}

function dataFor(el) {
  return JSON.parse(document.getElementById(el.dataset.swSource).textContent);
}

function axis(c, label) {
  return { stroke: c.ink, font: c.font, labelFont: c.font, grid: { stroke: c.grid, width: 1 }, ticks: { stroke: c.grid }, label };
}

// Every chart on the page: [element, plots, how to draw it again]
const drawn = [];

const entryOf = (el) => drawn.find((e) => e.el === el);

// Pacific time, like the rest of the pages, whatever the viewer's computer says
const PACIFIC = (ts) => uPlot.tzDate(new Date(ts * 1e3), 'America/Los_Angeles');

// Drag zooms time only. The y axis refits to what's visible, and a spike stays
// on the chart. Double-click still resets.
const DRAG = { x: true, y: false, setScale: true };

// The score and rain plots under one chart move together, whichever was zoomed
function followScale(el) {
  return (u, key) => {
    const entry = entryOf(el);
    if (key !== 'x' || !entry || !entry.plots.length || entry.following) return;
    entry.following = true;
    const view = { min: u.scales.x.min, max: u.scales.x.max };
    for (const other of entry.plots) if (other !== u) other.setScale('x', view);
    entry.following = false;
    entry.view = view;
    updateReset(el);
  };
}

// Based on uPlot's high-low-bands and area-fill demos. uPlot's legend is off,
// scoreKey below does that job.
function scoreChart(el, d) {
  const c = colours();
  const width = el.clientWidth || 600;
  const sync = uPlot.sync('sw-' + el.dataset.swSource);
  const flagAt = d.x.indexOf(d.flag);
  const plots = [];
  plots.push(
    new uPlot(
      {
        width,
        height: 240,
        legend: { show: false },
        cursor: { sync: { key: sync.key }, drag: DRAG },
        tzDate: PACIFIC,
        scales: { x: { time: true } },
        series: [
          {},
          { stroke: c.ink, width: 2 },
          { stroke: c.muted, width: 1.5 },
          { stroke: c.muted, width: 1, dash: [4, 4] },
        ],
        bands: [{ series: [2, 3], fill: c.band }],
        axes: [axis(c), axis(c, 'Score')],
        hooks: {
          setScale: [followScale(el)],
          setCursor: [(u) => readOut(el, d, u.cursor.idx)],
          draw: [
            (u) => {
              // The dot sits on the score and goes when the score is hidden
              if (flagAt < 0 || !u.series[1].show) return;
              const cx = u.valToPos(d.x[flagAt], 'x', true);
              const cy = u.valToPos(d.score[flagAt], 'y', true);
              const ctx = u.ctx;
              ctx.save();
              ctx.beginPath();
              ctx.arc(cx, cy, 6 * devicePixelRatio, 0, 2 * Math.PI);
              ctx.fillStyle = c.flag;
              ctx.fill();
              ctx.restore();
            },
          ],
        },
      },
      [d.x, d.score, d.line, d.dry],
      el,
    ),
  );
  if (d.rain.some((v) => v)) {
    plots.push(
      new uPlot(
        {
          width,
          height: 110,
          cursor: { sync: { key: sync.key }, drag: DRAG },
          tzDate: PACIFIC,
          scales: { x: { time: true } },
          legend: { show: false },
          hooks: { setScale: [followScale(el)], setCursor: [(u) => readOut(el, d, u.cursor.idx)] },
          series: [{}, { stroke: c.rain, fill: c.rainFill, width: 1 }],
          axes: [axis(c), axis(c, 'Rain, mm')],
        },
        [d.x, d.rain.map((v) => v ?? 0)],
        el,
      ),
    );
  }
  for (const u of plots) u.over.addEventListener('mouseleave', () => readOut(el, d, null));
  scoreKey(el, d, plots);
  return plots;
}

// The score chart's key, with a real tick box per line. It sits after the
// chart because the chart is role="img", which hides its contents from
// screen readers.
const SCORE_KEY = [
  {
    id: 'score',
    name: 'Score.',
    about: (d) => `${d.about || 'How unusual the reading is.'} Higher is more unusual.`,
  },
  {
    id: 'line',
    name: 'Alert line.',
    about: () => "An alert starts when the score goes above this line. It's raised during and after rain.",
  },
  {
    id: 'dry',
    name: 'Alert line without rain.',
    about: () => 'Dashed. The shaded area between the two lines is the rain allowance.',
  },
  { id: 'rain', name: 'Rain.', about: () => 'Rainfall in mm, in the lower chart.' },
];
// Where each key entry lives: [plot, series], or the full rain plot
const SERIES_OF = { score: [0, 1], line: [0, 2], dry: [0, 3], rain: [1, null] };
const POINT_AT = 'Hover over the chart to see exact values.';
const STAMP = uPlot.fmtDate('{D} {MMM}, {h}:{mm} {aa}');

function keyOf(el) {
  const next = el.nextElementSibling;
  return next && next.classList.contains('sw-key') ? next : null;
}

function showLine(plots, id, shown) {
  const [plot, series] = SERIES_OF[id];
  const u = plots[plot];
  if (!u) return;
  if (series === null) u.root.style.display = shown ? '' : 'none';
  else u.setSeries(series, { show: shown });
}

function keyEntry(tag, swatch, name, about) {
  const entry = document.createElement(tag);
  entry.className = 'sw-key__item';
  const mark = document.createElement('span');
  mark.className = `sw-key__swatch sw-key__swatch--${swatch}`;
  const text = document.createElement('span');
  const strong = document.createElement('strong');
  strong.textContent = name;
  text.append(strong, ' ' + about);
  entry.append(mark, text);
  return entry;
}

function scoreKey(el, d, plots) {
  const old = keyOf(el);
  if (old) old.remove();
  // Kept on the element. A dark mode redraw keeps hidden lines hidden
  const hidden = new Set((el.dataset.swHidden || '').split(',').filter(Boolean));
  const key = document.createElement('div');
  key.className = 'sw-key';
  if (!printing) {
    const readout = document.createElement('p');
    readout.className = 'sw-key__readout';
    readout.textContent = POINT_AT;
    const hint = document.createElement('p');
    hint.className = 'sw-key__hint';
    hint.textContent = 'Untick a line to hide it.';
    key.append(readout, hint);
  }
  const list = document.createElement('ul');
  list.className = 'sw-key__list';
  for (const line of SCORE_KEY) {
    if (!plots[SERIES_OF[line.id][0]]) continue;
    const li = document.createElement('li');
    if (printing) {
      li.append(keyEntry('span', line.id, line.name, line.about(d)));
    } else {
      const label = keyEntry('label', line.id, line.name, line.about(d));
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = !hidden.has(line.id);
      box.addEventListener('change', () => {
        showLine(plots, line.id, box.checked);
        if (box.checked) hidden.delete(line.id);
        else hidden.add(line.id);
        el.dataset.swHidden = [...hidden].join(',');
      });
      label.prepend(box);
      li.append(label);
      if (!box.checked) showLine(plots, line.id, false);
    }
    list.append(li);
  }
  if (d.x.indexOf(d.flag) >= 0) {
    const li = document.createElement('li');
    // In line with the samples above, which sit after a tick box
    if (!printing) li.className = 'sw-key__plain';
    li.append(keyEntry('span', 'dot', 'Red dot.', 'First reading above the alert line.'));
    list.append(li);
  }
  key.append(list);
  el.after(key);
}

// Reads out every value under the pointer
function readOut(el, d, i) {
  const key = keyOf(el);
  const readout = key && key.querySelector('.sw-key__readout');
  if (!readout) return;
  if (i == null || d.x[i] == null) {
    readout.textContent = POINT_AT;
    return;
  }
  const num = (v) => (v == null ? 'none' : v.toLocaleString('en-US', { maximumFractionDigits: 2 }));
  const parts = [
    STAMP(PACIFIC(d.x[i])),
    `Score ${num(d.score[i])}`,
    `Alert line ${num(d.line[i])}`,
    `Without rain ${num(d.dry[i])}`,
  ];
  if (d.rain[i] != null) parts.push(`Rain ${num(d.rain[i])} mm`);
  readout.textContent = parts.join(' · ');
}

// The reading on Alerts, with the unusual ones as red dots. No legend, readout
// or zoom. The dots are what matters.
function readingChart(el, d) {
  const c = colours();
  return [
    new uPlot(
      {
        width: el.clientWidth || 600,
        height: 220,
        legend: { show: false },
        cursor: { show: false },
        select: { show: false },
        tzDate: PACIFIC,
        scales: { x: { time: true } },
        series: [
          {},
          { stroke: c.ink, width: 1.5, points: { show: false } },
          { stroke: c.flag, paths: () => null, points: { show: true, size: 9, stroke: c.flag, fill: c.flag } },
        ],
        axes: [axis(c), axis(c, d.label)],
      },
      [d.x, d.y, d.unusual],
      el,
    ),
  ];
}

// Based on uPlot's sparklines demo
function sparkChart(el, d) {
  const c = colours();
  return [
    new uPlot(
      {
        width: 180,
        height: 34,
        pxAlign: false,
        legend: { show: false },
        cursor: { show: false },
        select: { show: false },
        tzDate: PACIFIC,
        scales: { x: { time: true, range: [d.start, d.end] } },
        axes: [{ show: false }, { show: false }],
        series: [{}, { stroke: c.ink, width: 1.5, points: { show: false } }],
      },
      [d.x, d.y],
      el,
    ),
  ];
}

function rainChart(el, d) {
  const c = colours();
  const top = Math.max(d.peak, 1.1);
  const storm = d.hours.map((h) => (h < d.storm_hours ? top + 0.1 : null));
  return [
    new uPlot(
      {
        width: el.clientWidth || 480,
        height: 200,
        legend: { show: false },
        scales: { x: { time: false }, y: { range: [0.9, top + 0.15] } },
        series: [
          { label: 'Hours' },
          { label: 'Raining', stroke: 'transparent', fill: c.rainFill, points: { show: false } },
          { label: 'The line', stroke: c.ink, width: 2, points: { show: false } },
        ],
        axes: [axis(c, 'Hours since rain started'), axis(c, 'Multiple of dry level')],
      },
      [d.hours, storm, d.multiple],
      el,
    ),
  ];
}

const draw = { score: scoreChart, reading: readingChart, spark: sparkChart, rain: rainChart };

function drawChart(el, data) {
  const entry = entryOf(el);
  if (entry) {
    for (const plot of entry.plots) plot.destroy();
    // Empty while the new plots are made. Their first scale doesn't reach the old ones
    entry.plots = [];
    entry.data = data;
    entry.plots = draw[el.dataset.swChart](el, data);
    // Switching to dark mode redraws the chart, and shouldn't lose a zoom
    if (entry.view && zoomed(entry.view, fullOf(el))) show(el, entry.view);
  } else {
    drawn.push({ el, data, plots: draw[el.dataset.swChart](el, data) });
  }
  zoomControls(el);
}

// Zoom buttons above every full-size chart. Sparklines don't get them.

function fullOf(el) {
  const xs = entryOf(el).plots[0].data[0];
  return { min: xs[0], max: xs[xs.length - 1] };
}

function viewOf(el) {
  const u = entryOf(el).plots[0];
  return { min: u.scales.x.min, max: u.scales.x.max };
}

function show(el, view) {
  const entry = entryOf(el);
  entry.view = view;
  for (const u of entry.plots) u.setScale('x', view);
  updateReset(el);
}

function updateReset(el) {
  const row = el.previousElementSibling;
  const reset = row && row.classList.contains('sw-zoom') ? row.querySelector('[data-sw-reset]') : null;
  if (reset) reset.disabled = !zoomed(viewOf(el), fullOf(el));
}

function zoomButton(text, act) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'sw-button sw-button--secondary sw-button--small';
  button.textContent = text;
  button.addEventListener('click', act);
  return button;
}

function zoomControls(el) {
  if (printing || el.dataset.swZoom !== 'time') return;
  const before = el.previousElementSibling;
  if (before && before.classList.contains('sw-zoom')) return;
  const row = document.createElement('div');
  row.className = 'sw-zoom';
  row.setAttribute('role', 'group');
  row.setAttribute('aria-label', 'How much of the chart to show');
  const ranges = document.createElement('div');
  ranges.className = 'sw-zoom__group';
  for (const [text, hours] of RANGES) {
    ranges.append(zoomButton(text, () => show(el, ending(hours, fullOf(el)))));
  }
  const zooms = document.createElement('div');
  zooms.className = 'sw-zoom__group';
  zooms.append(
    zoomButton('Zoom in', () => show(el, zoom(viewOf(el), 0.5, fullOf(el)))),
    zoomButton('Zoom out', () => show(el, zoom(viewOf(el), 2, fullOf(el)))),
  );
  const reset = zoomButton('Reset view', () => show(el, fullOf(el)));
  reset.dataset.swReset = 'yes';
  zooms.append(reset);
  row.append(ranges, zooms);
  el.before(row);
  updateReset(el);
}

function redrawAll() {
  for (const entry of drawn) drawChart(entry.el, entry.data);
}

for (const el of document.querySelectorAll('[data-sw-chart]')) drawChart(el, dataFor(el));

let resizing = null;
window.addEventListener('resize', () => {
  clearTimeout(resizing);
  resizing = setTimeout(() => {
    for (const { el, plots } of drawn) {
      if (['score', 'reading', 'rain'].includes(el.dataset.swChart)) {
        for (const plot of plots) plot.setSize({ width: el.clientWidth, height: plot.height });
      }
    }
  }, 100);
});

// The print page opens the print dialog once its charts and font are ready

if (printing) {
  const open = () => window.print();
  const again = document.getElementById('sw-print');
  if (again) again.addEventListener('click', open);
  document.fonts.ready.then(() => setTimeout(open, 300));
}

// Note downloads. With no boxes ticked, every note shown is included.

const exportForm = document.getElementById('sw-export');
if (exportForm) {
  const boxes = [...exportForm.querySelectorAll('input[name=note]')];
  const all = document.getElementById('sw-select-all');
  const count = document.getElementById('sw-export-count');
  const update = () => {
    const ticked = boxes.filter((b) => b.checked).length;
    count.textContent = ticked
      ? `${ticked} of ${boxes.length} note${boxes.length === 1 ? '' : 's'} selected.`
      : 'No notes selected, so all notes shown will be exported.';
    all.checked = ticked === boxes.length;
    all.indeterminate = ticked > 0 && ticked < boxes.length;
  };
  for (const el of exportForm.querySelectorAll('[data-sw-js-only]')) el.hidden = false;
  all.addEventListener('change', () => {
    for (const b of boxes) b.checked = all.checked;
    update();
  });
  for (const b of boxes) b.addEventListener('change', update);
  update();
}

// The test email. The tick boxes sit in the recipients table, tied to the form with form=

const testForm = document.getElementById('sw-test-email');
const testBoxes = [...document.querySelectorAll('input[name=to][form=sw-test-email]')];
if (testForm && testBoxes.length) {
  const all = document.getElementById('sw-test-all');
  const count = document.getElementById('sw-test-count');
  const none = count.textContent;
  const update = () => {
    const ticked = testBoxes.filter((b) => b.checked).length;
    count.textContent = ticked
      ? `${ticked} of ${testBoxes.length} ${testBoxes.length === 1 ? 'person' : 'people'} ticked.`
      : none;
    all.checked = ticked === testBoxes.length;
    all.indeterminate = ticked > 0 && ticked < testBoxes.length;
  };
  all.hidden = false;
  all.addEventListener('change', () => {
    for (const b of testBoxes) b.checked = all.checked;
    update();
  });
  for (const b of testBoxes) b.addEventListener('change', update);
  update();
}

// Confirm actions that can't be undone, like deleting a note

document.addEventListener('submit', (event) => {
  const ask = event.target.dataset.swConfirm;
  if (ask && !window.confirm(ask)) event.preventDefault();
});

// Choices that reveal more fields. Without JavaScript everything stays shown.

function syncReveals() {
  for (const panel of document.querySelectorAll('[data-sw-reveal-for]')) {
    const input = document.getElementById(panel.dataset.swRevealFor);
    panel.hidden = !(input && input.checked);
  }
}
syncReveals();
document.addEventListener('change', (event) => {
  if (event.target.matches('input[type=radio]')) syncReveals();
});

// The Settings map: a site's square turns green when it's ticked, black when not

for (const box of document.querySelectorAll('input[name=watch]')) {
  box.addEventListener('change', () => {
    for (const square of document.querySelectorAll('[data-sw-site]')) {
      if (square.dataset.swSite !== box.value) continue;
      square.classList.toggle('sw-map__node--watched', box.checked);
      square.classList.toggle('sw-map__node--unwatched', !box.checked);
      const title = square.parentNode.querySelector('title');
      // Keeps the site's name from the page and swaps only what follows the colon
      if (title) title.textContent = `${title.textContent.split(':')[0]}: ${box.checked ? 'Alerts on' : 'Alerts off'}`;
    }
  });
}

// History's filters apply straight away. The button is for when JavaScript is off.

const filters = document.querySelector('form[data-sw-autosubmit]');
if (filters) {
  for (const button of filters.querySelectorAll('[data-sw-nojs]')) button.hidden = true;
  filters.addEventListener('change', () => filters.submit());
}

// The settings draft is saved on every change. Switching pages or reloading
// keeps it.

const form = document.querySelector('form[data-sw-autosave]');
let leaving = false;

function setUnsaved(count, text) {
  document.body.dataset.unsaved = String(count);
  const note = document.getElementById('sw-unsaved');
  const line = document.getElementById('sw-unsaved-text');
  const discard = document.getElementById('sw-discard');
  if (note) note.hidden = !count;
  if (line) line.textContent = text;
  if (discard) discard.hidden = !count;
}

if (form) {
  let timer = null;
  const save = async () => {
    const body = new FormData(form);
    body.delete('save');
    const response = await fetch(form.action || window.location.href, {
      method: 'POST',
      body,
      headers: { 'X-Requested-With': 'fetch' },
      credentials: 'same-origin',
    });
    // A number that doesn't parse is shown by the form itself on Save changes
    if (!response.ok) return;
    const data = await response.json();
    setUnsaved(data.changes, data.banner);
    for (const [model, rules] of Object.entries(data.rate_previews || {})) {
      for (const [rule, text] of Object.entries(rules)) {
        const el = document.getElementById(`sw-rate-preview-${model}-${rule}`);
        if (el) el.textContent = text;
      }
    }
    for (const [model, text] of Object.entries(data.previews || {})) {
      const el = document.getElementById('sw-preview-' + model);
      if (el) el.textContent = text;
    }
    for (const [model, curve] of Object.entries(data.curves || {})) {
      const el = document.querySelector(`[data-sw-chart=rain][data-sw-model="${CSS.escape(model)}"]`);
      if (el) drawChart(el, curve);
    }
  };
  form.addEventListener('change', () => {
    clearTimeout(timer);
    timer = setTimeout(save, 300);
  });
}

// Warn before leaving with unsaved changes. Moving between the settings pages
// and review doesn't count.

// On the website the dashboard sits under /strawberrywatch, which the page says in data-root
const ROOT = document.body.dataset.root || '';
const SETTINGS_PATHS = ['/settings', '/settings/advanced', '/settings/review', '/settings/email', '/setup', '/token'].map(
  (path) => ROOT + path,
);
const ASK = 'Leave this page? Your unsaved changes will still be here when you come back.';
const unsavedNow = () => Number(document.body.dataset.unsaved || 0) > 0;

if (document.body.dataset.settingsPage === 'yes') {
  document.addEventListener('submit', () => {
    leaving = true;
  });
  document.addEventListener('click', (event) => {
    const link = event.target.closest('a[href]');
    if (!link) return;
    const url = new URL(link.href, window.location.href);
    if (url.origin === window.location.origin && SETTINGS_PATHS.includes(url.pathname)) {
      leaving = true;
      return;
    }
    if (unsavedNow() && !window.confirm(ASK)) {
      event.preventDefault();
      return;
    }
    leaving = true;
  });
  window.addEventListener('beforeunload', (event) => {
    if (unsavedNow() && !leaving) {
      event.preventDefault();
      event.returnValue = '';
    }
  });
}

// A stop slider says its stop's name to screen readers, not the number behind it

for (const slider of document.querySelectorAll('input[type=range][data-sw-stops]')) {
  const names = slider.dataset.swStops.split('|');
  slider.addEventListener('input', () => slider.setAttribute('aria-valuetext', names[Number(slider.value)]));
}

// Reporting marks. Based on Uptime Kuma's src/components/HeartbeatBar.vue
// (MIT): one tab stop per row and the arrow keys walk the hours. A tap opens
// the label, since phones have no hover.

for (const row of document.querySelectorAll('.sw-ticks')) {
  const marks = [...row.querySelectorAll('.sw-tick')];
  const move = (from, to) => {
    from.tabIndex = -1;
    to.tabIndex = 0;
    to.focus();
  };
  row.addEventListener('keydown', (event) => {
    const i = marks.indexOf(event.target);
    if (i < 0) return;
    const to = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: marks.length - 1 }[event.key];
    if (to === undefined || !marks[to]) return;
    event.preventDefault();
    move(marks[i], marks[to]);
  });
  row.addEventListener('click', (event) => {
    const mark = event.target.closest('.sw-tick');
    if (!mark) return;
    const opening = !mark.classList.contains('is-open');
    for (const open of document.querySelectorAll('.sw-tick.is-open')) open.classList.remove('is-open');
    mark.classList.toggle('is-open', opening);
  });
}
document.addEventListener('click', (event) => {
  if (event.target.closest('.sw-tick')) return;
  for (const open of document.querySelectorAll('.sw-tick.is-open')) open.classList.remove('is-open');
});
