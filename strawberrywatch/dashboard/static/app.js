// The website's menu and dark mode button, the charts, the Settings map
// following its tick boxes, saving the settings draft as it's edited, and
// asking before leaving changes unsaved. Chart data comes from
// <script type="application/json"> blocks the server fills with tojson, and
// nothing here writes HTML: text goes in with textContent only.

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
      // Storage can be blocked; the mode then lasts for this page only.
    }
    showMode(dark);
    redrawAll();
  });
}

// The page that becomes a PDF is drawn light whatever the screen mode, since
// it's going on paper. The mode itself isn't changed.
const printing = document.body.dataset.swPrint === 'yes';
if (printing) {
  document.body.classList.remove('dark-mode');
  document.documentElement.classList.remove('dark');
}

// Charts. Colours come from the stylesheet's tokens, read when a chart is
// drawn, so switching mode redraws them in the other set.

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

// From uPlot's high-low-bands and area-fill demos: the line as a band that
// rises when rain raises it, and rain as a filled area beneath, on its own scale
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
        cursor: { sync: { key: sync.key } },
        scales: { x: { time: true } },
        series: [
          {},
          { label: 'How unusual', stroke: c.ink, width: 2 },
          { label: 'The line, with rain', stroke: c.muted, width: 1.5 },
          { label: 'The line, dry', stroke: c.muted, width: 1, dash: [4, 4] },
        ],
        bands: [{ series: [2, 3], fill: c.band }],
        axes: [axis(c), axis(c, 'How unusual')],
        hooks: {
          draw: [
            (u) => {
              if (flagAt < 0) return;
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
          cursor: { sync: { key: sync.key } },
          scales: { x: { time: true } },
          series: [{}, { label: 'Rain, mm', stroke: c.rain, fill: c.rainFill, width: 1 }],
          axes: [axis(c), axis(c, 'Rain, mm')],
        },
        [d.x, d.rain.map((v) => v ?? 0)],
        el,
      ),
    );
  }
  return plots;
}

// From uPlot's timeline-discrete demo, as bars: one mark per expected reading,
// nothing where it never came
function stripChart(el, d) {
  const c = colours();
  return [
    new uPlot(
      {
        width: 288,
        height: 18,
        legend: { show: false },
        cursor: { show: false },
        select: { show: false },
        padding: [1, 0, 1, 0],
        scales: { x: { time: false }, y: { range: [0, 1] } },
        axes: [{ show: false }, { show: false }],
        series: [{}, { stroke: c.ink, fill: c.ink, paths: uPlot.paths.bars({ size: [0.5, 100] }), points: { show: false } }],
      },
      [d.x, d.y],
      el,
    ),
  ];
}

// From uPlot's sparklines demo: one small line per sensor
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
        axes: [axis(c, 'Hours from the start of the rain'), axis(c, 'Times normal')],
      },
      [d.hours, storm, d.multiple],
      el,
    ),
  ];
}

const draw = { score: scoreChart, strip: stripChart, spark: sparkChart, rain: rainChart };

function drawChart(el, data) {
  const entry = drawn.find((e) => e.el === el);
  if (entry) {
    for (const plot of entry.plots) plot.destroy();
    entry.data = data;
    entry.plots = draw[el.dataset.swChart](el, data);
  } else {
    drawn.push({ el, data, plots: draw[el.dataset.swChart](el, data) });
  }
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
      if (el.dataset.swChart === 'score' || el.dataset.swChart === 'rain') {
        for (const plot of plots) plot.setSize({ width: el.clientWidth, height: plot.height });
      }
    }
  }, 100);
});

// Once its charts are drawn and its font is in, the print page opens the
// print window, where Save as PDF is one of the printers

if (printing) {
  const open = () => window.print();
  const again = document.getElementById('sw-print');
  if (again) again.addEventListener('click', open);
  document.fonts.ready.then(() => setTimeout(open, 300));
}

// Ticking notes to download. With nothing ticked, every note shown goes.

const exportForm = document.getElementById('sw-export');
if (exportForm) {
  const boxes = [...exportForm.querySelectorAll('input[name=note]')];
  const all = document.getElementById('sw-select-all');
  const count = document.getElementById('sw-export-count');
  const update = () => {
    const ticked = boxes.filter((b) => b.checked).length;
    count.textContent = ticked
      ? `${ticked} of ${boxes.length} note${boxes.length === 1 ? '' : 's'} ticked.`
      : 'Nothing ticked, so every note shown is included.';
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

// Asking before something that can't be undone, like deleting a note

document.addEventListener('submit', (event) => {
  const ask = event.target.dataset.swConfirm;
  if (ask && !window.confirm(ask)) event.preventDefault();
});

// A choice that opens more questions. Without script everything stays shown.

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
      if (title) title.textContent = `${box.value}: ${box.checked ? 'Watched' : 'Not watched'}`;
    }
  });
}

// History's filters: a choice updates the list straight away, so the button is
// only needed without script

const filters = document.querySelector('form[data-sw-autosubmit]');
if (filters) {
  for (const button of filters.querySelectorAll('[data-sw-nojs]')) button.hidden = true;
  filters.addEventListener('change', () => filters.submit());
}

// The settings draft: every change is kept as it's made, so switching pages or
// reloading never loses it. Save changes goes on to the check page.

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

// Asking before leaving with unsaved changes. Moving between the settings
// pages and the check page doesn't count as leaving.

const SETTINGS_PATHS = ['/settings', '/settings/advanced', '/settings/review', '/settings/email', '/setup', '/token'];
const ASK = "You have changes that aren't saved yet. Leave anyway? They'll still be here when you come back.";
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
