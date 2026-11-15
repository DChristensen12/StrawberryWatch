/* StrawberryWatch Workbench. Vanilla JS, no build step */

(function () {
  "use strict";

  const words = JSON.parse(document.body.dataset.words || "{}");
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const TZ = "America/Los_Angeles";

  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fill = (template, values) =>
    Object.entries(values).reduce((out, [k, v]) => out.split(`{{ ${k} }}`).join(v), template);
  const count = (n, word) =>
    `${n.toLocaleString("en-US")} ${n === 1 ? word : words.plurals[word]}`;

  function joinList(items) {
    if (items.length <= 1) return items.join("");
    return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
  }

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "X-CSRFToken": csrf, "X-Requested-With": "fetch" },
      body,
    });
  }

  // pywebview's bridge, or null in a plain browser
  function bridge() {
    return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
  }

  function blobToBase64(blob) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1]);
      reader.readAsDataURL(blob);
    });
  }

  async function saveBlob(name, blob) {
    const api = bridge();
    if (api) {
      return api.save_file(name, await blobToBase64(blob));
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return null;
  }

  function showExported(path) {
    const note = document.getElementById("exported");
    if (!note || !path) return;
    note.textContent = fill(note.dataset.template, { path });
    note.hidden = false;
  }

  // Exports that download from a URL
  document.querySelectorAll(".wb-export").forEach((el) => {
    el.addEventListener("click", async (event) => {
      event.preventDefault();
      const response = await fetch(el.dataset.href || el.getAttribute("href"));
      if (response.ok) showExported(await saveBlob(el.dataset.saveAs, await response.blob()));
    });
  });

  // Selects that apply as soon as they change
  document.querySelectorAll(".wb-submit-on-change").forEach((select) => {
    select.addEventListener("change", () => {
      select.form.querySelectorAll(".wb-reading-keep").forEach((input) => (input.disabled = false));
      select.form.submit();
    });
  });

  // Times on the creek's own clock

  const parts = (ts, options) =>
    Object.fromEntries(
      new Intl.DateTimeFormat("en-US", { timeZone: TZ, hourCycle: "h23", ...options })
        .formatToParts(new Date(ts * 1000))
        .map((p) => [p.type, p.value])
    );

  function dayLabel(ts) {
    const p = parts(ts, { day: "numeric", month: "numeric" });
    return `${p.day} ${words.months[Number(p.month) - 1]}`;
  }

  function hourLabel(ts) {
    const p = parts(ts, { hour: "numeric", minute: "2-digit" });
    const label = words.hours[Number(p.hour) % 24];
    return p.minute === "00" ? label : label.replace(/ (am|pm)$/, `:${p.minute} $1`);
  }

  function localInput(ts) {
    const p = parts(ts, { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
    return `${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}`;
  }

  function fromLocalInput(value) {
    // The offset at that moment, found by formatting a first guess
    const guess = Date.parse(`${value}:00Z`) / 1000;
    const p = parts(guess, { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
    const shown = Date.parse(`${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}:00Z`) / 1000;
    return guess + (guess - shown);
  }

  const tzDate = (ts) => uPlot.tzDate(new Date(ts * 1000), TZ);

  function axisStyle() {
    return {
      stroke: token("--wb-axis"),
      grid: { stroke: token("--wb-grid"), width: 1 },
      ticks: { show: false },
      font: '14px "Sen", sans-serif',
    };
  }

  function chartSize(el, ratio) {
    const width = Math.max(el.clientWidth, 240);
    return { width, height: Math.round(width * ratio) };
  }

  async function json(url) {
    const response = await fetch(url);
    return response.ok ? response.json() : null;
  }

  // Step 1: Pick data

  const pick = document.getElementById("pick-form");
  if (pick) {
    const pairs = JSON.parse(pick.dataset.pairs || "{}");
    const group = pick.querySelector(".wb-map-pick");
    const siteBoxes = () => [...pick.querySelectorAll('input[name="sites"]')];
    const readingBoxes = () => [...pick.querySelectorAll('input[name="readings"]')];
    const start = document.getElementById("start-date");
    const end = document.getElementById("end-date");

    pick.querySelectorAll(".wb-dot").forEach((dot) => {
      dot.style.left = `${dot.dataset.left}%`;
      dot.style.top = `${dot.dataset.top}%`;
    });

    function update() {
      const ticked = siteBoxes().filter((b) => b.checked);
      const readings = readingBoxes().filter((b) => b.checked).map((b) => b.value);
      group.setAttribute(
        "aria-label",
        fill(group.dataset.template, { ticked_sites: joinList(ticked.map((b) => b.dataset.name)) })
      );
      group.querySelectorAll("text[data-site]").forEach((label) => {
        const box = siteBoxes().find((b) => b.value === label.dataset.site);
        label.classList.toggle("is-bold", Boolean(box && box.checked));
      });
      const from = Date.parse(start.value);
      const to = Date.parse(end.value);
      const days = Number.isFinite(from) && Number.isFinite(to) && to >= from ? (to - from) / 86400000 + 1 : 0;
      let existing = 0;
      ticked.forEach((b) => (existing += readings.filter((r) => (pairs[b.value] || []).includes(r)).length));
      const line = document.getElementById("picked");
      line.textContent = fill(line.dataset.template, {
        days: count(days, "day"),
        sites: count(ticked.length, "site"),
        readings: count(readings.length, "reading"),
        total: count(96 * days * existing, "reading"),
      });
    }

    pick.addEventListener("change", update);
    pick.addEventListener("input", update);

    pick.querySelectorAll("[data-model-sites]").forEach((button) => {
      button.addEventListener("click", () => {
        const sites = button.dataset.modelSites.split(",");
        const readings = button.dataset.modelReadings.split(",");
        siteBoxes().forEach((b) => (b.checked = sites.includes(b.value)));
        readingBoxes().forEach((b) => (b.checked = readings.includes(b.value)));
        update();
      });
    });

    pick.querySelectorAll(".wb-shortcuts [data-from]").forEach((button) => {
      button.addEventListener("click", () => {
        start.value = button.dataset.from;
        end.value = button.dataset.to;
        update();
      });
    });

    const around = document.getElementById("around-event");
    const past = document.getElementById("past-event");
    around.addEventListener("click", () => {
      past.hidden = !past.hidden;
      around.setAttribute("aria-expanded", String(!past.hidden));
    });
    document.getElementById("past-event-select").addEventListener("change", (event) => {
      const option = event.target.selectedOptions[0];
      if (option && option.dataset.from) {
        start.value = option.dataset.from;
        end.value = option.dataset.to;
        update();
      }
    });

    const warning = document.getElementById("not-set-up");
    pick.querySelectorAll('input[name="source"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        warning.hidden = radio.dataset.setUp === "yes";
      });
    });
  }

  // Background work: poll every 2 seconds, then move on

  const job = document.getElementById("job");
  if (job && job.dataset.job) {
    const running = job.querySelector(".wb-job-running");
    const failed = job.querySelector(".wb-job-error");
    const empty = job.querySelector(".wb-job-empty");
    running.hidden = false;
    const poll = async () => {
      const state = await json(`${document.body.dataset.root || ""}/jobs/${job.dataset.job}`);
      if (!state || state.state === "running") {
        setTimeout(poll, 2000);
        return;
      }
      running.hidden = true;
      if (state.state === "done") {
        window.location = job.dataset.next;
      } else if (state.message === "empty" && empty) {
        empty.hidden = false;
      } else if (failed) {
        failed.textContent = fill(failed.dataset.template, { reason: state.message });
        failed.hidden = false;
      } else {
        window.location = job.dataset.next;
      }
    };
    setTimeout(poll, 2000);
  }

  // Step 2: Explore

  const explore = document.getElementById("explore");
  if (explore && window.uPlot) {
    const query = `site=${encodeURIComponent(explore.dataset.site)}&reading=${encodeURIComponent(explore.dataset.reading)}`;
    // See https://github.com/leeoniya/uPlot/blob/master/demos/sync-cursor.html
    const sync = uPlot.sync("wb");
    const charts = [];

    const base = (el, ratio) => ({
      ...chartSize(el, ratio),
      cursor: { sync: { key: sync.key } },
      legend: { show: false },
    });

    json(`${explore.dataset.series}?${query}`).then((d) => {
      const el = document.getElementById("chart-series");
      if (!d || !el) return;
      const ink = token("--wb-chart");
      const flag = token("--wb-flag");
      const chart = new uPlot(
        {
          ...base(el, 0.28),
          tzDate,
          scales: { x: { time: true } },
          series: [
            {},
            { stroke: ink, width: 2.5, spanGaps: false, points: { show: false } },
            {
              show: false,
              stroke: flag,
              fill: flag,
              width: 0,
              paths: () => null,
              points: { show: true, size: 7, stroke: flag, fill: flag },
            },
          ],
          axes: [
            { ...axisStyle(), values: (u, splits) => splits.map(dayLabel) },
            { ...axisStyle(), size: 56 },
          ],
        },
        [d.times, d.values, d.flags],
        el
      );
      charts.push(chart);
      const toggle = document.getElementById("show-flags");
      toggle.addEventListener("click", () => {
        const on = toggle.getAttribute("aria-pressed") !== "true";
        toggle.setAttribute("aria-pressed", String(on));
        chart.setSeries(2, { show: on });
      });
    });

    json(`${explore.dataset.day}?${query}`).then((d) => {
      const el = document.getElementById("chart-day");
      if (!d || !el) return;
      const ink = token("--wb-chart");
      const water = token("--wb-water");
      // See https://github.com/leeoniya/uPlot/blob/master/demos/high-low-bands.html
      charts.push(
        new uPlot(
          {
            ...base(el, 0.47),
            scales: { x: { time: false, range: [0, 23] } },
            bands: [{ series: [1, 2], fill: `${water}59` }],
            series: [
              {},
              { stroke: "transparent", points: { show: false } },
              { stroke: "transparent", points: { show: false } },
              { stroke: ink, width: 2.5, points: { show: false } },
            ],
            axes: [
              {
                ...axisStyle(),
                splits: () => [0, 6, 12, 18, 23],
                values: (u, splits) => splits.map((h) => words.hours[h]),
              },
              { ...axisStyle(), size: 56 },
            ],
          },
          [d.hours, d.high, d.low, d.median],
          el
        )
      );
    });

    json(`${explore.dataset.histogram}?${query}`).then((d) => {
      const el = document.getElementById("chart-histogram");
      if (!d || !el || !d.edges.length) return;
      const ink = token("--wb-chart");
      const lefts = d.edges.slice(0, -1);
      charts.push(
        new uPlot(
          {
            ...base(el, 0.47),
            cursor: { show: false },
            scales: { x: { time: false, range: [d.edges[0], d.edges[d.edges.length - 1]] } },
            series: [
              {},
              {
                stroke: ink,
                fill: ink,
                paths: uPlot.paths.bars({ size: [0.9, Infinity], align: 1 }),
                points: { show: false },
              },
            ],
            axes: [axisStyle(), { ...axisStyle(), size: 48 }],
          },
          [lefts, d.counts],
          el
        )
      );
    });

    const notes = document.getElementById("notes");
    notes.addEventListener("blur", () => {
      const body = new FormData();
      body.append("notes", notes.value);
      post(explore.dataset.notes, body);
    });

    document.getElementById("export-csv").addEventListener("click", async () => {
      const response = await fetch(explore.dataset.csv);
      if (response.ok) showExported(await saveBlob("readings.csv", await response.blob()));
    });

    document.getElementById("export-png").addEventListener("click", async () => {
      const canvases = charts.map((c) => c.ctx.canvas);
      if (!canvases.length) return;
      const width = Math.max(...canvases.map((c) => c.width));
      const height = canvases.reduce((sum, c) => sum + c.height, 0);
      const out = document.createElement("canvas");
      out.width = width;
      out.height = height;
      const ctx = out.getContext("2d");
      ctx.fillStyle = token("--wb-surface");
      ctx.fillRect(0, 0, width, height);
      let y = 0;
      canvases.forEach((c) => {
        ctx.drawImage(c, 0, y);
        y += c.height;
      });
      const data = out.toDataURL("image/png");
      if (bridge()) {
        showExported(await bridge().save_file("charts.png", data.split(",")[1]));
        return;
      }
      const body = new FormData();
      body.append("data", data);
      const response = await post(explore.dataset.png, body);
      if (response.ok) await saveBlob("charts.png", await response.blob());
    });
  }

  // Step 3: Recalibrate

  const checkForm = document.getElementById("check-form");
  if (checkForm) {
    const save = document.getElementById("save-button");
    const hint = document.getElementById("save-hint");
    checkForm.addEventListener("change", () => {
      const model = checkForm.querySelector('input[name="model"]:checked')?.value || "";
      const choice = checkForm.querySelector('input[name="false_alarms"]:checked')?.value || "";
      document.getElementById("save-model").value = model;
      document.getElementById("save-choice").value = choice;
      if (save) {
        const ready = `${model}:${choice}` === checkForm.dataset.checkKey;
        save.disabled = !ready;
        hint.hidden = ready;
      }
    });
  }

  // Events

  const eventForm = document.getElementById("event-form");
  if (eventForm) {
    const alertKinds = eventForm.dataset.alertKinds.split(",");
    const starts = document.getElementById("event-starts");
    const ends = document.getElementById("event-ends");
    if (eventForm.dataset.new === "yes") {
      eventForm.querySelectorAll('input[name="kind"]').forEach((radio) => {
        radio.addEventListener("change", () => {
          const yes = alertKinds.includes(radio.value);
          eventForm.querySelector(`input[name="should_alert"][value="${yes ? "yes" : "no"}"]`).checked = true;
        });
      });
    }
    eventForm.addEventListener("submit", (event) => {
      const sites = eventForm.querySelectorAll('input[name="sites"]:checked').length;
      if (!sites || !starts.value || !ends.value || ends.value < starts.value) {
        event.preventDefault();
        document.getElementById("event-problem").hidden = false;
      }
    });

    const block = document.getElementById("event-chart-block");
    if (block && window.uPlot) {
      const el = document.getElementById("chart-event");
      let chart = null;
      const draw = (u) => {
        const lo = fromLocalInput(starts.value);
        const hi = fromLocalInput(ends.value);
        if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
        const left = u.valToPos(lo, "x", true);
        const right = u.valToPos(hi, "x", true);
        const { top, height } = u.bbox;
        const ctx = u.ctx;
        ctx.save();
        ctx.fillStyle = `${token("--wb-water")}47`;
        ctx.fillRect(left, top, right - left, height);
        ctx.strokeStyle = token("--wb-ink");
        ctx.lineWidth = 3 * devicePixelRatio;
        [left, right].forEach((x) => {
          ctx.beginPath();
          ctx.moveTo(x, top);
          ctx.lineTo(x, top + height);
          ctx.stroke();
        });
        ctx.restore();
      };
      const load = async (reading) => {
        const url = `${block.dataset.series}?event=${encodeURIComponent(block.dataset.event)}&site=${encodeURIComponent(block.dataset.site)}&reading=${reading}`;
        const d = await json(url);
        if (!d) return;
        if (chart) chart.destroy();
        chart = new uPlot(
          {
            ...chartSize(el, 0.4),
            tzDate,
            legend: { show: false },
            cursor: { drag: { x: true, y: false, setScale: false } },
            scales: { x: { time: true } },
            series: [{}, { stroke: token("--wb-chart"), width: 2.5, points: { show: false } }],
            axes: [
              { ...axisStyle(), values: (u, splits) => splits.map(hourLabel) },
              { ...axisStyle(), size: 56 },
            ],
            hooks: {
              draw: [draw],
              setSelect: [
                (u) => {
                  if (u.select.width < 2) return;
                  // Snap to the quarter hour, as the inputs step by 15 minutes
                  const snap = (ts) => Math.round(ts / 900) * 900;
                  starts.value = localInput(snap(u.posToVal(u.select.left, "x")));
                  ends.value = localInput(snap(u.posToVal(u.select.left + u.select.width, "x")));
                  u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
                  u.redraw();
                },
              ],
            },
          },
          [d.times, d.values],
          el
        );
      };
      load(block.dataset.reading);
      block.querySelectorAll("[data-reading]").forEach((button) => {
        if (!button.classList.contains("wb-toggle")) return;
        button.addEventListener("click", () => {
          block.querySelectorAll(".wb-toggle").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
          load(button.dataset.reading);
        });
      });
      [starts, ends].forEach((input) => input.addEventListener("change", () => chart && chart.redraw()));
    }
  }

  const deleteForm = document.getElementById("delete-form");
  if (deleteForm) {
    deleteForm.addEventListener("submit", (event) => {
      if (!window.confirm(document.getElementById("delete-question").textContent)) event.preventDefault();
    });
  }

  // Settings

  document.querySelectorAll("[data-reveal]").forEach((button) => {
    button.addEventListener("click", () => {
      const form = document.getElementById(button.dataset.reveal);
      form.hidden = false;
      form.querySelector('input[type="password"]').focus();
    });
  });

  document.querySelectorAll(".wb-autosave").forEach((form) => {
    form.addEventListener("change", (event) => {
      const radio = event.target;
      post(form.action, new FormData(form));
      if (radio.name === "theme") document.documentElement.dataset.theme = radio.value;
      if (radio.name === "size") document.documentElement.dataset.size = radio.value;
    });
  });

  const choose = document.getElementById("choose-folder");
  if (choose) {
    choose.addEventListener("click", async () => {
      const api = bridge();
      if (!api) return;
      const path = await api.choose_folder();
      if (path) {
        document.getElementById("csv-folder").value = path;
        document.getElementById("csv-form").submit();
      }
    });
  }

  const open = document.getElementById("open-folder");
  if (open) {
    open.addEventListener("click", () => bridge() && bridge().open_folder());
  }
})();
