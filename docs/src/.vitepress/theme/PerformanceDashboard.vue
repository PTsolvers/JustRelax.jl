<script setup>
import { onMounted, ref } from "vue"

const root = ref(null)

onMounted(() => {
  const historyURL = "https://raw.githubusercontent.com/PTsolvers/JustRelax.jl/benchmark-data/benchmark_history.json";
  const $ = id => root.value.querySelector(`#${id}`);
  const colors = ["#0f9d8a", "#2f7ed8", "#d19a00", "#8e5bd6", "#d64545", "#2e9e57"];
  const metricDefinitions = {
    runtime: { label: "median runtime", unit: "ms", value: b => 1e3 * b.time_median_seconds, better: -1 },
    gflops: { label: "effective performance", unit: "GFLOP/s", value: b => b.effective_gflops_per_second, better: 1 },
    bandwidth: { label: "effective memory throughput T_eff", unit: "GB/s", value: b => b.effective_bandwidth_gb_per_second, better: 1 },
    allocations: { label: "allocations", unit: "allocations", value: b => b.allocations, better: -1 }
  };
  const state = { hardware: "", group: "", metric: "runtime" };
  let data, runs;

  // Runs are grouped by backend; the device of each run appears in tooltips and the results table.
  const hardwareKey = run => run.backend;
  const latestFingerprint = () => hardwareRuns(state.hardware).at(-1)?.hardware_fingerprint;
  const deviceName = run => run.benchmarks[0]?.metadata?.device || run.hardware_fingerprint;
  const hardwareRuns = key => runs.filter(run => hardwareKey(run) === key);
  const observations = key => hardwareRuns(key).flatMap(run => run.benchmarks.map(benchmark => ({ run, benchmark })));
  const groupObservations = () => observations(state.hardware).filter(item => item.benchmark.group === state.group);
  const shortCommit = commit => commit.slice(0, 8);
  const formatNumber = value => {
    if (!Number.isFinite(value)) return "—";
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (Math.abs(value) >= 10) return value.toFixed(1);
    if (Math.abs(value) >= 1) return value.toFixed(2);
    return value.toPrecision(3);
  };
  const escapeHTML = value => String(value).replace(/[&<>'"]/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);

  function init() {
    $("jp-download").href = historyURL;
    const hardware = [...new Set(runs.map(hardwareKey))];
    $("jp-hardware").innerHTML = hardware.map(key => `<option>${escapeHTML(key)}</option>`).join("");
    state.hardware = hardware[0] || "";
    $("jp-hardware").addEventListener("change", event => {
      state.hardware = event.target.value;
      state.group = groups()[0] || "";
      loadRooflineProfile();
      render();
    });
    $("jp-metric").addEventListener("change", event => {
      state.metric = event.target.value;
      renderHistory();
    });
    for (const id of ["jp-peak-compute", "jp-peak-bandwidth"]) {
      $(id).addEventListener("input", () => {
        saveRooflineProfile();
        renderRoofline();
      });
    }
    state.group = groups()[0] || "";
    loadRooflineProfile();
    render();
  }

  function groups() {
    return [...new Set(observations(state.hardware).map(item => item.benchmark.group))].sort();
  }

  function render() {
    renderStats();
    renderReadiness();
    renderNavigation();
    renderHistory();
    renderRoofline();
    renderTable();
  }

  function renderStats() {
    const latest = runs.at(-1);
    const values = [
      [new Set(runs.map(run => run.commit)).size, "commits tracked"],
      [runs.reduce((sum, run) => sum + run.benchmarks.length, 0), "data points"],
      [new Set(runs.map(hardwareKey)).size, "backends"],
      [latest ? new Date(latest.timestamp_utc).toLocaleDateString() : "—", "last result"]
    ];
    $("jp-stats").innerHTML = values.map(([value, label]) => `<div class="jp-stat"><span class="jp-stat-value">${escapeHTML(value)}</span><span class="jp-stat-label">${label}</span></div>`).join("");
  }

  function renderReadiness() {
    const selectedRuns = hardwareRuns(state.hardware);
    if (!selectedRuns.length) {
      $("jp-readiness").innerHTML = "<strong>Data readiness:</strong> No benchmark results have been recorded yet.";
      return;
    }
    const hasHistory = new Set(selectedRuns.map(run => run.commit)).size > 1;
    const dirty = selectedRuns.some(run => run.dirty);
    const historyText = hasHistory ? "Commit history is active." : "This is the first recorded commit; regression history starts here.";
    const dirtyText = dirty ? " The selected data includes a dirty worktree and must not be treated as a release baseline." : "";
    $("jp-readiness").innerHTML = `<strong>Data readiness:</strong> ${historyText}${dirtyText}`;
  }

  function renderNavigation() {
    $("jp-groups").innerHTML = groups().map(group => `<button class="jp-pill${group === state.group ? " active" : ""}" type="button" data-group="${escapeHTML(group)}">${escapeHTML(group)}</button>`).join("");
    $("jp-groups").querySelectorAll("[data-group]").forEach(button => button.addEventListener("click", () => {
      state.group = button.dataset.group;
      render();
    }));
  }

  function seriesForGroup() {
    const byName = new Map();
    for (const item of groupObservations()) {
      if (!byName.has(item.benchmark.name)) byName.set(item.benchmark.name, []);
      byName.get(item.benchmark.name).push(item);
    }
    return [...byName.entries()].map(([name, items]) => [name, items.sort((a, b) => Date.parse(a.run.timestamp_utc) - Date.parse(b.run.timestamp_utc))]);
  }

  function renderHistory() {
    const definition = metricDefinitions[state.metric];
    const series = seriesForGroup();
    $("jp-history-copy").textContent = `${definition.label}; ${definition.better < 0 ? "lower" : "higher"} is better.`;
    $("jp-history-legend").innerHTML = series.map(([name], index) => `<span><span class="jp-swatch" style="background:${colors[index % colors.length]}"></span>${escapeHTML(name)}</span>`).join("");
    const chart = $("jp-history-chart");
    if (!series.length) { chart.innerHTML = '<div class="jp-empty">No results in this group.</div>'; return; }

    const commits = hardwareRuns(state.hardware);
    const values = series.flatMap(([, items]) => items.map(item => definition.value(item.benchmark))).filter(Number.isFinite);
    const width = 920, height = 340, margin = { top: 18, right: 24, bottom: 62, left: 72 };
    const x = index => commits.length === 1 ? (margin.left + width - margin.right) / 2 : margin.left + index * (width - margin.left - margin.right) / (commits.length - 1);
    let minimum = Math.min(...values), maximum = Math.max(...values);
    if (minimum === maximum) { const pad = Math.abs(minimum || 1) * .2; minimum -= pad; maximum += pad; }
    else { const pad = (maximum - minimum) * .12; minimum -= pad; maximum += pad; }
    minimum = Math.max(0, minimum);
    const y = value => height - margin.bottom - (value - minimum) * (height - margin.top - margin.bottom) / (maximum - minimum);
    const runIndex = new Map(commits.map((run, index) => [run, index]));
    const ticks = 5;
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(definition.label)} by commit">`;
    for (let index = 0; index <= ticks; index++) {
      const value = minimum + (maximum - minimum) * index / ticks;
      const py = y(value);
      svg += `<line class="gridline" x1="${margin.left}" y1="${py}" x2="${width - margin.right}" y2="${py}"/><text x="${margin.left - 10}" y="${py + 4}" text-anchor="end">${formatNumber(value)}</text>`;
    }
    commits.forEach((run, index) => {
      svg += `<text transform="translate(${x(index)},${height - margin.bottom + 18}) rotate(-35)" text-anchor="end">${shortCommit(run.commit)}</text>`;
    });
    svg += `<line class="axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"/><line class="axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/><text x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">${definition.unit}</text>`;
    series.forEach(([, items], seriesIndex) => {
      const color = colors[seriesIndex % colors.length];
      const points = items.map(item => [x(runIndex.get(item.run)), y(definition.value(item.benchmark)), item]);
      if (points.length > 1) svg += `<path d="${points.map(([px, py], index) => `${index ? "L" : "M"}${px},${py}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
      for (const [px, py, item] of points) svg += `<circle cx="${px}" cy="${py}" r="5" fill="${color}"><title>${escapeHTML(item.benchmark.name)}\n${shortCommit(item.run.commit)} · ${escapeHTML(deviceName(item.run))}\n${formatNumber(definition.value(item.benchmark))} ${definition.unit}</title></circle>`;
    });
    chart.innerHTML = svg + "</svg>";
  }

  const rooflineStorageKey = () => `justrelax-roofline:${state.hardware}:${latestFingerprint()}`;

  function loadRooflineProfile() {
    const metadata = hardwareRuns(state.hardware).at(-1)?.benchmarks?.[0]?.metadata || {};
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(rooflineStorageKey()) || "{}"); } catch (_) {}
    $("jp-peak-compute").value = saved.compute || metadata.peak_compute_gflops || "";
    $("jp-peak-bandwidth").value = saved.bandwidth || metadata.peak_memory_bandwidth_gb_per_second || "";
  }

  function saveRooflineProfile() {
    const profile = { compute: $("jp-peak-compute").value, bandwidth: $("jp-peak-bandwidth").value };
    try { localStorage.setItem(rooflineStorageKey(), JSON.stringify(profile)); } catch (_) {}
  }

  function latestBenchmarks() {
    const latest = new Map();
    const fingerprint = latestFingerprint();
    for (const item of groupObservations()) if (item.run.hardware_fingerprint === fingerprint) latest.set(item.benchmark.name, item);
    return [...latest.values()];
  }

  function renderRoofline() {
    const items = latestBenchmarks().filter(item => item.benchmark.arithmetic_intensity_flops_per_byte > 0 && item.benchmark.effective_gflops_per_second > 0);
    const chart = $("jp-roofline-chart");
    const compute = Number($("jp-peak-compute").value);
    const bandwidth = Number($("jp-peak-bandwidth").value);
    const ready = compute > 0 && bandwidth > 0;
    if (!items.length) {
      chart.innerHTML = '<div class="jp-empty">No roofline metrics in this group: the benchmarks record no arithmetic intensity or GFLOP/s.</div>';
      $("jp-roof-note").textContent = "";
      return;
    }

    const width = 660, height = 410, margin = { top: 22, right: 24, bottom: 66, left: 78 };
    const pointX = items.map(item => item.benchmark.arithmetic_intensity_flops_per_byte);
    const pointY = items.map(item => item.benchmark.effective_gflops_per_second);
    const ridge = ready ? compute / bandwidth : NaN;
    const xmin = Math.max(1e-3, Math.min(...pointX, ...(ready ? [ridge] : [])) / 3);
    const xmax = Math.max(...pointX, ...(ready ? [ridge] : [])) * 3;
    const ymin = Math.max(1e-4, Math.min(...pointY) / 3);
    const roofMax = ready ? Math.max(compute, bandwidth * xmax) : 0;
    const ymax = Math.max(...pointY, roofMax) * 1.7;
    const lx = value => margin.left + (Math.log10(value) - Math.log10(xmin)) * (width - margin.left - margin.right) / (Math.log10(xmax) - Math.log10(xmin));
    const ly = value => height - margin.bottom - (Math.log10(value) - Math.log10(ymin)) * (height - margin.top - margin.bottom) / (Math.log10(ymax) - Math.log10(ymin));
    const logTicks = (min, max) => {
      const ticks = [];
      for (let exponent = Math.floor(Math.log10(min)); exponent <= Math.ceil(Math.log10(max)); exponent++) {
        const value = 10 ** exponent;
        if (value >= min && value <= max) ticks.push(value);
      }
      return ticks;
    };

    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Roofline chart">`;
    for (const value of logTicks(xmin, xmax)) svg += `<line class="gridline" x1="${lx(value)}" y1="${margin.top}" x2="${lx(value)}" y2="${height - margin.bottom}"/><text x="${lx(value)}" y="${height - margin.bottom + 19}" text-anchor="middle">${formatNumber(value)}</text>`;
    for (const value of logTicks(ymin, ymax)) svg += `<line class="gridline" x1="${margin.left}" y1="${ly(value)}" x2="${width - margin.right}" y2="${ly(value)}"/><text x="${margin.left - 10}" y="${ly(value) + 4}" text-anchor="end">${formatNumber(value)}</text>`;
    svg += `<line class="axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"/><line class="axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/><text x="${(margin.left + width - margin.right) / 2}" y="${height - 18}" text-anchor="middle">Arithmetic intensity (FLOP/byte)</text><text x="20" y="${height / 2}" text-anchor="middle" transform="rotate(-90 20 ${height / 2})">Effective GFLOP/s</text>`;

    if (ready) {
      const boundaryX = Math.min(Math.max(ridge, xmin), xmax);
      const boundaryY = Math.min(compute, bandwidth * boundaryX);
      if (ridge > xmin) svg += `<path d="M${lx(xmin)},${ly(bandwidth * xmin)} L${lx(boundaryX)},${ly(boundaryY)}" fill="none" stroke="#2f7ed8" stroke-width="3"/><text class="bound-label" x="${lx(Math.sqrt(xmin * boundaryX))}" y="${ly(bandwidth * Math.sqrt(xmin * boundaryX)) - 10}" text-anchor="middle">memory ceiling</text>`;
      if (ridge < xmax) svg += `<path d="M${lx(boundaryX)},${ly(boundaryY)} L${lx(xmax)},${ly(compute)}" fill="none" stroke="#d64545" stroke-width="3"/><text class="bound-label" x="${lx(Math.sqrt(boundaryX * xmax))}" y="${ly(compute) - 10}" text-anchor="middle">compute ceiling</text>`;
      if (ridge >= xmin && ridge <= xmax) svg += `<line x1="${lx(ridge)}" y1="${margin.top}" x2="${lx(ridge)}" y2="${height - margin.bottom}" stroke="#d19a00" stroke-width="1.5" stroke-dasharray="6 6"/><text x="${lx(Math.sqrt(xmin * ridge))}" y="${margin.top + 18}" text-anchor="middle">memory-bound</text><text x="${lx(Math.sqrt(ridge * xmax))}" y="${margin.top + 18}" text-anchor="middle">compute-bound</text>`;
    }

    items.forEach((item, index) => {
      const benchmark = item.benchmark;
      svg += `<circle cx="${lx(benchmark.arithmetic_intensity_flops_per_byte)}" cy="${ly(benchmark.effective_gflops_per_second)}" r="7" fill="${colors[index % colors.length]}"><title>${escapeHTML(benchmark.name)}\nAI ${formatNumber(benchmark.arithmetic_intensity_flops_per_byte)} FLOP/byte\n${formatNumber(benchmark.effective_gflops_per_second)} GFLOP/s</title></circle>`;
    });
    chart.innerHTML = svg + "</svg>";

    $("jp-roof-note").textContent = ready ? `Ridge point: ${formatNumber(ridge)} FLOP/byte. The ceilings are specific to this browser and backend, measured on ${deviceName(hardwareRuns(state.hardware).at(-1))}.` : "No device peak values are recorded. Enter published or measured peaks to draw the memory- and compute-bound ceilings; the dashboard will not invent them.";
  }

  function renderTable() {
    const series = seriesForGroup();
    if (!series.length) { $("jp-results").innerHTML = '<div class="jp-empty">No results.</div>'; return; }
    const rows = series.map(([name, items]) => {
      const current = items.at(-1);
      const previous = items.slice(0, -1).filter(item => item.run.hardware_fingerprint === current.run.hardware_fingerprint).slice(-5);
      const benchmark = current.benchmark;
      const mean = previous.reduce((sum, item) => sum + item.benchmark.time_median_seconds, 0) / previous.length;
      const delta = 100 * (benchmark.time_median_seconds / mean - 1);
      const deltaClass = delta < 0 ? "jp-good" : delta > 0 ? "jp-bad" : "";
      const commitURL = `${data.repository_url}/commit/${current.run.commit}`;
      return `<tr>
        <td><strong>${escapeHTML(name)}</strong>${current.run.dirty ? '<span class="jp-tag">dirty</span>' : ''}<span class="jp-desc">${escapeHTML(deviceName(current.run))}</span><span class="jp-desc">${escapeHTML(benchmark.performance_model_description || "")}</span></td>
        <td class="num">${formatNumber(1e3 * benchmark.time_median_seconds)} ms</td>
        <td class="num">${previous.length ? `${formatNumber(1e3 * mean)} ms<span class="jp-desc">${previous.length} run${previous.length > 1 ? "s" : ""}</span>` : "—"}</td>
        <td class="num ${deltaClass}">${Number.isFinite(delta) ? `${delta > 0 ? "+" : ""}${delta.toFixed(2)}%` : "first run"}</td>
        <td class="num">${formatNumber(benchmark.arithmetic_intensity_flops_per_byte)}</td>
        <td class="num">${formatNumber(benchmark.effective_gflops_per_second)}</td>
        <td class="num">${formatNumber(benchmark.effective_bandwidth_gb_per_second)}</td>
        <td class="num">${formatNumber(100 * benchmark.effective_bandwidth_gb_per_second / benchmark.metadata.peak_memory_bandwidth_gb_per_second)}%</td>
        <td class="num"><a href="${commitURL}">${shortCommit(current.run.commit)}</a><span class="jp-desc">${escapeHTML(benchmark.metadata.commit_subject || "")}</span></td>
      </tr>`;
    }).join("");
    $("jp-results").innerHTML = `<table><thead><tr><th>Benchmark</th><th>Median</th><th>Recent mean</th><th>Change</th><th>FLOP/byte</th><th>GFLOP/s</th><th>T_eff (GB/s)</th><th>of STREAM peak</th><th>Commit</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  fetch(historyURL)
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(payload => {
      if (!Array.isArray(payload.runs)) throw new Error("benchmark history has no runs array");
      data = payload;
      runs = [...data.runs].sort((a, b) => Date.parse(a.timestamp_utc) - Date.parse(b.timestamp_utc));
      init();
    })
    .catch(error => {
      root.value.innerHTML = `<div class="jp-empty">Could not load the benchmark history from the benchmark-data branch (${escapeHTML(error.message)}).</div>`;
    });
})
</script>

<template>
  <div id="justrelax-performance" ref="root">
  <div class="jp-stats" id="jp-stats"></div>
  <div class="jp-controls">
    <label>Backend<select id="jp-hardware"></select></label>
    <label>History metric<select id="jp-metric">
      <option value="runtime">Median runtime</option>
      <option value="gflops">Effective GFLOP/s</option>
      <option value="bandwidth">T_eff</option>
      <option value="allocations">Allocations</option>
    </select></label>
    <a class="jp-button" id="jp-download" download>Download JSON</a>
  </div>
  <div class="jp-status" id="jp-readiness"></div>
  <nav class="jp-pills" id="jp-groups" aria-label="Benchmark groups"></nav>
  <div class="jp-grid">
    <section class="jp-card">
      <div class="jp-card-head">
        <div><strong>Commit history</strong><p class="jp-copy" id="jp-history-copy"></p></div>
        <div class="jp-legend" id="jp-history-legend"></div>
      </div>
      <div class="jp-chart" id="jp-history-chart"></div>
    </section>
    <section class="jp-card">
      <strong>Latest roofline position</strong>
      <p class="jp-copy">Measured time with modeled FLOPs and bytes.</p>
      <div class="jp-roof-controls">
        <label>Peak compute (GFLOP/s)<input id="jp-peak-compute" type="number" min="0" step="any" placeholder="Required for ceiling"></label>
        <label>Peak bandwidth (GB/s)<input id="jp-peak-bandwidth" type="number" min="0" step="any" placeholder="Required for ceiling"></label>
      </div>
      <div class="jp-chart" id="jp-roofline-chart"></div>
      <p class="jp-copy" id="jp-roof-note"></p>
    </section>
  </div>
  <section class="jp-card">
    <strong>Latest results</strong>
    <p class="jp-copy">Change is relative to the mean median runtime of up to five earlier runs of the same benchmark on the same device.</p>
    <div class="jp-table" id="jp-results"></div>
  </section>
</div>
</template>

<style>
.VPDoc:has(#justrelax-performance) .container,
.VPDoc:has(#justrelax-performance) .content,
.VPDoc:has(#justrelax-performance) .content-container { max-width: none !important; }
#justrelax-performance {
  --jp-panel: var(--vp-c-bg-soft);
  --jp-border: var(--vp-c-divider);
  --jp-muted: var(--vp-c-text-2);
  --jp-accent: var(--vp-c-brand-1);
  --jp-good: var(--vp-c-success-1);
  --jp-bad: var(--vp-c-danger-1);
  --jp-warn: var(--vp-c-warning-1);
  --jp-axis: var(--vp-c-text-3);
  --jp-mono: var(--vp-font-family-mono);
}

#justrelax-performance label { display: grid; gap: 4px; color: var(--jp-muted); font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
#justrelax-performance select,
#justrelax-performance input { width: 100%; min-height: 2.3rem; padding: 4px 8px; color: inherit; background: var(--jp-panel); border: 1px solid var(--jp-border); border-radius: 6px; font: inherit; text-transform: none; letter-spacing: 0; font-weight: 400; }

#justrelax-performance .jp-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 1rem 0; }
#justrelax-performance .jp-stat { padding: 12px; background: var(--jp-panel); border: 1px solid var(--jp-border); border-radius: 8px; }
#justrelax-performance .jp-stat-value { display: block; color: var(--jp-accent); font: 700 1.3rem/1.2 var(--jp-mono); }
#justrelax-performance .jp-stat-label { display: block; margin-top: 4px; color: var(--jp-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }

#justrelax-performance .jp-controls { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr) auto; gap: 10px; align-items: end; margin-bottom: 1rem; }
#justrelax-performance .jp-button { display: inline-flex; align-items: center; min-height: 2.3rem; padding: 4px 12px; border: 1px solid var(--jp-border); border-radius: 6px; white-space: nowrap; }
#justrelax-performance .jp-button:hover { border-color: var(--jp-accent); }

#justrelax-performance .jp-status { margin-bottom: 1rem; padding: 10px 14px; color: var(--jp-muted); background: var(--jp-panel); border: 1px solid var(--jp-border); border-left: 4px solid var(--jp-warn); border-radius: 6px; }
#justrelax-performance .jp-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1rem; }
#justrelax-performance .jp-pill { padding: 4px 12px; color: var(--jp-muted); background: transparent; border: 1px solid var(--jp-border); border-radius: 999px; cursor: pointer; font: inherit; }
#justrelax-performance .jp-pill:hover, #justrelax-performance .jp-pill.active { color: #fff; background: var(--jp-accent); border-color: var(--jp-accent); }

#justrelax-performance .jp-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }
#justrelax-performance .jp-card { min-width: 0; margin-bottom: 12px; padding: 14px; background: var(--jp-panel); border: 1px solid var(--jp-border); border-radius: 8px; }
#justrelax-performance .jp-card-head { display: flex; justify-content: space-between; gap: 12px; }
#justrelax-performance .jp-copy { margin: 2px 0 8px !important; color: var(--jp-muted); font-size: .8rem; line-height: 1.4; }
#justrelax-performance .jp-legend { display: flex; gap: 8px; flex-wrap: wrap; justify-content: end; color: var(--jp-muted); font-size: .7rem; }
#justrelax-performance .jp-roof-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
#justrelax-performance .jp-swatch { display: inline-block; width: 9px; height: 9px; margin-right: 4px; border-radius: 50%; }

#justrelax-performance .jp-chart svg { display: block; width: 100%; height: auto; max-height: 440px; }
#justrelax-performance .jp-chart text { fill: var(--jp-muted); font-family: var(--jp-mono); font-size: 11px; }
#justrelax-performance .jp-chart .axis { stroke: var(--jp-axis); }
#justrelax-performance .jp-chart .gridline { stroke: var(--jp-border); stroke-dasharray: 3 6; }
#justrelax-performance .jp-chart circle { stroke: var(--jp-panel); stroke-width: 2; }
#justrelax-performance .jp-chart .bound-label { font-weight: 700; font-size: 12px; }

#justrelax-performance .jp-table { overflow-x: auto; }
#justrelax-performance .jp-table table { width: 100%; margin: 0; font-size: .8rem; }
#justrelax-performance .jp-table td.num { font-family: var(--jp-mono); white-space: nowrap; }
#justrelax-performance .jp-desc { display: block; color: var(--jp-muted); font-size: .72rem; }
#justrelax-performance .jp-good { color: var(--jp-good); }
#justrelax-performance .jp-bad { color: var(--jp-bad); }
#justrelax-performance .jp-tag { margin-left: 6px; padding: 1px 6px; color: var(--jp-warn); border: 1px solid var(--jp-warn); border-radius: 999px; font: .65rem var(--jp-mono); text-transform: uppercase; }
#justrelax-performance .jp-empty { padding: 3rem 1rem; color: var(--jp-muted); text-align: center; }

@media (max-width: 900px) {
  #justrelax-performance .jp-controls { grid-template-columns: 1fr; }
  #justrelax-performance .jp-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
