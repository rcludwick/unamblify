// unamblify — Copyright (c) 2026 Rob Ludwick.
// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.
//
// The dashboard. Vanilla ES2020, no build step. Hash-routed pages: Runs,
// New run, Capture, Samples (the prepared ⋈ captured navigator: clean /
// chip / model rows with waveforms, spectrograms, differences and an A/B
// play head), Run (loss / LR / sys / eval charts, logs, config,
// checkpoints, compare panel), Compare, Host. Talks to /api/* (spec §8)
// and listens on the two SSE streams. With a --token the token is kept in
// localStorage and sent as a Bearer header; SSE is read through fetch so
// the header can be sent (EventSource cannot).

'use strict';

// ── tiny DOM + format helpers ─────────────────────────────────────────

const $ = (sel, root = document) => root.querySelector(sel);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'html') el.innerHTML = v;
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, v);
    }
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };

const fmt = {
  num(v, d = 4) {
    if (v == null || Number.isNaN(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
    return Number(v.toFixed(a >= 100 ? 1 : a >= 10 ? 2 : d)).toString();
  },
  int(v) { return v == null ? '–' : Math.round(v).toLocaleString(); },
  pct(a, b) { return b ? `${Math.min(100, 100 * a / b).toFixed(1)}%` : '–'; },
  dur(s) {
    if (s == null || !Number.isFinite(s)) return '–';
    s = Math.max(0, Math.round(s));
    const d = Math.floor(s / 86400), hh = Math.floor(s % 86400 / 3600), mm = Math.floor(s % 3600 / 60), ss = s % 60;
    if (d) return `${d}d ${hh}h`;
    if (hh) return `${hh}h ${mm}m`;
    if (mm) return `${mm}m ${ss}s`;
    return `${ss}s`;
  },
  ago(iso) {
    if (!iso) return '–';
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return iso;
    return `${fmt.dur((Date.now() - t) / 1000)} ago`;
  },
  time(iso) {
    if (!iso) return '–';
    const t = Date.parse(iso);
    return Number.isNaN(t) ? iso : new Date(t).toLocaleString();
  },
  ms(t) { return new Date(t).toLocaleTimeString(); },
  bytes(b) {
    if (b == null) return '–';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return `${b.toFixed(i ? 1 : 0)} ${u[i]}`;
  },
};

function toast(msg, kind = '') {
  const el = h('div', { class: `toast ${kind}` }, msg);
  $('#toasts').append(el);
  setTimeout(() => el.remove(), kind === 'error' ? 7000 : 3500);
}

// Inline confirmation in place of window.confirm: swaps `anchor` for a
// yes/no strip and restores it afterwards.
function confirmInline(anchor, text, onYes, danger = false) {
  const strip = h('span', { class: `confirm ${danger ? 'danger' : ''}` },
    h('span', null, text),
    h('button', { class: `btn sm ${danger ? 'danger' : 'warn'}`, onclick: () => { restore(); onYes(); } }, 'Yes'),
    h('button', { class: 'btn sm', onclick: () => restore() }, 'No'));
  const parent = anchor.parentNode;
  parent.replaceChild(strip, anchor);
  function restore() { if (strip.parentNode) strip.parentNode.replaceChild(anchor, strip); }
  return restore;
}

// ── API client ────────────────────────────────────────────────────────

const auth = {
  get token() { try { return localStorage.getItem('unamblify.token') || ''; } catch { return ''; } },
  set token(t) { try { t ? localStorage.setItem('unamblify.token', t) : localStorage.removeItem('unamblify.token'); } catch { /* private mode */ } },
  headers() { const t = auth.token; return t ? { Authorization: `Bearer ${t}` } : {}; },
};

class ApiError extends Error {
  constructor(status, msg) { super(msg); this.status = status; }
}

async function api(method, path, body) {
  const opts = { method, headers: { ...auth.headers() } };
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  if (res.status === 401) { needToken(); throw new ApiError(401, 'unauthorized: set the token on the Host page'); }
  const ct = res.headers.get('content-type') || '';
  const payload = ct.includes('json') ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) throw new ApiError(res.status, (payload && payload.error) || `${res.status} ${res.statusText}`);
  return payload;
}
const GET = (p) => api('GET', p);
const POST = (p, b = {}) => api('POST', p, b);
const DEL = (p) => api('DELETE', p);

async function fetchBlob(path) {
  const res = await fetch(path, { headers: auth.headers() });
  if (!res.ok) throw new ApiError(res.status, `${res.status} fetching ${path}`);
  return res.blob();
}

let tokenPrompted = false;
function needToken() {
  if (tokenPrompted) return;
  tokenPrompted = true;
  toast('This server needs a bearer token. Set it on the Host page.', 'error');
  if (!location.hash.startsWith('#/host')) location.hash = '#/host';
}

// SSE over fetch (so the Authorization header goes along). `onEvent(name,
// data)`; reconnects with backoff; `close()` stops it.
function sse(url, onEvent, onState = () => {}) {
  let stopped = false, ctrl = null, backoff = 1000;
  (async () => {
    while (!stopped) {
      try {
        ctrl = new AbortController();
        const res = await fetch(url, { headers: { ...auth.headers(), Accept: 'text/event-stream' }, signal: ctrl.signal });
        if (res.status === 401) { onState('auth'); needToken(); return; }
        if (!res.ok || !res.body) throw new Error(`${res.status}`);
        onState('open');
        backoff = 1000;
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let i;
          while ((i = buf.indexOf('\n\n')) >= 0) {
            const chunk = buf.slice(0, i);
            buf = buf.slice(i + 2);
            let ev = 'message';
            const data = [];
            for (const line of chunk.split('\n')) {
              if (line.startsWith('event:')) ev = line.slice(6).trim();
              else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
            }
            if (!data.length) continue;
            let parsed = null;
            try { parsed = JSON.parse(data.join('\n')); } catch { parsed = data.join('\n'); }
            try { onEvent(ev, parsed); } catch (e) { console.error('event handler', ev, e); }
          }
        }
        onState('closed');
      } catch (e) {
        if (stopped) return;
        onState('closed');
      }
      await new Promise((r) => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 15000);
    }
  })();
  return { close() { stopped = true; if (ctrl) ctrl.abort(); } };
}

// ── charts (uPlot) ────────────────────────────────────────────────────

const PALETTE = ['#0a84ff', '#30d158', '#ff9f0a', '#bf5af2', '#64d2ff', '#ff453a', '#ffd60a', '#ff375f', '#8e8e93'];
const KEY_COLOR = {
  'loss/total': '#0a84ff', 'loss/stft': '#30d158', 'loss/mel': '#ff9f0a', 'loss/sisdr': '#bf5af2',
  'loss/onset': '#64d2ff', 'loss/tail': '#ff453a', 'lr': '#ff9f0a',
  'sys/steps_per_s': '#0a84ff', 'sys/samples_per_s': '#30d158', 'sys/cpu': '#ff9f0a', 'sys/gpu_util': '#bf5af2', 'sys/gpu_mem_gb': '#64d2ff',
  'eval/lsd': '#0a84ff', 'eval/lsd_first1s': '#30d158', 'eval/lsd_last1s': '#ff9f0a', 'eval/mel': '#bf5af2', 'eval/sisdr': '#64d2ff', 'eval/babble': '#ff453a',
};
let colorIdx = 0;
const colorFor = (key) => KEY_COLOR[key] || (KEY_COLOR[key] = PALETTE[colorIdx++ % PALETTE.length]);

const AXIS = { stroke: '#8e8e93', grid: { stroke: '#2c2c35', width: 1 }, ticks: { stroke: '#2c2c35', width: 1 }, font: '11px -apple-system, system-ui, sans-serif' };

// A chart over `keys` with one series per key; `data(key)` returns
// {step, v} arrays. `update()` re-joins and redraws (throttled).
function makeChart(container, opts = {}) {
  const el = h('div', { class: 'chart' });
  container.append(el);
  let u = null, keys = [], visible = new Set(), pending = false;
  const getData = opts.data;
  const width = () => Math.max(200, el.clientWidth || container.clientWidth || 600);
  function build() {
    if (u) { u.destroy(); u = null; }
    clear(el);
    if (!keys.length) { el.append(h('div', { class: 'empty' }, opts.emptyText || 'No data yet')); return; }
    const series = [{ label: 'step' }];
    for (const k of keys) series.push({ label: opts.label ? opts.label(k) : k, stroke: opts.color ? opts.color(k) : colorFor(k), width: 1.5, dash: opts.dash ? opts.dash(k) : undefined, spanGaps: true, show: visible.has(k), points: { show: opts.points || false, size: 5 } });
    u = new uPlot({
      width: width(), height: opts.height || 220,
      scales: { x: { time: false }, y: { distr: opts.log ? 3 : 1 } },
      axes: [{ ...AXIS }, { ...AXIS, size: 60, values: (_, t) => t.map((v) => fmt.num(v, 3)) }],
      series,
      legend: { show: false },
      cursor: { drag: { x: true, y: false }, sync: opts.sync ? { key: opts.sync } : undefined },
      hooks: { setCursor: [(uu) => opts.onCursor && opts.onCursor(uu)] },
    }, joined(), el);
  }
  function joined() {
    const tables = keys.map((k) => { const d = getData(k) || { step: [], v: [] }; return [d.step, d.v]; });
    if (!tables.length) return [[]];
    return uPlot.join(tables);
  }
  function update() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (!u) { build(); return; }
      u.setData(joined());
    });
  }
  const ro = new ResizeObserver(() => { if (u) u.setSize({ width: width(), height: opts.height || 220 }); });
  ro.observe(el);
  return {
    el,
    setKeys(ks, defaultVisible) {
      keys = ks.slice();
      if (defaultVisible) visible = new Set(ks.filter(defaultVisible));
      else for (const k of ks) if (!visible.has(k) && !this._touched) visible.add(k);
      build();
    },
    toggle(k) { if (visible.has(k)) visible.delete(k); else visible.add(k); this._touched = true; if (u) u.setSeries(keys.indexOf(k) + 1, { show: visible.has(k) }); },
    isVisible: (k) => visible.has(k),
    update,
    destroy() { ro.disconnect(); if (u) u.destroy(); },
    get keys() { return keys; },
    get u() { return u; },
  };
}

// A metric key's group: an `eval/lsd@dstar` per-mode column groups under
// its mode, everything else under '' (the whole-set columns first).
const modeOf = (k) => { const i = k.indexOf('@'); return i < 0 ? '' : k.slice(i + 1); };
const baseKey = (k) => { const i = k.indexOf('@'); return i < 0 ? k : k.slice(0, i); };

// Toggle buttons for a chart's series plus the latest value of each. With
// `groupOf`, the buttons are laid out one row per group (a mode), each
// labelled, and the per-group keys drop the `@mode` suffix.
function seriesToggles(chart, getLast, groupOf) {
  const el = h('div', { class: 'toggles' });
  const render = () => {
    clear(el);
    let last = null;
    for (const k of chart.keys) {
      const g = groupOf ? groupOf(k) : '';
      if (groupOf && g !== last) { el.append(h('span', { class: 'muted small mono togglegroup', style: { flexBasis: '100%', marginTop: last === null ? '0' : '4px' } }, g ? `@${g}` : 'all clips')); last = g; }
      const label = groupOf && g ? baseKey(k).replace(/^(loss|eval|sys)\//, '') : k.replace(/^(loss|eval|sys)\//, '');
      const b = h('button', { class: `btn sm toggle ${chart.isVisible(k) ? 'on' : ''}`, onclick: () => { chart.toggle(k); render(); } },
        h('i', { class: 'swatch', style: { background: colorFor(k) } }), label,
        h('span', { class: 'muted mono' }, ` ${fmt.num(getLast(k))}`));
      el.append(b);
    }
  };
  render();
  return { el, render };
}

// ── router ────────────────────────────────────────────────────────────

const app = $('#app');
let current = null;   // { leave() }
const pages = {};

function route() {
  const hash = location.hash || '#/runs';
  const [pathPart, queryPart] = hash.slice(1).split('?');
  const parts = pathPart.split('/').filter(Boolean);
  const q = new URLSearchParams(queryPart || '');
  const name = parts[0] || 'runs';
  const page = pages[name] || pages.runs;
  if (current && current.leave) { try { current.leave(); } catch (e) { console.error(e); } }
  current = null;
  for (const a of document.querySelectorAll('#nav a')) a.classList.toggle('active', a.dataset.page === (name === 'run' ? 'runs' : name));
  clear(app);
  window.scrollTo(0, 0);
  try { current = page(parts.slice(1), q) || {}; }
  catch (e) { console.error(e); app.append(h('div', { class: 'card' }, h('h2', null, 'Error'), h('pre', null, String(e && e.stack || e)))); }
}
window.addEventListener('hashchange', route);

// ── global event stream (runs + capture) ──────────────────────────────

const bus = new EventTarget();
const connEl = $('#conn');
function setConn(state) {
  const dot = $('.dot', connEl), txt = connEl.lastElementChild;
  dot.className = `dot ${state === 'open' ? 'on' : state === 'closed' ? 'off' : ''}`;
  txt.textContent = state === 'open' ? 'live' : state === 'auth' ? 'token needed' : 'reconnecting';
}
// Every (re)connect after the first means a gap: pages that show live
// state refetch on `reconnect` (the streams carry no replay).
let globalOpens = 0;
sse('/api/events', (ev, data) => bus.dispatchEvent(new CustomEvent(ev, { detail: data })), (state) => {
  setConn(state);
  if (state === 'open' && globalOpens++ > 0) bus.dispatchEvent(new CustomEvent('reconnect'));
});

// Shared pieces.
const pill = (state, text) => h('span', { class: `pill ${state || 'unknown'}` }, text || state || 'unknown');
const runLink = (r) => h('a', { href: `#/run/${encodeURIComponent(r.id)}`, class: 'clip', title: r.id }, r.name || r.id);
const progressBar = (a, b, cls = '') => h('div', { class: `bar ${cls}` }, h('i', { style: { width: b ? `${Math.min(100, 100 * a / b)}%` : '0%' } }));

// ── page: Runs ────────────────────────────────────────────────────────

pages.runs = () => {
  const head = h('div', { class: 'page-head' }, h('h1', null, 'Runs'), h('a', { class: 'btn primary', href: '#/new' }, 'New run'));
  const card = h('div', { class: 'card' });
  app.append(head, card);
  let timer = null;
  async function load() {
    // Never rebuild under an open confirm strip: a status write from any
    // live run would otherwise wipe the Yes/No the user is looking at.
    if (card.querySelector('.confirm')) return;
    let rows;
    try { rows = await GET('/api/runs'); } catch (e) { clear(card).append(h('div', { class: 'empty' }, `Could not load runs: ${e.message}`)); return; }
    if (card.querySelector('.confirm')) return;
    clear(card);
    if (!rows.length) { card.append(h('div', { class: 'empty' }, 'No runs yet. Start one from New run.')); return; }
    const tbody = h('tbody');
    for (const r of rows) {
      const st = r.status || {};
      const cfg = r.config || {};
      const actions = h('div', { class: 'btngroup' });
      if (r.supervised) {
        const b = h('button', { class: 'btn sm warn' }, 'Stop');
        b.onclick = () => confirmInline(b, `Stop ${r.name}? A checkpoint is written first.`, async () => {
          try { await POST(`/api/runs/${encodeURIComponent(r.id)}/stop`); toast(`Stopped ${r.name}`, 'ok'); } catch (e) { toast(e.message, 'error'); }
          load();
        });
        actions.append(b);
      } else if (st.status === 'queued') {
        actions.append(h('button', { class: 'btn sm good', onclick: async () => { try { await POST(`/api/runs/${encodeURIComponent(r.id)}/resume`); toast(`Started ${r.name}`, 'ok'); } catch (e) { toast(e.message, 'error'); } load(); } }, 'Start'));
      } else if (st.status === 'stopped' || st.status === 'failed') {
        actions.append(h('button', { class: 'btn sm good', onclick: async () => { try { await POST(`/api/runs/${encodeURIComponent(r.id)}/resume`); toast(`${r.checkpoints > 0 ? 'Resumed' : 'Started'} ${r.name}`, 'ok'); } catch (e) { toast(e.message, 'error'); } load(); } }, r.checkpoints > 0 ? 'Resume' : 'Start over'));
        const d = h('button', { class: 'btn sm danger' }, 'Delete');
        d.onclick = () => confirmInline(d, `Delete ${r.name} and its checkpoints?`, async () => {
          try { await DEL(`/api/runs/${encodeURIComponent(r.id)}`); toast(`Deleted ${r.name}`, 'ok'); } catch (e) { toast(e.message, 'error'); }
          load();
        }, true);
        actions.append(d);
      } else if (st.status === 'finished') {
        const d = h('button', { class: 'btn sm danger' }, 'Delete');
        d.onclick = () => confirmInline(d, `Delete ${r.name} and its checkpoints?`, async () => {
          try { await DEL(`/api/runs/${encodeURIComponent(r.id)}`); } catch (e) { toast(e.message, 'error'); }
          load();
        }, true);
        actions.append(d);
      }
      tbody.append(h('tr', null,
        h('td', null, h('div', { class: 'stack', style: { gap: '0' } }, runLink(r), h('span', { class: 'muted small mono' }, r.id))),
        h('td', null, pill(st.status)),
        h('td', { class: 'nowrap' }, h('div', { class: 'row', style: { gap: '8px', flexWrap: 'nowrap' } }, progressBar(st.step, st.total_steps, st.status === 'finished' ? 'green' : ''), h('span', { class: 'small mono' }, `${fmt.int(st.step)} / ${fmt.int(st.total_steps)}`))),
        h('td', { class: 'num mono' }, fmt.num(r.last_loss)),
        h('td', null, cfg.model ? `${cfg.model.profile} ll${cfg.model.lookahead}` : '–'),
        h('td', null, cfg.data ? `${cfg.data.mode} · ${cfg.data.source}` : '–'),
        h('td', null, st.device || '–'),
        h('td', { class: 'num' }, r.checkpoints),
        h('td', { class: 'nowrap muted small', title: st.updated }, fmt.ago(st.updated)),
        h('td', null, actions)));
    }
    card.append(h('div', { class: 'tablewrap' }, h('table', null,
      h('thead', null, h('tr', null, h('th', null, 'Run'), h('th', null, 'Status'), h('th', null, 'Progress'), h('th', { class: 'num' }, 'loss/total'), h('th', null, 'Model'), h('th', null, 'Data'), h('th', null, 'Device'), h('th', { class: 'num' }, 'Ckpts'), h('th', null, 'Updated'), h('th', null, ''))),
      tbody)));
  }
  const onRuns = () => { clearTimeout(timer); timer = setTimeout(load, 200); };
  bus.addEventListener('runs', onRuns);
  bus.addEventListener('reconnect', onRuns);
  load();
  const tick = setInterval(load, 15000);
  return { leave() { bus.removeEventListener('runs', onRuns); bus.removeEventListener('reconnect', onRuns); clearInterval(tick); clearTimeout(timer); } };
};

// ── page: New run ─────────────────────────────────────────────────────

pages.new = () => {
  app.append(h('div', { class: 'page-head' }, h('h1', null, 'New run')));
  const sel = h('select', null, h('option', { value: '' }, 'Choose a config…'));
  const ta = h('textarea', { rows: 18, placeholder: 'name = "smoke"\n[model]\nprofile = "lite"\nlookahead = 5\n[train]\nsteps = 20\ndevice = "cpu"\n' });
  const name = h('input', { type: 'text', placeholder: '(from config)' });
  const device = h('select', null, h('option', { value: '' }, '(from config)'));
  const steps = h('input', { type: 'number', min: 1, placeholder: '(from config)' });
  const out = h('div', { class: 'stack' });
  const startBtn = h('button', { class: 'btn primary', disabled: true }, 'Start run');
  const validateBtn = h('button', { class: 'btn' }, 'Validate');
  const left = h('div', { class: 'card stack' }, h('h2', null, 'Config'),
    h('div', { class: 'row' }, sel, h('span', { class: 'muted small' }, 'from the configs directory, or paste TOML below')), ta);
  const right = h('div', { class: 'card stack' }, h('h2', null, 'Overrides'),
    h('div', { class: 'form' }, h('label', { class: 'field' }, 'Name', name), h('label', { class: 'field' }, 'Device', device), h('label', { class: 'field' }, 'Steps', steps)),
    h('div', { class: 'row' }, validateBtn, startBtn), out);
  app.append(h('div', { class: 'grid two' }, left, right));

  GET('/api/configs').then((cfgs) => { for (const c of cfgs) sel.append(h('option', { value: c.name }, `${c.name}${c.ok ? '' : ' (invalid)'}`)); }).catch((e) => toast(e.message, 'error'));
  GET('/api/health').then((hh) => { for (const d of hh.devices || []) device.append(h('option', { value: d.name }, `${d.name} — ${d.note}`)); }).catch(() => {});
  sel.onchange = async () => {
    if (!sel.value) return;
    try { const c = await GET(`/api/configs/${encodeURIComponent(sel.value)}`); ta.value = c.toml; validate(); } catch (e) { toast(e.message, 'error'); }
  };
  const overrides = () => { const o = {}; if (name.value.trim()) o.name = name.value.trim(); if (device.value) o.device = device.value; if (steps.value) o.steps = Number(steps.value); return o; };
  async function validate() {
    clear(out);
    startBtn.disabled = true;
    if (!ta.value.trim()) return;
    try {
      const r = await POST('/api/configs/validate', { toml: ta.value });
      const c = r.config, o = overrides();
      const dl = h('dl', { class: 'kv' });
      const kv = (k, v) => dl.append(h('dt', null, k), h('dd', null, v));
      kv('name', o.name || c.name); kv('model', `${c.model.profile}, lookahead ${c.model.lookahead}`);
      kv('data', `${c.data.source}${c.data.shards ? ` (${c.data.shards})` : ''}, ${c.data.mode}, crop ${c.data.crop_s} s`);
      kv('train', `${fmt.int(o.steps || c.train.steps)} steps · batch ${c.train.batch} · lr ${c.train.lr} · ${o.device || c.train.device}${c.train.gan ? ' · GAN' : ''}${c.train.amp ? ' · AMP' : ''}`);
      kv('ckpt / eval', `every ${c.ckpt.every_steps} (keep ${c.ckpt.keep}) / every ${c.eval.every_steps}, ${c.eval.max_items} clips`);
      out.append(h('div', { class: 'row' }, pill('ok', 'valid')), dl);
      startBtn.disabled = false;
    } catch (e) { out.append(h('div', { class: 'row' }, pill('bad', 'invalid'), h('span', { class: 'mono small' }, e.message))); }
  }
  validateBtn.onclick = validate;
  let vt = null;
  ta.oninput = () => { clearTimeout(vt); vt = setTimeout(validate, 400); };
  startBtn.onclick = async () => {
    startBtn.disabled = true;
    try { const r = await POST('/api/runs', { toml: ta.value, overrides: overrides() }); toast(`Started ${r.name}`, 'ok'); location.hash = `#/run/${encodeURIComponent(r.id)}`; }
    catch (e) { toast(e.message, 'error'); startBtn.disabled = false; }
  };
  return {};
};

// ── page: Capture ─────────────────────────────────────────────────────

pages.capture = () => {
  app.append(h('div', { class: 'page-head' }, h('h1', null, 'Capture'), h('span', { class: 'muted small' }, 'Encode → decode of the prepared corpus: AMBE modes through the ThumbDV (one mode per stick), Codec 2 modes (M17) in software. Decode-only siblings (unamblify augment) are listed read-only.')));
  const vsCard = h('div', { class: 'card stack' });
  app.append(vsCard);
  const grid = h('div', { class: 'grid' });
  app.append(grid);
  const cards = {};
  // The voice-set matrix: which corpora each capture set holds. Refreshed
  // on a slower beat than the cards; the server caches per manifest.
  async function loadVoiceSets() {
    let v;
    try { v = await GET('/api/voicesets'); }
    catch (e) { clear(vsCard).append(h('h2', null, 'Voice sets'), h('div', { class: 'empty' }, e.message)); return; }
    const sets = (v.sets || []).filter((s) => s.utterances > 0);
    clear(vsCard);
    vsCard.append(h('div', { class: 'card-head' }, h('h2', null, 'Voice sets'), h('span', { class: 'muted small' }, 'corpora captured per mode — utterances, hover a cell for hours')));
    if (!sets.length) { vsCard.append(h('div', { class: 'empty' }, 'nothing captured yet')); return; }
    const corpora = [...new Set(sets.flatMap((s) => s.corpora.map((c) => c.corpus)))].sort();
    const cell = (s, corpus) => s.corpora.find((c) => c.corpus === corpus);
    const head = h('tr', null, h('th', { class: 'l' }, 'corpus'),
      ...sets.map((s) => h('th', { title: s.name }, s.label, s.kind ? h('span', { class: 'muted small' }, ' +' + s.kind) : null)));
    const bodyRows = corpora.map((corpus) => h('tr', null,
      h('td', { class: 'mono l' }, corpus),
      ...sets.map((s) => { const c = cell(s, corpus); return h('td', c ? { title: fmt.num(c.hours, 1) + ' h' } : null, c ? fmt.int(c.utterances) : '–'); })));
    const totalRow = h('tr', { class: 'total' }, h('td', { class: 'l' }, 'total'),
      ...sets.map((s) => h('td', { title: fmt.num(s.hours, 1) + ' h' }, fmt.int(s.utterances))));
    vsCard.append(h('div', { class: 'tablewrap' }, h('table', { class: 'vsmatrix' }, h('thead', null, head), h('tbody', null, ...bodyRows, totalRow))));
  }
  const logStick = {}; // name -> false once the reader scrolls up; default follows the tail
  let timer = null;
  async function load() {
    let v;
    try { v = await GET('/api/capture?log_tail=40'); } catch (e) { clear(grid).append(h('div', { class: 'card empty' }, e.message)); return; }
    for (const m of v.modes) render(m);
  }
  function render(m) {
    const s = m.status || {};
    const name = m.name || m.mode;
    // `stopping`: stop requested and the utterance in flight is finishing, or the final status is
    // written and the process is still exiting. No controls then: a second stop changes nothing.
    const state = m.stopping ? 'stopping' : m.alive ? (s.state || 'running') : (s.state === 'error' ? 'error' : 'idle');
    const card = cards[name] || (cards[name] = h('div', { class: 'card stack' }));
    if (!card.parentNode) grid.append(card);
    // Never re-render under an open confirm strip or a focused args field;
    // the next poll (or the confirm's own action) picks the update up.
    if (card.querySelector('.confirm') || (card.contains(document.activeElement) && document.activeElement.tagName === 'INPUT')) return;
    // The whole card is rebuilt on every poll, which would throw the log
    // panel back to its oldest lines each time. `logStick[name]` is the
    // reader's intent, not a per-render measurement: it stays true (follow
    // the tail) until the reader actually scrolls up, and turns true again
    // when they scroll back to the bottom. Measuring position each render
    // instead was fragile — a fresh panel's long lines wrap after the pin,
    // leaving it short of the bottom, which then read as "scrolled up" and
    // stuck there.
    const oldLog = card.querySelector('.logtail');
    const logKeepTop = oldLog ? oldLog.scrollTop : 0;
    const stick = logStick[name] !== false;
    clear(card);
    const eta = s.eta_s != null ? fmt.dur(s.eta_s) : '–';
    const actions = h('div', { class: 'btngroup' });
    const act = async (verb) => { try { await POST(`/api/capture/${m.mode}/${verb}`); toast(`${m.mode}: ${verb}`, 'ok'); } catch (e) { toast(e.message, 'error'); } setTimeout(load, 300); };
    if (m.readonly) {
      // A decode-only sibling: `unamblify augment --mode M --kind K` writes it; pause / stop through its own control.json.
      actions.append(h('span', { class: 'muted small' }, `decode-only sibling of ${m.mode}: unamblify augment --mode ${m.mode} --kind ${m.kind} (read-only here)`));
    } else if (m.stopping) {
      actions.append(h('span', { class: 'muted small' }, ['stopped', 'done', 'error'].includes(s.state) ? 'stopped; the harness is exiting' : 'stopping after the utterance in flight'));
    } else if (m.alive) {
      if (s.state === 'paused' || m.control === 'pause') actions.append(h('button', { class: 'btn sm good', onclick: () => act('resume') }, 'Resume'));
      else { const b = h('button', { class: 'btn sm warn' }, 'Pause'); b.onclick = () => confirmInline(b, 'Pause after the current utterance?', () => act('pause')); actions.append(b); }
      const st = h('button', { class: 'btn sm danger' }, 'Stop');
      st.onclick = () => confirmInline(st, 'Stop after the current utterance? Resume later picks up where it left off.', () => act('stop'), true);
      actions.append(st);
    } else {
      const args = h('input', { type: 'text', placeholder: m.software ? 'extra args, e.g. --corpus vctk --limit 200' : 'extra args, e.g. --port /dev/cu.usbserial-X --limit 200', class: 'mono small', style: { minWidth: '220px' } });
      // Software modes take --jobs N (encoder threads); the harness defaults to the physical cores.
      const jobs = m.software ? h('input', { type: 'number', min: '1', step: '1', placeholder: 'jobs', title: '--jobs: encoder threads (default: physical cores)', class: 'mono small', style: { width: '70px' } }) : null;
      actions.append(args, jobs, h('button', { class: 'btn sm primary', onclick: async () => {
        const a = args.value.trim() ? args.value.trim().split(/\s+/) : [];
        if (jobs && jobs.value.trim()) a.push('--jobs', jobs.value.trim());
        try { await POST(`/api/capture/${m.mode}/start`, { args: a }); toast(`${m.mode}: started`, 'ok'); } catch (e) { toast(e.message, 'error'); }
        setTimeout(load, 400);
      } }, 'Start'));
    }
    const canary = s.canary_ok == null ? pill('unknown', 'canary –') : s.canary_ok ? pill('ok', 'canary ok') : pill('bad', 'canary mismatch');
    const badge = m.readonly ? pill('queued', `augment · ${m.kind}`) : m.software ? pill('info', 'software') : pill('unknown', 'chip');
    const workers = (s.ports || []).length ? (m.software ? `${s.ports.length} thread${s.ports.length === 1 ? '' : 's'}` : s.ports.join(', ')) : (s.port || '–');
    card.append(
      h('div', { class: 'card-head' }, h('h2', null, m.label || name), h('span', { class: 'mono small muted' }, name), badge, pill(state === 'error' ? 'failed' : state, state + (m.supervised ? '' : m.alive ? ' (external)' : '')), canary),
      h('div', { class: 'row', style: { gap: '10px' } }, progressBar(s.done || m.manifest_rows || 0, s.total || 0, 'big green'), h('span', { class: 'mono small nowrap' }, `${fmt.int(s.done != null ? s.done : m.manifest_rows)} / ${fmt.int(s.total)}${s.failed ? ` · ${s.failed} failed` : ''}`)),
      h('div', { class: 'stats' },
        h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.num(s.frames_s, 1)), h('span', { class: 'k' }, 'frames / s')),
        h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.num(s.utt_per_hour, 0)), h('span', { class: 'k' }, 'utt / hour')),
        h('div', { class: 'stat' }, h('span', { class: 'v' }, eta), h('span', { class: 'k' }, 'eta')),
        h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.pct(s.done, s.total)), h('span', { class: 'k' }, 'done'))),
      h('dl', { class: 'kv' },
        h('dt', null, 'current'), h('dd', { class: 'mono' }, s.current_key || '–'),
        h('dt', null, m.software ? 'workers' : 'port'), h('dd', { class: 'mono' }, workers),
        h('dt', null, m.software ? 'codec' : 'chip'), h('dd', { class: 'mono small' }, [s.prodid, s.version].filter(Boolean).join(' ') || (m.software ? `codec2 · ${m.frame_ms} ms frames` : '–')),
        h('dt', null, 'control'), h('dd', null, m.control || '–', m.pid ? h('span', { class: 'muted' }, ` · pid ${m.pid}`) : null),
        h('dt', null, 'updated'), h('dd', { title: s.updated }, fmt.ago(s.updated)),
        ...(s.paused_s > 0 ? [h('dt', null, 'paused'), h('dd', { title: 'excluded from the rates and ETA' }, `${fmt.dur(s.paused_s)} total`)] : [])),
      actions,
      h('div', { class: 'logtail' }, (m.log_tail || []).length ? m.log_tail.map((r) => h('div', { class: r.level }, h('span', { class: 't' }, fmt.ms(r.t)), ' ', r.msg)) : h('span', { class: 'muted' }, 'no log yet')));
    const newLog = card.querySelector('.logtail');
    if (newLog) {
      const atBottom = () => newLog.scrollHeight - newLog.scrollTop - newLog.clientHeight < 24;
      // A real scroll updates the intent; the pin below fires this too,
      // harmlessly reasserting stick = true.
      newLog.addEventListener('scroll', () => { logStick[name] = atBottom(); });
      if (stick) {
        newLog.scrollTop = newLog.scrollHeight;
        // Long lines wrap a frame later and grow the panel; re-pin once
        // layout settles so the newest line is actually in view.
        requestAnimationFrame(() => { if (logStick[name] !== false) newLog.scrollTop = newLog.scrollHeight; });
      } else {
        newLog.scrollTop = logKeepTop;
      }
    }
  }
  const onCap = () => { clearTimeout(timer); timer = setTimeout(load, 250); };
  bus.addEventListener('capture', onCap);
  bus.addEventListener('reconnect', onCap);
  load();
  loadVoiceSets();
  const tick = setInterval(load, 5000);
  const vsTick = setInterval(loadVoiceSets, 20000);
  return { leave() { bus.removeEventListener('capture', onCap); bus.removeEventListener('reconnect', onCap); clearInterval(tick); clearInterval(vsTick); clearTimeout(timer); } };
};

// ── page: Host ────────────────────────────────────────────────────────

pages.host = () => {
  app.append(h('div', { class: 'page-head' }, h('h1', null, 'Host')));
  const sysCard = h('div', { class: 'card stack' }, h('h2', null, 'System'));
  const tokenInput = h('input', { type: 'password', value: auth.token, placeholder: 'bearer token', autocomplete: 'off' });
  const tokenCard = h('div', { class: 'card stack' }, h('h2', null, 'Access'),
    h('p', { class: 'muted small', style: { margin: 0 } }, 'When the server runs with --token, /api/* needs it. Stored in this browser only.'),
    h('div', { class: 'row' }, tokenInput, h('button', { class: 'btn primary', onclick: () => { auth.token = tokenInput.value.trim(); tokenPrompted = false; toast('Token saved; reloading', 'ok'); setTimeout(() => location.reload(), 300); } }, 'Save'),
      h('button', { class: 'btn', onclick: () => { auth.token = ''; tokenInput.value = ''; toast('Token cleared'); } }, 'Clear')));
  const devCard = h('div', { class: 'card stack' }, h('h2', null, 'Devices'));
  app.append(h('div', { class: 'grid two' }, sysCard, h('div', { class: 'stack' }, tokenCard, devCard)));
  const driveCard = h('div', { class: 'card stack' }, h('div', { class: 'card-head' }, h('h2', null, 'Drive temperature')));
  app.append(driveCard);
  const driveData = {};   // device -> {step, v}, step = minutes ago (negative)
  const driveChart = makeChart(driveCard, { data: (k) => driveData[k], label: (k) => k.replace('/dev/', ''), height: 180, emptyText: 'No SMART readings yet' });
  const driveTable = h('div', { class: 'stack' });
  driveCard.append(driveTable);
  // What the drive is being asked to do, under what it costs in heat: the
  // two only explain each other on one time axis.
  driveCard.append(h('div', { class: 'card-head' }, h('h2', null, 'Drive I/O'), h('span', { class: 'muted small' }, 'MiB/s, averaged over each sample')));
  const ioData = {};      // "<disk> read" / "<disk> write" -> {step, v}
  const ioChart = makeChart(driveCard, { data: (k) => ioData[k], label: (k) => k, height: 160, emptyText: 'No I/O samples yet' });
  const ioNow = h('div', { class: 'stack' });
  driveCard.append(ioNow);
  // The hottest sensor is the one that matters: an enclosure can run a flash
  // die far above the controller's composite reading.
  const hottestOf = (d) => { const all = [d.composite_c, ...(d.sensors || [])].filter((x) => x != null); return all.length ? Math.max(...all) : null; };
  const cpuCard = h('div', { class: 'card stack' }, h('div', { class: 'card-head' }, h('h2', null, 'CPU / GPU')));
  app.append(cpuCard);
  const cpuData = {};   // series -> {step, v}, step = minutes ago (negative)
  // Both series are percentages, so one 0-100 axis is honest.
  const cpuChart = makeChart(cpuCard, { data: (k) => cpuData[k], label: (k) => k, height: 180, emptyText: 'No samples yet' });
  const cpuNow = h('div', { class: 'stack' });
  cpuCard.append(cpuNow);
  // The key takes the chart's own colour for each series, so it cannot
  // drift from the lines.
  const keyRow = (k, val) => h('tr', null,
    h('td', { style: { width: '14px' } }, h('i', { class: 'swatch', style: { background: colorFor(k), width: '10px', height: '10px', borderRadius: '2px', display: 'inline-block' } })),
    h('td', { class: 'mono' }, k),
    h('td', { class: 'num mono' }, val));
  async function load() {
    try {
      const s = await GET('/api/sys');
      clear(sysCard).append(h('h2', null, 'System'),
        h('div', { class: 'stats' },
          h('div', { class: 'stat' }, h('span', { class: 'v' }, `${fmt.num(s.cpu_percent, 0)}%`), h('span', { class: 'k' }, 'cpu')),
          h('div', { class: 'stat' }, h('span', { class: 'v' }, `${fmt.bytes(s.mem_used)}`), h('span', { class: 'k' }, `of ${fmt.bytes(s.mem_total)}`)),
          h('div', { class: 'stat' }, h('span', { class: 'v' }, s.load.map((x) => x.toFixed(1)).join(' ')), h('span', { class: 'k' }, 'load 1 / 5 / 15')),
          h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.dur(s.uptime_s)), h('span', { class: 'k' }, 'uptime'))),
        h('dl', { class: 'kv' },
          h('dt', null, 'host'), h('dd', null, s.host), h('dt', null, 'os'), h('dd', null, `${s.os} · ${s.kernel} · ${s.arch}`),
          h('dt', null, 'cpus'), h('dd', null, `${s.cpus} logical${s.physical_cores ? `, ${s.physical_cores} physical` : ''}`),
          h('dt', null, 'server'), h('dd', null, `pid ${s.pid}, up ${fmt.dur(s.server_uptime_s)}`),
          h('dt', null, 'exe'), h('dd', { class: 'mono small' }, s.exe), h('dt', null, 'runs'), h('dd', { class: 'mono small' }, s.runs_dir), h('dt', null, 'data'), h('dd', { class: 'mono small' }, s.data_root)));
      clear(devCard).append(h('h2', null, 'Devices'), h('div', { class: 'tablewrap' }, h('table', null, h('tbody', null, s.devices.map((d) => h('tr', null, h('td', { class: 'mono' }, d.name), h('td', { class: 'muted' }, d.note)))))));
    } catch (e) { clear(sysCard).append(h('h2', null, 'System'), h('div', { class: 'empty' }, e.message)); }
  }
  async function loadDrives() {
    let v;
    try { v = await GET('/api/drives'); } catch (e) { clear(driveTable).append(h('div', { class: 'empty' }, e.message)); return; }
    const keys = v.drives.filter((d) => hottestOf(d) != null).map((d) => d.device);
    const last = v.history.length ? v.history[v.history.length - 1].t : 0;
    for (const k of keys) driveData[k] = { step: [], v: [] };
    for (const s2 of v.history) for (const k of keys) {
      const c = s2.temps[k];
      if (c == null) continue;
      driveData[k].step.push((s2.t - last) / 60);
      driveData[k].v.push(c);
    }
    if (driveChart.keys.join() !== keys.join()) driveChart.setKeys(keys, () => true); else driveChart.update();
    // Only drives that have moved something: an idle card reader is two flat lines and two key rows of nothing.
    const ioDevs = [...new Set(v.history.flatMap((s2) => Object.entries(s2.io || {}).filter(([, io]) => io.r > 0 || io.w > 0).map(([d]) => d)))].sort();
    const ioKeys = ioDevs.flatMap((d) => [`${d.replace('/dev/', '')} read`, `${d.replace('/dev/', '')} write`]);
    // Start every series at the temperature chart's first sample, with a gap, so the two charts share a time axis
    // from the first minute instead of after a day.
    const t0 = v.history.length ? (v.history[0].t - last) / 60 : 0;
    for (const k of ioKeys) ioData[k] = { step: [t0], v: [null] };
    for (const s2 of v.history) for (const d of ioDevs) {
      const io = (s2.io || {})[d];
      if (!io) continue;
      const n = d.replace('/dev/', '');
      for (const [k, val] of [[`${n} read`, io.r], [`${n} write`, io.w]]) { ioData[k].step.push((s2.t - last) / 60); ioData[k].v.push(val); }
    }
    if (ioChart.keys.join() !== ioKeys.join()) ioChart.setKeys(ioKeys, () => true); else ioChart.update();
    const nowIo = v.history.length ? (v.history[v.history.length - 1].io || {}) : {};
    clear(ioNow).append(ioKeys.length ? h('div', { class: 'tablewrap' }, h('table', null, h('tbody', null, ioDevs.flatMap((d) => {
      const n = d.replace('/dev/', ''), io = nowIo[d];
      return [keyRow(`${n} read`, io ? `${fmt.num(io.r, 1)} MiB/s` : '--'), keyRow(`${n} write`, io ? `${fmt.num(io.w, 1)} MiB/s` : '--')];
    })))) : h('div', { class: 'muted small' }, 'Rates appear after the second sample.'));
    clear(driveTable).append(h('div', { class: 'tablewrap' }, h('table', null, h('tbody', null, v.drives.map((d) => {
      const c = hottestOf(d);
      return h('tr', null,
        h('td', { style: { width: '14px' } }, c == null ? '' : h('i', { class: 'swatch', style: { background: colorFor(d.device), width: '10px', height: '10px', borderRadius: '2px', display: 'inline-block' } })),
        h('td', { class: 'mono' }, d.device.replace('/dev/', '')),
        h('td', { class: 'clip' }, d.model),
        h('td', { class: 'num mono' }, c == null ? '--' : `${c} C`),
        h('td', { class: 'muted small' }, d.note || ''));
    })))));
  }
  async function loadCpu() {
    let v;
    try { v = await GET('/api/host'); } catch (e) { clear(cpuNow).append(h('div', { class: 'empty' }, e.message)); return; }
    const hs = v.history || [];
    const last = hs.length ? hs[hs.length - 1].t : 0;
    const step = hs.map((s2) => (s2.t - last) / 60);
    const anyGpu = hs.some((s2) => s2.gpu != null);
    cpuData['cpu %'] = { step, v: hs.map((s2) => s2.cpu) };
    if (anyGpu) cpuData['gpu %'] = { step, v: hs.map((s2) => (s2.gpu == null ? null : s2.gpu)) };
    const keys = anyGpu ? ['cpu %', 'gpu %'] : ['cpu %'];
    if (cpuChart.keys.join() !== keys.join()) cpuChart.setKeys(keys, () => true); else cpuChart.update();
    const cur = hs.length ? hs[hs.length - 1] : null;
    const span = hs.length > 1 ? (hs[hs.length - 1].t - hs[0].t) / 3600 : 0;
    // Load average is not a percentage, so it is a stat, not a line the
    // 0-100 scale would flatten.
    clear(cpuNow).append(
      h('div', { class: 'tablewrap' }, h('table', null, h('tbody', null,
        keyRow('cpu %', cur ? `${fmt.num(cur.cpu, 1)}%` : '--'),
        ...(anyGpu ? [keyRow('gpu %', cur && cur.gpu != null ? `${fmt.num(cur.gpu, 0)}%` : '--')] : [])))),
      h('div', { class: 'stats' },
        h('div', { class: 'stat' }, h('span', { class: 'v' }, cur ? fmt.num(cur.load1, 2) : '--'), h('span', { class: 'k' }, 'load 1m')),
        h('div', { class: 'stat' }, h('span', { class: 'v' }, cur && cur.gpu_mem_gb != null ? `${fmt.num(cur.gpu_mem_gb, 1)} GB` : '--'), h('span', { class: 'k' }, 'gpu memory')),
        h('div', { class: 'stat' }, h('span', { class: 'v' }, `${fmt.num(span, 1)} h`), h('span', { class: 'k' }, `${hs.length} samples`))));
  }
  load();
  loadDrives();
  loadCpu();
  const tick = setInterval(load, 5000);
  const dtick = setInterval(() => { loadDrives(); loadCpu(); }, 30000);
  return { leave() { clearInterval(tick); clearInterval(dtick); driveChart.destroy(); ioChart.destroy(); cpuChart.destroy(); } };
};

// ── page: Compare (overlay one metric across runs) ────────────────────

pages.compare = (_parts, q) => {
  app.append(h('div', { class: 'page-head' }, h('h1', null, 'Compare')));
  const picked = new Set((q.get('ids') || '').split(',').filter(Boolean));
  let key = q.get('key') || 'loss/total';
  const list = h('div', { class: 'card stack' }, h('h2', null, 'Runs'));
  const keySel = h('select');
  const chartCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Overlay'), keySel));
  const legend = h('div', { class: 'legend' });
  app.append(h('div', { class: 'grid', style: { gridTemplateColumns: 'minmax(260px, 1fr) minmax(0, 3fr)' } }, list, chartCard));
  chartCard.append(legend);
  const data = {};   // id -> {step, v}
  const chart = makeChart(chartCard, { data: (id) => data[id], label: (id) => id, color: (id) => colorFor(`run:${id}`), emptyText: 'Pick runs on the left' });
  let runs = [];
  async function loadRuns() {
    runs = await GET('/api/runs');
    clear(list).append(h('h2', null, 'Runs'));
    for (const r of runs) {
      const cb = h('input', { type: 'checkbox', checked: picked.has(r.id) });
      cb.onchange = () => { if (cb.checked) picked.add(r.id); else picked.delete(r.id); sync(); refresh(); };
      list.append(h('label', { class: 'row', style: { gap: '8px' } }, cb, h('i', { class: 'swatch', style: { background: colorFor(`run:${r.id}`), width: '10px', height: '10px', borderRadius: '2px', display: 'inline-block' } }), h('span', { class: 'clip' }, r.name), pill((r.status || {}).status)));
    }
  }
  function sync() { const qq = new URLSearchParams(); if (picked.size) qq.set('ids', [...picked].join(',')); qq.set('key', key); history.replaceState(null, '', `#/compare?${qq}`); }
  async function refresh() {
    const ids = [...picked];
    const keys = new Set();
    await Promise.all(ids.map(async (id) => {
      try { const m = await GET(`/api/runs/${encodeURIComponent(id)}/metrics?keys=${encodeURIComponent(key)}`); data[id] = m.series[key] || { step: [], v: [] }; } catch { data[id] = { step: [], v: [] }; }
      try { const info = await GET(`/api/runs/${encodeURIComponent(id)}`); for (const k of info.metric_keys || []) keys.add(k); } catch { /* ignore */ }
    }));
    const opts = [...keys].sort();
    if (!opts.includes(key)) opts.unshift(key);
    clear(keySel);
    for (const k of opts) keySel.append(h('option', { value: k, selected: k === key }, k));
    chart.setKeys(ids);
    clear(legend);
    for (const id of ids) { const r = runs.find((x) => x.id === id); legend.append(h('span', null, h('i', { style: { background: colorFor(`run:${id}`) } }), r ? r.name : id, h('span', { class: 'muted' }, ` last ${fmt.num((data[id].v || []).slice(-1)[0])}`))); }
  }
  keySel.onchange = () => { key = keySel.value; sync(); refresh(); };
  loadRuns().then(refresh).catch((e) => toast(e.message, 'error'));
  return { leave() { chart.destroy(); } };
};

// ── colour maps (hand-coded LUTs) ─────────────────────────────────────

// Viridis, 11 anchor stops, linearly interpolated to 256 entries.
const VIRIDIS_STOPS = [[68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112], [122, 209, 81], [189, 223, 38], [253, 231, 37]];
// Diverging for differences: teal (negative) → ink (zero) → orange (positive).
const DIVERGING_STOPS = [[100, 210, 255], [40, 100, 140], [22, 22, 28], [150, 95, 20], [255, 159, 10]];
function buildLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const p = i / 255 * (stops.length - 1), j = Math.min(stops.length - 2, Math.floor(p)), f = p - j;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[j][c] + (stops[j + 1][c] - stops[j][c]) * f;
  }
  return lut;
}
const LUT_VIRIDIS = buildLut(VIRIDIS_STOPS);
const LUT_DIVERGING = buildLut(DIVERGING_STOPS);
function lutCss(lut) {
  const st = [];
  for (let i = 0; i <= 8; i++) { const k = Math.round(i / 8 * 255) * 3; st.push(`rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]}) ${i / 8 * 100}%`); }
  return `linear-gradient(90deg, ${st.join(', ')})`;
}

// spec.json → {mels, frames, mats[name] -> Float32Array[mels*frames] in
// [mel][frame] order, framesOf[name]}. Every array-valued key is a matrix
// (a checkpoint's clean / degraded / out; a sample's clean16 or
// degraded-dstar; an infer output's out). The trainer writes [frame][mel]
// with `frames` and `n_mels` declared; the declared shape decides the
// orientation (a clip with exactly n_mels frames would otherwise be
// guessed wrong), and the n_mels heuristic is only for files that declare
// neither. Flat arrays with n_mels/frames given are accepted too.
// Matrices of different lengths keep their own frame count in framesOf;
// `frames` is the longest.
function parseSpec(spec) {
  const nMels = spec.n_mels || spec.mels || null;
  const nFrames = spec.frames || null;
  // The trainer writes `hop` (samples) and `rate` (Hz); `hop_s` is accepted too.
  const hopS = spec.hop_s || (spec.hop && spec.rate ? spec.hop / spec.rate : 0.01);
  const out = { mels: nMels, frames: spec.frames || null, dbMin: spec.db_min, dbMax: spec.db_max, hopS, mats: {}, framesOf: {}, speechFrames: spec.speech_frames || null };
  for (const [name, m] of Object.entries(spec)) {
    if (!Array.isArray(m) || !m.length) continue;
    let mels, frames, get;
    if (Array.isArray(m[0])) {
      const frameMajor = nFrames && m.length === nFrames && m[0].length === (nMels || m[0].length) ? true
        : nMels && m.length === nMels && m[0].length === (nFrames || m[0].length) ? false
          : Boolean(nMels && m.length !== nMels && m[0].length === nMels);
      if (frameMajor) { frames = m.length; mels = m[0].length; get = (r, c) => m[c][r]; }
      else { mels = m.length; frames = m[0].length; get = (r, c) => m[r][c]; }
    } else {
      mels = nMels || 80; frames = Math.floor(m.length / mels);
      get = spec.layout === 'frame_major' ? (r, c) => m[c * mels + r] : (r, c) => m[r * frames + c];
    }
    const a = new Float32Array(mels * frames);
    for (let r = 0; r < mels; r++) for (let c = 0; c < frames; c++) a[r * frames + c] = get(r, c);
    out.mats[name] = a;
    out.framesOf[name] = frames;
    out.mels = mels; out.frames = Math.max(out.frames || 0, frames);
  }
  if (out.dbMin == null || out.dbMax == null) {
    let lo = Infinity, hi = -Infinity;
    for (const a of Object.values(out.mats)) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    out.dbMin = out.dbMin == null ? lo : out.dbMin; out.dbMax = out.dbMax == null ? hi : out.dbMax;
  }
  return out;
}

// a − b over the frames both have ([mel][frame] arrays of `mels` rows,
// aFrames / bFrames columns): {d, frames, m} with m the symmetric range
// for the diverging LUT, clamped to [1, 40] dB.
function diffMat(a, aFrames, b, bFrames, mels) {
  const frames = Math.min(aFrames, bFrames);
  const d = new Float32Array(mels * frames);
  let m = 0;
  for (let r = 0; r < mels; r++) for (let c = 0; c < frames; c++) {
    const v = a[r * aFrames + c] - b[r * bFrames + c];
    d[r * frames + c] = v;
    const x = Math.abs(v);
    if (x > m) m = x;
  }
  return { d, frames, m: Math.min(Math.max(m, 1), 40) };
}

// Fill a `.cbar` strip with `lut` and its label row with the range.
function colorBar(cbar, cbarLbl, lut, lo, hi, text) {
  cbar.style.background = lutCss(lut);
  clear(cbarLbl).append(h('span', null, `${fmt.num(lo, 1)} dB`), h('span', { class: 'spacer' }), h('span', null, text), h('span', { class: 'spacer' }), h('span', null, `${fmt.num(hi, 1)} dB`));
}

function paintSpec(canvas, data, mels, frames, lo, hi, lut) {
  const off = document.createElement('canvas');
  off.width = frames; off.height = mels;
  const ctx = off.getContext('2d');
  const img = ctx.createImageData(frames, mels);
  const span = hi - lo || 1;
  for (let r = 0; r < mels; r++) {
    const y = mels - 1 - r;   // low mel at the bottom
    for (let c = 0; c < frames; c++) {
      const v = (data[r * frames + c] - lo) / span;
      const k = Math.max(0, Math.min(255, Math.round(v * 255))) * 3;
      const o = (y * frames + c) * 4;
      img.data[o] = lut[k]; img.data[o + 1] = lut[k + 1]; img.data[o + 2] = lut[k + 2]; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const w = canvas.clientWidth || 300, hh = canvas.clientHeight || 180;
  canvas.width = w * devicePixelRatio; canvas.height = hh * devicePixelRatio;
  const g = canvas.getContext('2d');
  g.imageSmoothingEnabled = true;
  g.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function paintWave(canvas, samples) {
  const w = canvas.clientWidth || 300, hh = canvas.clientHeight || 180;
  canvas.width = w * devicePixelRatio; canvas.height = hh * devicePixelRatio;
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (!samples) return;
  const mid = canvas.height / 2, amp = canvas.height * 0.45, n = samples.length, per = n / canvas.width;
  g.fillStyle = 'rgba(255,255,255,0.55)';
  for (let x = 0; x < canvas.width; x++) {
    const a = Math.floor(x * per), b = Math.min(n, Math.floor((x + 1) * per) + 1);
    let lo = 1, hi = -1;
    for (let i = a; i < b; i++) { const v = samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo > hi) continue;
    g.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
  }
}

// ── compare panel (one checkpoint, one clip) ──────────────────────────

let audioCtx = null;
const getAudioCtx = () => (audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)());

function comparePanel(runId, ckpt) {
  const el = h('div', { class: 'card stack' });
  const clips = ckpt.clips || [];
  const clipSel = h('select');
  // `<clip>@<mode>`: a clip rendered for one of the run's modes.
  const clipLabel = (name) => { const i = name.lastIndexOf('@'); return i < 0 ? name : `${name.slice(0, i)}  ·  ${name.slice(i + 1)}`; };
  for (const c of clips) clipSel.append(h('option', { value: c.name }, clipLabel(c.name)));
  const modes = [['none', 'Spectrograms'], ['out-deg', 'out − degraded'], ['out-clean', 'out − clean']];
  let mode = 'none', showWave = false;
  const modeBtns = h('div', { class: 'btngroup' });
  const waveBtn = h('button', { class: 'btn sm toggle', onclick: () => { showWave = !showWave; waveBtn.classList.toggle('on', showWave); drawWaves(); } }, 'Waveform');
  const head = h('div', { class: 'card-head' }, h('h2', null, `Checkpoint step ${fmt.int(ckpt.step)}`), clips.length > 1 ? clipSel : h('span', { class: 'mono small' }, clips[0] ? clipLabel(clips[0].name) : ''), modeBtns, waveBtn);
  const specs = h('div', { class: 'specs' });
  const cbar = h('div', { class: 'cbar' });
  const cbarLbl = h('div', { class: 'row small muted' });
  const tailNote = h('p', { class: 'small muted tailnote', hidden: true });
  const audios = h('div', { class: 'audios' });
  el.append(head, specs, h('div', { class: 'stack', style: { gap: '4px' } }, cbar, cbarLbl, tailNote), audios);
  if (!clips.length) { el.append(h('div', { class: 'empty' }, 'No rendered clips in this checkpoint')); return el; }

  const base = `/api/runs/${encodeURIComponent(runId)}/checkpoints/${ckpt.dir}`;
  let spec = null, buffers = {}, urls = {}, dead = false;
  const canvases = {}, waves = {};
  const variants = ['clean', 'degraded', 'out'];

  function renderModeBtns() {
    clear(modeBtns);
    for (const [m, label] of modes) modeBtns.append(h('button', { class: `btn sm toggle ${mode === m ? 'on' : ''}`, onclick: () => { mode = m; renderModeBtns(); draw(); } }, label));
  }
  renderModeBtns();

  function draw() {
    clear(specs);
    if (!spec) { specs.append(h('div', { class: 'empty' }, 'Loading spectrograms…')); return; }
    const panes = mode === 'none' ? variants.map((v) => [v, spec.mats[v], LUT_VIRIDIS, spec.dbMin, spec.dbMax])
      : (() => {
        const ref = mode === 'out-deg' ? 'degraded' : 'clean';
        const a = spec.mats.out, b = spec.mats[ref];
        if (!a || !b) return [];
        const { d, m } = diffMat(a, spec.frames, b, spec.frames, spec.mels);
        return [[ref, spec.mats[ref], LUT_VIRIDIS, spec.dbMin, spec.dbMax], ['out', a, LUT_VIRIDIS, spec.dbMin, spec.dbMax], [`out − ${ref}`, d, LUT_DIVERGING, -m, m]];
      })();
    for (const [name, mat, lut, lo, hi] of panes) {
      if (!mat) continue;
      const c = h('canvas'), w = h('canvas', { class: 'wave' }), cur = h('div', { class: 'cursor' });
      // The eval appends synthetic key-down garbage after the real audio
      // (spec.speech_frames marks the boundary); show it, don't hide it.
      const hasTail = spec.speechFrames && spec.frames && spec.speechFrames < spec.frames;
      const lbl = name === 'degraded' && hasTail ? 'degraded · chip output, then synthetic key-down garbage' : name;
      const box = h('div', { class: 'spec' }, c, w, h('span', { class: 'lbl' }, lbl), cur);
      if (hasTail) box.append(h('div', { class: 'tailmark', style: { left: `${(100 * spec.speechFrames / spec.frames).toFixed(2)}%` }, title: 'end of the chip\'s audio; what follows is injected by the eval (target: silence)' }));
      specs.append(box);
      canvases[name] = c; waves[name] = w;
      requestAnimationFrame(() => { paintSpec(c, mat, spec.mels, spec.frames, lo, hi, lut); drawWaves(); });
      const v = variants.includes(name) ? name : 'out';
      box.onclick = (e) => { const r = box.getBoundingClientRect(); const t = (e.clientX - r.left) / r.width * (spec.frames * spec.hopS); playWindow(v, t, 1.0); };
      box.title = 'Click to play 1 s from here';
    }
    const last = panes[panes.length - 1];
    if (last) colorBar(cbar, cbarLbl, last[2], last[3], last[4], mode === 'none' ? 'shared dB range, log-mel' : 'difference, dB');
    tailNote.hidden = !(spec.speechFrames && spec.frames && spec.speechFrames < spec.frames);
    if (!tailNote.hidden) tailNote.textContent = `The dashed line marks the end of the chip's audio (${(spec.speechFrames * spec.hopS).toFixed(2)} s). The eval appends ${((spec.frames - spec.speechFrames) * spec.hopS).toFixed(1)} s of synthetic key-down garbage to the degraded input — noise, a stuck frame, or bursts, seeded per clip — with silence as the target; the babble metric scores what the model emits there. The Samples page plays the chip's raw output without it.`;
  }

  function drawWaves() {
    for (const [name, w] of Object.entries(waves)) {
      const v = variants.includes(name) ? name : 'out';
      const b = buffers[v];
      paintWave(w, showWave && b ? b.getChannelData(0) : null);
    }
  }

  function playWindow(v, start, len) {
    const b = buffers[v];
    if (!b) { toast('audio not decoded yet'); return; }
    const ctx = getAudioCtx();
    if (ctx.state === 'suspended') ctx.resume();
    const src = ctx.createBufferSource();
    src.buffer = b; src.connect(ctx.destination);
    const s = Math.max(0, Math.min(b.duration - 0.05, start));
    src.start(0, s, Math.min(len, b.duration - s));
  }

  async function loadClip(name) {
    clear(audios);
    for (const u of Object.values(urls)) URL.revokeObjectURL(u);
    urls = {}; buffers = {}; spec = null;
    draw();
    const clip = clips.find((c) => c.name === name) || clips[0];
    // Spectrograms.
    if (clip.spec) {
      GET(`${base}/spec/${encodeURIComponent(clip.name)}`).then((s) => { if (dead) return; spec = parseSpec(s); draw(); })
        .catch((e) => { if (!dead) clear(specs).append(h('div', { class: 'empty' }, `spec.json: ${e.message}`)); });
    } else clear(specs).append(h('div', { class: 'empty' }, 'No spec.json for this clip'));
    // Audio players + slices.
    for (const v of variants) {
      if (!clip.variants.includes(v)) continue;
      const audio = h('audio', { controls: true, preload: 'none' });
      const first = h('button', { class: 'btn sm', disabled: true, onclick: () => playWindow(v, 0, 1) }, 'First 1 s');
      const last = h('button', { class: 'btn sm', disabled: true, onclick: () => { const b = buffers[v]; if (b) playWindow(v, Math.max(0, b.duration - 1), 1); } }, 'Last 1 s');
      const tailed = spec && spec.speechFrames && spec.frames && spec.speechFrames < spec.frames;
      const title = v === 'degraded' ? (tailed ? 'degraded + synthetic tail' : 'degraded') : v;
      const lastTitle = v === 'degraded' && tailed ? 'The injected key-down garbage' : v === 'out' && tailed ? 'What the model made of the injected garbage (should be silence)' : '';
      last.title = lastTitle;
      audios.append(h('div', { class: 'audio' }, h('div', { class: 'row' }, h('strong', null, title), h('span', { class: 'muted small mono' }, `${clip.name}.${v}.wav`)), audio, h('div', { class: 'btngroup' }, first, last)));
      fetchBlob(`${base}/audio/${encodeURIComponent(`${clip.name}.${v}.wav`)}`).then(async (blob) => {
        if (dead) return;
        urls[v] = URL.createObjectURL(blob);
        audio.src = urls[v];
        try {
          const buf = await blob.arrayBuffer();
          buffers[v] = await getAudioCtx().decodeAudioData(buf);
          first.disabled = last.disabled = false;
          drawWaves();
        } catch (e) { console.warn('decode', v, e); }
      }).catch((e) => toast(`${v}: ${e.message}`, 'error'));
    }
  }
  clipSel.onchange = () => loadClip(clipSel.value);
  loadClip(clips[0].name);
  el.destroy = () => { dead = true; for (const u of Object.values(urls)) URL.revokeObjectURL(u); };
  return el;
}

// ── page: Samples (prepared ⋈ captured navigator) ─────────────────────

const sampleUrl = (key, rest = '') => `/api/samples/${key.split('/').map(encodeURIComponent).join('/')}${rest}`;

// One row of the sample detail: label, <audio>, waveform strip,
// spectrogram. `load(audioUrl, specUrl)` fetches both; `spec()` is the
// parsed spec once it arrived; `paint(lo, hi)` (re)draws.
function sampleRow(label, sub, players) {
  const audio = h('audio', { controls: true, preload: 'none' });
  const wave = h('canvas', { class: 'wavestrip' });
  const canvas = h('canvas');
  const cur = h('div', { class: 'cursor' });
  const box = h('div', { class: 'spec' }, canvas, h('span', { class: 'lbl' }, label), cur);
  const status = h('span', { class: 'muted small' });
  const el = h('div', { class: 'srow' },
    h('div', { class: 'stack', style: { gap: '4px' } }, h('strong', null, label), h('span', { class: 'muted small mono' }, sub), status),
    h('div', { class: 'stack', style: { gap: '6px' } }, audio, wave, box));
  let spec = null, buffer = null, url = null, dead = false, name = null, range = null;
  const row = { el, audio, label, get spec() { return spec; }, get name() { return name; }, get ready() { return Boolean(spec); } };
  players.add(row);
  audio.addEventListener('timeupdate', () => {
    if (!spec || !buffer) { cur.style.display = 'none'; return; }
    cur.style.display = 'block';
    cur.style.left = `${100 * audio.currentTime / buffer.duration}%`;
  });
  box.onclick = (e) => {
    if (!buffer) return;
    const r = box.getBoundingClientRect();
    audio.currentTime = (e.clientX - r.left) / r.width * buffer.duration;
    audio.play().catch(() => {});
  };
  box.title = 'Click to play from here';
  // `lo, hi`: the shared dB range; remembered so a repaint (the audio
  // arriving after the spec, a resize) keeps it.
  row.paint = (lo, hi) => {
    if (lo != null) range = [lo, hi];
    const [l, hh] = range || (spec ? [spec.dbMin, spec.dbMax] : [-100, 0]);
    if (spec) requestAnimationFrame(() => { if (!dead) paintSpec(canvas, spec.mats[name], spec.mels, spec.framesOf[name], l, hh, LUT_VIRIDIS); });
    requestAnimationFrame(() => { if (!dead) paintWave(wave, buffer ? buffer.getChannelData(0) : null); });
  };
  row.load = async (audioUrl, specUrl, onSpec) => {
    status.textContent = 'loading…';
    const a = fetchBlob(audioUrl).then(async (blob) => {
      if (dead) return;
      url = URL.createObjectURL(blob);
      audio.src = url;
      try { buffer = await getAudioCtx().decodeAudioData(await blob.arrayBuffer()); } catch (e) { console.warn('decode', label, e); }
      row.paint();
    });
    const s = GET(specUrl).then((sj) => {
      if (dead) return;
      spec = parseSpec(sj);
      name = Object.keys(spec.mats)[0];
      if (onSpec) onSpec(row);
    });
    try { await Promise.all([a, s]); status.textContent = buffer ? `${fmt.num(buffer.duration, 2)} s` : ''; }
    catch (e) { status.textContent = e.message; toast(`${label}: ${e.message}`, 'error'); }
  };
  row.destroy = () => { dead = true; audio.pause(); if (url) URL.revokeObjectURL(url); players.delete(row); };
  return row;
}

pages.samples = (_parts, q) => {
  // `mode` is one capture set, or two joined by a comma: the utterances
  // both sets hold, so the detail can stack them (dstar against dstar+perens).
  const [mode1 = '', mode2 = ''] = (q.get('mode') || '').split(',');
  const filters = { corpus: q.get('corpus') || '', split: q.get('split') || '', speaker: q.get('speaker') || '', mode: mode1, mode2, q: q.get('q') || '' };
  let page = Math.max(1, Number(q.get('page')) || 1), total = 0, items = [], selKey = q.get('key') || null, facets = null;
  const PER = 50;
  let dead = false;

  // ── left: filters + list ──
  const sel = (name, first) => { const s = h('select', { class: 'small' }, h('option', { value: '' }, first)); s.onchange = () => { filters[name] = s.value; page = 1; if (name === 'corpus') fillSpeakers(); if ((name === 'mode' || name === 'mode2') && selKey) { detailKey = null; select(selKey); } load(); }; return s; };
  const corpusSel = sel('corpus', 'All corpora'), splitSel = sel('split', 'All splits'), speakerSel = sel('speaker', 'All speakers'), modeSel = sel('mode', 'Any mode'), mode2Sel = sel('mode2', '… and in');
  mode2Sel.title = 'Keep only utterances this set captured too, to compare the two side by side';
  const search = h('input', { type: 'text', placeholder: 'key contains…', class: 'mono small', value: filters.q });
  let st = null;
  search.oninput = () => { clearTimeout(st); st = setTimeout(() => { filters.q = search.value.trim(); page = 1; load(); }, 300); };
  const randomBtn = h('button', { class: 'btn sm', onclick: async () => { try { const it = await GET(`/api/samples/random?${qs()}`); select(it.key, true); } catch (e) { toast(e.message, 'error'); } } }, 'Random');
  const count = h('span', { class: 'muted small' });
  const prev = h('button', { class: 'btn sm', onclick: () => { page = Math.max(1, page - 1); load(); } }, '‹ Prev');
  const next = h('button', { class: 'btn sm', onclick: () => { page += 1; load(); } }, 'Next ›');
  const tbody = h('tbody');
  const listWrap = h('div', { class: 'tablewrap slist' }, h('table', null, h('thead', null, h('tr', null, h('th', null, 'Key'), h('th', null, 'Speaker'), h('th', { class: 'num' }, 's'), h('th', null, 'Captured'))), tbody));
  const left = h('div', { class: 'card stack' },
    h('div', { class: 'card-head' }, h('h2', null, 'Samples'), count),
    h('div', { class: 'filters' }, corpusSel, splitSel, speakerSel, modeSel, mode2Sel, search, randomBtn),
    listWrap,
    h('div', { class: 'row' }, prev, next, h('span', { class: 'spacer' }), h('span', { class: 'muted small' }, '↑ ↓ select · space play / pause')));

  // ── right: the selected sample ──
  const detail = h('div', { class: 'card stack' }, h('div', { class: 'empty' }, 'Pick a sample on the left, or press Random.'));
  app.append(h('div', { class: 'page-head' }, h('h1', null, 'Samples'), h('span', { class: 'muted small' }, 'every prepared utterance the chip has been through, with what a checkpoint makes of it')));
  app.append(h('div', { class: 'samples-grid' }, left, detail));

  const qs = () => { const p = new URLSearchParams(); for (const [k, v] of Object.entries(filters)) if (v && k !== 'mode' && k !== 'mode2') p.set(k, v); const sets = [filters.mode, filters.mode2].filter(Boolean); if (sets.length) p.set('mode', sets.join(',')); return p.toString(); };
  function sync() { const p = new URLSearchParams(qs()); if (page > 1) p.set('page', String(page)); if (selKey) p.set('key', selKey); history.replaceState(null, '', `#/samples?${p}`); }

  function fillSpeakers() {
    if (!facets) return;
    clear(speakerSel).append(h('option', { value: '' }, 'All speakers'));
    for (const s of facets.speakers) if (!filters.corpus || s.corpus === filters.corpus) speakerSel.append(h('option', { value: s.name, selected: s.name === filters.speaker }, `${s.name} (${s.count})`));
    if (filters.speaker && !facets.speakers.some((s) => s.name === filters.speaker && (!filters.corpus || s.corpus === filters.corpus))) { filters.speaker = ''; speakerSel.value = ''; }
  }
  async function loadFacets() {
    try { facets = await GET('/api/samples/facets'); } catch (e) { toast(`facets: ${e.message}`, 'error'); return; }
    clear(corpusSel).append(h('option', { value: '' }, `All corpora (${fmt.int(facets.total)})`));
    for (const c of facets.corpora) corpusSel.append(h('option', { value: c.name, selected: c.name === filters.corpus }, `${c.name} (${fmt.int(c.count)})`));
    clear(splitSel).append(h('option', { value: '' }, 'All splits'));
    for (const c of facets.splits) splitSel.append(h('option', { value: c.name, selected: c.name === filters.split }, `${c.name} (${fmt.int(c.count)})`));
    clear(modeSel).append(h('option', { value: '' }, 'Any mode'));
    for (const c of facets.modes) modeSel.append(h('option', { value: c.name, selected: c.name === filters.mode }, `${c.label ? `${c.label} · ` : ''}${c.name} (${fmt.int(c.count)})`));
    clear(mode2Sel).append(h('option', { value: '' }, '… and in'));
    for (const c of facets.modes) mode2Sel.append(h('option', { value: c.name, selected: c.name === filters.mode2 }, `and in ${c.name}`));
    fillSpeakers();
  }
  const chips = (modes) => h('span', { class: 'row', style: { gap: '4px', display: 'inline-flex' } }, modes.map((m) => h('span', { class: 'pill chip', title: `${m.label || m.mode}: ${fmt.int(m.frames)} frames of ${m.frame_ms || 20} ms` }, m.mode)));
  function renderList() {
    clear(tbody);
    if (!items.length) { tbody.append(h('tr', null, h('td', { colspan: 4, class: 'empty' }, 'No samples match'))); }
    for (const it of items) {
      const tr = h('tr', { class: it.key === selKey ? 'sel' : '', onclick: () => select(it.key) },
        h('td', { class: 'mono small clip', title: it.key }, it.key),
        h('td', null, it.speaker),
        h('td', { class: 'num mono' }, fmt.num(it.duration_s, 2)),
        h('td', null, chips(it.modes)));
      tbody.append(tr);
    }
    const lo = total ? (page - 1) * PER + 1 : 0, hi = Math.min(total, page * PER);
    count.textContent = total ? `${fmt.int(lo)}–${fmt.int(hi)} of ${fmt.int(total)}` : '0 samples';
    prev.disabled = page <= 1;
    next.disabled = hi >= total;
    sync();
  }
  async function load() {
    let v;
    try { v = await GET(`/api/samples?${qs()}&page=${page}&per_page=${PER}`); } catch (e) { toast(e.message, 'error'); return; }
    if (dead) return;
    items = v.items; total = v.total;
    if (page > 1 && !items.length && total) { page = Math.max(1, Math.ceil(total / PER)); return load(); }
    renderList();
  }

  // ── detail ──
  const players = new Set();   // sampleRow objects
  let abOn = false, head = 0, focused = null, detailKey = null, rows = [], diffMode = 'none', chipMode = '', showAllSets = false;
  let modelRow = null, pollTimer = null, runs = [], runInfo = null;
  const pref = { get run() { try { return localStorage.getItem('unamblify.samples.run') || ''; } catch { return ''; } }, set run(v) { try { localStorage.setItem('unamblify.samples.run', v); } catch { /* private mode */ } },
    get step() { try { return localStorage.getItem('unamblify.samples.step') || ''; } catch { return ''; } }, set step(v) { try { localStorage.setItem('unamblify.samples.step', v); } catch { /* private mode */ } } };

  // One play head when A/B is on: starting a row seeks it to the head
  // and pauses every other row, so switching rows compares one instant.
  function wirePlayer(row) {
    const a = row.audio;
    a.addEventListener('play', () => {
      focused = row;
      if (!abOn) return;
      if (Math.abs(a.currentTime - head) > 0.05 && head < (a.duration || Infinity)) a.currentTime = head;
      for (const o of players) if (o !== row && !o.audio.paused) o.audio.pause();
    });
    a.addEventListener('timeupdate', () => { if (focused === row) head = a.currentTime; });
    a.addEventListener('pause', () => { if (focused === row) head = a.currentTime; });
    // Scrubbing any row moves the shared head (a programmatic seek from
    // `play` above lands on the same value, so it is harmless here).
    a.addEventListener('seeked', () => { focused = row; head = a.currentTime; });
    row.el.addEventListener('click', () => { focused = row; });
  }

  function destroyDetail() {
    clearTimeout(pollTimer); pollTimer = null;
    for (const r of rows) r.destroy();
    rows = []; modelRow = null; focused = null;
  }

  function diffPane() {
    const pane = h('div', { class: 'stack', style: { gap: '4px' } });
    const cbar = h('div', { class: 'cbar' }), lbl = h('div', { class: 'row small muted' });
    const specBox = h('div', { class: 'spec' });
    pane.append(specBox, cbar, lbl);
    pane.update = () => {
      clear(specBox);
      const chip = rows.find((r) => r.kind === 'chip' && r.mode === chipMode) || rows.find((r) => r.kind === 'chip');
      const clean = rows.find((r) => r.kind === 'clean');
      const model = modelRow;
      const [aRow, bRow, text] = diffMode === 'chip-clean' ? [chip, clean, `chip (${chip && chip.mode}) − clean`]
        : diffMode === 'model-clean' ? [model, clean, 'model − clean'] : [model, chip, `model − chip (${chip && chip.mode})`];
      if (!aRow || !bRow || !aRow.ready || !bRow.ready) {
        const missing = [!aRow && (diffMode === 'chip-clean' ? 'no chip output' : 'render the model first'), !bRow && (diffMode === 'model-chip' ? 'no chip output' : 'no clean row')].filter(Boolean).join(', ');
        specBox.append(h('div', { class: 'empty' }, aRow && bRow ? 'Waiting for both spectrograms…' : `Nothing to compare: ${missing}`)); cbar.style.background = 'none'; clear(lbl); return;
      }
      const sa = aRow.spec, sb = bRow.spec;
      const { d, frames, m } = diffMat(sa.mats[aRow.name], sa.framesOf[aRow.name], sb.mats[bRow.name], sb.framesOf[bRow.name], sa.mels);
      const c = h('canvas');
      specBox.append(c, h('span', { class: 'lbl' }, text));
      requestAnimationFrame(() => paintSpec(c, d, sa.mels, frames, -m, m, LUT_DIVERGING));
      colorBar(cbar, lbl, LUT_DIVERGING, -m, m, 'difference, dB (teal: first quieter, orange: first louder)');
    };
    return pane;
  }

  async function select(key, fromRandom = false) {
    if (!key) return;
    selKey = key;
    for (const tr of tbody.children) tr.classList.toggle('sel', tr.firstChild && tr.firstChild.title === key);
    if (fromRandom && !items.some((i) => i.key === key)) { filters.q = key; search.value = key; page = 1; load(); }
    sync();
    if (detailKey === key) return;
    detailKey = key;
    destroyDetail();
    clear(detail).append(h('div', { class: 'empty' }, 'Loading…'));
    let it;
    try { it = await GET(sampleUrl(key)); } catch (e) { clear(detail).append(h('div', { class: 'empty' }, e.message)); return; }
    if (dead || detailKey !== key) return;
    renderDetail(it);
  }

  // A decode-only sibling's mutation, for the captured line: `· drops 12 frames`.
  const capAug = (c) => (c && c.aug ? ` · ${c.aug.kind} ${fmt.int((c.aug.positions || []).length)} frame${(c.aug.positions || []).length === 1 ? '' : 's'}${c.aug.subst ? ` (${c.aug.subst})` : ''}` : '');
  // An augmented twin's parent and what was done to it (prepare --noise-share / --ham-chain-share).
  function twinRows(it) {
    const p = it.prepared || {};
    if (!p.parent) return [];
    const a = p.aug || {};
    const parts = [];
    if (a.noise) parts.push(`noise ${a.noise.noise_set} ${a.noise.noise_clip.replace(/^raw\/[^/]+\//, '')} @ ${fmt.num(a.noise.noise_offset_s, 1)} s, SNR ${fmt.num(a.noise.snr_db, 1)} dB`);
    if (a.chain) {
      const c = a.chain, st = [];
      if (c.shelf_db != null) st.push(`proximity +${fmt.num(c.shelf_db, 1)} dB`);
      if (c.mic) st.push(`mic tilt ${fmt.num(c.mic.tilt_db, 1)} dB, resonance ${fmt.num(c.mic.resonance_hz, 0)} Hz +${fmt.num(c.mic.resonance_db, 1)} dB`);
      if (c.pops) st.push(`pops ${fmt.num(c.pops.share * 100, 0)} % @ ${fmt.num(c.pops.freq_hz, 0)} Hz ${fmt.num(c.pops.level_dbfs, 0)} dBFS`);
      if (c.clip) st.push(`${c.clip.kind} clip ${fmt.num(c.clip.drive_db, 1)} dB`);
      if (c.agc) st.push(`AGC ${fmt.num(c.agc.attack_ms, 0)}/${fmt.num(c.agc.release_ms, 0)} ms + limiter`);
      parts.push(`chain: ${st.join(', ')}`);
    }
    if (a.post_gain_db) parts.push(`post gain ${fmt.num(a.post_gain_db, 1)} dB`);
    return [
      h('dt', null, 'twin of'), h('dd', null, h('a', { href: '#', class: 'mono', onclick: (e) => { e.preventDefault(); select(p.parent, true); } }, p.parent), h('span', { class: 'muted small' }, ' · the clean target is the parent\'s file; this key\'s clean row is the augmented input')),
      h('dt', null, 'augment'), h('dd', { class: 'small' }, parts.join(' · ') || '–'),
    ];
  }

  function renderDetail(it) {
    clear(detail);
    // With two sets chosen in the filters the detail is a side-by-side:
    // only those two chip rows, unless asked for the rest.
    const pair = [filters.mode, filters.mode2].filter(Boolean);
    const shown = pair.length === 2 && !showAllSets ? it.modes.filter((m) => pair.includes(m.mode)) : it.modes;
    const modes = shown.map((m) => m.mode);
    const labelOf = (id) => { const m = it.modes.find((x) => x.mode === id); return m && m.label ? m.label : id; };
    if (!modes.includes(chipMode)) chipMode = modes.includes(filters.mode) ? filters.mode : modes[0];
    const head_ = h('div', { class: 'card-head' }, h('h2', { class: 'mono', style: { fontSize: '14px', overflowWrap: 'anywhere' } }, it.key));
    const kv = h('dl', { class: 'kv' },
      h('dt', null, 'speaker'), h('dd', null, `${it.speaker}${it.prepared && it.prepared.gender ? ` (${it.prepared.gender})` : ''} · ${it.corpus}`),
      h('dt', null, 'split'), h('dd', null, pill(it.split === 'train' ? 'info' : it.split === 'dev' ? 'ok' : 'queued', it.split)),
      h('dt', null, 'duration'), h('dd', null, `${fmt.num(it.duration_s, 2)} s`),
      h('dt', null, 'captured'), h('dd', null, it.modes.map((m) => `${m.label ? `${m.label} (${m.mode})` : m.mode}: ${fmt.int(m.frames)} frames (${fmt.num(m.frames * (m.frame_ms || 20) / 1000, 2)} s)${capAug(it.captured && it.captured[m.mode])}`).join(' · ')),
      h('dt', null, 'chip'), h('dd', { class: 'mono small muted' }, Object.values(it.captured || {}).map((c) => `${c.prodid} ${c.version}`).filter((v, i, a) => a.indexOf(v) === i).join(', ') || '–'),
      ...twinRows(it));
    const diffBtns = h('div', { class: 'btngroup' });
    const abBtn = h('button', { class: 'btn sm toggle', title: 'One play head across every row', onclick: () => { abOn = !abOn; abBtn.classList.toggle('on', abOn); } }, 'A/B play head');
    const chipSel = h('select', { class: 'small' });
    for (const m of modes) chipSel.append(h('option', { value: m, selected: m === chipMode }, `chip: ${labelOf(m)} · ${m}`));
    chipSel.onchange = () => { chipMode = chipSel.value; diff.update(); };
    const setsBtn = pair.length === 2 && it.modes.length > 2
      ? h('button', { class: `btn sm toggle ${showAllSets ? 'on' : ''}`, title: 'Show every capture set this utterance is in, not only the two being compared', onclick: () => { showAllSets = !showAllSets; destroyDetail(); renderDetail(it); } }, `All ${it.modes.length} sets`)
      : null;
    const toggles = h('div', { class: 'row' }, diffBtns, modes.length > 1 ? chipSel : null, setsBtn, h('span', { class: 'spacer' }), abBtn);
    const rowsEl = h('div', { class: 'stack', style: { gap: '0' } });
    const diff = diffPane();
    const cbar = h('div', { class: 'cbar' }), cbarLbl = h('div', { class: 'row small muted' });
    detail.append(head_, kv, toggles, rowsEl, h('div', { class: 'stack', style: { gap: '4px' } }, cbar, cbarLbl), diff);
    const DIFFS = [['none', 'Spectrograms'], ['chip-clean', 'chip − clean'], ['model-clean', 'model − clean'], ['model-chip', 'model − chip']];
    const renderDiffBtns = () => { clear(diffBtns); for (const [m, l] of DIFFS) diffBtns.append(h('button', { class: `btn sm toggle ${diffMode === m ? 'on' : ''}`, onclick: () => { diffMode = m; renderDiffBtns(); diff.hidden = diffMode === 'none'; diff.update(); } }, l)); };
    renderDiffBtns();
    diff.hidden = diffMode === 'none';

    // Shared dB range: every spec is written on the same axes.
    let lo = -100, hi = 0;
    const onSpec = (row) => { lo = row.spec.dbMin; hi = row.spec.dbMax; row.paint(lo, hi); colorBar(cbar, cbarLbl, LUT_VIRIDIS, lo, hi, 'shared dB range, log-mel 80 × 16 ms'); diff.update(); };
    const clean = sampleRow(it.prepared && it.prepared.parent ? 'input (twin)' : 'clean', `${it.key}.16k.wav`, players);
    clean.kind = 'clean';
    rows.push(clean); rowsEl.append(clean.el); wirePlayer(clean);
    clean.load(sampleUrl(it.key, '/audio/clean16'), sampleUrl(it.key, '/spec/clean16'), onSpec);
    for (const m of shown) {
      const r = sampleRow(`chip · ${m.label || m.mode}`, `captured/${m.mode}/${it.key}.wav`, players);
      r.kind = 'chip'; r.mode = m.mode;
      rows.push(r); rowsEl.append(r.el); wirePlayer(r);
      r.load(sampleUrl(it.key, `/audio/degraded-${m.mode}`), sampleUrl(it.key, `/spec/degraded-${m.mode}`), onSpec);
    }
    focused = clean;

    // Model row: run + checkpoint, render on demand, poll until ready.
    const runSel = h('select', { class: 'small' }), ckSel = h('select', { class: 'small' });
    const renderBtn = h('button', { class: 'btn sm primary', disabled: true }, 'Render');
    const modelStatus = h('span', { class: 'row small muted', style: { gap: '6px' } });
    const modelHost = h('div', { class: 'stack', style: { gap: '6px' } });
    const modelEl = h('div', { class: 'srow' },
      h('div', { class: 'stack', style: { gap: '6px' } }, h('strong', null, 'model'), runSel, ckSel, h('div', { class: 'row' }, renderBtn, modelStatus)),
      modelHost);
    rowsEl.append(modelEl);
    const modelQ = () => `?run=${encodeURIComponent(runSel.value)}&step=${encodeURIComponent(ckSel.value)}`;
    function showModel(v) {
      if (modelRow) { modelRow.destroy(); modelRow = null; }
      clear(modelHost);
      if (!v.ready) { modelHost.append(h('div', { class: 'empty' }, v.error ? v.error : v.running ? 'Rendering…' : 'Not rendered for this checkpoint yet')); diff.update(); return; }
      modelRow = sampleRow(`model · step ${fmt.int(v.step)}`, `${runSel.selectedOptions[0] ? runSel.selectedOptions[0].textContent : v.run}`, players);
      modelRow.kind = 'model';
      modelRow.el.classList.add('inner');
      modelHost.append(modelRow.el); wirePlayer(modelRow);
      modelRow.load(v.audio, v.spec, onSpec);
    }
    async function checkModel() {
      clearTimeout(pollTimer); pollTimer = null;
      if (!runSel.value || !ckSel.value) { clear(modelHost).append(h('div', { class: 'empty' }, runs.length ? 'Pick a run and a checkpoint' : 'No runs with checkpoints yet')); renderBtn.disabled = true; return; }
      let v;
      try { v = await GET(sampleUrl(it.key, `/model${modelQ()}`)); } catch (e) { clear(modelStatus).append(h('span', null, e.message)); renderBtn.disabled = true; return; }
      if (dead || detailKey !== it.key) return;
      if (v.running) { clear(modelStatus).append(h('span', { class: 'spinner' }), 'rendering…'); renderBtn.disabled = true; pollTimer = setTimeout(checkModel, 1000); if (!modelRow) showModel(v); return; }
      clear(modelStatus);
      if (v.error) modelStatus.append(h('span', { class: 'pill bad' }, 'failed'), h('span', { title: v.error }, v.error));
      renderBtn.disabled = v.ready;
      renderBtn.textContent = v.ready ? 'Rendered' : v.error ? 'Retry' : 'Render';
      showModel(v);
    }
    renderBtn.onclick = async () => {
      renderBtn.disabled = true;
      try { await POST(sampleUrl(it.key, '/model'), { run: runSel.value, step: Number(ckSel.value) }); clear(modelStatus).append(h('span', { class: 'spinner' }), 'rendering…'); pollTimer = setTimeout(checkModel, 800); }
      catch (e) { toast(e.message, 'error'); renderBtn.disabled = false; }
    };
    async function fillCkpts() {
      clear(ckSel);
      pref.run = runSel.value;
      if (!runSel.value) { checkModel(); return; }
      try { runInfo = await GET(`/api/runs/${encodeURIComponent(runSel.value)}`); } catch (e) { toast(e.message, 'error'); return; }
      const cks = (runInfo.checkpoints || []).filter((c) => c.model);
      const best = runInfo.status && runInfo.status.best;
      for (const c of cks.slice().reverse()) ckSel.append(h('option', { value: String(c.step) }, `step ${fmt.int(c.step)}${best && best.step === c.step ? ' ★ best' : ''}`));
      if (!cks.length) ckSel.append(h('option', { value: '' }, 'no checkpoints'));
      const want = pref.step;
      if (want && cks.some((c) => String(c.step) === want)) ckSel.value = want;
      checkModel();
    }
    runSel.onchange = fillCkpts;
    ckSel.onchange = () => { pref.step = ckSel.value; checkModel(); };
    (async () => {
      try { runs = (await GET('/api/runs')).filter((r) => r.checkpoints > 0); } catch (e) { toast(e.message, 'error'); runs = []; }
      if (dead || detailKey !== it.key) return;
      clear(runSel);
      if (!runs.length) runSel.append(h('option', { value: '' }, 'no runs with checkpoints'));
      for (const r of runs) runSel.append(h('option', { value: r.id, selected: r.id === pref.run }, `${r.name} · ${r.config && r.config.model ? `${r.config.model.profile} ll${r.config.model.lookahead}` : r.id}`));
      if (runs.length && !runs.some((r) => r.id === pref.run)) runSel.value = runs[0].id;
      fillCkpts();
    })();
  }

  // Keyboard: ↑ / ↓ move the selection, space toggles the focused row.
  function onKey(e) {
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!items.length) return;
      e.preventDefault();
      let i = items.findIndex((x) => x.key === selKey);
      i = e.key === 'ArrowDown' ? Math.min(items.length - 1, i + 1) : Math.max(0, i < 0 ? 0 : i - 1);
      select(items[i].key);
      const tr = tbody.children[i];
      if (tr && tr.scrollIntoView) tr.scrollIntoView({ block: 'nearest' });
    } else if (e.key === ' ' && !(t && t.tagName === 'BUTTON')) {
      const row = focused || rows[0];
      if (!row) return;
      e.preventDefault();
      if (row.audio.paused) row.audio.play().catch(() => {}); else row.audio.pause();
    }
  }
  document.addEventListener('keydown', onKey);

  loadFacets();
  load().then(() => { if (selKey) select(selKey); });
  return { leave() { dead = true; document.removeEventListener('keydown', onKey); destroyDetail(); } };
};

// ── virtualised log view ──────────────────────────────────────────────

function logView() {
  const ROW = 20, OVER = 12;
  const rows = [];
  const spacer = h('div', { class: 'spacer' });
  const box = h('div', { class: 'logview' }, spacer);
  let follow = true, raf = false;
  const followBtn = h('button', { class: 'btn sm toggle on', onclick: () => { follow = !follow; followBtn.classList.toggle('on', follow); if (follow) box.scrollTop = box.scrollHeight; } }, 'Follow');
  function render() {
    raf = false;
    spacer.style.height = `${rows.length * ROW}px`;
    const start = Math.max(0, Math.floor(box.scrollTop / ROW) - OVER), end = Math.min(rows.length, Math.ceil((box.scrollTop + box.clientHeight) / ROW) + OVER);
    clear(spacer);
    for (let i = start; i < end; i++) {
      const r = rows[i];
      spacer.append(h('div', { class: `line ${r.level}`, style: { top: `${i * ROW}px` } }, h('span', { class: 't' }, fmt.ms(r.t)), h('span', { class: 'lv' }, r.level), h('span', null, r.msg)));
    }
  }
  const schedule = () => { if (!raf) { raf = true; requestAnimationFrame(render); } };
  box.onscroll = () => { const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - ROW; if (!atEnd && follow) { follow = false; followBtn.classList.remove('on'); } schedule(); };
  return {
    el: h('div', { class: 'stack' }, h('div', { class: 'row' }, h('span', { class: 'muted small', id: 'logcount' }, '0 lines'), h('span', { class: 'spacer' }), followBtn), box),
    append(newRows) {
      let last = rows.length ? rows[rows.length - 1].seq : -1;
      for (const r of newRows) if (r.seq > last) { rows.push(r); last = r.seq; }
      $('#logcount', this.el).textContent = `${rows.length} lines`;
      schedule();
      if (follow) requestAnimationFrame(() => { box.scrollTop = box.scrollHeight; });
    },
    get lastSeq() { return rows.length ? rows[rows.length - 1].seq : 0; },
  };
}

function renderToml(text) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const lines = text.split('\n').map((line) => {
    const m = line.match(/^(\s*)(\[[^\]]*\])(.*)$/);
    if (m) return `${m[1]}<span class="h">${esc(m[2])}</span><span class="c">${esc(m[3])}</span>`;
    const kv = line.match(/^(\s*)([A-Za-z0-9_.-]+)(\s*=\s*)(.*)$/);
    if (!kv) return line.trim().startsWith('#') ? `<span class="c">${esc(line)}</span>` : esc(line);
    let [, ind, k, eq, v] = kv;
    let comment = '';
    const ci = v.indexOf('#');
    if (ci >= 0 && !/^"[^"]*#/.test(v)) { comment = v.slice(ci); v = v.slice(0, ci); }
    const cls = /^"/.test(v.trim()) ? 's' : /^(true|false)\s*$/.test(v) ? 'b' : /^[-+0-9.eE_]+\s*$/.test(v) ? 'n' : '';
    return `${ind}<span class="k">${esc(k)}</span>${esc(eq)}<span class="${cls}">${esc(v)}</span><span class="c">${esc(comment)}</span>`;
  });
  return h('pre', { class: 'toml', html: lines.join('\n') });
}

// ── page: Run ─────────────────────────────────────────────────────────

pages.run = (parts) => {
  const id = decodeURIComponent(parts[0] || '');
  if (!id) { location.hash = '#/runs'; return {}; }
  const series = {};   // key -> {step, t, v}
  const charts = {};
  let info = null, es = null, dirty = false, flushTimer = null;
  const panels = [];
  const head = h('div', { class: 'page-head' }, h('h1', null, id));
  const statusRow = h('div', { class: 'row' });
  const stats = h('div', { class: 'stats' });
  const actions = h('div', { class: 'btngroup' });
  const headCard = h('div', { class: 'card stack' }, statusRow, stats);
  app.append(head, headCard);

  const lossCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Loss')));
  const lrCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Learning rate')));
  const sysCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'System')));
  const evalCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Eval'), h('span', { class: 'muted small' }, 'whole clip, first second, last second; dashed = one mode')));
  app.append(h('div', { class: 'grid two' }, lossCard, lrCard, sysCard, evalCard));
  const lastOf = (k) => (series[k] && series[k].v.length ? series[k].v[series[k].v.length - 1] : null);
  const data = (k) => series[k];
  charts.loss = makeChart(lossCard, { data, sync: 'run' });
  charts.lr = makeChart(lrCard, { data, height: 160, sync: 'run' });
  charts.sys = makeChart(sysCard, { data, height: 160, sync: 'run' });
  // Per-mode eval columns (`eval/lsd@dstar`) are dashed, so a mode's
  // curve reads against the whole-set mean of the same metric.
  charts.eval = makeChart(evalCard, { data, points: true, sync: 'run', dash: (k) => (modeOf(k) ? [5, 4] : undefined) });
  const toggles = {};
  const groups = { loss: (k) => k.startsWith('loss/'), lr: (k) => k === 'lr', sys: (k) => k.startsWith('sys/'), eval: (k) => k.startsWith('eval/') };
  // Whole-set eval keys first, then each mode's block, in mode order.
  const evalOrder = (a, b) => (modeOf(a) || '').localeCompare(modeOf(b) || '') || a.localeCompare(b);
  function rebuildKeys() {
    const keys = Object.keys(series).sort();
    for (const [g, pred] of Object.entries(groups)) {
      const ks = keys.filter(pred);
      if (g === 'eval') ks.sort(evalOrder);
      if (ks.join() === charts[g].keys.join()) continue;
      charts[g].setKeys(ks, g === 'loss' ? (k) => k === 'loss/total' || ks.length <= 3 : g === 'eval' ? (k) => k === 'eval/lsd' || (k.startsWith('eval/lsd') && !modeOf(k) && !ks.some(modeOf)) || baseKey(k) === 'eval/lsd' : () => true);
      if (toggles[g]) toggles[g].el.remove();
      toggles[g] = seriesToggles(charts[g], lastOf, g === 'eval' && ks.some(modeOf) ? modeOf : null);
      const card = { loss: lossCard, lr: lrCard, sys: sysCard, eval: evalCard }[g];
      card.insertBefore(toggles[g].el, charts[g].el);
    }
  }
  function flush() {
    flushTimer = null;
    if (!dirty) return;
    dirty = false;
    rebuildKeys();
    for (const c of Object.values(charts)) c.update();
    for (const t of Object.values(toggles)) t.render();
  }
  const scheduleFlush = () => { dirty = true; if (!flushTimer) flushTimer = setTimeout(flush, 250); };
  function addRows(rows) {
    for (const r of rows) {
      const s = series[r.k] || (series[r.k] = { step: [], t: [], v: [] });
      if (s.step.length && r.step <= s.step[s.step.length - 1]) continue;
      s.step.push(r.step); s.t.push(r.t); s.v.push(r.v);
    }
    scheduleFlush();
  }

  // Logs / config / checkpoints.
  const logs = logView();
  const logCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Logs')), logs.el);
  const cfgCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Config')));
  const ckCard = h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Checkpoints')));
  const cmpHost = h('div', { class: 'stack' });
  app.append(h('div', { class: 'grid two' }, logCard, cfgCard), ckCard, cmpHost);
  let openPanel = null;
  function showCompare(ck) {
    if (openPanel) { openPanel.destroy && openPanel.destroy(); openPanel.remove(); }
    openPanel = comparePanel(id, ck);
    cmpHost.append(openPanel);
    openPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function renderCheckpoints(cks) {
    clear(ckCard).append(h('div', { class: 'card-head' }, h('h2', null, 'Checkpoints'), h('span', { class: 'muted small' }, `${cks.length}`)));
    if (!cks.length) { ckCard.append(h('div', { class: 'empty' }, 'No checkpoints yet')); return; }
    const best = info && info.status && info.status.best;
    const tb = h('tbody');
    for (const ck of cks.slice().reverse()) {
      const meta = ck.meta || {};
      const metaStr = Object.entries(meta).filter(([k]) => k !== 'step').slice(0, 4).map(([k, v]) => `${k}=${typeof v === 'number' ? fmt.num(v) : v}`).join('  ');
      tb.append(h('tr', null,
        h('td', { class: 'mono' }, fmt.int(ck.step), best && best.step === ck.step ? h('span', { class: 'pill ok', style: { marginLeft: '8px' } }, `best ${best.metric}`) : null),
        h('td', { class: 'muted small nowrap' }, fmt.time(new Date(ck.mtime_ms).toISOString())),
        h('td', null, ck.model ? pill('ok', 'model') : pill('bad', 'no model'), ' ', ck.optim ? pill('info', 'optim') : null),
        h('td', { class: 'mono small muted' }, metaStr),
        h('td', { class: 'num' }, ck.clips.length),
        h('td', null, ck.clips.length ? h('button', { class: 'btn sm', onclick: () => showCompare(ck) }, 'Compare') : null)));
    }
    ckCard.append(h('div', { class: 'tablewrap' }, h('table', null, h('thead', null, h('tr', null, h('th', null, 'Step'), h('th', null, 'Written'), h('th', null, 'Files'), h('th', null, 'Meta'), h('th', { class: 'num' }, 'Clips'), h('th'))), tb)));
  }
  let actionsKey = null;
  function renderHead() {
    const st = (info && info.status) || {};
    clear(head).append(h('h1', null, (info && info.name) || id), pill(st.status), h('span', { class: 'muted small mono' }, id), actions);
    // Rebuild the buttons only when what they act on changed, and never
    // under an open confirm strip (status writes arrive every few seconds).
    const key = `${Boolean(info && info.supervised)}|${st.status}|${(info && info.checkpoints || []).length > 0}`;
    if (key !== actionsKey && !actions.querySelector('.confirm')) {
      actionsKey = key;
      clear(actions);
      if (info && info.supervised) {
        const b = h('button', { class: 'btn warn' }, 'Stop');
        b.onclick = () => confirmInline(b, 'Stop this run? A checkpoint is written first.', async () => { try { await POST(`/api/runs/${encodeURIComponent(id)}/stop`); toast('Stopped', 'ok'); } catch (e) { toast(e.message, 'error'); } reload(); });
        actions.append(b);
      } else if (st.status === 'queued') {
        actions.append(h('button', { class: 'btn good', onclick: async () => { try { await POST(`/api/runs/${encodeURIComponent(id)}/resume`); toast('Started', 'ok'); } catch (e) { toast(e.message, 'error'); } reload(); } }, 'Start'));
      } else if (st.status === 'stopped' || st.status === 'failed') {
        const hasCk = (info && info.checkpoints || []).length > 0;
        actions.append(h('button', { class: 'btn good', onclick: async () => { try { await POST(`/api/runs/${encodeURIComponent(id)}/resume`); toast(hasCk ? 'Resumed' : 'Started', 'ok'); } catch (e) { toast(e.message, 'error'); } reload(); } }, hasCk ? 'Resume' : 'Start over'));
      }
      actions.append(h('a', { class: 'btn', href: `#/compare?ids=${encodeURIComponent(id)}` }, 'Compare…'));
    }
    clear(statusRow).append(h('div', { class: 'row', style: { flex: '1', gap: '10px' } }, progressBar(st.step, st.total_steps, st.status === 'finished' ? 'green' : ''), h('span', { class: 'mono small nowrap' }, `${fmt.int(st.step)} / ${fmt.int(st.total_steps)} (${fmt.pct(st.step, st.total_steps)})`)));
    const sps = lastOf('sys/steps_per_s');
    const eta = sps && st.total_steps ? (st.total_steps - st.step) / sps : null;
    clear(stats).append(
      h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.num(lastOf('loss/total'))), h('span', { class: 'k' }, 'loss/total')),
      h('div', { class: 'stat' }, h('span', { class: 'v' }, st.best ? fmt.num(st.best.value) : '–'), h('span', { class: 'k' }, st.best ? `best ${st.best.metric} @ ${fmt.int(st.best.step)}` : 'best')),
      h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.num(sps, 2)), h('span', { class: 'k' }, 'steps / s')),
      h('div', { class: 'stat' }, h('span', { class: 'v' }, st.status === 'running' ? fmt.dur(eta) : '–'), h('span', { class: 'k' }, 'eta')),
      h('div', { class: 'stat' }, h('span', { class: 'v' }, st.device || '–'), h('span', { class: 'k' }, st.host || 'device')),
      h('div', { class: 'stat' }, h('span', { class: 'v' }, fmt.ago(st.updated)), h('span', { class: 'k' }, `started ${fmt.time(st.started)}`)));
  }

  async function reload() {
    try {
      info = await GET(`/api/runs/${encodeURIComponent(id)}`);
    } catch (e) { clear(headCard).append(h('div', { class: 'empty' }, e.message)); return; }
    renderHead();
    renderCheckpoints(info.checkpoints || []);
    clear(cfgCard).append(h('div', { class: 'card-head' }, h('h2', null, 'Config'), h('span', { class: 'muted small mono' }, 'config.toml (resolved)')), renderToml(info.config_toml || ''));
  }
  // `incremental`: only what arrived after the last step / log seq seen
  // (a reconnect after a gap); otherwise the whole history.
  async function loadHistory(incremental = false) {
    const lastStep = incremental ? Math.max(0, ...Object.values(series).map((s) => (s.step.length ? s.step[s.step.length - 1] : 0))) : 0;
    try {
      const m = await GET(`/api/runs/${encodeURIComponent(id)}/metrics${lastStep ? `?after_step=${lastStep}` : ''}`);
      for (const [k, s] of Object.entries(m.series)) {
        const cur = series[k] || (series[k] = { step: [], t: [], v: [] });
        if (lastStep) { addRows(s.step.map((st, i) => ({ k, step: st, t: s.t[i], v: s.v[i] }))); continue; }
        // Merge: keep the live rows that arrived while history loaded.
        const live = cur.step.length ? { step: cur.step, t: cur.t, v: cur.v } : null;
        cur.step = s.step.slice(); cur.t = s.t.slice(); cur.v = s.v.slice();
        if (live) for (let i = 0; i < live.step.length; i++) if (live.step[i] > cur.step[cur.step.length - 1]) { cur.step.push(live.step[i]); cur.t.push(live.t[i]); cur.v.push(live.v[i]); }
      }
      scheduleFlush();
      renderHead();
    } catch (e) { toast(`metrics: ${e.message}`, 'error'); }
    try { logs.append(await GET(`/api/runs/${encodeURIComponent(id)}/logs?${incremental && logs.lastSeq ? `after=${logs.lastSeq}` : 'tail=2000'}`)); } catch (e) { toast(`logs: ${e.message}`, 'error'); }
  }
  // SSE first so nothing is missed, then the history. A reconnect (the
  // laptop slept, the server restarted) refetches what the gap dropped.
  let opens = 0;
  es = sse(`/api/runs/${encodeURIComponent(id)}/events`, (ev, d) => {
    if (ev === 'metric' || ev === 'sys') addRows(d);
    else if (ev === 'log') logs.append(d);
    else if (ev === 'status') { info = info || {}; info.status = d; renderHead(); }
    else if (ev === 'checkpoint') { info = info || { checkpoints: [] }; info.checkpoints = [...(info.checkpoints || []).filter((c) => c.step !== d.step), d].sort((a, b) => a.step - b.step); renderCheckpoints(info.checkpoints); toast(`Checkpoint at step ${fmt.int(d.step)}`); }
    else if (ev === 'lagged') { loadHistory(true); }
  }, (state) => { if (state === 'open' && opens++ > 0) { reload(); loadHistory(true); } });
  reload().then(() => loadHistory());
  // Global run-list events arrive on every status write of every live run;
  // one debounced reload per burst is plenty.
  let runsTimer = null;
  const onRuns = () => { clearTimeout(runsTimer); runsTimer = setTimeout(reload, 400); };
  bus.addEventListener('runs', onRuns);
  const tick = setInterval(() => { if (info && info.status && info.status.status === 'running') renderHead(); }, 10000);
  return { leave() { es.close(); bus.removeEventListener('runs', onRuns); clearInterval(tick); clearTimeout(flushTimer); clearTimeout(runsTimer); for (const c of Object.values(charts)) c.destroy(); if (openPanel && openPanel.destroy) openPanel.destroy(); } };
};

// ── boot ──────────────────────────────────────────────────────────────

route();
