const DATA = {
  manifest: 'data/sample_manifest.json'
};

const state = {
  groundTruth: '',
  originalScript: '',
  characterMap: null,
  rules: '',
  manifest: null,
  issues: [],
  decisions: {},
  activeFilter: 'all',
  mapValid: true,
  rulesValid: true,
  running: false,
  backendAvailable: false,
  lastRun: null
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function readText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return response.text();
}

async function readJSON(path) {
  return JSON.parse(await readText(path));
}

function durationLabel(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

function applyManifest(manifest) {
  const run = manifest.verified_run || {};
  $('#projectLabel').textContent = manifest.project_label || 'Sample episode';
  $('#sourceTitle').textContent = manifest.source_title || 'Sample episode';
  $('#sourceSubtitle').textContent = manifest.source_subtitle || 'Verified public source';
  $('#sourceTimecode').textContent = manifest.source_timecode || '--:-- — --:--';
  $('#verifiedRunTime').textContent = durationLabel(run.processing_seconds);
  $('#asrCountSide').textContent = run.transcript_segments ?? '-';
  $('#frameCountSide').textContent = run.evidence_frames ?? '-';
  $('#flagCountSide').textContent = run.reference_flags ?? '-';
  $('#transcriptCount').textContent = run.transcript_segments ?? '-';
  $('#frameCount').textContent = run.evidence_frames ?? '-';
  $('#paperEpisodeLabel').textContent = `${String(manifest.episode_label || 'Episode').toUpperCase()} · INDIAN ADAPTATION`;
  $('#sourceEpisodeLink').href = manifest.source_url || '#';
  $('#groundTruthDownload').href = manifest.inputs?.ground_truth || '#';
  $('#draftDownload').href = manifest.inputs?.script || '#';
}

function esc(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 2200);
}

function switchTab(name) {
  $$('.tab').forEach(button => {
    const active = button.dataset.tab === name;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $$('.tab-panel').forEach(panel => panel.classList.remove('active'));
  $(`#${name}Tab`).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function parseMap() {
  try {
    return JSON.parse($('#mapInput').value);
  } catch {
    return null;
  }
}

function setInputState(id, status, label) {
  const badge = $(id);
  badge.textContent = label;
  badge.classList.toggle('changed', status === 'changed');
  badge.classList.toggle('invalid', status === 'invalid');
}

function syncRunAvailability() {
  const draftReady = Boolean($('#scriptInput').value.trim());
  const ready = draftReady && state.mapValid && state.rulesValid && !state.running;
  $('#runButton').disabled = !ready;
  const sectionState = $('.section-state');
  sectionState.classList.toggle('invalid', !ready && !state.running);
  sectionState.innerHTML = ready
    ? '<span class="status-dot"></span>All required inputs ready'
    : state.running
      ? '<span class="status-dot"></span>Review running'
      : '<span class="status-dot"></span>Resolve input errors to continue';
}

function validateMap() {
  const message = $('#mapValidation');
  const map = parseMap();
  const characters = map?.characters;
  state.mapValid = Array.isArray(characters) && characters.length > 0 && characters.every(character =>
    (character.source_name || character.romanized_name) && character.adapted_name
  );
  message.className = `field-message ${state.mapValid ? 'valid' : 'error'}`;
  message.textContent = state.mapValid
    ? `${characters.length} valid character mapping${characters.length === 1 ? '' : 's'}.`
    : 'Use valid JSON with a non-empty characters array and source/adapted names.';
  const changed = state.mapValid && JSON.stringify(map) !== JSON.stringify(state.characterMap);
  setInputState('#mapInputState', state.mapValid ? (changed ? 'changed' : 'ready') : 'invalid', state.mapValid ? (changed ? 'Updated' : 'Ready') : 'Fix JSON');
  syncRunAvailability();
  return map;
}

function validateRules() {
  const requirements = SceneProofModel.parseRequirements($('#rulesInput').value);
  state.rulesValid = requirements.length > 0;
  const message = $('#rulesValidation');
  message.className = `field-message ${state.rulesValid ? 'valid' : 'error'}`;
  message.textContent = state.rulesValid
    ? `${requirements.length} production requirement${requirements.length === 1 ? '' : 's'} detected.`
    : 'Add at least one numbered adaptation requirement.';
  const changed = state.rulesValid && $('#rulesInput').value.trim() !== state.rules.trim();
  setInputState('#rulesInputState', state.rulesValid ? (changed ? 'changed' : 'ready') : 'invalid', state.rulesValid ? (changed ? 'Updated' : 'Ready') : 'Add rules');
  syncRunAvailability();
  return requirements;
}

function runAlignmentModel(script = $('#scriptInput').value) {
  const characterMap = parseMap();
  if (!characterMap || !state.rulesValid) throw new Error('Resolve the input validation errors before running the model.');
  return SceneProofModel.review({
    evidence: state.groundTruth,
    script,
    characterMap,
    rules: $('#rulesInput').value
  });
}

function hydrateBackendIssues(serverIssues, script) {
  const browserResult = runAlignmentModel(script);
  const local = new Map(browserResult.issues.map(issue => [issue.id, issue]));
  return serverIssues.map((item, index) => {
    const known = local.get(item.id);
    return {
      ...(known || {
        id: item.id || `model-${index + 1}`,
        order: 90 + index,
        target: () => false,
        apply: value => value
      }),
      severity: item.severity || known?.severity || 'warning',
      type: item.category || known?.type || 'Semantic mismatch',
      title: item.title || known?.title || 'Possible source mismatch',
      explanation: item.explanation || known?.explanation || '',
      evidence: item.evidence || known?.evidence || '',
      timecode: item.timecode || known?.timecode || 'unknown',
      proposal: item.proposal || known?.proposal || '',
      confidence: item.confidence ?? known?.confidence ?? 0,
      detectedBy: item.detected_by || known?.detectedBy || 'configured model'
    };
  });
}

function setPipelineStage(stage, status) {
  const node = $(`.model-node[data-stage="${stage}"]`);
  if (!node) return;
  node.classList.remove('running', 'complete', 'failed');
  if (status) node.classList.add(status);
}

function resetPipelineStages() {
  $$('.model-node').forEach(node => node.classList.remove('running', 'complete', 'failed'));
}

function showRunResult(result) {
  state.lastRun = result;
  ['ingest', 'speech', 'vision', 'qa'].forEach(stage => setPipelineStage(stage, 'complete'));
  const media = result.media || {};
  const transcript = result.transcript || {};
  $('#runFacts').innerHTML = `
    <span><b>${Number(media.duration_seconds || 0).toFixed(1)}s</b> source video</span>
    <span><b>${transcript.segment_count || 0}</b> Whisper segments</span>
    <span><b>${media.sampled_frame_count || 0}</b> sampled frames</span>
    <span><b>${result.issues?.length || 0}</b> review issues</span>`;
  $('#pipelineNote').textContent = `${result.mode} · processed in ${media.processing_seconds || '?'}s. ${result.limitations?.[0] || 'Raw evidence saved.'}`;
  $('#runtimeBadge').textContent = 'Live multimodal run';
  $('#runtimeChip').textContent = 'Backend run complete';
  $('#runtimeChip').classList.add('live');
  $('#flagCountSide').textContent = result.issues?.length || 0;
}

function lineClass(line) {
  if (/^(EPISODE|CONTINUITY|---|SCENE|Scene heading:|Actions and dialogue:)/.test(line)) return 'heading';
  if (/^\*\*[^*]+\*\*$/.test(line)) return 'character';
  if (/^Start:/.test(line)) return 'timecode';
  return '';
}

function issueMarkup(issue) {
  const decision = state.decisions[issue.id];
  const status = decision ? decision : '';
  return `
    <section class="review-box ${decision ? 'decided' : ''}" id="issue-${issue.id}" data-status="${esc(status)}">
      <div class="issue-box">
        <div class="box-title"><span>${issue.severity.toUpperCase()}</span>${esc(issue.type)}</div>
        <p><strong>${esc(issue.title)}.</strong> ${esc(issue.explanation)}</p>
        <div class="evidence-line"><span>Source ${esc(issue.timecode)}</span><span>${esc(issue.evidence)}</span></div>
        <div class="model-line"><span>${esc(issue.detectedBy || 'configured model')}</span><span>${Math.round((issue.confidence || 0) * 100)}% confidence</span></div>
      </div>
      <div class="proposal-box">
        <div class="box-title"><span>PROPOSED EDIT</span>Evidence-backed suggestion</div>
        <p>${esc(issue.proposal)}</p>
        <div class="decision-row">
          ${['accepted', 'rejected', 'intentional'].map(value => `<button data-issue="${issue.id}" data-decision="${value}" class="${decision === value ? 'selected' : ''}">${value === 'accepted' ? 'Accept edit' : value === 'rejected' ? 'Reject' : 'Mark intentional'}</button>`).join('')}
        </div>
      </div>
    </section>`;
}

function visibleIssues() {
  if (state.activeFilter === 'critical') return state.issues.filter(issue => issue.severity === 'critical');
  if (state.activeFilter === 'warning') return state.issues.filter(issue => issue.severity !== 'critical');
  if (state.activeFilter === 'pending') return state.issues.filter(issue => !state.decisions[issue.id]);
  return state.issues;
}

function timecodeSeconds(value) {
  const match = String(value || '').match(/(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)/);
  if (!match) return 0;
  return (Number(match[1] || 0) * 3600) + (Number(match[2]) * 60) + Number(match[3]);
}

function renderTimeline() {
  const duration = Math.max(1, timecodeSeconds(String(state.manifest?.source_timecode || '').split(/[—-]/).pop()));
  $('#timelineRange').textContent = `00:00 — ${durationLabel(duration)}`;
  $('#timelineTrack').innerHTML = state.issues.map(issue => {
    const position = Math.min(100, Math.max(0, (timecodeSeconds(issue.timecode) / duration) * 100));
    const classes = ['timeline-dot', issue.severity === 'critical' ? 'critical' : '', state.decisions[issue.id] ? 'decided' : ''].filter(Boolean).join(' ');
    return `<button class="${classes}" style="left:${position}%" type="button" data-scroll="${esc(issue.id)}" aria-label="${esc(issue.title)} at ${esc(issue.timecode)}"></button>`;
  }).join('');
  $$('.timeline-dot').forEach(button => button.addEventListener('click', () => {
    if (!visibleIssues().some(issue => issue.id === button.dataset.scroll)) applyReviewFilter('all');
    $(`#issue-${button.dataset.scroll}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));
}

function updateFilterButtons() {
  const counts = {
    all: state.issues.length,
    critical: state.issues.filter(issue => issue.severity === 'critical').length,
    warning: state.issues.filter(issue => issue.severity !== 'critical').length,
    pending: state.issues.filter(issue => !state.decisions[issue.id]).length
  };
  $('#filterAllCount').textContent = counts.all;
  $('#filterCriticalCount').textContent = counts.critical;
  $('#filterWarningCount').textContent = counts.warning;
  $('#filterPendingCount').textContent = counts.pending;
  $$('.filter-button').forEach(button => {
    const active = button.dataset.filter === state.activeFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

function applyReviewFilter(filter) {
  state.activeFilter = filter;
  updateFilterButtons();
  renderIssueRail();
  renderAnnotatedScript();
}

function renderAnnotatedScript() {
  const script = $('#scriptInput').value;
  const lines = script.split('\n');
  const unmatched = [...visibleIssues()];
  let html = '';

  lines.forEach(line => {
    const matches = unmatched.filter(issue => issue.target(line));
    const flagged = matches.length ? ' flagged' : '';
    html += `<div class="script-line ${lineClass(line)}${flagged}">${line ? esc(line) : '&nbsp;'}</div>`;
    matches.forEach(issue => {
      html += issueMarkup(issue);
      unmatched.splice(unmatched.indexOf(issue), 1);
    });
  });

  unmatched.forEach(issue => { html += issueMarkup(issue); });
  $('#annotatedScript').innerHTML = html;
  bindDecisionButtons();
}

function renderIssueRail() {
  const issues = visibleIssues();
  $('#issueRail').innerHTML = issues.length ? issues.map((issue, index) => `
    <button class="rail-item ${state.decisions[issue.id] ? 'decided' : ''}" data-scroll="${issue.id}">
      <span>${String(index + 1).padStart(2, '0')}</span>
      <div><strong>${esc(issue.title)}</strong><small>${esc(issue.type)} · ${esc(issue.timecode)}</small></div>
    </button>
  `).join('') : '<p class="rail-empty">No issues match this filter.</p>';

  $$('.rail-item').forEach(button => button.addEventListener('click', () => {
    $$('.rail-item').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    $(`#issue-${button.dataset.scroll}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));
}

function bindDecisionButtons() {
  $$('.decision-row button').forEach(button => button.addEventListener('click', () => {
    state.decisions[button.dataset.issue] = button.dataset.decision;
    renderAnnotatedScript();
    renderIssueRail();
    updateFilterButtons();
    renderTimeline();
    renderCleanOutput();
    showToast(button.dataset.decision === 'accepted' ? 'Edit accepted' : button.dataset.decision === 'rejected' ? 'Suggestion rejected' : 'Difference marked intentional');
  }));
}

function buildCleanScript() {
  let output = $('#scriptInput').value;
  state.issues
    .filter(issue => state.decisions[issue.id] === 'accepted')
    .sort((a, b) => (a.order || 0) - (b.order || 0))
    .forEach(issue => { output = issue.apply(output); });
  return output;
}

function renderCleanOutput() {
  const decided = Object.keys(state.decisions).length;
  const accepted = Object.values(state.decisions).filter(value => value === 'accepted').length;
  const rejected = Object.values(state.decisions).filter(value => value === 'rejected').length;
  const intentional = Object.values(state.decisions).filter(value => value === 'intentional').length;
  $('#resolvedCount').textContent = decided;
  $('#cleanScript').textContent = buildCleanScript();

  const banner = $('#decisionBanner');
  banner.innerHTML = decided
    ? `<span class="ready-dot"></span><strong>${decided} of ${state.issues.length} decided.</strong><span>${accepted} accepted · ${rejected} rejected · ${intentional} intentional</span>`
    : '<span class="ready-dot"></span><strong>No decisions yet.</strong><span>Return to the annotated script and review each suggestion.</span>';
}

async function runReview() {
  const button = $('#runButton');
  const videoFile = $('#videoInput').files?.[0];
  validateMap();
  validateRules();
  if (!$('#scriptInput').value.trim() || !state.mapValid || !state.rulesValid) {
    showToast('Resolve the highlighted input errors first');
    syncRunAvailability();
    return;
  }
  state.running = true;
  syncRunAvailability();
  button.classList.add('is-running');
  button.setAttribute('aria-busy', 'true');
  $('#runMode').textContent = 'Review in progress';
  $('#runHint').textContent = videoFile ? 'Processing the selected episode evidence' : 'Fitting a fresh model on the current inputs';
  resetPipelineStages();

  if (videoFile && state.backendAvailable) {
    button.querySelector('span').textContent = 'Running video AI…';
    ['ingest', 'speech', 'vision', 'qa'].forEach(stage => setPipelineStage(stage, 'running'));
    const form = new FormData();
    form.append('source_video', videoFile);
    form.append('adapted_script', $('#scriptInput').value);
    form.append('character_map', $('#mapInput').value);
    form.append('rules', $('#rulesInput').value);
    try {
      const response = await fetch('/api/analyze', { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'The backend could not process this video');
      state.decisions = {};
      state.issues = hydrateBackendIssues(result.issues || [], $('#scriptInput').value);
      showRunResult(result);
      finishReview(button, 'Run review');
      return;
    } catch (error) {
      console.error(error);
      ['ingest', 'speech', 'vision', 'qa'].forEach(stage => setPipelineStage(stage, 'failed'));
      $('#pipelineNote').textContent = `Run failed: ${error.message}`;
      showToast('Video pipeline failed — see status');
      state.running = false;
      button.classList.remove('is-running');
      button.removeAttribute('aria-busy');
      button.querySelector('span').textContent = 'Retry review';
      syncRunAvailability();
      return;
    }
  }

  if (videoFile && !state.backendAvailable) {
    $('#pipelineNote').textContent = 'Hosted demo mode: the selected video is preview-only because GitHub Pages cannot execute Whisper. The review will continue against the supplied, timestamped Episode 1 evidence.';
    showToast('Video preview loaded — using verified source evidence');
  }

  button.querySelector('span').textContent = 'Loading verified evidence…';
  setPipelineStage('ingest', 'complete');
  setPipelineStage('speech', 'complete');
  setPipelineStage('vision', 'complete');
  await new Promise(resolve => setTimeout(resolve, 450));
  try {
    state.decisions = {};
    const modelResult = runAlignmentModel($('#scriptInput').value);
    state.issues = modelResult.issues;
    setPipelineStage('qa', 'complete');
    $('#runtimeBadge').textContent = 'In-browser model run';
    $('#runtimeChip').textContent = 'Model fitted on current inputs';
    $('#pipelineNote').textContent = `Fresh model run: ${modelResult.metadata.documentsFitted} documents, ${modelResult.metadata.featureCount} learned features, ${state.issues.length} evidence-linked anomalies. No show-specific issue catalogue was used.`;
    finishReview(button, 'Run review');
  } catch (error) {
    console.error(error);
    setPipelineStage('qa', 'failed');
    $('#pipelineNote').textContent = `Run failed: ${error.message}`;
    state.running = false;
    button.classList.remove('is-running');
    button.removeAttribute('aria-busy');
    button.querySelector('span').textContent = 'Retry review';
    syncRunAvailability();
    showToast('Review could not run — check the inputs');
  }
}

function finishReview(button, buttonLabel) {
    state.running = false;
    state.activeFilter = 'all';
    $('#issueBadge').textContent = state.issues.length;
    $('#flagCountSide').textContent = state.issues.length;
    $('.review-toolbar h2').textContent = `${state.issues.length} issue${state.issues.length === 1 ? '' : 's'} need a decision`;
    $('.critical-chip b').textContent = state.issues.filter(issue => issue.severity === 'critical').length;
    $('.warning-chip b').textContent = state.issues.filter(issue => issue.severity !== 'critical').length;
    $('#reviewEmpty').hidden = true;
    $('#reviewContent').hidden = false;
    updateFilterButtons();
    renderTimeline();
    renderIssueRail();
    renderAnnotatedScript();
    renderCleanOutput();
    button.classList.remove('is-running');
    button.removeAttribute('aria-busy');
    button.querySelector('span').textContent = buttonLabel;
    syncRunAvailability();
    $('#runMode').textContent = 'Review complete';
    $('#runHint').textContent = 'Inspect each evidence-linked issue and record a decision';
    switchTab('review');
    showToast(`${state.issues.length} evidence-linked issues found`);
}

function renderMap() {
  const map = parseMap();
  const characters = map?.characters || [];
  $('#mappingList').innerHTML = characters.map(character => `
    <div class="map-row">
      <i>${esc((character.source_name || character.romanized_name || 'N').slice(0, 1))}</i>
      <div><small>${esc(character.romanized_name)}</small><strong>${esc(character.adapted_name)}</strong></div>
    </div>
  `).join('');
  $('#mapCount').textContent = `${characters.length} mapping${characters.length === 1 ? '' : 's'}`;
  $('#mapInputMeta').textContent = `${characters.length} approved mapping${characters.length === 1 ? '' : 's'}`;
}

function renderRules() {
  const requirements = SceneProofModel.parseRequirements($('#rulesInput').value);
  $('#ruleTags').innerHTML = requirements.slice(0, 6).map(rule => `<span>${esc(rule.title)}</span>`).join('');
  $('#rulesCount').textContent = `${requirements.length} rule${requirements.length === 1 ? '' : 's'}`;
  $('#rulesInputMeta').textContent = `${requirements.length} production requirement${requirements.length === 1 ? '' : 's'}`;
}

function updateScriptStats() {
  const value = $('#scriptInput').value;
  const words = value.trim() ? value.trim().split(/\s+/).length : 0;
  const blocks = (value.match(/^\*\*/gm) || []).length;
  $('#scriptStats').textContent = `${words} words · ${blocks} blocks`;
  $('#draftInputMeta').textContent = `${words} words · ${blocks} blocks`;
}

function downloadText(filename, content, type = 'text/plain') {
  const blob = new Blob([content], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function reviewDownload() {
  const lines = state.issues.map((issue, index) => {
    const decision = state.decisions[issue.id] || 'pending';
    return `${index + 1}. [${issue.severity.toUpperCase()}] ${issue.title}\n   Source: ${issue.timecode}\n   Decision: ${decision}\n   Evidence: ${issue.evidence}\n   Proposed edit: ${issue.proposal}`;
  }).join('\n\n');
  const slug = state.manifest?.file_slug || 'episode';
  const label = state.manifest?.episode_label || 'Episode';
  downloadText(`sceneproof-${slug}-review.txt`, `SCENEPROOF REVIEW — ${label.toUpperCase()}\n\n${lines}\n\n--- ORIGINAL WRITER SCRIPT ---\n\n${$('#scriptInput').value}`);
}

async function resetDemo() {
  const videoInput = $('#videoInput');
  const video = $('#videoPreview');
  videoInput.value = '';
  video.pause();
  video.removeAttribute('src');
  video.load();
  video.style.display = 'none';
  $('#videoPlaceholder').style.display = '';
  $('#scriptInput').value = state.originalScript;
  $('#mapInput').value = JSON.stringify(state.characterMap, null, 2);
  $('#rulesInput').value = state.rules;
  state.decisions = {};
  state.issues = [];
  state.activeFilter = 'all';
  state.mapValid = true;
  state.rulesValid = true;
  state.running = false;
  state.lastRun = null;
  $('#flagCountSide').textContent = state.manifest?.verified_run?.reference_flags ?? '-';
  resetPipelineStages();
  $('#runFacts').innerHTML = `
    <span><b>Speech model</b> Whisper small</span>
    <span><b>Comparison</b> Fresh vectors per run</span>
    <span><b>Decisions</b> Human-controlled edits</span>
    <span><b>Optional</b> Configured vision model</span>`;
  renderMap();
  renderRules();
  updateScriptStats();
  setInputState('#draftInputState', 'ready', 'Ready');
  setInputState('#mapInputState', 'ready', 'Ready');
  setInputState('#rulesInputState', 'ready', 'Ready');
  $('#mapValidation').className = 'field-message';
  $('#mapValidation').textContent = '';
  $('#rulesValidation').className = 'field-message';
  $('#rulesValidation').textContent = '';
  $('#issueBadge').textContent = '0';
  $('#reviewEmpty').hidden = false;
  $('#reviewContent').hidden = true;
  $('#cleanScript').textContent = state.originalScript;
  applyManifest(state.manifest || {});
  $('#sourceMode').textContent = 'Verified sample';
  const button = $('#runButton');
  button.classList.remove('is-running');
  button.removeAttribute('aria-busy');
  button.querySelector('span').textContent = 'Run review';
  syncRunAvailability();
  switchTab('inputs');
  showToast('Demo inputs restored');
}

function bindUI() {
  $$('.tab').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.tab)));
  $$('.input-pill').forEach(button => button.addEventListener('click', () => {
    $$('.input-pill').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
    $(`#${button.dataset.focus}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));

  $('#runButton').addEventListener('click', runReview);
  $('#resetButton').addEventListener('click', resetDemo);
  $('#loadScriptButton').addEventListener('click', () => {
    $('#scriptInput').value = state.originalScript;
    updateScriptStats();
    setInputState('#draftInputState', 'ready', 'Ready');
    syncRunAvailability();
    showToast('Sample flawed script loaded');
  });
  $('#scriptInput').addEventListener('input', () => {
    updateScriptStats();
    setInputState('#draftInputState', $('#scriptInput').value.trim() ? 'changed' : 'invalid', $('#scriptInput').value.trim() ? 'Updated' : 'Required');
    syncRunAvailability();
  });
  $('#mapInput').addEventListener('input', () => {
    validateMap();
    renderMap();
  });
  $('#rulesInput').addEventListener('input', () => {
    validateRules();
    renderRules();
  });
  $('#emptySetupButton').addEventListener('click', () => switchTab('inputs'));
  $$('.filter-button').forEach(button => button.addEventListener('click', () => applyReviewFilter(button.dataset.filter)));
  $('#nextIssueButton').addEventListener('click', () => {
    const next = state.issues.find(issue => !state.decisions[issue.id]);
    if (!next) {
      showToast('Every issue has a decision');
      return;
    }
    applyReviewFilter('pending');
    requestAnimationFrame(() => $(`#issue-${next.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
  });
  $('#themeButton').addEventListener('click', event => {
    const enabled = document.body.classList.toggle('high-contrast');
    event.currentTarget.setAttribute('aria-pressed', String(enabled));
  });

  $('#videoInput').addEventListener('change', event => {
    const file = event.target.files?.[0];
    if (!file) return;
    const video = $('#videoPreview');
    video.src = URL.createObjectURL(file);
    video.style.display = 'block';
    $('#videoPlaceholder').style.display = 'none';
    $('#sourceMode').textContent = state.backendAvailable ? 'Ready for AI run' : 'Preview only';
    $('#runMode').textContent = state.backendAvailable ? 'Ready for fresh video analysis' : 'Backend required for this video';
    $('#runHint').textContent = state.backendAvailable ? 'Whisper and frame extraction will run now' : 'The hosted static demo cannot execute local models';
    showToast(`Loaded ${file.name} locally`);
  });

  $('#downloadReviewButton').addEventListener('click', reviewDownload);
  $('#downloadCleanButton').addEventListener('click', () => {
    const slug = state.manifest?.file_slug || 'episode';
    downloadText(`${slug}-clean-revised-script.txt`, buildCleanScript());
  });
}

async function checkBackend() {
  try {
    const response = await fetch('/api/health', { headers: { Accept: 'application/json' } });
    const contentType = response.headers.get('content-type') || '';
    if (!response.ok || !contentType.includes('application/json')) throw new Error('No local API');
    const health = await response.json();
    state.backendAvailable = health.new_video_support === true;
  } catch {
    state.backendAvailable = false;
  }
  $('#runtimeChip').textContent = state.backendAvailable ? 'Live AI backend connected' : 'Hosted verified demo';
  $('#runtimeChip').classList.toggle('live', state.backendAvailable);
  $('#runtimeBadge').textContent = state.backendAvailable ? 'New-video AI ready' : 'Verified sample mode';
  $('#runMode').textContent = state.backendAvailable ? 'Ready for multimodal inference' : 'Ready with verified evidence';
  $('#runHint').textContent = state.backendAvailable ? 'Upload any episode or run the supplied sample' : 'New videos require the included Python backend';
}

async function init() {
  try {
    const manifest = await readJSON(DATA.manifest);
    const inputs = manifest.inputs || {};
    const [groundTruth, script, mapText, rules] = await Promise.all([
      readText(inputs.ground_truth),
      readText(inputs.script),
      readText(inputs.character_map),
      readText(inputs.rules)
    ]);
    state.manifest = manifest;
    state.groundTruth = groundTruth;
    state.originalScript = script;
    state.characterMap = JSON.parse(mapText);
    state.rules = rules;
    applyManifest(manifest);
    $('#scriptInput').value = script;
    $('#mapInput').value = JSON.stringify(state.characterMap, null, 2);
    $('#rulesInput').value = rules;
    renderMap();
    renderRules();
    updateScriptStats();
    setInputState('#draftInputState', 'ready', 'Ready');
    setInputState('#mapInputState', 'ready', 'Ready');
    setInputState('#rulesInputState', 'ready', 'Ready');
    $('#cleanScript').textContent = script;
    bindUI();
    await checkBackend();
    syncRunAvailability();
  } catch (error) {
    console.error(error);
    $('#scriptInput').value = 'Demo data could not be loaded. Refresh the page to try again.';
    showToast('Demo data could not be loaded');
  }
}

init();
