(function attachSceneProofModel(root) {
  'use strict';

  const BLOCK_RE = /\*\*([^*]+)\*\*:?\s*\nStart:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)\s*\n([\s\S]*?)(?=\n\n\*\*|\n--- SCRIPT END|$)/g;
  const RULE_RE = /^\s*(\d+)\.\s*([^:]+):\s*(.+?)\s*$/;
  const STOP_WORDS = new Set('a an and are as at be been before but by can did do does for from had has have he her hers him his how i in into is it its me my no not of on or our she so than that the their them they this to was we were what when where which while who will with you your'.split(' '));
  const DEFAULT_CONFIG = Object.freeze({
    relevanceFloor: 0.06,
    candidateRatio: 0.80,
    maxCandidatesPerRule: 2,
    minimumTimeOverlap: 0.15,
    preservationFloor: 0.60,
    criticalConfidence: 0.75
  });

  function clean(value) {
    return String(value || '').replace(/\s+/g, ' ').trim();
  }

  function seconds(value) {
    const parts = String(value).trim().split(':').map(Number);
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    return parts[0] || 0;
  }

  function formatTime(value) {
    const safe = Math.max(Number(value) || 0, 0);
    const minutes = Math.floor(safe / 60);
    const remainder = safe - minutes * 60;
    return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(2).padStart(5, '0')}`;
  }

  function slug(value) {
    return clean(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'issue';
  }

  function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function parseBlocks(screenplay) {
    const blocks = [];
    for (const match of String(screenplay || '').matchAll(BLOCK_RE)) {
      blocks.push({
        label: clean(match[1]),
        start: seconds(match[2]),
        end: seconds(match[3]),
        text: clean(match[4])
      });
    }
    return blocks;
  }

  function parseRequirements(rules) {
    return String(rules || '').split('\n').flatMap(line => {
      const match = line.match(RULE_RE);
      return match ? [{ number: Number(match[1]), title: clean(match[2]), text: clean(match[3]), fullText: `${clean(match[2])}. ${clean(match[3])}` }] : [];
    });
  }

  function characterPairs(characterMap) {
    return (characterMap?.characters || []).flatMap(character => {
      const target = clean(character.adapted_name);
      if (!target) return [];
      return ['source_name', 'romanized_name'].flatMap(field => {
        const source = clean(character[field]);
        return source && source.toLocaleLowerCase() !== target.toLocaleLowerCase() ? [[source, target]] : [];
      });
    });
  }

  function normalizeSource(text, pairs) {
    return [...pairs]
      .sort((left, right) => right[0].length - left[0].length)
      .reduce((value, [source, target]) => value.replace(new RegExp(escapeRegExp(source), 'gi'), target), text);
  }

  function extractForbiddenTerms(requirement) {
    if (!/\b(?:do not retain|must not contain|forbid|avoid using|remove the term)\b/i.test(requirement.text)) return [];
    const quoted = [...requirement.text.matchAll(/["“]([^"”]{2,60})["”]/g)].map(match => clean(match[1]));
    if (quoted.length) return quoted;
    const match = requirement.text.match(/(?:term|word|phrase)\s+([A-Za-z][A-Za-z0-9_-]{1,40})/i);
    return match ? [match[1]] : [];
  }

  function isPreservationRule(requirement) {
    return !extractForbiddenTerms(requirement).length && /\b(?:preserve|must|required|required to|do not (?:soften|remove|omit)|protect|retain)\b/i.test(requirement.text);
  }

  function termCounts(document) {
    const words = (clean(document).toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(word => !STOP_WORDS.has(word));
    const features = new Map();
    const add = key => features.set(key, (features.get(key) || 0) + 1);
    words.forEach(word => add(`w:${word}`));
    for (let index = 0; index < words.length - 1; index += 1) add(`b:${words[index]}_${words[index + 1]}`);
    const characters = ` ${words.join(' ')} `;
    for (let size = 3; size <= 5; size += 1) {
      for (let index = 0; index <= characters.length - size; index += 1) add(`c:${characters.slice(index, index + size)}`);
    }
    return features;
  }

  function fitVectors(documents) {
    const counts = documents.map(termCounts);
    const documentFrequency = new Map();
    counts.forEach(features => features.forEach((_, key) => documentFrequency.set(key, (documentFrequency.get(key) || 0) + 1)));
    const vectors = counts.map(features => {
      const vector = new Map();
      let norm = 0;
      features.forEach((count, key) => {
        const tf = 1 + Math.log(count);
        const idf = Math.log((1 + documents.length) / (1 + documentFrequency.get(key))) + 1;
        const weight = tf * idf;
        vector.set(key, weight);
        norm += weight * weight;
      });
      norm = Math.sqrt(norm) || 1;
      vector.forEach((weight, key) => vector.set(key, weight / norm));
      return vector;
    });
    return { vectors, featureCount: documentFrequency.size };
  }

  function cosine(left, right) {
    let score = 0;
    const [small, large] = left.size <= right.size ? [left, right] : [right, left];
    small.forEach((weight, key) => { score += weight * (large.get(key) || 0); });
    return score;
  }

  function overlap(left, right) {
    const intersection = Math.max(0, Math.min(left.end, right.end) - Math.max(left.start, right.start));
    return intersection / Math.max(Math.min(left.end - left.start, right.end - right.start), 0.01);
  }

  function sceneTimecode(screenplay) {
    const start = String(screenplay).match(/^Scene start:\s*([0-9:.]+)/im);
    const end = String(screenplay).match(/^Scene end:\s*([0-9:.]+)/im);
    return start && end ? `${start[1]}–${end[1]}` : 'unknown';
  }

  function directIssues(script, adaptedBlocks, pairs, requirements) {
    const issues = [];
    pairs.forEach(([source, target]) => {
      if (!new RegExp(escapeRegExp(source), 'i').test(script)) return;
      const block = adaptedBlocks.find(item => new RegExp(escapeRegExp(source), 'i').test(item.text));
      issues.push({
        id: `name-${slug(source)}`,
        severity: 'warning',
        type: 'Character-map mismatch',
        title: 'A source character name remains in the adaptation',
        explanation: 'This flag was generated by comparing the draft with the supplied character map.',
        evidence: `Character map: ${source} → ${target}`,
        timecode: block ? `${formatTime(block.start)}–${formatTime(block.end)}` : 'unknown',
        proposal: `Replace “${source}” with “${target}”.`,
        confidence: 1,
        detectedBy: 'input-derived constraint',
        targetText: source,
        apply: value => value.replace(new RegExp(escapeRegExp(source), 'g'), target)
      });
    });

    requirements.forEach(requirement => extractForbiddenTerms(requirement).forEach(term => {
      if (!new RegExp(escapeRegExp(term), 'i').test(script)) return;
      const block = adaptedBlocks.find(item => new RegExp(escapeRegExp(term), 'i').test(item.text));
      const replacementMatch = requirement.text.match(/\binto\s+(?:an?\s+)?(.+?)(?:\s+hosting|[.,]|$)/i);
      const replacement = clean(replacementMatch?.[1]) || 'target-culture equivalent';
      issues.push({
        id: `rule-${requirement.number}-${slug(term)}`,
        severity: 'warning',
        type: 'Forbidden-term violation',
        title: 'The draft retains a term forbidden by the supplied rules',
        explanation: `The model derived this constraint from requirement ${requirement.number}, not from a built-in term list.`,
        evidence: requirement.fullText,
        timecode: block ? `${formatTime(block.start)}–${formatTime(block.end)}` : sceneTimecode(script),
        proposal: `Replace “${term}” with “${replacement}”.`,
        confidence: 1,
        detectedBy: 'input-derived constraint',
        targetText: term,
        apply: value => value.replace(new RegExp(escapeRegExp(term), 'gi'), replacement)
      });
    }));
    return issues;
  }

  function semanticAnomalies(sourceBlocks, adaptedBlocks, requirements, config) {
    const preservationRules = requirements.filter(isPreservationRule);
    if (!sourceBlocks.length || !adaptedBlocks.length || !preservationRules.length) return { anomalies: [], metadata: { documents: 0, features: 0 } };
    const documents = [...sourceBlocks.map(item => item.text), ...adaptedBlocks.map(item => item.text), ...preservationRules.map(item => item.fullText)];
    const { vectors, featureCount } = fitVectors(documents);
    const sourceCount = sourceBlocks.length;
    const adaptedCount = adaptedBlocks.length;
    const anomalies = [];

    preservationRules.forEach((requirement, ruleOffset) => {
      const ruleVector = vectors[sourceCount + adaptedCount + ruleOffset];
      const relevance = sourceBlocks.map((_, index) => cosine(vectors[index], ruleVector));
      const topRelevance = Math.max(...relevance);
      if (topRelevance < config.relevanceFloor) return;
      const floor = Math.max(config.relevanceFloor, topRelevance * config.candidateRatio);
      const candidates = relevance.map((score, index) => ({ score, index })).filter(item => item.score >= floor).sort((a, b) => b.score - a.score).slice(0, config.maxCandidatesPerRule);

      candidates.forEach(candidate => {
        const source = sourceBlocks[candidate.index];
        const overlapping = adaptedBlocks.map((target, index) => ({ target, index })).filter(item => overlap(source, item.target) >= config.minimumTimeOverlap);
        if (!overlapping.length) {
          anomalies.push({ requirement, source, targetIndex: null, preservationScore: 0, reason: 'missing' });
          return;
        }
        const target = overlapping.sort((left, right) => cosine(vectors[candidate.index], vectors[sourceCount + right.index]) - cosine(vectors[candidate.index], vectors[sourceCount + left.index]))[0];
        const sourceToTarget = cosine(vectors[candidate.index], vectors[sourceCount + target.index]);
        const targetToRule = cosine(vectors[sourceCount + target.index], ruleVector);
        const relativeRuleCoverage = Math.min(targetToRule / Math.max(candidate.score, 0.001), 1);
        const preservationScore = Math.max(sourceToTarget, relativeRuleCoverage);
        if (preservationScore < config.preservationFloor) anomalies.push({ requirement, source, targetIndex: target.index, preservationScore, reason: 'semantic drift' });
      });
    });
    return { anomalies, metadata: { documents: documents.length, features: featureCount } };
  }

  function groupAnomalies(anomalies) {
    const groups = [];
    [...anomalies].sort((a, b) => a.source.start - b.source.start || a.requirement.number - b.requirement.number).forEach(anomaly => {
      const group = groups.find(items => (anomaly.targetIndex !== null && anomaly.targetIndex === items[0].targetIndex) || items.some(item => Math.min(anomaly.source.end, item.source.end) > Math.max(anomaly.source.start, item.source.start)));
      if (group) group.push(anomaly); else groups.push([anomaly]);
    });
    return groups;
  }

  function anomalyIssue(group, adaptedBlocks, config) {
    const start = Math.min(...group.map(item => item.source.start));
    const end = Math.max(...group.map(item => item.source.end));
    const ruleNumbers = [...new Set(group.map(item => item.requirement.number))].sort((a, b) => a - b);
    const evidence = [...new Set(group.map(item => item.source.text))].join(' ');
    const missing = group.every(item => item.reason === 'missing');
    const score = Math.max(...group.map(item => item.preservationScore));
    const targetIndex = group.find(item => item.targetIndex !== null)?.targetIndex ?? null;
    const targetText = targetIndex === null ? '' : adaptedBlocks[targetIndex].text;
    const rulesLabel = ruleNumbers.join(', ');
    const replacement = `[MODEL-PROPOSED RESTORED BEAT — verify dialogue] ${evidence}`;
    const anchor = targetText || [...adaptedBlocks].reverse().find(block => block.end <= start)?.text || '';
    return {
      id: `alignment-rules-${ruleNumbers.join('-')}-${Math.floor(start * 100)}`,
      severity: 1 - score >= config.criticalConfidence ? 'critical' : 'warning',
      type: missing ? 'Missing required beat' : 'Semantic drift',
      title: missing ? 'A rule-protected source beat is missing' : 'The adaptation changes a rule-protected meaning',
      explanation: `The per-run alignment model linked this evidence to requirement(s) ${rulesLabel} and found ${missing ? 'no overlapping adapted block' : `low semantic preservation (score ${score.toFixed(2)})`}.`,
      evidence,
      timecode: `${formatTime(start)}–${formatTime(end)}`,
      proposal: `Restore the meaning while applying requirement(s) ${rulesLabel}: ${evidence}`,
      confidence: Number((1 - score).toFixed(3)),
      detectedBy: 'per-run TF-IDF + timestamp alignment model',
      targetText: anchor,
      apply: value => {
        if (targetText && value.includes(targetText)) return value.replace(targetText, replacement);
        return anchor && value.includes(anchor) ? value.replace(anchor, `${anchor}\n\n${replacement}`) : `${value}\n\n${replacement}`;
      }
    };
  }

  function review({ evidence, script, characterMap, rules, config: suppliedConfig }) {
    const config = { ...DEFAULT_CONFIG, ...(suppliedConfig || {}) };
    const pairs = characterPairs(characterMap || {});
    const sourceBlocks = parseBlocks(evidence).map(block => ({ ...block, text: normalizeSource(block.text, pairs) }));
    const adaptedBlocks = parseBlocks(script);
    const requirements = parseRequirements(rules);
    const direct = directIssues(script, adaptedBlocks, pairs, requirements);
    const { anomalies, metadata } = semanticAnomalies(sourceBlocks, adaptedBlocks, requirements, config);
    const learned = groupAnomalies(anomalies).map(group => anomalyIssue(group, adaptedBlocks, config));
    const issues = [...direct, ...learned].sort((left, right) => seconds(left.timecode.split('–')[0]) - seconds(right.timecode.split('–')[0]));
    issues.forEach((issue, index) => {
      issue.order = index + 1;
      issue.target = line => issue.targetText ? line.includes(issue.targetText) : false;
    });
    return {
      issues,
      metadata: {
        model: 'per-run TF-IDF word/character vectors + timestamp alignment',
        sourceBlocks: sourceBlocks.length,
        adaptedBlocks: adaptedBlocks.length,
        requirements: requirements.length,
        documentsFitted: metadata.documents,
        featureCount: metadata.features,
        hardcodedShowKnowledge: false,
        thresholds: config
      }
    };
  }

  const api = { review, parseBlocks, parseRequirements };
  root.SceneProofModel = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
