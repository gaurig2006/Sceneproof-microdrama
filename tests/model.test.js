const assert = require('assert');
const fs = require('fs');
const path = require('path');
const model = require('../dist/model.js');

const data = path.join(__dirname, '..', 'dist', 'data');
const text = name => fs.readFileSync(path.join(data, name), 'utf8');

const result = model.review({
  evidence: text('moonlight_is_not_late_ep01_ground_truth.txt'),
  script: text('moonlight_is_not_late_ep01_flawed_indian_adaptation.txt'),
  characterMap: JSON.parse(text('moonlight_is_not_late_character_map.json')),
  rules: text('moonlight_is_not_late_adaptation_rules.txt')
});

assert.strictEqual(result.issues.length, 5);
assert.strictEqual(result.metadata.hardcodedShowKnowledge, false);
assert.ok(result.metadata.featureCount > 100);
assert.ok(result.issues.some(issue => issue.detectedBy.includes('TF-IDF')));

const unseen = model.review({
  evidence: '**ACTION:**\nStart: (00:10.00) | End: (00:15.00)\nRiya tears the contract in front of Dev.\n',
  script: '**ACTION:**\nStart: (00:00.00) | End: (00:05.00)\nRiya enters the office.\n',
  characterMap: { characters: [] },
  rules: '1. Ending: Preserve Riya tearing the contract in front of Dev.'
});

assert.strictEqual(unseen.issues.length, 1);
assert.strictEqual(unseen.issues[0].type, 'Missing required beat');
console.log('browser alignment model tests passed');
