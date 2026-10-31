const fs = require('node:fs');
const {spawnSync, execFileSync} = require('node:child_process');
const {resolve, join} = require('node:path');
const {createHash} = require('node:crypto');
const [base, head, tools, out] = process.argv.slice(2).map(x => resolve(x));
const hash = x => createHash('sha256').update(x).digest('hex');
for (const path of ['Cargo.lock', 'crates/metrics/tests/breakdown_order.rs']) {
  if (!fs.readFileSync(join(base, path)).equals(fs.readFileSync(join(head, path)))) throw Error('Frozen fixture drift: ' + path);
}
fs.mkdirSync(out + '.logs', {recursive: true});
function run(root, label, key, args, seconds = 90, command = 'cargo') {
  const started = Date.now();
  const r = spawnSync(command, args, {cwd: root, encoding: 'utf8', timeout: seconds * 1000, maxBuffer: 32 * 1024 * 1024,
    env: {...process.env, CARGO_TARGET_DIR: '/tmp/latency-oct31-target', CARGO_PROFILE_DEV_DEBUG: '0', CARGO_PROFILE_TEST_DEBUG: '0'}});
  const text = (r.stdout || '') + (r.stderr || '');
  fs.writeFileSync(join(out + '.logs', label + '-' + key + '.log'), text);
  return {command: [command, ...args], exit: r.status, signal: r.signal, error: r.error?.message, ms: Date.now() - started,
    sha256: hash(text), passedNames: [...text.matchAll(/^test ([^\n]+) \.\.\. ok$/gm)].map(x => x[1]).sort(), pass: r.status === 0 && !r.error && !r.signal};
}
function measure(root, label) {
  const locked = ['--locked', '--offline', '-j2'];
  const checks = {
    probe: run(root, label, 'probe', ['+1.88.0', 'test', '-p', 'llm-latency-lens-metrics', '--test', 'breakdown_order', ...locked]),
    metrics: run(root, label, 'metrics', ['+1.88.0', 'test', '-p', 'llm-latency-lens-metrics', '--lib', ...locked]),
  };
  for (const op of ['test', 'build', 'clippy']) checks[op] = run(root, label, op,
    ['+1.88.0', op, '--workspace', '--all-features', ...locked, ...(op === 'clippy' ? ['--all-targets', '--', '-D', 'warnings'] : [])]);
  checks.fmt = run(root, label, 'fmt', ['+1.88.0', 'fmt', '--all', '--', '--check'], 30);
  checks.audit = run(root, label, 'audit', ['audit', '--file', 'Cargo.lock', '--no-fetch', '--json'], 30);
  checks.npmAudit = run(root, label, 'npm-audit', ['audit', '--json'], 30, 'npm');
  return checks;
}
(async () => {
  const baseline = measure(base, 'baseline'), candidate = measure(head, 'candidate');
  const lib = join(tools, 'node_modules/@metaharness/flywheel');
  if (require(join(lib, 'package.json')).version !== '0.1.12') throw Error('Tool version drift');
  const fw = await import(join(lib, 'dist/index.js'));
  const revision = root => execFileSync('git', ['rev-parse', 'HEAD'], {cwd: root, encoding: 'utf8'}).trim();
  const b = revision(base), h = revision(head), path = 'crates/metrics/src/aggregator.rs', parentBytes = fs.readFileSync(join(base, path), 'utf8');
  const names = ['full_breakdowns_are_sorted', 'provider_subset_breakdowns_are_sorted', 'model_subset_breakdowns_are_sorted'];
  function score(checks, suite) {
    const primary = suite === 'holdout' ? names.filter(x => checks.probe.passedNames.includes(x)).length / names.length
      : checks.metrics.passedNames.length / 35;
    return {primary, noopRate: 1 - primary, costPerWin: 0, regressed: Object.values(checks).some(x => !x.pass)};
  }
  const result = await fw.runFlywheelGenerations({rootPolicy: {revision: b}, proposer: async () => ({value: h,
    summary: 'Lexical provider and model breakdown ordering across all aggregation paths', inverse: {path, parentBytes, sha256: hash(parentBytes)}}),
    evaluator: async (p, s) => score(p.revision === b ? baseline : candidate, s.id),
    holdout: {id: 'holdout', items: names}, anchor: {id: 'anchor', items: Array.from({length: 35}, (_, i) => `existing-metrics-${i}`)},
    mutationTargets: ['revision'], maxGenerations: 1, signer: fw.makeSigner(), promotionRule: fw.meetsPromotionRule, dataSource: 'LIVE'});
  const fingerprint = fw.gateFingerprint(fw.meetsPromotionRule), replay = fw.verifyReplayBundle(result.replayBundle, {pinnedGateFingerprint: fingerprint});
  const lostPassingNames = baseline.metrics.passedNames.filter(x => !candidate.metrics.passedNames.includes(x));
  fs.writeFileSync(out, JSON.stringify({baselineRevision: b, candidateRevision: h, baseline, candidate, lostPassingNames, result, replay, fingerprint,
    rollback: {path, parentBytes, sha256: hash(parentBytes)}, limitations: 'Original metrics suite has35tests; broad failures block promotion. Neutral unmeasured cost; noopRate is failed-case proxy. No external service integration or production performance claim.'}, null, 2));
  console.log(JSON.stringify({promotions: result.promotions.length, replay, lostPassingNames}));
})().catch(e => {console.error(e); process.exitCode = 1;});
