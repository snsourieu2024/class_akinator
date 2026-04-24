import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { readFileSync } from 'fs';

const html = readFileSync('./index.html', 'utf8');
const mMatch = html.match(/const KNOWLEDGE_MATRIX = (\[[\s\S]*?\]);/);
if (!mMatch) throw new Error('matrix not found');
const FULL = eval(mMatch[1]);
const target = 34; // Elshan
const row = FULL[target];
const QUESTIONS_LEN = 38;
const K = 14;

const Mmat = new Matrix(FULL);
const svd = new SingularValueDecomposition(Mmat);
const V = svd.rightSingularVectors;
const VkCols = V.subMatrixColumn([...Array(K).keys()]);
const Zmat = Mmat.mmul(VkCols);
const Z = [];
for (let i = 0; i < FULL.length; i++) {
  const r = [];
  for (let c = 0; c < K; c++) r.push(Zmat.get(i, c));
  Z.push(r);
}
const Vk = [];
for (let r = 0; r < K; r++) {
  const rowv = [];
  for (let j = 0; j < QUESTIONS_LEN; j++) rowv.push(V.get(j, r));
  Vk.push(rowv);
}

function Hbin(p) {
  if (p <= 0 || p >= 1) return 0;
  return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
}
function norm01(arr) {
  const mn = Math.min(...arr), mx = Math.max(...arr);
  if (mx === mn) return arr.map(() => 0.5);
  return arr.map(x => (x - mn) / (mx - mn));
}
function vecNorm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function beliefFromScores(scores) {
  const shifted = scores.map(x => Math.max(x, 1e-12));
  const s = shifted.reduce((a, x) => a + x, 0);
  return shifted.map(x => x / s);
}
function computeRawScores(aObj) {
  const idxs = Object.keys(aObj).map(Number);
  const out = [];
  for (let i = 0; i < FULL.length; i++) {
    let s = 0;
    for (const j of idxs) {
      const aj = aObj[j] ? 1 : 0;
      s += 1 - Math.abs(FULL[i][j] - aj);
    }
    out.push(idxs.length ? s : 1);
  }
  return out;
}
function projectUser(aObj) {
  const idxs = Object.keys(aObj).map(Number).sort((a, b) => a - b);
  const z = new Array(K).fill(0);
  for (let r = 0; r < K; r++) {
    let s = 0;
    for (const j of idxs) s += Vk[r][j] * (aObj[j] ? 1 : 0);
    z[r] = s;
  }
  return z;
}
function computeSvdCosineScores(aObj) {
  const zu = projectUser(aObj);
  const nu = vecNorm(zu);
  const out = [];
  for (let i = 0; i < FULL.length; i++) {
    const zi = Z[i];
    const ni = vecNorm(zi);
    if (nu < 1e-15 || ni < 1e-15) out.push(0);
    else out.push(dot(zi, zu) / (ni * nu));
  }
  return out;
}
function finalScores(aObj) {
  const nAns = Object.keys(aObj).length;
  const raw = computeRawScores(aObj);
  const rawN = norm01(raw);
  if (nAns < 4) return rawN;
  const svdS = computeSvdCosineScores(aObj);
  const svdN = norm01(svdS);
  return rawN.map((r, i) => 0.6 * svdN[i] + 0.4 * r);
}
function pickQuestion(aObj) {
  const answered = new Set(Object.keys(aObj).map(Number));
  const scores = finalScores(aObj);
  const b = beliefFromScores(scores);
  let bestJ = -1, bestH = -1;
  for (let j = 0; j < QUESTIONS_LEN; j++) {
    if (answered.has(j)) continue;
    let p = 0;
    for (let i = 0; i < FULL.length; i++) p += b[i] * FULL[i][j];
    const h = Hbin(p);
    if (h > bestH) { bestH = h; bestJ = j; }
  }
  return bestJ;
}
function shouldGuess(scores, nAnswered) {
  if (nAnswered < 3) return false;
  const idx = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
  const s0 = idx[0].s, s1 = idx[1].s;
  if (s0 <= 0) return false;
  return s0 - s1 > 0.25 * s0;
}

const QUESTION_NAMES = [
  'European', 'Spanish', 'American', 'Caribbean', 'Eastern European', 'Latin American',
  'Arab', 'Asian', 'North American', 'African', 'male', 'tall', 'dark hair', 'blonde',
  'red hair', 'blue/dyed', 'long hair', 'curly', 'short hair', 'dark eyes', 'light eyes',
  'facial hair', 'glasses', 'hoodie', 'cap', 'outgoing', 'quiet', 'funny', 'competitive',
  'chill', 'asks in class', 'mixed bg', 'late', 'front row', 'back row', 'flatmates', 'family', 'alone',
];

const answers = {};
for (let t = 0; t < 20; t++) {
  const j = pickQuestion(answers);
  const truth = row[j] === 1;
  answers[j] = truth;
  const scores = finalScores(answers);
  const top = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
  console.log(
    'step', t + 1, '| asked:', QUESTION_NAMES[j], '(' + j + ') =', truth ? 'Y' : 'N',
    '| top:', top[0].i, top[0].s.toFixed(3), '2nd:', top[1].i, top[1].s.toFixed(3),
  );
  if (shouldGuess(scores, Object.keys(answers).length)) {
    console.log('>>> GUESS person index', top[0].i, top[0].i === target ? '(Elshan OK)' : '(wrong)');
    break;
  }
}
