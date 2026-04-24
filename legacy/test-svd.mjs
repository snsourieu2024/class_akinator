import { svd, matrix, multiply, transpose, diag } from 'mathjs';

const r = svd([[1, 0], [0, 1]]);
console.log('s', r.s);
console.log('u rows', r.u.size());
console.log('v rows', r.v.size());
