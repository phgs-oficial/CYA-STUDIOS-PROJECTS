/* ========= SETTINGS ========= */
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

const TILE_COUNT = 20;
const CANVAS_SIZE = 400;
canvas.width = CANVAS_SIZE; canvas.height = CANVAS_SIZE;
const TILE = CANVAS_SIZE / TILE_COUNT;

const ACTIONS = ['STRAIGHT','LEFT','RIGHT']; // relative actions
const DIRS = ['up','right','down','left']; // order matters for left/right rotation

/* hyperparams */
const LEARNING_RATE = 0.1;   // alpha
const GAMMA = 0.9;           // gamma
let EPSILON = 1.0;           // exploration start
const EPSILON_MIN = 0.05;
const EPSILON_DECAY = 0.9995;

/* training speed */
const LOOPS_PER_TICK = 1200; // how many steps per interval while training (acelerado)
const TRAIN_INTERVAL_MS = 0; // setInterval delay for training ticks

/* ========= GLOBAL STATE ========= */
let wins = 0, deaths = 0, episodes = 0;
const winsSpan = document.getElementById('wins');
const deathsSpan = document.getElementById('deaths');
const episodesSpan = document.getElementById('episodes');

let runInterval = null;
let trainInterval = null;

/* ========= GAME (corrected movement & collision) ========= */
class Game {
  constructor() { this.reset(); }

  reset() {
    this.head = { x: Math.floor(TILE_COUNT/2), y: Math.floor(TILE_COUNT/2) };
    this.dir = 'right';
    this.snake = [ {...this.head} ]; // tail..head
    this.length = 3;
    this.placeFruit();
    this.alive = true;
    this.stepCount = 0;
  }

  placeFruit() {
    let tries = 0;
    while (tries < 1000) {
      const p = { x: Math.floor(Math.random()*TILE_COUNT), y: Math.floor(Math.random()*TILE_COUNT) };
      if (!this.snake.some(s => s.x === p.x && s.y === p.y)) { this.fruit = p; return; }
      tries++;
    }
    this.fruit = { x:0, y:0 }; // fallback
  }

  // compute next head from a direction string
  nextHeadForDir(dir) {
    const nh = { x: this.head.x, y: this.head.y };
    if (dir === 'up') nh.y -= 1;
    if (dir === 'down') nh.y += 1;
    if (dir === 'left') nh.x -= 1;
    if (dir === 'right') nh.x += 1;
    return nh;
  }

  // apply relative action: STRAIGHT / LEFT / RIGHT
  applyActionRelative(action) {
    // find current dir index in DIRS
    const idx = DIRS.indexOf(this.dir);
    let newIdx = idx;
    if (action === 'LEFT') newIdx = (idx + 3) % 4; // -1 mod 4
    else if (action === 'RIGHT') newIdx = (idx + 1) % 4;
    // STRAIGHT => same idx
    this.dir = DIRS[newIdx];
  }

  // perform a game step using a relative action; returns {reward, done}
  stepWithAction(action) {
    // compute next direction and head
    this.applyActionRelative(action);
    const nh = this.nextHeadForDir(this.dir);

    // willGrow if fruit at nextHead
    const willGrow = (nh.x === this.fruit.x && nh.y === this.fruit.y);

    // collision with walls
    if (nh.x < 0 || nh.x >= TILE_COUNT || nh.y < 0 || nh.y >= TILE_COUNT) {
      this.resetAfterDeath();
      return { reward:-1, done:true };
    }

    // collision with body
    // when not growing, tail will move off (so we can ignore tail cell)
    const bodyToCheck = willGrow ? this.snake : this.snake.slice(1);
    if (bodyToCheck.some(s => s.x === nh.x && s.y === nh.y)) {
      this.resetAfterDeath();
      return { reward:-1, done:true };
    }

    // move: push new head
    this.snake.push({ x: nh.x, y: nh.y });
    this.head = { x: nh.x, y: nh.y };

    if (willGrow) {
      this.length++;
      // eaten fruit
      this.placeFruit();
      this.stepCount = 0;
      return { reward: +1, done:false };
    } else {
      // normal move: maintain length
      while (this.snake.length > this.length) this.snake.shift();
      this.stepCount++;
      return { reward: 0, done:false };
    }
  }

  resetAfterDeath() {
    deaths++;
    deathsSpan.innerText = deaths;
    // reset game to initial
    this.reset();
  }

  getStateVector() {
    // state = danger ahead/left/right (0/1), dir one-hot (4), food left/right/up/down (4) => 11 bits
    // compute offsets based on current dir
    const head = this.head;

    const dir = this.dir;
    let forward = {x:0,y:0}, left = {x:0,y:0}, right = {x:0,y:0};
    if (dir === 'up') { forward={x:0,y:-1}; left={x:-1,y:0}; right={x:1,y:0}; }
    if (dir === 'down') { forward={x:0,y:1}; left={x:1,y:0}; right={x:-1,y:0}; }
    if (dir === 'left') { forward={x:-1,y:0}; left={x:0,y:1}; right={x:0,y:-1}; }
    if (dir === 'right') { forward={x:1,y:0}; left={x:0,y:-1}; right={x:0,y:1}; }

    const test = (ox, oy) => {
      const x = head.x + ox;
      const y = head.y + oy;
      if (x < 0 || x >= TILE_COUNT || y < 0 || y >= TILE_COUNT) return 1;
      if (this.snake.some(s => s.x === x && s.y === y)) return 1;
      return 0;
    };

    const dangerAhead = test(forward.x, forward.y);
    const dangerLeft = test(left.x, left.y);
    const dangerRight = test(right.x, right.y);

    const dirOneHot = [
      dir === 'up' ? 1:0,
      dir === 'right' ? 1:0,
      dir === 'down' ? 1:0,
      dir === 'left' ? 1:0
    ];

    const fx = this.fruit.x - head.x;
    const fy = this.fruit.y - head.y;
    const fruitLeft = fx < 0 ? 1:0;
    const fruitRight = fx > 0 ? 1:0;
    const fruitUp = fy < 0 ? 1:0;
    const fruitDown = fy > 0 ? 1:0;

    return [
      dangerAhead, dangerLeft, dangerRight,
      ...dirOneHot,
      fruitLeft, fruitRight, fruitUp, fruitDown
    ];
  }

  draw() {
    // background
    ctx.fillStyle = '#000';
    ctx.fillRect(0,0,CANVAS_SIZE,CANVAS_SIZE);

    // fruit
    ctx.fillStyle = '#f33';
    ctx.fillRect(this.fruit.x * TILE, this.fruit.y * TILE, TILE-1, TILE-1);

    // snake
    ctx.fillStyle = '#58d68d';
    for (let i=0;i<this.snake.length;i++){
      const p = this.snake[i];
      ctx.fillRect(p.x * TILE, p.y * TILE, TILE-1, TILE-1);
    }
  }
}

/* ========= Q-LEARNING AGENT ========= */
class QAgent {
  constructor() {
    this.q = {}; // map stateKey -> {STRAIGHT:val, LEFT:val, RIGHT:val}
    this.lr = LEARNING_RATE;
    this.gamma = GAMMA;
    this.epsilon = EPSILON;
    this.episodes = 0;
    this.score = 0;
  }

  stateKey(stateVec) { return stateVec.join(','); }

  ensureState(stateKey) {
    if (!this.q[stateKey]) {
      this.q[stateKey] = { STRAIGHT:0, LEFT:0, RIGHT:0 };
    }
    return this.q[stateKey];
  }

  chooseAction(stateVec) {
    const key = this.stateKey(stateVec);
    const table = this.ensureState(key);

    // epsilon-greedy
    if (Math.random() < this.epsilon) {
      return ACTIONS[Math.floor(Math.random()*ACTIONS.length)];
    }

    // pick best (tie-breaker random among best)
    const vals = ACTIONS.map(a => table[a]);
    const maxv = Math.max(...vals);
    const candidates = ACTIONS.filter((a,i)=>vals[i] === maxv);
    return candidates[Math.floor(Math.random()*candidates.length)];
  }

  update(stateVec, action, reward, nextStateVec) {
    const key = this.stateKey(stateVec);
    const nextKey = this.stateKey(nextStateVec);
    const table = this.ensureState(key);
    const nextTable = this.ensureState(nextKey);

    const bestNext = Math.max(nextTable.STRAIGHT, nextTable.LEFT, nextTable.RIGHT);
    const target = reward + this.gamma * bestNext;
    table[action] = table[action] + this.lr * (target - table[action]);

    // decay epsilon a bit
    if (this.epsilon > EPSILON_MIN) this.epsilon *= EPSILON_DECAY;
  }

  reset() {
    this.q = {};
    this.epsilon = EPSILON;
    this.episodes = 0;
    this.score = 0;
  }
}

/* ========= MAIN / UI ========= */
const game = new Game();
const agent = new QAgent();

// draw initial
game.draw();
updateSpans();

document.getElementById('runBtn').addEventListener('click', () => {
  stopAll();
  // run visual at 12 fps
  runInterval = setInterval(() => {
    stepVisual();
  }, 1000/12);
});

document.getElementById('trainBtn').addEventListener('click', () => {
  stopAll();
  // training: large loops per tick without render
  trainInterval = setInterval(() => {
    for (let i=0;i<LOOPS_PER_TICK;i++){
      stepTrain();
    }
    episodes++;
    episodesSpan.innerText = episodes;
  }, TRAIN_INTERVAL_MS);
});

document.getElementById('stopBtn').addEventListener('click', () => {
  stopAll();
});

function stopAll() {
  if (runInterval) { clearInterval(runInterval); runInterval = null; }
  if (trainInterval) { clearInterval(trainInterval); trainInterval = null; }
}

function updateSpans() {
  winsSpan.innerText = wins;
  deathsSpan.innerText = deaths;
  episodesSpan.innerText = episodes;
}

/* visual step: choose action, step, render, update q */
function stepVisual() {
  const state = game.getStateVector();
  const action = agent.chooseAction(state);
  const res = game.stepWithAction(action);
  const nextState = game.getStateVector();

  // update q
  agent.update(state, action, res.reward, nextState);

  if (res.reward > 0) { wins++; updateSpans(); }
  if (res.reward < 0) { /* death already counted in resetAfterDeath */ updateSpans(); }
  // draw
  game.draw();
}

/* train step: same but no draw and fast loops */
function stepTrain() {
  const state = game.getStateVector();
  const action = agent.chooseAction(state);
  const res = game.stepWithAction(action);
  const nextState = game.getStateVector();

  agent.update(state, action, res.reward, nextState);

  if (res.reward > 0) { wins++; }
  if (res.reward < 0) { /* death handled in game */ }
  // if game died, game.resetAfterDeath already incremented deaths; ensure counters reflect that
  // update global counters rarely (to avoid slowdown); we will update UI after each tick in trainInterval
}

/* keep UI counters updated periodically when training */
setInterval(() => {
  updateSpans();
}, 250);

/* keyboard for manual testing (optional) */
window.addEventListener('keydown', (e) => {
  const map = { ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right' };
  if (map[e.key]) {
    game.dir = map[e.key];
  }
});

console.log('SNAKE AI INICIADA');

