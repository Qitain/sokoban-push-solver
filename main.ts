import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Boxes cannot return from the dead hallway, so independent room optima add.
const MAX_EXPANSIONS = 250_000;
const WALK_LETTERS = ["u", "d", "l", "r"];
const PUSH_LETTERS = ["U", "D", "L", "R"];
const UNREACHABLE = 1_000_000;
const MAX_STAMP = 0x7fff_ffff;

interface Level {
  width: number;
  height: number;
  size: number;
  wall: Uint8Array;
  goals: number[];
  goalMask: Uint8Array;
  boxes: number[];
  player: number;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

interface Push {
  from: number;
  to: number;
}

interface Flood {
  deltas: number[];
  occupied: Uint8Array;
  seen: Int32Array;
  queue: Int32Array;
  stamp: number;
}

interface SearchNode {
  boxes: number[];
  low: number;
  g: number;
  f: number;
  order: number;
  key: number;
  parent: SearchNode | null;
  from: number;
  to: number;
}

interface SearchResult {
  pushes: Push[] | null;
  expansions: number;
  status: "solved" | "dead" | "exhausted" | "budget";
}

interface Matching {
  count: number;
  distances: Int32Array;
  dp: Float64Array;
  bits: Uint8Array;
}

// Numeric sorting makes box arrays canonical regardless of their history.
function numericOrder(a: number, b: number): number {
  return a - b;
}

// Builds a constant-time membership mask for a list of board cells.
function maskOf(size: number, squares: number[]): Uint8Array {
  const mask = new Uint8Array(size);
  for (const square of squares) mask[square] = 1;
  return mask;
}

// Parses XSB while keeping the outer frame and missing short-row cells walls.
function parseLevel(text: string): Level {
  const lines = text.split(/\r?\n/).filter((line) => line.length > 0);
  const height = lines.length;
  const width = lines.reduce((longest, line) => Math.max(longest, line.length), 0);
  if (width < 3 || height < 3) throw new Error("Invalid board dimensions");
  const size = width * height;
  const wall = new Uint8Array(size).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let row = 1; row < height - 1; row++) {
    for (let column = 1; column < width - 1 && column < lines[row].length; column++) {
      const tile = lines[row][column];
      if (tile === "#") continue;
      if (!" .$*@+".includes(tile)) throw new Error("Invalid XSB tile");
      const square = row * width + column;
      wall[square] = 0;
      if (tile === "." || tile === "*" || tile === "+") goals.push(square);
      if (tile === "$" || tile === "*") boxes.push(square);
      if (tile === "@" || tile === "+") {
        if (player !== -1) throw new Error("Multiple players");
        player = square;
      }
    }
  }
  if (player === -1) throw new Error("Missing player");
  boxes.sort(numericOrder);
  return { width, height, size, wall, goals, goalMask: maskOf(size, goals), boxes, player };
}

// Reverse pushes mark exactly the cells from which a lone box can reach a goal.
function liveSquares(level: Level): Uint8Array {
  const live = new Uint8Array(level.size);
  const queue = new Int32Array(level.size);
  const deltas = [-level.width, level.width, -1, 1];
  let head = 0;
  let tail = 0;
  for (const goal of level.goals) {
    if (!live[goal] && !level.wall[goal]) {
      live[goal] = 1;
      queue[tail++] = goal;
    }
  }
  while (head < tail) {
    const square = queue[head++];
    for (const delta of deltas) {
      const previous = square - delta;
      const support = previous - delta;
      if (level.wall[previous] === 0 && level.wall[support] === 0 && !live[previous]) {
        live[previous] = 1;
        queue[tail++] = previous;
      }
    }
  }
  return live;
}

// Removing dead cells on the player's hallway isolates rooms in row-major order.
function roomsOf(level: Level): Room[] | null {
  const live = liveSquares(level);
  const cut = new Uint8Array(level.size);
  const rowStart = Math.floor(level.player / level.width) * level.width;
  for (let square = rowStart; square < rowStart + level.width; square++) {
    if (!level.wall[square] && !live[square]) cut[square] = 1;
  }
  for (const box of level.boxes) if (cut[box]) return null;
  for (const goal of level.goals) if (cut[goal] || level.wall[goal]) return null;
  const boxMask = maskOf(level.size, level.boxes);
  const seen = new Uint8Array(level.size);
  const queue = new Int32Array(level.size);
  const deltas = [-level.width, level.width, -1, 1];
  const rooms: Room[] = [];
  let assignedGoals = 0;
  for (let first = 0; first < level.size; first++) {
    if (level.wall[first] || cut[first] || seen[first]) continue;
    const squares: number[] = [];
    const boxes: number[] = [];
    const goals: number[] = [];
    let head = 0;
    let tail = 1;
    queue[0] = first;
    seen[first] = 1;
    while (head < tail) {
      const square = queue[head++];
      squares.push(square);
      if (boxMask[square]) boxes.push(square);
      if (level.goalMask[square]) goals.push(square);
      for (const delta of deltas) {
        const next = square + delta;
        if (level.wall[next] === 0 && !cut[next] && !seen[next]) {
          seen[next] = 1;
          queue[tail++] = next;
        }
      }
    }
    if (boxes.length !== goals.length) return null;
    if (boxes.length > 0) {
      boxes.sort(numericOrder);
      goals.sort(numericOrder);
      assignedGoals += goals.length;
      rooms.push({ squares, boxes, goals });
    }
  }
  return assignedGoals === level.goals.length ? rooms : null;
}

// Creates reusable occupancy, queue and stamped reachability storage.
function makeFlood(level: Level): Flood {
  return {
    deltas: [-level.width, level.width, -1, 1],
    occupied: new Uint8Array(level.size),
    seen: new Int32Array(level.size),
    queue: new Int32Array(level.size),
    stamp: 0,
  };
}

// Stamps avoid clearing reachability between floods; wraparound is handled exactly.
function nextStamp(flood: Flood): number {
  if (flood.stamp === MAX_STAMP) {
    flood.seen.fill(0);
    flood.stamp = 0;
  }
  return ++flood.stamp;
}

// Floods the player's free region and returns its least cell as the canonical player.
function reachable(level: Level, start: number, flood: Flood): number {
  const stamp = nextStamp(flood);
  let head = 0;
  let tail = 1;
  let low = start;
  flood.queue[0] = start;
  flood.seen[start] = stamp;
  while (head < tail) {
    const square = flood.queue[head++];
    if (square < low) low = square;
    for (const delta of flood.deltas) {
      const next = square + delta;
      if (level.wall[next] === 0 && !flood.occupied[next] && flood.seen[next] !== stamp) {
        flood.seen[next] = stamp;
        flood.queue[tail++] = next;
      }
    }
  }
  return low;
}

// Wall-only goal BFS gives all floor-to-goal distances by undirected symmetry.
function wallDistances(level: Level, deltas: number[]): Int32Array {
  const count = level.goals.length;
  const distances = new Int32Array(level.size * count).fill(UNREACHABLE);
  const queue = new Int32Array(level.size);
  for (let goalIndex = 0; goalIndex < count; goalIndex++) {
    const goal = level.goals[goalIndex];
    let head = 0;
    let tail = 1;
    queue[0] = goal;
    distances[goal * count + goalIndex] = 0;
    while (head < tail) {
      const square = queue[head++];
      const distance = distances[square * count + goalIndex] + 1;
      for (const delta of deltas) {
        const next = square + delta;
        const index = next * count + goalIndex;
        if (level.wall[next] === 0 && distances[index] === UNREACHABLE) {
          distances[index] = distance;
          queue[tail++] = next;
        }
      }
    }
  }
  return distances;
}

// Allocates matching scratch once, using subset DP for the generated small rooms.
function makeMatching(level: Level, deltas: number[]): Matching {
  const count = level.boxes.length;
  const masks = 1 << count;
  const bits = new Uint8Array(masks);
  for (let mask = 1; mask < masks; mask++) bits[mask] = bits[mask >> 1] + (mask & 1);
  return {
    count,
    distances: wallDistances(level, deltas),
    dp: new Float64Array(masks),
    bits,
  };
}

// Subset DP evaluates every bijection, an admissible lower bound on remaining pushes.
function subsetMatching(boxes: number[], matching: Matching): number {
  const { count, distances, dp, bits } = matching;
  dp[0] = 0;
  for (let mask = 1; mask < dp.length; mask++) {
    const offset = boxes[bits[mask] - 1] * count;
    let best = UNREACHABLE;
    for (let goal = 0; goal < count; goal++) {
      const bit = 1 << goal;
      if (mask & bit) {
        const cost = dp[mask ^ bit] + distances[offset + goal];
        if (cost < best) best = cost;
      }
    }
    dp[mask] = best;
  }
  return dp[dp.length - 1];
}

// At most four boxes on 255 cells keep this base-size room key below 2^53.
function roomKey(boxes: number[], low: number, size: number): number {
  let key = low;
  for (const box of boxes) key = key * size + box;
  return key;
}

// Heap order is lower f, then larger g, then earlier insertion for determinism.
function preferred(a: SearchNode, b: SearchNode): boolean {
  return a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.order < b.order)));
}

// Binary heap retains the exact A* tie order without sorting the entire frontier.
class MinHeap {
  items: SearchNode[] = [];

  // Sifts a new node upward while preserving the deterministic priority invariant.
  add(node: SearchNode): void {
    let index = this.items.length;
    this.items.push(node);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (!preferred(node, this.items[parent])) break;
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.items[index] = node;
  }

  // Removes the minimum and sifts the last node downward to restore heap order.
  take(): SearchNode | null {
    if (this.items.length === 0) return null;
    const first = this.items[0];
    const last = this.items.pop()!;
    if (this.items.length > 0) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        if (left >= this.items.length) break;
        const right = left + 1;
        const child = right < this.items.length && preferred(this.items[right], this.items[left])
          ? right : left;
        if (!preferred(this.items[child], last)) break;
        this.items[index] = this.items[child];
        index = child;
      }
      this.items[index] = last;
    }
    return first;
  }
}

// A full wall/box 2x2 cannot move any contained box, so off-goal boxes are deadlocked.
function frozenBlock(level: Level, occupied: Uint8Array, pushedTo: number): boolean {
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 2; column++) {
      const a = pushedTo - row * level.width - column;
      const b = a + 1;
      const c = a + level.width;
      const d = c + 1;
      if ((level.wall[a] || occupied[a]) && (level.wall[b] || occupied[b])
          && (level.wall[c] || occupied[c]) && (level.wall[d] || occupied[d])
          && ((occupied[a] && !level.goalMask[a]) || (occupied[b] && !level.goalMask[b])
            || (occupied[c] && !level.goalMask[c]) || (occupied[d] && !level.goalMask[d]))) {
        return true;
      }
    }
  }
  return false;
}

// Captures legal pushes before successor floods overwrite the current region's stamp.
function collectPushes(level: Level, boxes: number[], live: Uint8Array,
  flood: Flood, candidates: Int32Array): number {
  let count = 0;
  const stamp = flood.stamp;
  for (let index = 0; index < boxes.length; index++) {
    const box = boxes[index];
    for (let direction = 0; direction < 4; direction++) {
      const delta = flood.deltas[direction];
      const to = box + delta;
      if (flood.seen[box - delta] !== stamp || level.wall[to] !== 0
          || flood.occupied[to] || !live[to]) continue;
      flood.occupied[box] = 0;
      flood.occupied[to] = 1;
      const frozen = frozenBlock(level, flood.occupied, to);
      flood.occupied[to] = 0;
      flood.occupied[box] = 1;
      if (!frozen) candidates[count++] = index * 4 + direction;
    }
  }
  return count;
}

// A solved state has every box on a goal, including boxes that started settled.
function boxesOnGoals(boxes: number[], goalMask: Uint8Array): boolean {
  for (const box of boxes) if (!goalMask[box]) return false;
  return true;
}

// Parent links recover only pushes; walking is reconstructed separately afterward.
function pushTrace(node: SearchNode): Push[] {
  const pushes: Push[] = [];
  for (let current = node; current.parent !== null; current = current.parent) {
    pushes.push({ from: current.from, to: current.to });
  }
  return pushes.reverse();
}

// Push A* merges equivalent player regions, reopens better costs and budgets expansions.
function solveRoom(level: Level, maxExpansions: number, useHeuristic: boolean): SearchResult {
  if (level.boxes.length > 4 || level.boxes.length !== level.goals.length || level.wall[level.player]) {
    return { pushes: null, expansions: 0, status: "dead" };
  }
  const live = liveSquares(level);
  for (const box of level.boxes) {
    if (!live[box]) return { pushes: null, expansions: 0, status: "dead" };
  }
  const flood = makeFlood(level);
  const boxes = level.boxes.slice().sort(numericOrder);
  const matching = useHeuristic ? makeMatching(level, flood.deltas) : null;
  const initialH = matching === null ? 0 : subsetMatching(boxes, matching);
  if (initialH >= UNREACHABLE) return { pushes: null, expansions: 0, status: "dead" };
  for (const box of boxes) flood.occupied[box] = 1;
  const initialLow = reachable(level, level.player, flood);
  for (const box of boxes) flood.occupied[box] = 0;
  const costs = new Map<number, number>();
  const initialKey = roomKey(boxes, initialLow, level.size);
  costs.set(initialKey, 0);
  const open = new MinHeap();
  open.add({ boxes, low: initialLow, g: 0, f: initialH, order: 0, key: initialKey,
    parent: null, from: -1, to: -1 });
  const candidates = new Int32Array(boxes.length * 4);
  const nextBoxes = boxes.slice();
  let order = 1;
  let expansions = 0;
  let current: SearchNode | null;
  while ((current = open.take()) !== null) {
    if (costs.get(current.key) !== current.g) continue;
    if (boxesOnGoals(current.boxes, level.goalMask)) {
      return { pushes: pushTrace(current), expansions, status: "solved" };
    }
    if (expansions >= maxExpansions) return { pushes: null, expansions, status: "budget" };
    expansions++;
    for (const box of current.boxes) flood.occupied[box] = 1;
    reachable(level, current.low, flood);
    const count = collectPushes(level, current.boxes, live, flood, candidates);
    for (let candidate = 0; candidate < count; candidate++) {
      const encoded = candidates[candidate];
      const index = encoded >> 2;
      const from = current.boxes[index];
      const to = from + flood.deltas[encoded & 3];
      const g = current.g + 1;
      flood.occupied[from] = 0;
      flood.occupied[to] = 1;
      for (let box = 0; box < nextBoxes.length; box++) {
        nextBoxes[box] = box === index ? to : current.boxes[box];
      }
      nextBoxes.sort(numericOrder);
      const low = reachable(level, from, flood);
      const key = roomKey(nextBoxes, low, level.size);
      const previous = costs.get(key);
      if (previous === undefined || g < previous) {
        const h = matching === null ? 0 : subsetMatching(nextBoxes, matching);
        if (h < UNREACHABLE) {
          costs.set(key, g);
          open.add({ boxes: nextBoxes.slice(), low, g, f: g + h, order: order++,
            key, parent: current, from, to });
        }
      }
      flood.occupied[to] = 0;
      flood.occupied[from] = 1;
    }
    for (const box of current.boxes) flood.occupied[box] = 0;
  }
  return { pushes: null, expansions, status: "exhausted" };
}

// A failed bounded A* gets the same expansion-bounded search with h=0.
function solveWithFallback(level: Level): Push[] | null {
  for (let attempt = 0; attempt < 2; attempt++) {
    const result = solveRoom(level, MAX_EXPANSIONS, attempt === 0);
    if (result.pushes !== null) return result.pushes;
  }
  return null;
}

// BFS finds a shortest walking route, with u,d,l,r resolving equal-length routes.
function shortestWalk(level: Level, start: number, target: number, flood: Flood,
  previous: Int32Array, via: Uint8Array): string | null {
  if (start === target) return "";
  const stamp = nextStamp(flood);
  let head = 0;
  let tail = 1;
  flood.queue[0] = start;
  flood.seen[start] = stamp;
  while (head < tail && flood.seen[target] !== stamp) {
    const square = flood.queue[head++];
    for (let direction = 0; direction < 4; direction++) {
      const next = square + flood.deltas[direction];
      if (level.wall[next] !== 0 || flood.occupied[next] || flood.seen[next] === stamp) continue;
      flood.seen[next] = stamp;
      previous[next] = square;
      via[next] = direction;
      flood.queue[tail++] = next;
      if (next === target) break;
    }
  }
  if (flood.seen[target] !== stamp) return null;
  const letters: string[] = [];
  for (let square = target; square !== start; square = previous[square]) {
    letters.push(WALK_LETTERS[via[square]]);
  }
  return letters.reverse().join("");
}

// Replays push positions into legal shortest walks and uppercase pushes on the real board.
function reconstructMoves(level: Level, pushes: Push[]): string | null {
  const flood = makeFlood(level);
  const previous = new Int32Array(level.size);
  const via = new Uint8Array(level.size);
  for (const box of level.boxes) flood.occupied[box] = 1;
  let player = level.player;
  const pieces: string[] = [];
  for (const push of pushes) {
    const delta = push.to - push.from;
    const direction = flood.deltas.indexOf(delta);
    if (direction < 0 || !flood.occupied[push.from] || level.wall[push.to]
        || flood.occupied[push.to]) return null;
    const walk = shortestWalk(level, player, push.from - delta, flood, previous, via);
    if (walk === null) return null;
    pieces.push(walk, PUSH_LETTERS[direction]);
    flood.occupied[push.from] = 0;
    flood.occupied[push.to] = 1;
    player = push.from;
  }
  return pieces.join("");
}

// Legal move replay needs only displacements to inherit the player's ending square.
function endOf(level: Level, start: number, moves: string): number {
  const deltas = [-level.width, level.width, -1, 1];
  let player = start;
  for (const letter of moves) player += deltas[WALK_LETTERS.indexOf(letter.toLowerCase())];
  return player;
}

// Solves rooms in scan order, treating other rooms' current boxes as immovable walls.
function solveRooms(level: Level, rooms: Room[]): string | null {
  const positions = rooms.map((room) => room.boxes.slice());
  const pieces: string[] = [];
  let player = level.player;
  for (let index = 0; index < rooms.length; index++) {
    const room = rooms[index];
    const wall = level.wall.slice();
    for (let other = 0; other < rooms.length; other++) {
      if (other !== index) for (const box of positions[other]) wall[box] = 1;
    }
    const local: Level = { ...level, wall, boxes: room.boxes, goals: room.goals,
      goalMask: maskOf(level.size, room.goals), player };
    const pushes = solveWithFallback(local);
    if (pushes === null) return null;
    const moves = reconstructMoves(local, pushes);
    if (moves === null) return null;
    pieces.push(moves);
    player = endOf(level, player, moves);
    positions[index] = room.goals.slice();
  }
  return pieces.join("");
}

// A failed room decomposition or search yields a legal empty answer.
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms === null) return "";
  return solveRooms(level, rooms) ?? "";
}

// Processes sorted input directories and always writes an answer.
function main(): void {
  const input = process.env.MAKEFUN_INPUT_DIR || "/input";
  const output = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  const ids = readdirSync(input, { withFileTypes: true })
    .filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
  for (const id of ids) {
    let moves = "";
    try {
      const level = parseLevel(readFileSync(join(input, id, "level.txt"), "utf8"));
      moves = solveLevel(level);
    } catch {
      moves = "";
    }
    mkdirSync(join(output, id), { recursive: true });
    writeFileSync(join(output, id, "solution.txt"), moves + "\n");
  }
}

main();
