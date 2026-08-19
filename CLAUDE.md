# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement-learning demo that accompanies a blog post (see `README.md`):
tabular **Q-learning** is turned loose in a labyrinth it has never seen and has
to walk from the top-left square to the bottom-right one, discovering the walls
by bumping into them.

The whole program is `maze_rl.py` — `Maze` is both the grid and the
environment (`Maze.step` is the transition function), `q_learning()` trains,
`greedy_path()` reads the route off the trained table, `shortest_path()` is an
independent breadth-first oracle, `maze_from_image()` turns a PNG into a
`Maze`, `read_png_luminance()` decodes the PNG with nothing but `zlib`, and
`main()` wires up argparse. There is no package structure, no test suite and no
build step.

`tools/make_banner.py` imports the solver, trains it three times over and draws
`docs/banner-{dark,light}.svg` from the real Q-tables. It is **local-only and
gitignored** — the generated banners are committed, the generator is not — so
it may be missing from a fresh clone. Do not add it back to git.

## Running

```
python maze_rl.py                        # the built-in 5x5 demo maze
python maze_rl.py -i df_maze.png         # the 40x40 maze, ~7 s
python maze_rl.py -i df_maze.png -m bfs  # skip the learning entirely
python tools/make_banner.py              # regenerate both banners (local-only script)
```

Python 3.10+, no third-party dependencies. The system `python3` on this machine
is 3.9 and fails at import on `dataclass(slots=True)`; use
`~/miniconda3/envs/py_313/bin/python` (or `py_310`) to actually run it.

## Conventions worth preserving

- The drawn maze goes to **stdout**; every summary line and the `-v` training
  trace go to **stderr**, so the picture stays redirectable.
- Exit codes carry the verdict: `0` the policy walked out, `1` it got lost or
  the goal is unreachable, `2` bad arguments or an unreadable image.
- **A move costs `-1`, including one into a wall; the goal is terminal and pays
  nothing.** That is what makes "maximise reward" and "take the shortest route"
  the same objective, and it is why `gamma` defaults to `1.0`. Do not
  reintroduce a discount below 1 as a default: past ~650 moves from the exit,
  `gamma=0.95` gives neighbouring squares bit-identical values and the learner
  never converges on `df_maze.png`.
- The Q-table starts at zero while every true value is negative, so untried
  directions look best. That optimism, not the `epsilon`-greedy noise, is what
  makes the agent explore — keep it.
- The agent is not told where the walls are. It may attempt any of the four
  moves from any square; illegal ones simply do not move it. Do not "helpfully"
  mask actions during training.
- Mazes are grids of open/wall squares. An `n`-cell-wide image maze becomes
  `2n + 1` squares wide (odd coordinates are cells, even ones are the doorways
  between them), which is why one agent handles both the demo grid and the PNG.
  Start is the first open square in reading order, goal the last.
- `Maze.render()` output is readable back by `Maze.from_text()`, route and all;
  `OPEN_GLYPHS` is what keeps that true.
- The banner must stay derived from real training runs. If the artwork stops
  matching what `maze_rl.py` learns, that is a bug, not a style choice. `SEED`
  and `STAGES` are fixed on purpose: re-rendering must produce byte-identical
  SVGs.
- The mathematics, the alternative approaches and the literature live in
  `README.md` — if the algorithm changes, that is what has to stay true.

## Checking correctness

`q_learning()` and `shortest_path()` are independent implementations and must
agree on the length of the route; that is the cheapest regression test there
is. `df_maze.png` is a *perfect* maze (3 199 open squares, 3 198 passages, a
spanning tree), so the shortest route is also the only one — matching it means
the policy walked it without a single wrong turn.

```python
from pathlib import Path
from maze_rl import DEMO_MAZE, Maze, greedy_path, maze_from_image, q_learning, shortest_path

for maze in (Maze.from_text(DEMO_MAZE), maze_from_image(Path("df_maze.png"))):
    best = len(shortest_path(maze)) - 1
    walked = greedy_path(maze, q_learning(maze, episodes=2000, seed=0))
    assert walked[-1] == maze.goal and len(walked) - 1 == best
```

Known answers: the built-in demo maze → `8` moves; `df_maze.png` → `908` moves
across a 40×40 maze whose furthest square is `918` moves from the exit. A run
that ends anywhere other than `maze.goal`, or outside the grid, is a bug.

## Branches

Work happens on `refactor/2026-revisit`; `main` is the default/PR target.
