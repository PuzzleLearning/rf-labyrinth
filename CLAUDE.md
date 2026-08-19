# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement-learning demo that accompanies a blog post
(<https://oskarj.wordpress.com/2023/01/23/reinforcement-learning-rf-with-python/>):
tabular **Q-learning** is turned loose on a labyrinth and has to find its way
from the top-left corner to the bottom-right one.

Everything lives in `solution/`, as two programs, each duplicated as a script
and as a notebook:

- `maze.py` / `maze.ipynb` — a hard-coded 5×5 grid, `0` = corridor, `1` = wall.
- `maze_from_image.py` / `maze_from_image.ipynb` — the same learner, but the
  maze is read from `df_maze.png` (504×504 RGBA) with OpenCV, thresholded to
  black/white, and **every pixel is treated as a state**.

There is no package structure, no test suite, no CLI and no build step.
Dependencies are pinned loosely in `requirements.txt`: `numpy`, `tqdm`,
`opencv-python`.

## Running

```
python solution/maze.py                # 5x5 grid, 10000 episodes
cd solution && python maze_from_image.py   # reads ./df_maze.png, 100 episodes
```

The image version resolves `df_maze.png` relative to the working directory, so
it only runs from inside `solution/`.

## State of the code (read this before changing anything)

The learner does not work, and the bugs are load-bearing rather than cosmetic —
any refactor has to fix them, not preserve them:

- **Nothing stops the agent leaving the grid.** The loop guard is
  `abs(state[0]) < len(labyrinth[0])`, so a row index of `-1` passes the test
  and then indexes NumPy from the far edge. `maze.ipynb` ends at state
  `(-1, 4)`; `maze_from_image.ipynb` ends at `(-504, 87)`.
- **Walls do not block.** Stepping into a wall costs `-1` but the agent moves
  in anyway, so a wall is a toll booth, not an obstacle.
- **The goal pays nothing.** Reaching it gives reward `0`, exactly like any
  other free square, so every Q-value is `≤ 0` and there is no gradient to
  follow towards the exit. The episode ends by wandering off the board, not by
  arriving.
- **`maze_from_image.py` does not import.** It compares `state == goal` but
  only ever defines `end`; the notebook papers over this with a `goal = end`
  cell.
- **Pixels are not states.** A 504×504 image gives a 504×504×4 Q-table for a
  maze that has roughly 40×40 actual cells, which is why the notebook records
  `100 episodes [24:52<00:00, 14.92s/it]`.

`maze.py` and `maze.ipynb` are the same program twice over, as are the image
pair; the notebooks carry stale outputs from the buggy runs.

## Conventions worth preserving

- The maze convention in `maze.py` is `0` = free, `1` = wall; in the image it
  is white = free, black = wall. Keep both readable as "open square is where
  the agent may stand".
- Start is the top-left cell, goal is the bottom-right one.
- The hyperparameters (`alpha`, `gamma`, `epsilon`, `episodes`) are the visible
  surface of the article — keep them named, keep them adjustable.

## Checking correctness

The maze has an exact answer, so there is a free oracle: a breadth-first search
from start to goal gives the shortest path. A working Q-learner's greedy policy
must produce a path of exactly that length. Any run where the two disagree — or
where the agent finishes outside the grid — is a bug.

## Branches

Work happens on `refactor/2026-revisit`; `main` is the default/PR target.
