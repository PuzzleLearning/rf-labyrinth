#!/usr/bin/env python3
"""Teach a Q-learning agent to walk out of a labyrinth.

The problem
-----------
A labyrinth is a grid of squares, some open and some walled off. An agent is
dropped into the top-left open square, wants to reach the bottom-right one,
and knows nothing about the layout: at every square it may try to step up,
down, left or right, and it discovers a wall only by walking into it -- which
costs it a move and leaves it exactly where it was.

Q-learning solves that without ever being shown the map. It keeps one number
per (square, direction) pair -- roughly "how much of the journey is left if I
go this way" -- and after every single step it nudges that number towards
what the step actually revealed. Nobody tells the agent where the exit is. It
walks, it bumps into things, and the numbers slowly arrange themselves into a
map good enough to walk the shortest route.

The maze comes either from the small hard-coded demo grid below or from a PNG
where light pixels are corridor and dark pixels are wall; see
``maze_from_image``. Because a maze has an exact answer, the module also
carries a breadth-first search: it is not part of the learning, it is there to
mark the learner's homework.

Usage
-----
    python maze_rl.py                        # the built-in 5x5 demo maze
    python maze_rl.py -i df_maze.png         # the 40x40 maze shipped here
    python maze_rl.py -i df_maze.png -m bfs  # skip the learning, just solve it
    python maze_rl.py -e 200 -v              # a short run, traced episode by episode

Requires Python 3.10 or newer. It has no third-party dependencies.
"""

import argparse
import random
import struct
import sys
import time
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

#: A square of the maze, as ``(row, column)``.
Square = tuple[int, int]

#: The Q-table: ``q[row * cols + col][action]``. A plain list of lists is
#: quicker than a dict here and needs no third-party array library.
QTable = list[list[float]]

#: The four moves the agent may attempt, as ``(row delta, column delta)``:
#: up, down, left, right, in the order the Q-table stores them.
MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: What a move costs, wall or no wall. Reaching the goal ends the episode and
#: pays nothing extra, so the agent's only way to score better is to take
#: fewer steps -- which makes "maximise the return" and "find the shortest
#: path" literally the same problem.
STEP_COST = -1.0

#: The maze used when no image is given: the 5x5 grid this repository started
#: with, two pillars of wall between four open corridors.
DEMO_MAZE = """\
.....
.#.#.
.#.#.
.#.#.
.....
"""

DEFAULT_EPISODES = 2000
DEFAULT_ALPHA = 0.8
DEFAULT_GAMMA = 1.0
DEFAULT_EPSILON = 0.2
DEFAULT_EPSILON_FINAL = 0.01

#: Characters used to draw a maze, as ``(wall, corridor, trail)``.
BOX_GLYPHS = ("█", " ", "·")
ASCII_GLYPHS = ("#", " ", ".")

#: What :meth:`Maze.from_text` reads as corridor; everything else is wall.
#: Every glyph :meth:`Maze.render` emits is in here, so a drawn maze -- route
#: and all -- can be pasted straight back in.
OPEN_GLYPHS = frozenset(". 0·SG")


# --------------------------------------------------------------------------
# The maze, which doubles as the environment the agent acts in
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Maze:
    """A labyrinth: a rectangular grid of open and walled-off squares.

    Attributes:
        grid: ``grid[row][col]`` is true where the agent may stand.
        start: Where every episode begins.
        goal: The square that ends an episode.
    """

    grid: tuple[tuple[bool, ...], ...]
    start: Square
    goal: Square

    @property
    def rows(self) -> int:
        """How many rows of squares the maze has."""
        return len(self.grid)

    @property
    def cols(self) -> int:
        """How many columns of squares the maze has."""
        return len(self.grid[0])

    @property
    def open_squares(self) -> int:
        """How many squares are corridor rather than wall."""
        return sum(sum(row) for row in self.grid)

    def is_open(self, square: Square) -> bool:
        """Whether ``square`` is inside the maze and free to stand on."""
        row, col = square
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col]

    def step(self, square: Square, action: int) -> tuple[Square, float, bool]:
        """Take one action and report what happened.

        Args:
            square: Where the agent currently stands.
            action: An index into :data:`MOVES`.

        Returns:
            ``(next_square, reward, reached_goal)``. Walking into a wall or off
            the edge is allowed and simply does not move the agent -- it still
            costs a step, which is how a wall teaches the agent anything at
            all.
        """
        row, col = square
        d_row, d_col = MOVES[action]
        target = (row + d_row, col + d_col)
        moved_to = target if self.is_open(target) else square
        return moved_to, STEP_COST, moved_to == self.goal

    @classmethod
    def from_grid(cls, grid: list[list[bool]]) -> "Maze":
        """Build a maze from a rectangular grid of "is this square open?".

        The start is the first open square in reading order and the goal is the
        last one, which for a maze drawn the usual way means the top-left and
        bottom-right corners.
        """
        if not grid or not grid[0]:
            raise ValueError("the maze is empty")
        if len({len(row) for row in grid}) != 1:
            raise ValueError("the maze is not rectangular")

        opens = [(r, c) for r, row in enumerate(grid) for c, ok in enumerate(row) if ok]
        if not opens:
            raise ValueError("the maze has no open squares at all")

        return cls(tuple(tuple(row) for row in grid), opens[0], opens[-1])

    @classmethod
    def from_text(cls, text: str) -> "Maze":
        """Read a maze drawn with :data:`OPEN_GLYPHS` for corridor.

        Blank lines are skipped, so a maze may be written as a triple-quoted
        block without fighting the indentation.
        """
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("the maze is empty")
        width = max(len(line) for line in lines)
        return cls.from_grid(
            [[ch in OPEN_GLYPHS for ch in line.ljust(width)] for line in lines]
        )

    def render(self, path: list[Square] | None = None, *, ascii_only: bool = False) -> str:
        """Draw the maze, optionally with ``path`` marked out through it."""
        wall, corridor, trail = ASCII_GLYPHS if ascii_only else BOX_GLYPHS
        walked = set(path or ())

        lines = []
        for row_index, row in enumerate(self.grid):
            glyphs = []
            for col_index, is_open in enumerate(row):
                square = (row_index, col_index)
                if square == self.start:
                    glyphs.append("S")
                elif square == self.goal:
                    glyphs.append("G")
                elif not is_open:
                    glyphs.append(wall)
                elif square in walked:
                    glyphs.append(trail)
                else:
                    glyphs.append(corridor)
            lines.append("".join(glyphs))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The oracle: breadth-first search, which knows the map and cannot be beaten
# --------------------------------------------------------------------------


def shortest_path(maze: Maze) -> list[Square] | None:
    """Return the shortest route from start to goal, or ``None`` if walled in.

    Every move costs the same, so breadth-first search is exactly right: the
    first time it reaches a square it has reached it by the fewest moves
    possible. This is the answer the learner has to match; it is not how the
    learner works.
    """
    came_from: dict[Square, Square | None] = {maze.start: None}
    queue = deque([maze.start])

    while queue:
        square = queue.popleft()
        if square == maze.goal:
            break
        row, col = square
        for d_row, d_col in MOVES:
            neighbour = (row + d_row, col + d_col)
            if neighbour not in came_from and maze.is_open(neighbour):
                came_from[neighbour] = square
                queue.append(neighbour)

    if maze.goal not in came_from:
        return None

    path: list[Square] = []
    cursor: Square | None = maze.goal
    while cursor is not None:
        path.append(cursor)
        cursor = came_from[cursor]
    return path[::-1]


# --------------------------------------------------------------------------
# The learner
# --------------------------------------------------------------------------


def default_step_limit(maze: Maze) -> int:
    """How long a single episode is allowed to run before it is cut short.

    Early episodes are close to a random walk and can wander for hundreds of
    thousands of steps; truncating them costs nothing (Q-learning learns from
    every individual step, not from finished episodes) and makes training an
    order of magnitude quicker.
    """
    return 6 * maze.open_squares


def exploration_rate(start: float, final: float, episode: int, episodes: int) -> float:
    """Linearly fade the exploration rate from ``start`` to ``final``."""
    if episodes <= 1:
        return start
    return start + (final - start) * episode / (episodes - 1)


def q_learning(
    maze: Maze,
    *,
    episodes: int = DEFAULT_EPISODES,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    epsilon_final: float = DEFAULT_EPSILON_FINAL,
    step_limit: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> QTable:
    """Learn a Q-table for ``maze`` by walking it over and over again.

    The update is the textbook one::

        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

    Args:
        maze: The labyrinth to learn, which is also the environment.
        episodes: How many walks from the start the agent gets.
        alpha: Learning rate. The maze never changes and never surprises the
            agent twice, so a large value is not reckless here -- it just
            means each observation is believed the first time.
        gamma: Discount applied to the next square's value. ``1.0`` is the
            honest setting for a shortest-path problem: a move costs the same
            whether it is the first or the thousandth.
        epsilon: Probability of ignoring the Q-table and moving at random, at
            the first episode.
        epsilon_final: The same probability at the last episode; it fades
            linearly in between.
        step_limit: Steps allowed per episode; defaults to
            :func:`default_step_limit`.
        seed: Seed for the private random number generator, so runs repeat.
        verbose: Report every episode on stderr.

    Returns:
        The Q-table, indexed ``q[row * maze.cols + col][action]``.
    """
    if episodes < 0:
        raise ValueError(f"episodes must not be negative, got {episodes}")

    rng = random.Random(seed)
    cols = maze.cols
    limit = default_step_limit(maze) if step_limit is None else step_limit

    # Everything starts at zero while every true value is negative, so an
    # untried direction always looks better than a tried one. That is not an
    # accident: this optimism is what pushes the agent to sweep the maze
    # systematically instead of loitering near the start.
    q: QTable = [[0.0] * len(MOVES) for _ in range(maze.rows * cols)]

    for episode in range(episodes):
        explore = exploration_rate(epsilon, epsilon_final, episode, episodes)
        square = maze.start
        steps = 0
        reached = square == maze.goal

        while not reached and steps < limit:
            values = q[square[0] * cols + square[1]]
            if rng.random() < explore:
                action = rng.randrange(len(MOVES))
            else:
                # Ties are broken at random; picking the first-listed action
                # instead would send an untaught agent marching into the same
                # wall for ever.
                best = max(values)
                action = rng.choice([a for a, v in enumerate(values) if v == best])

            moved_to, reward, reached = maze.step(square, action)
            future = 0.0 if reached else max(q[moved_to[0] * cols + moved_to[1]])
            values[action] += alpha * (reward + gamma * future - values[action])

            square = moved_to
            steps += 1

        if verbose:
            outcome = "reached the goal" if reached else "gave up"
            print(
                f"episode {episode + 1:>6}/{episodes}: {steps:>7} steps,"
                f" epsilon {explore:.3f}, {outcome}",
                file=sys.stderr,
            )

    return q


def greedy_path(maze: Maze, q: QTable, *, step_limit: int | None = None) -> list[Square]:
    """Walk the maze following the learned policy, taking no random moves.

    The walk stops at the goal, at the step limit, when the policy asks for a
    move into a wall, or as soon as it revisits a square -- all three of the
    latter mean the table has not converged yet. Check ``path[-1] ==
    maze.goal`` to tell success from failure.
    """
    cols = maze.cols
    limit = default_step_limit(maze) if step_limit is None else step_limit

    square = maze.start
    path = [square]
    seen = {square}

    while square != maze.goal and len(path) <= limit:
        values = q[square[0] * cols + square[1]]
        action = max(range(len(MOVES)), key=values.__getitem__)
        moved_to, _, _ = maze.step(square, action)
        if moved_to == square or moved_to in seen:
            break
        square = moved_to
        path.append(square)
        seen.add(square)

    return path


# --------------------------------------------------------------------------
# Reading a maze out of a PNG
# --------------------------------------------------------------------------

#: Samples per pixel for each PNG colour type.
PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _unfilter(raw: bytes, height: int, stride: int, pixel_bytes: int) -> list[bytes]:
    """Undo the per-scanline filters PNG applies before compression.

    Each decompressed scanline is prefixed with a filter byte saying how it was
    encoded relative to the pixel on its left (``a``), the one above (``b``)
    and the one above-left (``c``); see section 9 of the PNG specification.
    """
    if len(raw) < height * (1 + stride):
        raise ValueError("the PNG image data stops short of its declared size")

    lines: list[bytes] = []
    previous = bytearray(stride)
    offset = 0

    for _ in range(height):
        filter_type = raw[offset]
        line = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += 1 + stride

        if filter_type == 1:  # Sub
            for i in range(pixel_bytes, stride):
                line[i] = (line[i] + line[i - pixel_bytes]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - pixel_bytes] if i >= pixel_bytes else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = line[i - pixel_bytes] if i >= pixel_bytes else 0
                up_left = previous[i - pixel_bytes] if i >= pixel_bytes else 0
                up = previous[i]
                d_left, d_up, d_diag = (
                    abs(up - up_left),
                    abs(left - up_left),
                    abs(left + up - 2 * up_left),
                )
                if d_left <= d_up and d_left <= d_diag:
                    guess = left
                elif d_up <= d_diag:
                    guess = up
                else:
                    guess = up_left
                line[i] = (line[i] + guess) & 0xFF
        elif filter_type != 0:  # None
            raise ValueError(f"unknown PNG filter type {filter_type}")

        lines.append(bytes(line))
        previous = line

    return lines


def _samples(line: bytes, depth: int, count: int) -> list[int]:
    """Split one unfiltered scanline into ``count`` raw samples."""
    if depth == 8:
        return list(line[:count])
    if depth == 16:
        # Sixteen bits of precision are wasted on deciding "wall or corridor";
        # the high byte carries the same answer.
        return [line[2 * i] for i in range(count)]

    per_byte = 8 // depth
    mask = (1 << depth) - 1
    return [
        (line[i // per_byte] >> (8 - depth * (i % per_byte + 1))) & mask
        for i in range(count)
    ]


def read_png_luminance(path: Path) -> tuple[int, int, bytes]:
    """Decode a PNG and return ``(width, height, brightness)``.

    Enough of the format to read a picture of a maze and no more: any colour
    type and bit depth, but not Adam7-interlaced files. ``brightness`` holds
    one byte per pixel in row-major order, ``0`` for black and ``255`` for
    white.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")

    header: tuple[int, ...] | None = None
    palette = b""
    compressed: list[bytes] = []
    offset = 8

    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"PLTE":
            palette = chunk
        elif kind == b"IDAT":
            compressed.append(chunk)
        elif kind == b"IEND":
            break
        offset += 12 + length  # length, type, payload, CRC

    if header is None:
        raise ValueError(f"{path} has no PNG header chunk")
    width, height, depth, colour, _compression, _filter, interlace = header
    if interlace:
        raise ValueError(f"{path} is interlaced, which this reader cannot undo")
    if colour not in PNG_CHANNELS:
        raise ValueError(f"{path} uses unknown PNG colour type {colour}")

    channels = PNG_CHANNELS[colour]
    stride = (width * channels * depth + 7) // 8
    pixel_bytes = max(1, channels * depth // 8)
    lines = _unfilter(zlib.decompress(b"".join(compressed)), height, stride, pixel_bytes)

    top = (1 << depth) - 1
    brightness = bytearray(width * height)
    for y, line in enumerate(lines):
        samples = _samples(line, depth, width * channels)
        for x in range(width):
            if colour == 3:
                index = 3 * samples[x]
                red, green, blue = palette[index : index + 3]
            elif colour in (0, 4):
                red = green = blue = samples[x * channels] * 255 // top
            else:
                first = x * channels
                red, green, blue = (
                    sample * 255 // top for sample in samples[first : first + 3]
                )
            brightness[y * width + x] = (299 * red + 587 * green + 114 * blue) // 1000

    return width, height, bytes(brightness)


def wall_lines(darkness: list[int]) -> list[int]:
    """Find the grid lines a maze's walls are drawn on.

    Args:
        darkness: ``darkness[i]`` counts the dark pixels in image column (or
            row) ``i``.

    Returns:
        One index per line, taken from the middle of its band of pixels.

    A maze is drawn on a lattice: walls only ever appear along a fixed set of
    columns and rows. Those lines are far darker than the ones running down
    the middle of a corridor -- even a wall-free lattice line still collects
    every junction it crosses -- so a cutoff halfway between the darkest and
    lightest line separates them cleanly. Recovering the lattice is what turns
    a 504x504 picture into a 40x40 maze, and it is the difference between a
    Q-table of a quarter of a million states and one of a few thousand.
    """
    cutoff = (min(darkness) + max(darkness)) / 2
    lines: list[int] = []
    band: list[int] = []

    for index, count in enumerate(darkness):
        if count >= cutoff:
            band.append(index)
        elif band:
            lines.append(band[len(band) // 2])
            band = []
    if band:
        lines.append(band[len(band) // 2])

    return lines


def maze_from_image(path: Path, *, threshold: int = 128) -> Maze:
    """Read a maze from a PNG in which light pixels are corridor.

    The picture is reduced to its lattice of cells (see :func:`wall_lines`) and
    then blown back up into a grid where odd coordinates are cells and even
    ones are the gaps between them: an ``n``-cell-wide maze becomes
    ``2n + 1`` squares wide, which is the same "open square or wall square"
    shape as :data:`DEMO_MAZE` and lets one agent handle both.
    """
    width, height, brightness = read_png_luminance(path)
    lit = [
        [brightness[y * width + x] >= threshold for x in range(width)]
        for y in range(height)
    ]

    columns = wall_lines([sum(1 for y in range(height) if not lit[y][x]) for x in range(width)])
    rows = wall_lines([row.count(False) for row in lit])
    if len(columns) < 2 or len(rows) < 2:
        raise ValueError(
            f"{path} does not look like a maze drawn on a grid: found "
            f"{len(columns)} vertical and {len(rows)} horizontal wall lines"
        )

    cell_cols, cell_rows = len(columns) - 1, len(rows) - 1
    centre_x = [(columns[c] + columns[c + 1]) // 2 for c in range(cell_cols)]
    centre_y = [(rows[r] + rows[r + 1]) // 2 for r in range(cell_rows)]

    grid = [[False] * (2 * cell_cols + 1) for _ in range(2 * cell_rows + 1)]
    for r in range(cell_rows):
        for c in range(cell_cols):
            grid[2 * r + 1][2 * c + 1] = lit[centre_y[r]][centre_x[c]]
            if c + 1 < cell_cols:  # is the wall to the right of this cell missing?
                grid[2 * r + 1][2 * c + 2] = lit[centre_y[r]][columns[c + 1]]
            if r + 1 < cell_rows:  # ... and the one below it?
                grid[2 * r + 2][2 * c + 1] = lit[rows[r + 1]][centre_x[c]]

    return Maze.from_grid(grid)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line. ``argv`` defaults to :data:`sys.argv`."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        epilog="The maze goes to stdout and the summary to stderr, so the "
        "drawing stays easy to redirect. See README.md for what the "
        "hyperparameters do and why gamma defaults to 1.",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=Path,
        metavar="PNG",
        help="read the maze from a PNG (light pixels are corridor); "
        "without this the built-in 5x5 demo maze is used",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=("q-learning", "bfs"),
        default="q-learning",
        help="which solver draws the route (default: q-learning). The "
        "breadth-first answer is computed either way, to mark the learner",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        metavar="N",
        help=f"training episodes (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        metavar="F",
        help=f"learning rate (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        metavar="F",
        help=f"discount factor (default: {DEFAULT_GAMMA})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        metavar="F",
        help=f"exploration rate at the first episode (default: {DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--epsilon-final",
        type=float,
        default=DEFAULT_EPSILON_FINAL,
        metavar="F",
        help=f"exploration rate at the last episode (default: {DEFAULT_EPSILON_FINAL})",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        metavar="N",
        help="steps allowed per episode (default: six per open square)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        metavar="B",
        help="brightness at which an image pixel counts as corridor (default: 128)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="seed for the agent's random choices (default: 0)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="draw with # and . instead of box-drawing characters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="report every training episode on stderr",
    )
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> str | None:
    """Return a complaint about the arguments, or ``None`` if they are fine."""
    if args.episodes < 0:
        return "--episodes must not be negative"
    if not 0.0 < args.alpha <= 1.0:
        return "--alpha must be greater than 0 and at most 1"
    if not 0.0 < args.gamma <= 1.0:
        return "--gamma must be greater than 0 and at most 1"
    if not 0.0 <= args.epsilon <= 1.0:
        return "--epsilon must be between 0 and 1"
    if not 0.0 <= args.epsilon_final <= 1.0:
        return "--epsilon-final must be between 0 and 1"
    if args.step_limit is not None and args.step_limit < 1:
        return "--step-limit must be at least 1"
    if not 0 <= args.threshold <= 255:
        return "--threshold must be between 0 and 255"
    return None


def moves(count: int) -> str:
    """Render a number of moves, with the noun in the right number."""
    return f"{count} move" if count == 1 else f"{count} moves"


def main(argv: list[str] | None = None) -> int:
    """Run the solver as a command line program and return its exit code."""
    args = parse_args(argv)

    complaint = _validate(args)
    if complaint is not None:
        print(f"error: {complaint}", file=sys.stderr)
        return 2

    try:
        if args.image:
            maze = maze_from_image(args.image, threshold=args.threshold)
        else:
            maze = Maze.from_text(DEMO_MAZE)
    except (OSError, ValueError, zlib.error) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    print(
        f"maze: {maze.rows} x {maze.cols} squares, {maze.open_squares} of them open;"
        f" {maze.start} -> {maze.goal}",
        file=sys.stderr,
    )

    best = shortest_path(maze)
    if best is None:
        print("error: the goal cannot be reached from the start", file=sys.stderr)
        return 1
    print(f"breadth-first search: {moves(len(best) - 1)}", file=sys.stderr)

    if args.method == "bfs":
        print(maze.render(best, ascii_only=args.ascii))
        return 0

    started = time.perf_counter()
    q = q_learning(
        maze,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_final=args.epsilon_final,
        step_limit=args.step_limit,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - started
    print(
        f"Q-learning: {args.episodes} episodes in {elapsed:.1f}s"
        f" (alpha={args.alpha}, gamma={args.gamma},"
        f" epsilon {args.epsilon} -> {args.epsilon_final}, seed {args.seed})",
        file=sys.stderr,
    )

    walked = greedy_path(maze, q, step_limit=args.step_limit)
    print(maze.render(walked, ascii_only=args.ascii))

    if walked[-1] != maze.goal:
        print(
            f"learned policy: lost after {moves(len(walked) - 1)} -- not converged,"
            f" try more episodes than {args.episodes}",
            file=sys.stderr,
        )
        return 1

    excess = (len(walked) - 1) - (len(best) - 1)
    verdict = (
        "the shortest route there is"
        if excess == 0
        else f"{moves(excess)} more than it needed"
    )
    print(f"learned policy: {moves(len(walked) - 1)} -- {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
