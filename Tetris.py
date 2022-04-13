"""
Contains classes for the game Tetris
Requires numpy to run

The Tetris implementation will be based on the TETR.IO implementation of the game.
"""
import time
import threading
import numpy as np

from typing import Callable, Optional, Union, Iterator
from collections import deque
from random import shuffle


class Tetromino:
    """
    Models a tetromino.
    Wall kicks for tetrominoes follow Tetr.io kicks
    """
    NO_TYPE: int = 0
    I_TYPE: int = 1
    J_TYPE: int = 2
    L_TYPE: int = 3
    O_TYPE: int = 4
    S_TYPE: int = 5
    T_TYPE: int = 6
    Z_TYPE: int = 7
    DEFAULT_WALL_KICKS: tuple = (
        ((),  # 0 -> 0
         ((-1, 0), (-1, -1), (0, 2), (-1, 2)),  # 0 -> 1
         ((0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0)),  # 0 -> 2
         ((1, 0), (1, -1), (0, 2), (1, 2))),  # 0 -> 3

        (((1, 0), (1, 1), (0, -2), (1, -2)),  # 1 -> 0
         (),  # 1 -> 1
         ((1, 0), (1, 1), (0, -2), (1, -2)),  # 1 -> 2
         ((1, 0), (1, -2), (1, -1), (0, -2), (0, -1))),  # 1 -> 3

        (((0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)),  # 2 -> 0
         ((-1, 0), (-1, -1), (0, 2), (-1, 2)),  # 2 -> 1
         (),  # 2 -> 2
         ((1, 0), (1, -1), (0, 2), (1, 2))),  # 2 -> 3

        (((-1, 0), (-1, 1), (0, -2), (-1, -2)),  # 3 -> 0
         ((-1, 0), (-1, -2), (-1, -1), (0, -2), (0, -1)),  # 3 -> 1
         ((-1, 0), (-1, 1), (0, -2), (-1, -2)),  # 3 -> 2
         ()))  # 3 -> 3

    def __init__(self, x: int, y: int, rotation: int, rotation_states: tuple, wall_kicks: tuple = None) -> None:
        """
        Initializer
        :param x: X coordinate position
        :param y: Y coordinate position
        :param rotation: Rotation (one of 4 states: 0, 1, 2 or 3)
        :param rotation_states: Data of the tetrominoes in each rotation state (Used by child classes)
        :param wall_kicks: The valid wall kicks of the tetromino
        """
        self.x: int = x
        self.y: int = y
        self.rotation: int = rotation
        self._saved_x: int = x
        self._saved_y: int = y
        self._saved_rotation: int = rotation
        self.initial_x: int = x
        self.initial_y: int = y
        self.initial_rotation: int = rotation
        self._rotation_states: tuple = rotation_states
        self._wall_kicks: tuple = wall_kicks if wall_kicks is not None \
            else Tetromino.DEFAULT_WALL_KICKS
        self._box_size: int = self._rotation_states[0].shape[0]  # Length of the bounding box square of the first state

    def get_wall_kicks(self, rotation: int, clockwise_rotations: int) -> tuple:
        """
        Returns a tuple of wall kicks based on the current and input rotation.
        The tuple is in order of wall kick application
        :param rotation: The initial rotation state
        :param clockwise_rotations: The number of clockwise rotations to reach the final rotation state
        :return: A tuple of wall kicks in order of application
        """
        return self._wall_kicks[rotation][(rotation + clockwise_rotations) % 4]

    def move(self, x: int = 0, y: int = 0, clockwise_rotations: int = 0) -> None:
        """
        Moves the tetromino
        :param x: x value to move by
        :param y: y value to move by
        :param clockwise_rotations: number of clockwise rotations to be made
        :return: None
        """
        self.x += x
        self.y += y
        self.rotation = (self.rotation + clockwise_rotations) % 4

    def box(self) -> np.ndarray:
        """
        Returns the tetromino data in a square bounding box.
        :return: The tetromino data (numpy array of boolean values stating the location of each block).
        """
        return self._rotation_states[self.rotation]

    def box_ranges(self) -> tuple:
        """
        Returns the bounding box range of the tetromino in the grid.
        :return: X0, X1, Y0, Y1 (where (X0, Y0) is the top left of the bounding box and (X1, Y1) is the bottom right)
        """
        return self.x, self.x + self._box_size, self.y, self.y + self._box_size

    def save(self) -> None:
        """
        Saves the current tetromino position and rotation state to be loaded later.
        This allows for previews of the piece, where the actual position of the piece is stored, while the current
        position is changed. When a successful position is found, the preview would be saved, else the original
        position will be loaded back.
        :return: None
        """
        self._saved_x = self.x
        self._saved_y = self.y
        self._saved_rotation = self.rotation

    def load(self) -> None:
        """
        Loads the saved tetromino position and rotation state.
        :return: None
        """
        self.x = self._saved_x
        self.y = self._saved_y
        self.rotation = self._saved_rotation

    def reset(self) -> None:
        """
        Resets the tetromino position and rotation state back to the initial values.
        :return: None
        """
        self.x = self.initial_x
        self.y = self.initial_y
        self.rotation = self.initial_rotation

    def get_type(self) -> int:
        """
        Returns the type of the tetromino, to be overridden by the subclasses
        :return: int enum of a tetromino type in class Tetromino
        """
        return Tetromino.NO_TYPE


class Grid:
    """
    Models the tetris playing grid.
    """
    BORDER_WIDTH: int = 4  # Borders on the left, right and bottom of the grid
    TOP_BUFFER: int = 8  # Buffer on the top of the grid

    GRID_WIDTH: int = 10  # Width of default tetris grid is 10
    GRID_HEIGHT: int = 20  # Height of default tetris grid is 20

    EMPTY_SPACE: bool = False  # Boolean enum for an empty space
    FILLED_SPACE: bool = True  # Boolean enum for a filled space

    EMPTY_ROW: np.ndarray = np.array(  # Empty row with borders
        (FILLED_SPACE,) * BORDER_WIDTH + (EMPTY_SPACE,) * GRID_WIDTH + (FILLED_SPACE,) * BORDER_WIDTH, dtype=bool)
    FULL_ROW: np.ndarray = np.array(  # Full row with borders
        (FILLED_SPACE,) * BORDER_WIDTH + (FILLED_SPACE,) * GRID_WIDTH + (FILLED_SPACE,) * BORDER_WIDTH, dtype=bool)

    EMPTY_GRID: np.ndarray = np.array(  # Empty initial grid to be copied
        (EMPTY_ROW,) * TOP_BUFFER + (EMPTY_ROW,) * GRID_HEIGHT + (FULL_ROW,) * BORDER_WIDTH, dtype=bool)
    BOUNDS_GRID: np.ndarray = np.array(  # Grid that defines the true tetris grid
        (FULL_ROW,) * TOP_BUFFER + (EMPTY_ROW,) * GRID_HEIGHT + (FULL_ROW,) * BORDER_WIDTH, dtype=bool)

    def __init__(self) -> None:
        self.grid: np.ndarray = Grid.EMPTY_GRID.copy()

    def is_empty(self) -> bool:
        """
        Checks if the grid is empty
        :return: whether the grid is empty
        """
        return np.array_equal(self.grid, Grid.EMPTY_GRID)

    def add_tetromino(self, tetromino: Tetromino) -> None:
        """
        Adds a tetromino to the grid.
        :param tetromino: Tetromino
        :return: None
        """
        x0, x1, y0, y1 = tetromino.box_ranges()
        self.grid[y0:y1, x0:x1] += tetromino.box()

    def check_tetromino(self, tetromino: Tetromino) -> bool:
        """
        Checks if a tetromino is valid and is able to fit in the grid without overlaps
        :param tetromino: Tetromino
        :return: Boolean of whether the tetromino is valid
        """
        x0, x1, y0, y1 = tetromino.box_ranges()
        return not np.any(self.grid[y0:y1, x0:x1] & tetromino.box())

    def check_whole_tetromino_overlap(self, tetromino: Tetromino) -> bool:
        """
        Checks if the entirety of the tetromino overlaps with the grid
        :param tetromino: Tetromino
        :return: Boolean of whether the whole tetromino will overlap the grid
        """
        x0, x1, y0, y1 = tetromino.box_ranges()
        return not np.any((self.grid[y0:y1, x0:x1] != tetromino.box()) & tetromino.box())

    def check_tetromino_touching_bounds(self, tetromino: Tetromino) -> bool:
        """
        Checks if any part of the tetromino is within the bounds of the grid
        :param tetromino: Tetromino
        :return: Boolean of whether the tetromino is touching bounds
        """
        x0, x1, y0, y1 = tetromino.box_ranges()
        return np.any(np.logical_not(Grid.BOUNDS_GRID[y0:y1, x0:x1]) & tetromino.box())

    def clear_get_cleared_rows(self) -> np.ndarray:
        """
        Clears any full rows, and returns the indexes of the full rows.
        :return: The indexes of the full rows in a numpy array.
        """
        cleared_rows = np.nonzero(np.all(self.grid[:-Grid.BORDER_WIDTH] == Grid.FULL_ROW, axis=1))[0]
        for i in cleared_rows:
            self.grid[1:i + 1] = self.grid[:i]
        return cleared_rows


class IPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "I" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((0, 0, 0, 0),
                  (1, 1, 1, 1),
                  (0, 0, 0, 0),
                  (0, 0, 0, 0)), dtype=bool),

        np.array(((0, 0, 1, 0),
                  (0, 0, 1, 0),
                  (0, 0, 1, 0),
                  (0, 0, 1, 0)), dtype=bool),

        np.array(((0, 0, 0, 0),
                  (0, 0, 0, 0),
                  (1, 1, 1, 1),
                  (0, 0, 0, 0)), dtype=bool),

        np.array(((0, 1, 0, 0),
                  (0, 1, 0, 0),
                  (0, 1, 0, 0),
                  (0, 1, 0, 0)), dtype=bool))
    WALL_KICKS: tuple = (
        ((),  # 0 -> 0
         ((-2, 0), (1, 0), (-2, 1), (1, -2)),  # 0 -> 1
         ((0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0)),  # 0 -> 2
         ((-1, 0), (2, 0), (-1, -2), (2, 1))),  # 0 -> 3

        (((2, 0), (-1, 0), (2, -1), (-1, 2)),  # 1 -> 0
         (),  # 1 -> 1
         ((-1, 0), (2, 0), (-1, -2), (2, 1)),  # 1 -> 2
         ((1, 0), (1, -2), (1, -1), (0, -2), (0, -1))),  # 1 -> 3

        (((0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)),  # 2 -> 0
         ((1, 0), (-2, 0), (1, 2), (-2, -1)),  # 2 -> 1
         (),  # 2 -> 2
         ((2, 0), (-1, 0), (2, -1), (-1, 2))),  # 2 -> 3

        (((1, 0), (-2, 0), (1, 2), (-2, -1)),  # 3 -> 0
         ((-1, 0), (-1, -2), (-1, -1), (0, -2), (0, -1)),  # 3 -> 1
         ((-2, 0), (1, 0), (-2, 1), (1, -2)),  # 3 -> 2
         ()))  # 3 -> 3

    def __init__(self, x: int = Grid.BORDER_WIDTH + 3, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0) -> None:
        super().__init__(x, y, rotation, IPiece.ROTATION_STATES, IPiece.WALL_KICKS)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.I_TYPE


class JPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "J" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((1, 0, 0),
                  (1, 1, 1),
                  (0, 0, 0)), dtype=bool),

        np.array(((0, 1, 1),
                  (0, 1, 0),
                  (0, 1, 0)), dtype=bool),

        np.array(((0, 0, 0),
                  (1, 1, 1),
                  (0, 0, 1)), dtype=bool),

        np.array(((0, 1, 0),
                  (0, 1, 0),
                  (1, 1, 0)), dtype=bool))

    def __init__(self, x: int = Grid.BORDER_WIDTH + 3, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0):
        super().__init__(x, y, rotation, JPiece.ROTATION_STATES)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.J_TYPE


class LPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "L" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((0, 0, 1),
                  (1, 1, 1),
                  (0, 0, 0)), dtype=bool),

        np.array(((0, 1, 0),
                  (0, 1, 0),
                  (0, 1, 1)), dtype=bool),

        np.array(((0, 0, 0),
                  (1, 1, 1),
                  (1, 0, 0)), dtype=bool),

        np.array(((1, 1, 0),
                  (0, 1, 0),
                  (0, 1, 0)), dtype=bool))

    def __init__(self, x: int = Grid.BORDER_WIDTH + 3, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0) -> None:
        super().__init__(x, y, rotation, LPiece.ROTATION_STATES)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.L_TYPE


class OPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "O" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((1, 1),
                  (1, 1)), dtype=bool),

        np.array(((1, 1),
                  (1, 1)), dtype=bool),

        np.array(((1, 1),
                  (1, 1)), dtype=bool),

        np.array(((1, 1),
                  (1, 1)), dtype=bool))
    WALL_KICKS: tuple = (
        ((),  # 0 -> 0
         (),  # 0 -> 1
         (),  # 0 -> 2
         (),),  # 0 -> 3

        ((),  # 1 -> 0
         (),  # 1 -> 1
         (),  # 1 -> 2
         (),),  # 1 -> 3

        ((),  # 2 -> 0
         (),  # 2 -> 1
         (),  # 2 -> 2
         (),),  # 2 -> 3

        ((),  # 3 -> 0
         (),  # 3 -> 1
         (),  # 3 -> 2
         (),))  # 3 -> 3

    def __init__(self, x: int = Grid.BORDER_WIDTH + 4, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0) -> None:
        super().__init__(x, y, rotation, OPiece.ROTATION_STATES, OPiece.WALL_KICKS)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.O_TYPE


class SPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "S" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((0, 1, 1),
                  (1, 1, 0),
                  (0, 0, 0)), dtype=bool),

        np.array(((0, 1, 0),
                  (0, 1, 1),
                  (0, 0, 1)), dtype=bool),

        np.array(((0, 0, 0),
                  (0, 1, 1),
                  (1, 1, 0)), dtype=bool),

        np.array(((1, 0, 0),
                  (1, 1, 0),
                  (0, 1, 0)), dtype=bool))

    def __init__(self, x: int = Grid.BORDER_WIDTH + 3, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0) -> None:
        super().__init__(x, y, rotation, SPiece.ROTATION_STATES)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.S_TYPE


class TPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "T" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((0, 1, 0),
                  (1, 1, 1),
                  (0, 0, 0)), dtype=bool),

        np.array(((0, 1, 0),
                  (0, 1, 1),
                  (0, 1, 0)), dtype=bool),

        np.array(((0, 0, 0),
                  (1, 1, 1),
                  (0, 1, 0)), dtype=bool),

        np.array(((0, 1, 0),
                  (1, 1, 0),
                  (0, 1, 0)), dtype=bool))

    def __init__(self, x: int = Grid.BORDER_WIDTH + 3, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0) -> None:
        super().__init__(x, y, rotation, TPiece.ROTATION_STATES)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.T_TYPE


class ZPiece(Tetromino):
    """
    Subclass of Tetromino modelling the "Z" piece.
    """
    ROTATION_STATES: tuple = (
        np.array(((1, 1, 0),
                  (0, 1, 1),
                  (0, 0, 0)), dtype=bool),

        np.array(((0, 0, 1),
                  (0, 1, 1),
                  (0, 1, 0)), dtype=bool),

        np.array(((0, 0, 0),
                  (1, 1, 0),
                  (0, 1, 1)), dtype=bool),

        np.array(((0, 1, 0),
                  (1, 1, 0),
                  (1, 0, 0)), dtype=bool))

    def __init__(self, x: int = Grid.BORDER_WIDTH + 3, y: int = Grid.TOP_BUFFER - 2, rotation: int = 0) -> None:
        super().__init__(x, y, rotation, ZPiece.ROTATION_STATES)

    def get_type(self) -> int:
        """
        Gives the type of the tetromino
        :return: int enum from Tetromino
        """
        return Tetromino.Z_TYPE


class TetrominoQueue:
    """
    Models the '7-bag' system of Tetris.
    To see the next pieces, do t_queue.next[n], where n is the 0-indexed position of the piece in queue.
    Slicing of next is not supported
    """

    def __init__(self, next_n: int = 5) -> None:
        """
        Initializer
        :param next_n: The number of next pieces to be shown, must be between 1 and 7 inclusive
        """
        self.next_n: int = next_n
        self.next: deque = deque(maxlen=next_n + 7)
        self.bag: list = [IPiece, JPiece, LPiece, OPiece, SPiece, TPiece, ZPiece]
        self.add_7_bag()

    def add_7_bag(self) -> None:
        """
        Adds a bag of 7 random unique tetrominoes to the queue.
        :return: None
        """
        shuffle(self.bag)
        for tetromino in self.bag:
            self.next.append(tetromino())

    def get_tetromino(self) -> Tetromino:
        """
        Gets the next tetromino.
        :return: The next tetromino.
        """
        if len(self.next) <= self.next_n:
            self.add_7_bag()
        return self.next.popleft()


class Game:
    """
    Models a Tetris game.
    Note that any unusual usage, such as not following moves set by tetris guidelines,
    or inputs after the end of the game can lead to unusual behaviour

    To add event handlers, use the GameEvents in the object
    """
    DEFAULT_LOCK_DELAY: float = 0.5  # The default number of seconds needed for the piece to lock into place
    INFINITY_LIMIT: int = 15  # Limit to the number of spins or moves that can reset the locking of the tetromino

    def __init__(self, autonomous: bool = True) -> None:
        """
        Initializer
        :param autonomous: whether threading is allowed for the game to run on its own
        """
        self.grid: Grid = Grid()
        self.tetromino_queue: TetrominoQueue = TetrominoQueue()
        self.current_tetromino: Optional[Tetromino] = None
        self.hold_tetromino: Optional[Tetromino] = None
        self.game_events: GameEvents = GameEvents(self, timed=autonomous)
        self.score: Score = Score(self)
        self._sync_lock: threading.Lock = threading.Lock()

        self.is_locking: bool = False  # whether the game is in a state of locking,
        # and thus whether the current tetromino should be able to be modified
        self.is_dead: bool = False
        self.has_held_once: bool = False
        self.infinity_count: int = 0
        self.was_on_filled: bool = False
        self.is_last_move_spin = False

        self.set_gravity(self.score.get_gravity())

    def start(self) -> None:
        """
        Starts the game events that handle timed events in the game
        :return: None
        """
        self.game_events.call_event(GameEvents.NEXT_EVENT)

    def end(self) -> None:
        """
        Ends the game, and stops the game from running timed events
        :return: None
        """
        self.is_dead = True
        self.game_events.call_event(GameEvents.DEATH_EVENT)
        self.game_events.timed = False

    def next(self) -> None:
        """
        The next tetromino is loaded and replaces the previous tetromino
        Used by the GameEvents class to control the game
        Should be used with caution, as unusual behaviour might emerge
        :return: None
        """
        with self._sync_lock:
            self.current_tetromino = self.tetromino_queue.get_tetromino()
            self.is_locking = False

        self._check_on_filled_call_event()
        self.game_events.call_event(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT)

    def lock(self) -> None:
        """
        The tetromino locks
        Used by the GameEvents class to control the game
        Should be used with caution, as unusual behaviour might emerge
        :return: None
        """
        if self.is_locking:
            return

        with self._sync_lock:
            self.is_locking = True

            self.grid.add_tetromino(self.current_tetromino)
            self.score.log_clear(self.grid.clear_get_cleared_rows())
            self.has_held_once = False
            self.infinity_count = 0
            self.is_last_move_spin = False

        self._check_death_call_event()
        self.game_events.call_event(GameEvents.GRID_CHANGE_EVENT)
        self.game_events.call_event(GameEvents.NEXT_EVENT)

    def step(self) -> None:
        """
        Steps the tetromino downwards (if possible)
        Used by the GameEvents class to control the game
        Should be used with caution, as unusual behaviour might emerge
        :return: None
        """
        if self.is_locking:
            return

        with self._sync_lock:
            self.current_tetromino.save()
            self.current_tetromino.move(0, 1)
            if not self.grid.check_tetromino(self.current_tetromino):
                self.current_tetromino.load()
                return

            self.score.log_step()
            self.is_last_move_spin = False

        self.game_events.call_event(GameEvents.MOVE_EVENT)
        self._check_on_filled_call_event()
        self.game_events.call_event(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT)

    def input_move(self, x: int = 0, y: int = 0) -> None:
        """
        Moves the tetromino (if possible)
        :param x: Movement along x-axis, left and right (negative value for left, positive value for right)
        :param y: Movement along y-axis, up and down (negative value for up, positive value for down)
        :return: None
        """
        if self.is_locking:
            return

        with self._sync_lock:
            self.current_tetromino.save()
            self.current_tetromino.move(x, y)
            if not self.grid.check_tetromino(self.current_tetromino):
                self.current_tetromino.load()
                return

            if x == 0 and y == 1:
                self.score.log_step()
            self.is_last_move_spin = False

        self.game_events.call_event(GameEvents.MOVE_EVENT)
        self._check_on_filled_call_event()
        self.game_events.call_event(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT)

    def input_rotate(self, clockwise_rotations) -> None:
        """
        Rotates the tetromino (if possible)
        :param clockwise_rotations: number of clockwise rotations, negative values are allowed
        :return: None
        """
        if self.is_locking:
            return

        with self._sync_lock:
            rotation_possible = False
            self.current_tetromino.save()

            # Checks rotation without wall kicks
            self.current_tetromino.move(0, 0, clockwise_rotations=clockwise_rotations)
            if self.grid.check_tetromino(self.current_tetromino):
                rotation_possible = True
            else:
                # Rotation with wall kicks
                self.current_tetromino.load()
                wall_kicks = self.current_tetromino.get_wall_kicks(self.current_tetromino.rotation, clockwise_rotations)
                for wall_kick in wall_kicks:
                    self.current_tetromino.load()
                    self.current_tetromino.move(wall_kick[0], wall_kick[1], clockwise_rotations=clockwise_rotations)
                    if self.grid.check_tetromino(self.current_tetromino):
                        rotation_possible = True
                        break

            if not rotation_possible:
                self.current_tetromino.load()
            else:
                self.is_last_move_spin = True

        if rotation_possible:
            self.game_events.call_event(GameEvents.ROTATE_EVENT)
            self._check_on_filled_call_event()
            self.game_events.call_event(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT)

    def input_hard_drop(self) -> None:
        """
        Hard drops the tetromino and locks it
        :return: None
        """
        self.score.log_hard_drop(self.move_until_stopped())
        self.game_events.call_event(GameEvents.LOCK_EVENT)

    def move_until_stopped(self, x: int = 0, y: int = 1) -> int:
        """
        Moves the current Tetromino in a specifiable direction until it is unable to move.
        The default direction is downwards
        :param x: the movement amount in the x direction
        :param y: the movement amount in the y direction
        :return: the number of times moved
        """
        if self.is_locking:
            return 0

        with self._sync_lock:
            num_times = 0
            self.current_tetromino.save()
            self.current_tetromino.move(x, y)
            while self.grid.check_tetromino(self.current_tetromino):
                num_times += 1
                self.current_tetromino.save()
                self.current_tetromino.move(x, y)

                if x == 0 and y == 1:
                    self.score.log_step()
                self.is_last_move_spin = False
            self.current_tetromino.load()

        if num_times > 0:
            self._check_on_filled_call_event()
            self.game_events.call_event(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT)

        return num_times

    def input_hold(self) -> None:
        """
        Holds the piece (if possible)
        :return: None
        """
        if self.is_locking:
            return

        if self.has_held_once:
            return

        if self.hold_tetromino is None:
            with self._sync_lock:
                self.hold_tetromino = self.current_tetromino
                self.current_tetromino = None
            self.next()
        else:
            with self._sync_lock:
                self.hold_tetromino, self.current_tetromino = self.current_tetromino, self.hold_tetromino

        with self._sync_lock:
            self.has_held_once = True
            self.hold_tetromino.reset()
            self.infinity_count = 0
            self.is_last_move_spin = False

        self.game_events.call_event(GameEvents.HOLD_EVENT)
        self.game_events.call_event(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT)

    def is_on_filled(self) -> bool:
        """
        Checks whether the current tetromino is on a filled or empty space
        :return: whether if it is on a filled space
        """
        with self._sync_lock:
            self.current_tetromino.save()
            self.current_tetromino.move(0, 1)
            is_tetromino_on_filled = not self.grid.check_tetromino(self.current_tetromino)
            if is_tetromino_on_filled or self.was_on_filled:
                self.infinity_count += 1
            self.current_tetromino.load()
        return is_tetromino_on_filled

    def _check_on_filled_call_event(self) -> None:
        """
        Checks if a tetromino is on a filled or empty space
        If it is on a filled space, the relevant event will be called
        :return: None
        """
        if self.is_locking:
            return

        is_on_filled = self.is_on_filled()
        if is_on_filled:
            if self.infinity_count > Game.INFINITY_LIMIT:
                self.game_events.call_event(GameEvents.LOCK_EVENT)
            self.game_events.call_event(GameEvents.TETROMINO_ON_FILLED_EVENT)

        with self._sync_lock:
            self.was_on_filled = is_on_filled

    def _check_death_call_event(self) -> None:
        """
        Checks if the game is lost
        If the game is lost, the relevant event will be called
        :return: None
        """
        if (not self.grid.check_tetromino(self.tetromino_queue.next[0]) or
            not self.grid.check_tetromino_touching_bounds(self.current_tetromino)) and \
                not self.is_dead:
            self.end()

    def set_gravity(self, new_gravity: float) -> None:
        """
        Updates the gravity of the Tetris game
        :param new_gravity: the new gravity (in seconds per row)
        :return: None
        """
        self.game_events.set_timer_duration(GameEvents.STEP_EVENT, new_gravity)


class Score:
    """
    Tracks the score and level of the tetris game.
    """
    LINES_CLEARED_SCORE: tuple = (0, 100, 300, 500, 800)
    T_SPIN_LINES_CLEARED_SCORE: tuple = (400, 800, 1200, 1600)
    MINI_T_SPIN_LINES_CLEARED_SCORE: tuple = (100, 200, 400)
    ALL_CLEAR_SCORE: int = 3500

    GRAVITY = [None, 1.0, 0.643, 0.404, 0.249, 0.150,
               0.0880, 0.0505, 0.0283, 0.0155, 0.00827,
               0.00431, 0.00219, 0.00108, 0.00052, 0.00024]  # seconds per row

    def __init__(self, game: Game, starting_score: int = 0, starting_level: int = 1):
        """
        Initializer
        :param game: the tetris game
        :param starting_score: the specifiable starting score
        :param starting_level: the specifiable starting level
        """
        self.game: Game = game
        self.score: int = starting_score
        self.level: int = starting_level
        self.current_level_lines_cleared: int = 0

        self.is_last_difficult: bool = False
        self.combo: int = -1
        self.is_step_soft_drop: bool = False  # whether the step is considered a soft drop

    def get_gravity(self) -> float:
        """
        Returns the current gravity based on the level
        :return: the gravity in seconds per row
        """
        i = self.level
        if self.level >= len(Score.GRAVITY):
            i = -1
        return Score.GRAVITY[i]

    def set_level(self, level: int) -> None:
        """
        Sets the level of the game and changes the gravity, use this function to set the level
        :param level: the level set
        """
        self.level = level
        self.game.set_gravity(self.get_gravity())

    def log_step(self) -> None:
        """
        Logs a step. If the step is an increase gravity caused by soft dropping, log the soft drop
        :return: None
        """
        if self.is_step_soft_drop:
            self.log_soft_drop()

    def log_soft_drop(self) -> None:
        """
        Logs a soft drop
        :return: None
        """
        self.score += 1

    def log_hard_drop(self, num_rows: int) -> None:
        """
        Logs a hard drop
        :param num_rows: the number of rows dropped
        :return: None
        """
        self.score += 2 * num_rows

    def log_clear(self, cleared_rows: np.ndarray) -> None:
        """
        Logs the locking of the tetromino and takes the list of lines cleared as the input,
        this is called after the tetromino has been added to the grid
        :param cleared_rows: the list of rows cleared
        :return: None
        """
        current_tetromino = self.game.current_tetromino
        grid = self.game.grid
        game_events = self.game.game_events

        n_cleared_rows = cleared_rows.shape[0]
        is_this_difficult = False
        total_add_score = 0

        default_scoring = True

        # Check valid conditions for T Spin
        if current_tetromino.get_type() == Tetromino.T_TYPE and self.game.is_last_move_spin:

            # Check T-Spin
            if grid.check_whole_tetromino_overlap(Score.TSpinCheckerLeft(current_tetromino)) or \
                    grid.check_whole_tetromino_overlap(Score.TSpinCheckerRight(current_tetromino)):
                default_scoring = False
                is_this_difficult = True

                # Add T-Spin score
                total_add_score += Score.T_SPIN_LINES_CLEARED_SCORE[n_cleared_rows] * self.level
                # Call event
                game_events.call_event(GameEvents.T_SPIN_EVENT)

            # Check Mini T-Spin
            elif grid.check_whole_tetromino_overlap(Score.MiniTSpinCheckerLeft(current_tetromino)) or \
                    grid.check_whole_tetromino_overlap(Score.MiniTSpinCheckerRight(current_tetromino)):
                default_scoring = False
                is_this_difficult = True

                # Add Mini T-Spin score
                total_add_score += Score.MINI_T_SPIN_LINES_CLEARED_SCORE[n_cleared_rows] * self.level
                # Call event
                game_events.call_event(GameEvents.MINI_T_SPIN_EVENT)

        if default_scoring:
            # Add default score
            total_add_score += Score.LINES_CLEARED_SCORE[n_cleared_rows] * self.level

        # Check all clear
        if self.game.grid.is_empty():
            total_add_score += Score.ALL_CLEAR_SCORE
            # Call event
            game_events.call_event(GameEvents.ALL_CLEAR_EVENT)

        # Check level up
        self.current_level_lines_cleared += n_cleared_rows
        self.level_up()

        # difficulty multiplier
        total_add_score_multiplier = 1
        if self.is_last_difficult and is_this_difficult:
            total_add_score_multiplier *= 1.5
        # combo Multiplier
        if n_cleared_rows >= 1:
            self.combo += 1
        else:
            self.combo = -1
        # modify score and other data
        self.score += total_add_score * total_add_score_multiplier
        self.is_last_difficult = is_this_difficult

    def level_up(self) -> None:
        """
        Attempts to level up if possible
        :return: None
        """
        # check level up
        limit = (self.level + 1) * 2
        if self.level <= 10:
            limit -= 1
        if self.current_level_lines_cleared > limit:
            self.set_level(self.level + 1)
            self.current_level_lines_cleared -= limit

    class TSpinCheckerLeft(Tetromino):
        """
        Subclass of Tetromino used to detect T Spins. Left-handed version
        """
        ROTATION_STATES: tuple = (
            np.array(((1, 1, 1),
                      (1, 1, 1),
                      (1, 0, 0)), dtype=bool),

            np.array(((1, 1, 1),
                      (0, 1, 1),
                      (0, 1, 1)), dtype=bool),

            np.array(((0, 0, 1),
                      (1, 1, 1),
                      (1, 1, 1)), dtype=bool),

            np.array(((1, 1, 0),
                      (1, 1, 0),
                      (1, 1, 1)), dtype=bool))

        def __init__(self, tetromino: Tetromino) -> None:
            super().__init__(tetromino.x, tetromino.y, tetromino.rotation, Score.TSpinCheckerLeft.ROTATION_STATES)

    class TSpinCheckerRight(Tetromino):
        """
        Subclass of Tetromino used to detect T Spins. Right-handed version
        """
        ROTATION_STATES: tuple = (
            np.array(((1, 1, 1),
                      (1, 1, 1),
                      (0, 0, 1)), dtype=bool),

            np.array(((0, 1, 1),
                      (0, 1, 1),
                      (1, 1, 1)), dtype=bool),

            np.array(((1, 0, 0),
                      (1, 1, 1),
                      (1, 1, 1)), dtype=bool),

            np.array(((1, 1, 1),
                      (1, 1, 0),
                      (1, 1, 0)), dtype=bool))

        def __init__(self, tetromino: Tetromino) -> None:
            super().__init__(tetromino.x, tetromino.y, tetromino.rotation, Score.TSpinCheckerRight.ROTATION_STATES)

    class MiniTSpinCheckerLeft(Tetromino):
        """
        Subclass of Tetromino used to detect mini T Spins. Left-handed version
        """
        ROTATION_STATES: tuple = (
            np.array(((1, 1, 0),
                      (1, 1, 1),
                      (1, 0, 1)), dtype=bool),

            np.array(((1, 1, 1),
                      (0, 1, 1),
                      (1, 1, 0)), dtype=bool),

            np.array(((1, 0, 1),
                      (1, 1, 1),
                      (0, 1, 1)), dtype=bool),

            np.array(((0, 1, 1),
                      (1, 1, 0),
                      (1, 1, 1)), dtype=bool))

        def __init__(self, tetromino: Tetromino) -> None:
            super().__init__(tetromino.x, tetromino.y, tetromino.rotation, Score.MiniTSpinCheckerLeft.ROTATION_STATES)

    class MiniTSpinCheckerRight(Tetromino):
        """
        Subclass of Tetromino used to detect mini T Spins. Right-handed version
        """
        ROTATION_STATES: tuple = (
            np.array(((0, 1, 1),
                      (1, 1, 1),
                      (1, 0, 1)), dtype=bool),

            np.array(((1, 1, 0),
                      (0, 1, 1),
                      (1, 1, 1)), dtype=bool),

            np.array(((1, 0, 1),
                      (1, 1, 1),
                      (1, 1, 0)), dtype=bool),

            np.array(((1, 1, 1),
                      (1, 1, 0),
                      (0, 1, 1)), dtype=bool))

        def __init__(self, tetromino: Tetromino) -> None:
            super().__init__(tetromino.x, tetromino.y, tetromino.rotation, Score.MiniTSpinCheckerRight.ROTATION_STATES)


class GameEvents:
    """
    Handles event calls and allows an external event handler for each event.
    It also helps to call events in the game, including timing events.
    Note, step event, lock event, tetromino on filled event and next event
    will not be called after the relevant events occur, but instead before the events.
    """
    # event type enums
    # auxiliary events for timing and calling Tetris game methods, these are not listeners
    STEP_EVENT: int = 0
    LOCK_EVENT: int = 1
    TETROMINO_ON_FILLED_EVENT: int = 2
    NEXT_EVENT: int = 3

    # auxiliary events for managing press-and-hold inputs
    SIDEWAYS_MOVEMENT_EVENT: int = 13
    INPUT_DELAY_END_EVENT: int = 14
    NEXT_DELAY_END_EVENT: int = 15

    # game_related listeners
    DEATH_EVENT: int = 4  # Listener, called when the game ends
    HOLD_EVENT: int = 5  # Listener, called when the current tetromino is held
    MOVE_EVENT: int = 6  # Listener, called when the current tetromino moves
    ROTATE_EVENT: int = 7  # Listener, called when the current tetromino rotates
    GRID_CHANGE_EVENT: int = 8  # Listener, called when the grid changes (excluding the current tetromino)
    TETROMINO_MOVEMENT_CHANGE_EVENT: int = 9  # Listener, called when the current tetromino moves, rotates or changes

    # score-related listeners
    T_SPIN_EVENT: int = 10  # Listener, called when a T-spin move is played
    MINI_T_SPIN_EVENT: int = 11  # Listener, called when a Mini T-spin move is played
    ALL_CLEAR_EVENT: int = 12  # Listener, called when an all clear happens

    # handler id used for handlers within the class
    INTERNAL_HANDLER_ID: int = 0

    def __init__(self, game: Game, timed: bool = False) -> None:
        """
        Initializer
        :param game: The game object
        :param timed: Whether the game event will have timer thread calls and threaded event calls
        """
        self.game: Game = game
        self.timed = timed
        self.threaded_external_event_handling = True  # whether the external event calls will be handled using threading
        self._sync_lock: threading.Lock = threading.Lock()
        # Event handlers
        self._event_handlers: dict = {
            GameEvents.STEP_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._step_handler},
            GameEvents.LOCK_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._lock_handler},
            GameEvents.TETROMINO_ON_FILLED_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._tetromino_on_filled_handler},
            GameEvents.NEXT_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._next_handler},
            GameEvents.SIDEWAYS_MOVEMENT_EVENT: {},
            GameEvents.INPUT_DELAY_END_EVENT: {},
            GameEvents.NEXT_DELAY_END_EVENT: {},
            GameEvents.DEATH_EVENT: {},
            GameEvents.HOLD_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._hold_handler},
            GameEvents.MOVE_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._move_handler},
            GameEvents.ROTATE_EVENT: {GameEvents.INTERNAL_HANDLER_ID: self._rotate_handler},
            GameEvents.GRID_CHANGE_EVENT: {},
            GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT: {},
            GameEvents.T_SPIN_EVENT: {},
            GameEvents.MINI_T_SPIN_EVENT: {},
            GameEvents.ALL_CLEAR_EVENT: {}
        }
        # The timer thread objects
        self._timers: dict = {}
        # The time a timer was started
        self._start_timer_time: dict = {
            GameEvents.STEP_EVENT: 0.,
            GameEvents.LOCK_EVENT: 0.,
            GameEvents.SIDEWAYS_MOVEMENT_EVENT: 0.,
            GameEvents.INPUT_DELAY_END_EVENT: 0.,
            GameEvents.NEXT_DELAY_END_EVENT: 0.
        }
        # The initial counter values used for resetting the counter (in seconds)
        # or the time needed for the timer to count to 0 (in seconds)
        self._timer_counter_initials: dict = {
            GameEvents.STEP_EVENT: 0,
            GameEvents.LOCK_EVENT: Game.DEFAULT_LOCK_DELAY,
            GameEvents.SIDEWAYS_MOVEMENT_EVENT: 0,
            GameEvents.INPUT_DELAY_END_EVENT: 0,
            GameEvents.NEXT_DELAY_END_EVENT: 0
        }
        # The counter values
        self._timer_counters: dict = {
            GameEvents.STEP_EVENT: self._timer_counter_initials[GameEvents.STEP_EVENT],
            GameEvents.LOCK_EVENT: self._timer_counter_initials[GameEvents.LOCK_EVENT],
            GameEvents.SIDEWAYS_MOVEMENT_EVENT: self._timer_counter_initials[GameEvents.SIDEWAYS_MOVEMENT_EVENT],
            GameEvents.INPUT_DELAY_END_EVENT: self._timer_counter_initials[GameEvents.INPUT_DELAY_END_EVENT],
            GameEvents.NEXT_DELAY_END_EVENT: self._timer_counter_initials[GameEvents.NEXT_DELAY_END_EVENT]
        }

    def get_event_handler_id(self, event_type: int) -> int:
        """
        Returns a unique event handler id for an event type
        :param event_type: the event type
        :return: a unique event handler id
        """
        i_candidate = 1
        while i_candidate in self._event_handlers[event_type]:
            i_candidate += 1
        return i_candidate

    def set_timer_duration(self, event_type: int, duration: float) -> None:
        """
        Sets the duration of a timer for it to countdown
        :param event_type: the event type
        :param duration: the new duration of the timer in seconds
        :return: None
        """
        with self._sync_lock:
            old_duration = self._timer_counter_initials[event_type]
            self._timer_counter_initials[event_type] = duration
            self._timer_counters[event_type] += (duration - old_duration)

        if self.stop_timer(event_type):
            self.start_timer(event_type)

    def call_event(self, event_type: int) -> None:
        """
        Calls an event
        e.g. call_event(GameEvents.LOCK_EVENT)
        :param event_type: The event type to be called, which is a static attribute in the GameEvents class
        :return: None
        """
        for handler_id, handler in self._event_handlers[event_type].items():
            if handler_id == self.INTERNAL_HANDLER_ID or not self.threaded_external_event_handling:
                handler()
            else:
                threading.Thread(target=handler, daemon=True).start()

    def add_event_handler(self, event_type: int, handler: Callable) -> int:
        """
        Adds an event handler
        e.g. add_event_handler(GameEvents.LOCK_EVENT, 3, my_handler)
        Note that the event handler will be called by a predetermined thread, and not by the main thread
        :param event_type: The event type for the handler to handle, which is a static attribute in the GameEvents class
        :param handler: The handler function to be called when the event is called
        :return: The id of the event handler needed to modify the event handler
        """
        handler_id = self.get_event_handler_id(event_type)
        self._event_handlers[event_type][handler_id] = handler
        return handler_id

    def remove_event_handler(self, event_type: int, handler_id: int) -> None:
        """
        Removes a previously added event handler
        e.g. remove_event_handler(GameEvents.LOCK_EVENT, 3)
        :param event_type: The event type the handler handles, which is a static attribute in the GameEvents class
        :param handler_id: The handler id, the same id used when adding the event handler
        :return: None
        """
        del self._event_handlers[event_type][handler_id]

    def start_timer(self, event_type: int) -> None:
        """
        Starts an event timer, which will call the event after a period of time
        :param event_type: The event type of the timer, which is a static attribute in the GameEvents class
        :return: None
        """
        if not self.timed:
            return

        self.stop_timer(event_type)
        with self._sync_lock:
            self._timers[event_type] = threading.Timer(self._timer_counters[event_type], self.call_event, (event_type,))
            self._timers[event_type].daemon = True
            self._start_timer_time[event_type] = time.time()
        self._timers[event_type].start()

    def stop_timer(self, event_type: int) -> bool:
        """
        Stops an event timer, and counts down the event timer counter
        :param event_type: The event type of the timer, which is a static attribute in the GameEvents class
        :return: whether the timer was successfully stopped
        """
        if not self.timed:
            return False

        with self._sync_lock:
            if event_type in self._timers:
                self._timers[event_type].cancel()
                current_time = time.time()
                del self._timers[event_type]
                self._timer_counters[event_type] -= min(current_time - self._start_timer_time[event_type],
                                                        self._timer_counters[event_type])
                return True
        return False

    def reset_timer(self, event_type: int) -> None:
        """
        Resets a timer, as well as the event timer counter
        :param event_type: The event type of the timer, which is a static attribute in the GameEvents class
        :return: None
        """
        if not self.timed:
            return

        self.stop_timer(event_type)
        with self._sync_lock:
            self._timer_counters[event_type] = self._timer_counter_initials[event_type]

    def _step_handler(self) -> None:
        """
        Event handler for a step event
        :return: None
        """
        self.reset_timer(GameEvents.STEP_EVENT)
        self.game.step()
        self.start_timer(GameEvents.STEP_EVENT)

    def _lock_handler(self) -> None:
        """
        Event handler for a lock event
        :return: None
        """
        self.reset_timer(GameEvents.LOCK_EVENT)
        self.reset_timer(GameEvents.STEP_EVENT)
        self.game.lock()

    def _tetromino_on_filled_handler(self) -> None:
        """
        Event handler for when the current tetromino is on a filled space
        Is called whenever the position is updated and the tetromino is found to be on a filled space
        :return: None
        """
        self.start_timer(GameEvents.LOCK_EVENT)

    def _next_handler(self) -> None:
        """
        Event handler for when the next tetromino is loaded
        :return: None
        """
        self.game.next()
        self.start_timer(GameEvents.STEP_EVENT)

    def _hold_handler(self) -> None:
        """
        Event handler for when the current tetromino is held
        :return: None
        """
        self.reset_timer(GameEvents.LOCK_EVENT)
        self.reset_timer(GameEvents.STEP_EVENT)
        self.start_timer(GameEvents.STEP_EVENT)

    def _move_handler(self) -> None:
        """
        Event handler for when the current tetromino is moved
        :return: None
        """
        self.reset_timer(GameEvents.LOCK_EVENT)

    def _rotate_handler(self) -> None:
        """
        Event handler for when the current tetromino is rotated
        :return: None
        """
        self.reset_timer(GameEvents.LOCK_EVENT)


class Controller:
    """
    Class that models the input interface from which the tetris game can be played.
    Uses Game, GameEvent and Score objects to run certain pieces of code.
    """

    def __init__(self, game: Game, arr: float = 0, das: float = 0.1, dcd: float = 0, sdf: float = 10,
                 infinite_sdf: bool = True) -> None:
        """
        Initializer
        :param game: the game
        :param arr: the Automatic Repeat Rate
        :param das: the Delayed Auto Shift
        :param dcd: the DAS Cut Delay
        :param sdf: the Soft Drop Factor
        :param infinite_sdf: whether the sdf is infinite
        """
        self.game = game
        self.press_binds = {}
        self.release_binds = {}
        self.is_moving_left: bool = False
        self.is_moving_right: bool = False

        self._event_handler_ids = (
            self.game.game_events.add_event_handler(GameEvents.INPUT_DELAY_END_EVENT, self._input_delay_end_handler),
            self.game.game_events.add_event_handler(GameEvents.NEXT_DELAY_END_EVENT, self._next_delay_end_handler),
            self.game.game_events.add_event_handler(GameEvents.SIDEWAYS_MOVEMENT_EVENT,
                                                    self._sideways_movement_handler),
            self.game.game_events.add_event_handler(GameEvents.GRID_CHANGE_EVENT, self._tetromino_change_handler),
            self.game.game_events.add_event_handler(GameEvents.HOLD_EVENT, self._tetromino_change_handler)
        )

        self.arr: float = arr
        self.das: float = das
        self.dcd: float = dcd
        self.sdf: float = sdf
        self.infinite_sdf: bool = infinite_sdf
        self.set_arr(arr)  # sideways movement rate, seconds per move (Automatic Repeat Rate)
        self.set_das(das)  # time between keypress and start of repeat action in seconds (Delayed Auto Shift)
        self.set_dcd(dcd)  # das delay when a piece locks and a next piece spawns in seconds (DAS Cut Delay)
        self.set_sdf(sdf, infinite_sdf)  # multiplier for the gravity of the piece when soft dropping (Soft Drop Factor)

    # TODO DAS broken
    def _tetromino_change_handler(self):
        self.game.game_events.reset_timer(GameEvents.SIDEWAYS_MOVEMENT_EVENT)

        self.game.game_events.reset_timer(GameEvents.NEXT_DELAY_END_EVENT)
        self.game.game_events.start_timer(GameEvents.NEXT_DELAY_END_EVENT)

    # TODO DAS broken
    def _input_delay_end_handler(self):
        self.game.game_events.call_event(GameEvents.SIDEWAYS_MOVEMENT_EVENT)
        self.game.game_events.reset_timer(GameEvents.INPUT_DELAY_END_EVENT)

    # TODO DAS broken
    def _next_delay_end_handler(self):
        self.game.game_events.reset_timer(GameEvents.NEXT_DELAY_END_EVENT)
        self.game.game_events.call_event(GameEvents.SIDEWAYS_MOVEMENT_EVENT)
        self.game.game_events.call_event(GameEvents.INPUT_DELAY_END_EVENT)

    def _sideways_movement_handler(self):
        if self.is_moving_left:
            x_movement = -1
        elif self.is_moving_right:
            x_movement = 1
        else:
            self.game.game_events.reset_timer(GameEvents.SIDEWAYS_MOVEMENT_EVENT)
            self.is_moving_left = False
            self.is_moving_right = False
            return

        self.game.input_move(x_movement, 0)
        self.game.game_events.reset_timer(GameEvents.SIDEWAYS_MOVEMENT_EVENT)
        self.game.game_events.start_timer(GameEvents.SIDEWAYS_MOVEMENT_EVENT)

    def set_arr(self, arr: float) -> None:
        """
        Sets the arr (Automatic Repeat Rate)
        :param arr: the new arr
        :return: None
        """
        if arr < 0:
            raise ValueError('ARR cannot be a negative value')
        self.arr: float = arr
        self.game.game_events.set_timer_duration(GameEvents.SIDEWAYS_MOVEMENT_EVENT, self.arr)

    def set_das(self, das: float) -> None:
        """
        Sets the das (Delayed Auto Shift)
        :param das: the new das
        :return: None
        """
        if das < 0:
            raise ValueError('DAS cannot be a negative value')
        self.das: float = das
        self.game.game_events.set_timer_duration(GameEvents.INPUT_DELAY_END_EVENT, self.das)

    def set_dcd(self, dcd: float) -> None:
        """
        Sets the dcd (DAS Cut Delay)
        :param dcd: the new dcd
        :return: None
        """
        if dcd < 0:
            raise ValueError('DCD cannot be a negative value')
        self.dcd: float = dcd
        self.game.game_events.set_timer_duration(GameEvents.NEXT_DELAY_END_EVENT, self.dcd)

    def set_sdf(self, sdf: float = None, infinite_sdf: bool = False) -> None:
        """
        Sets the sdf (Soft Drop Factor)
        :param sdf: the new sdf
        :param infinite_sdf: whether the sdf is infinite
        :return: None
        """
        if sdf < 1:
            raise ValueError('SDF must be equal to or larger than 1')
        self.infinite_sdf = infinite_sdf
        self.sdf: float = sdf

    def bind(self, key: Union[int, str], press_control: Callable = None, release_control: Callable = None) -> None:
        """
        Binds a key input to functions
        :param key: the key input
        :param press_control: the function to call when the key is pressed
        :param release_control: the function to call when the key is released
        :return: None
        """
        self.press_binds[key] = press_control
        self.release_binds[key] = release_control

    def unbind(self, key) -> None:
        """
        Unbinds a key input to its functions
        :param key: the key input
        :return: None
        """
        if key in self.press_binds:
            del self.press_binds[key]
        if key in self.release_binds:
            del self.release_binds[key]

    def press(self, key) -> None:
        """
        Calls the functions bind to the input key when pressed
        :param key: the input key
        :return: None
        """
        if key in self.press_binds and self.press_binds[key] is not None:
            self.press_binds[key]()

    def release(self, key) -> None:
        """
        Calls the functions bind to the input key when released
        :param key: the input key
        :return: None
        """
        if key in self.release_binds and self.release_binds[key] is not None:
            self.release_binds[key]()

    def soft_drop_press(self) -> None:
        """
        Handles the soft drop when the control is pressed and held down
        :return: None
        """
        self.game.score.is_step_soft_drop = True
        if self.infinite_sdf:
            self.game.move_until_stopped()
        else:
            if not self.game.game_events.timed:
                self.game.input_move(0, 1)
                return
            self.game.game_events.set_timer_duration(GameEvents.STEP_EVENT, self.game.score.get_gravity() / self.sdf)

    def soft_drop_release(self) -> None:
        """
        Handles the soft drop when the control is released
        :return: None
        """
        self.game.score.is_step_soft_drop = False
        if not self.infinite_sdf:
            if not self.game.game_events.timed:
                return
            self.game.game_events.set_timer_duration(GameEvents.STEP_EVENT, self.game.score.get_gravity())

    def soft_drop(self) -> None:
        """
        Presses and releases the soft drop input
        :return: None
        """
        self.soft_drop_press()
        self.soft_drop_release()

    def left_press(self) -> None:
        """
        Handles the left movement when the control is pressed and held down
        :return: None
        """
        self.game.input_move(-1, 0)
        if not self.game.game_events.timed:
            return
        self.game.game_events.start_timer(GameEvents.INPUT_DELAY_END_EVENT)
        self.is_moving_left = True

    def left_release(self) -> None:
        """
        Handles the left movement when the control is released
        :return: None
        """
        if not self.game.game_events.timed:
            return
        self.game.game_events.reset_timer(GameEvents.INPUT_DELAY_END_EVENT)
        self.is_moving_left = False

    def left(self) -> None:
        """
        Presses and releases the left input
        :return: None
        """
        self.left_press()
        self.left_release()

    def right_press(self):
        """
        Handles the right movement when the control is pressed and held down
        :return: None
        """
        self.game.input_move(1, 0)
        if not self.game.game_events.timed:
            return
        self.game.game_events.start_timer(GameEvents.INPUT_DELAY_END_EVENT)
        self.is_moving_right = True

    def right_release(self):
        """
        Handles the right movement when the control is released
        :return: None
        """
        if not self.game.game_events.timed:
            return
        self.game.game_events.reset_timer(GameEvents.INPUT_DELAY_END_EVENT)
        self.is_moving_right = False

    def right(self) -> None:
        """
        Presses and releases the right input
        :return: None
        """
        self.right_press()
        self.right_release()

    def hard_drop(self) -> None:
        """
        Handles a hard drop
        :return: None
        """
        self.game.input_hard_drop()

    def hold(self) -> None:
        """
        Handles a hard drop
        :return: None
        """
        self.game.input_hold()

    def rotate_cw(self) -> None:
        """
        Handles a clockwise rotation
        :return: None
        """
        self.game.input_rotate(1)

    def rotate_ccw(self) -> None:
        """
        Handles a counter-clockwise rotation
        :return: None
        """
        self.game.input_rotate(3)

    def rotate_180(self) -> None:
        """
        Handles a 180-degree rotation
        :return: None
        """
        self.game.input_rotate(2)


class ConsoleGraphics:
    """
    Very basic graphics class that renders the Tetris game in the console.
    """
    HOLD_PIECE_SECTION_WIDTH_DEFAULT: int = 13
    BOARD_SECTION_WIDTH_DEFAULT: int = 35
    NEXT_PIECE_SECTION_WIDTH_DEFAULT: int = 13
    DISPLAY_HEIGHT_DEFAULT: int = 25

    EMPTY_BLOCK = '  '
    FILLED_BLOCK = '██'
    JOIN_STR = '.'

    def __init__(self, game: Game, is_printing: bool = True, is_updating: bool = True,
                 clear_output_handler: Callable = None,
                 hold_piece_section_width: int = HOLD_PIECE_SECTION_WIDTH_DEFAULT,
                 board_section_width: int = BOARD_SECTION_WIDTH_DEFAULT,
                 next_piece_section_width: int = NEXT_PIECE_SECTION_WIDTH_DEFAULT,
                 display_height: int = DISPLAY_HEIGHT_DEFAULT) -> None:
        """
        Initializer
        :param game: the game
        :param is_printing: whether the graphics will automatically print the output
        :param is_updating: whether the graphics will automatically update its display
        :param clear_output_handler: the handler called to clear the previous output, does not do anything if 'None'
        :param hold_piece_section_width: the hold section display width
        :param next_piece_section_width: the next piece section display width
        :param display_height: the height of the display
        """
        self.game: Game = game
        self.is_printing: bool = is_printing
        self.is_updating: bool = is_updating
        self.clear_output_handler: Callable = clear_output_handler
        self.display_height: int = display_height

        self.hold_piece_section_width: int = hold_piece_section_width
        self.board_section_width: int = board_section_width
        self.next_piece_section_width: int = next_piece_section_width

        self.hold_piece_display_lines: list = ['' for _ in range(display_height)]
        self.board_display_lines: list = ['' for _ in range(display_height)]
        self.next_piece_display_lines: list = ['' for _ in range(display_height)]

        self.reset_display_lines(self.hold_piece_display_lines, self.hold_piece_section_width)
        self.reset_display_lines(self.board_display_lines, self.board_section_width)
        self.reset_display_lines(self.next_piece_display_lines, self.next_piece_section_width)

        self._event_handler_ids: tuple = ()
        if self.is_updating:
            self.is_updating = False
            self.set_updating(True)

    def set_updating(self, is_updating: bool = True) -> None:
        """
        Sets whether the graphic displays will constantly update to any changes to the Tetris game
        :param is_updating: whether the graphics is updating
        :return: None
        """
        if self.is_updating == is_updating:
            return
        self.is_updating = is_updating

        if self.is_updating:
            self._event_handler_ids: tuple = (
                self.game.game_events.add_event_handler(GameEvents.GRID_CHANGE_EVENT, self.update_display),
                self.game.game_events.add_event_handler(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT,
                                                        self.update_display),
                self.game.game_events.add_event_handler(GameEvents.HOLD_EVENT, self.update_display)
            )
        else:
            self.game.game_events.remove_event_handler(GameEvents.GRID_CHANGE_EVENT, self._event_handler_ids[0]),
            self.game.game_events.remove_event_handler(GameEvents.TETROMINO_MOVEMENT_CHANGE_EVENT,
                                                       self._event_handler_ids[1]),
            self.game.game_events.remove_event_handler(GameEvents.HOLD_EVENT, self._event_handler_ids[2])
            self._event_handler_ids = ()

    def update_display(self) -> None:
        """
        Updates all the display string list sections for output, and prints if the graphics is set to print
        :return: None
        """
        self._update_board_display_lines()
        self._update_hold_piece_display_lines()
        self._update_next_piece_display_lines()

        if self.is_printing:
            self.print_display()

    @staticmethod
    def grid_to_display_lines(grid: np.ndarray, display_lines: list, index_iterator: Iterator,
                              display_lines_index_map: Callable = lambda x: x) -> None:
        """
        Transcribes a grid of boolean values to display lines to draw blocks
        :param grid: the grid of values
        :param display_lines: the display lines
        :param index_iterator: the index iterator for iterating through the index values of the grid
        :param display_lines_index_map: the map function used for mapping the iterator values to the display line index
        :return: None
        """
        for row_i in index_iterator:
            row = grid[row_i]
            row_str_lst = []
            for block in row:
                if block:
                    row_str_lst.append(ConsoleGraphics.FILLED_BLOCK)
                else:
                    row_str_lst.append(ConsoleGraphics.EMPTY_BLOCK)
            display_lines[display_lines_index_map(row_i)] = ConsoleGraphics.JOIN_STR.join(row_str_lst)

    @staticmethod
    def reset_display_lines(display_lines: list, width: int) -> None:
        """
        Resets the given display lines to be empty
        :param display_lines: the given display lines
        :param width: the width of each display line
        :return: None
        """
        for i in range(len(display_lines)):
            display_lines[i] = ' ' * width

    def _update_board_display_lines(self) -> None:
        """
        Updates the board display section string list
        :return: None
        """
        self.reset_display_lines(self.board_display_lines, self.board_section_width)

        display_grid = self.get_display_grid(top_buffer_rows_included=2)
        self.grid_to_display_lines(display_grid, self.board_display_lines,
                                   range(0, display_grid.shape[0]))

    def _update_hold_piece_display_lines(self) -> None:
        """
        Updates the hold piece display section
        :return: None
        """
        self.reset_display_lines(self.hold_piece_display_lines, self.hold_piece_section_width)

        if not self.game.has_held_once:
            return
        hold_piece = type(self.game.hold_tetromino).ROTATION_STATES[0]

        self.grid_to_display_lines(hold_piece, self.hold_piece_display_lines,
                                   range(hold_piece.shape[0]))

    def _update_next_piece_display_lines(self) -> None:
        """
        Updates the next piece display section
        :return: None
        """
        self.reset_display_lines(self.next_piece_display_lines, self.next_piece_section_width)

        display_row_i = 0
        for next_i in range(self.game.tetromino_queue.next_n):
            next_piece = type(self.game.tetromino_queue.next[next_i]).ROTATION_STATES[0]

            self.grid_to_display_lines(next_piece, self.next_piece_display_lines,
                                       range(next_piece.shape[0]), lambda x: x + display_row_i)
            display_row_i += next_piece.shape[0]
            self.next_piece_display_lines[display_row_i] = ''
            display_row_i += 1

    def get_display_grid(self, top_buffer_rows_included: int = None,
                         include_current_tetromino: bool = True) -> np.ndarray:
        """
        Returns the grid that shows the play field with the tetromino
        :param top_buffer_rows_included: the number of rows in the top buffer to be included in the display grid
        :param include_current_tetromino: whether the current tetromino will be included
        :return: A numpy array of the grid
        """
        if top_buffer_rows_included is None or top_buffer_rows_included > Grid.TOP_BUFFER:
            top_buffer_rows_included = Grid.TOP_BUFFER

        display_grid = self.game.grid.grid.copy()
        if include_current_tetromino:
            x0, x1, y0, y1 = self.game.current_tetromino.box_ranges()
            display_grid[y0:y1, x0:x1] = display_grid[y0:y1, x0:x1] | self.game.current_tetromino.box()

        return display_grid[Grid.TOP_BUFFER - top_buffer_rows_included:-Grid.BORDER_WIDTH,
                            Grid.BORDER_WIDTH:-Grid.BORDER_WIDTH]

    def get_ui_display(self) -> str:
        """
        Returns a string that displays the game UI
        :return: the UI string
        """
        display = '=' * 50 + '\n'
        for line_i in range(self.display_height):
            display += f"{(self.hold_piece_display_lines[line_i]): <{self.hold_piece_section_width}}" + \
                       f"{(self.board_display_lines[line_i]): <{self.board_section_width}}" + \
                       f"{(self.next_piece_display_lines[line_i]): >{self.next_piece_section_width}}" + '\n'

        display += f'score: {self.game.score.score}\n'
        return display

    def print_display(self) -> None:
        """
        Prints the UI display
        :return: None
        """
        if self.clear_output_handler is not None:
            self.clear_output_handler()
        print(self.get_ui_display())
    