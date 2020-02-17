import cv2 as cv
import numpy as np
from math import sin, radians
from random import random


class HexBoard:
    BLUE = 1
    RED = 2
    EMPTY = 3

    def __init__(self, board_size):
        self.board = {}
        self.size = board_size
        for x in range(board_size):
            for y in range(board_size):
                self.board[x, y] = HexBoard.EMPTY

    def is_empty(self, coordinates):
        return self.board[coordinates] == HexBoard.EMPTY

    def is_color(self, coordinates, color):
        return self.board[coordinates] == color

    def get_color(self, coordinates):
        if coordinates == (-1, -1):
            return HexBoard.EMPTY
        return self.board[coordinates]

    def exists(self, coordinates):
        for coordinate in coordinates:
            if coordinate < 0 or coordinate > self.size - 1:
                return False
        return True

    def place(self, coordinates, color):
        self.board[coordinates] = color

    @staticmethod
    def get_opposite_color(current_color):
        if current_color == HexBoard.BLUE:
            return HexBoard.RED
        return HexBoard.BLUE

    def get_neighbors(self, coordinates):
        (cx, cy) = coordinates
        neighbors = []
        if cx - 1 >= 0:
            neighbors.append((cx - 1, cy))
        if cx + 1 < self.size:
            neighbors.append((cx + 1, cy))
        if cx - 1 >= 0 and cy + 1 <= self.size - 1:
            neighbors.append((cx - 1, cy + 1))
        if cx + 1 < self.size and cy - 1 >= 0:
            neighbors.append((cx + 1, cy - 1))
        if cy + 1 < self.size:
            neighbors.append((cx, cy + 1))
        if cy - 1 >= 0:
            neighbors.append((cx, cy - 1))
        return neighbors

    def border(self, color, move):
        (nx, ny) = move
        return (color == HexBoard.BLUE and nx == self.size - 1) or (color == HexBoard.RED and ny == self.size - 1)

    def traverse(self, color, move, visited):
        if not self.is_color(move, color) or (move in visited and visited[move]):
            return False
        if self.border(color, move):
            return True
        visited[move] = True
        for n in self.get_neighbors(move):
            if self.traverse(color, n, visited):
                return True
        return False

    def check_win(self, color):
        for i in range(self.size):
            if color == HexBoard.BLUE:
                move = (0, i)
            else:
                move = (i, 0)
            if self.traverse(color, move, {}):
                return True
        return False

    def possible_moves(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.is_empty((i, j)):
                    yield i, j

    def get_snake(self, color):
        for i in range(self.size):
            for j in range(self.size):
                if self.is_color((i, j), color):
                    yield i, j

    def eval(self):
        return random()

    def weight(self, coords, color):
        """
        :param coords: Coordinates of the hex to travel too.
        :param color:  The color of the player
        :return: A weight (int)

        If color == Red than we would like to go from top to bottom.
        So going below should be rewarded, so the weight is 'small'

        Weight is the negative value of x or y coordinate (depends on color).
        If the color is already your own color than the weight = weight - 0.5

        This give lower weight to hexes going down and even lower for hexes going down with the current color.
        """
        direction = 1 if color == HexBoard.RED else 0
        weight = -coords[direction]
        if self.is_color(coords, color):
            weight -= 0.5
        return weight

    def walk_path(self, color, current_state):
        """
        :param color: The color of the player
        :return: the length of the shortest path which is a list of coordinates

        This function simulates the dijkstra algorithm.
        At each hex looks at each neigbour and it travels to the neighbor which is the closest.

        Function starts at the top hexes if color = Red and the most left hexes if color = Blue.
        At each start hex it tries to find the shortest path to the Bottom (red) or Right (blue).

        Of all paths take the shortest one and return the length of that path.

        """
        paths = []
        score = []
        start = [(x, 0) for x in range(self.size)] if color == HexBoard.RED else [(0, y) for y in range(self.size)]
        end = [(x, self.size-1) for x in range(self.size)] if color == HexBoard.RED else [(self.size-1, y) for y in range(self.size)]

        # Filter out the impossible start positions
        start = [coords for coords in start if self.is_color(coords, color) or self.is_empty(coords)]

        for coords in start:
            placed = []
            if self.is_empty(coords):
                placed.append(coords)
                self.place(coords, color)
                # self.render()
            current_path = []
            current_score = []
            current_score.append(self.weight(coords, color))
            current_path.append(coords)
            while current_path[-1] not in end and [n for n in self.get_neighbors(current_path[-1]) if
                                                   self.is_empty(n)] != []:
                neighbor, s = self.best_neighbor(current_path[-1], color, current_path)
                self.place(neighbor, color)
                placed.append(neighbor)
                # self.render()
                current_score.append(s)
                current_path.append(neighbor)
            paths.append(current_path)

            if current_path[-1] in end:
                score.append(sum(current_score))
            else:
                # Paths with no ends in them are not good
                score.append(np.inf)
            for step in placed:
                if step not in current_state:
                    self.place(step, HexBoard.EMPTY)
                    # self.render()
        # Remove already filled coordinates
        return len([coords for coords in paths[score.index(min(score))] if coords not in current_state])

    def best_neighbor(self, hex, color, current_path):
        """
        :param hex: Hex that searches neighbours
        :param color: Color of hex
        :param current_path: The current path that has been traversed
        :return: Return the coordinates of the neighbour with the minimal weight which is not in the current_path

        Finds the neighbour with the smallest weight.
        """
        neighbors = [n for n in self.get_neighbors(hex) if (self.is_empty(n) or self.is_color(n, color)) and n not in current_path]
        neighbors_dist = [self.weight(coords, color) for coords in neighbors]
        min_dist = min(neighbors_dist)
        return neighbors[neighbors_dist.index(min_dist)], min_dist

    def dijkstra_eval(self):
        current_state = [coords for coords, color in self.board.items() if color != HexBoard.EMPTY]
        red_short = self.walk_path(HexBoard.RED, current_state)
        blue_short = self.walk_path(HexBoard.BLUE, current_state)
        return red_short - blue_short

    def alphabeta(self, depth, lower, upper):
        if self.check_win(HexBoard.RED):
            return 999 - depth
        if self.check_win(HexBoard.BLUE):
            return -999 + depth
        if depth == 5:
            return self.dijkstra_eval()

        g = np.inf if depth % 2 else -np.inf
        best_move, best_g = None, g

        for move in self.possible_moves():
            self.place(move, HexBoard.BLUE if depth % 2 else HexBoard.RED)
            # self.render()
            g = (min if depth % 2 else max)(g, self.alphabeta(depth + 1, lower, upper))
            self.place(move, HexBoard.EMPTY)
            # self.render()
            if depth % 2:
                upper = min(upper, g)
                if g < best_g:
                    best_move, best_g = move, g
                if lower >= g:
                    break
            else:
                lower = max(lower, g)
                if g > best_g:
                    best_move, best_g = move, g
                if g >= upper:
                    break

        if depth == 0:
            return best_move
        else:
            return g

    def print(self):
        print("   ", end="")
        for y in range(self.size):
            print(chr(y + ord('a')), "", end="")
        print("")
        print("-----------------------")
        for y in range(self.size):
            print(y, "|", end="")
            for z in range(y):
                print(" ", end="")
            for x in range(self.size):
                piece = self.board[x, y]
                if piece == HexBoard.BLUE:
                    print("b ", end="")
                elif piece == HexBoard.RED:
                    print("r ", end="")
                else:
                    if x == self.size:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("-----------------------")

    def render(self):
        # calculate all relevant lengths
        hex_long = int(round(
            2000 / (self.size * 3 - 1)
        ))
        hex_short = int(round(
            hex_long * sin(radians(30)) / sin(radians(60))
        ))
        hex_diag = int(round(
            hex_long * sin(radians(90)) / sin(radians(60))
        ))

        # create canvas
        canvas = cv.imread('background.jpg')
        canvas = cv.resize(canvas, (
            hex_long * (self.size * 3 - 1) + 12,
            hex_diag * self.size + hex_short * (self.size + 1) + 12,
        ))

        # render hexes
        for i in range(self.size):
            for j in range(self.size):
                h = i * (hex_diag + hex_short) + 6
                w = i * hex_long + j * hex_long * 2 + 6

                color = (255, 255, 255)
                if self.is_color((j, i), HexBoard.RED):
                    color = (0, 0, 255)
                if self.is_color((j, i), HexBoard.BLUE):
                    color = (255, 0, 0)

                points = np.array((
                    (w, h + hex_short),
                    (w + hex_long, h),
                    (w + hex_long * 2, h + hex_short),
                    (w + hex_long * 2, h + hex_short + hex_diag),
                    (w + hex_long, h + hex_diag + hex_short * 2),
                    (w, h + hex_short + hex_diag),
                ))

                cv.fillPoly(canvas, [points], color)
                cv.polylines(canvas, [points], True, (0, 0, 0), 12)
                cv.putText(
                    canvas,
                    str(i) + chr(ord('a') + j),
                    (w + int(hex_long / 1.75), h + hex_short + int(hex_diag / 1.5)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    16 / self.size,
                    (0, 0, 0),
                    16,
                )
                cv.putText(
                    canvas,
                    str(i) + chr(ord('a') + j),
                    (w + int(hex_long / 1.75), h + hex_short + int(hex_diag / 1.5)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    16 / self.size,
                    (255, 255, 255),
                    4,
                )

        # render borders
        points = [(6, hex_short + 6)]
        for i in range(self.size):
            points.append((i * hex_long * 2 + 6 + hex_long, 6))
            points.append(((i + 1) * hex_long * 2 + 6, hex_short + 6))
        points = np.array(points)

        cv.polylines(canvas, [points], False, (0, 0, 255), 12)
        points[:, 0] = canvas.shape[1] - points[:, 0]
        points[:, 1] = canvas.shape[0] - points[:, 1]
        cv.polylines(canvas, [points], False, (0, 0, 255), 12)

        points = []
        for i in range(self.size):
            points.append((6 + i * hex_long, 6 + hex_short + (hex_short + hex_diag) * i))
            points.append((6 + i * hex_long, 6 + hex_short + hex_diag + (hex_short + hex_diag) * i))
        points = np.array(points)

        cv.polylines(canvas, [points], False, (255, 0, 0), 12)
        points[:, 0] = canvas.shape[1] - points[:, 0]
        points[:, 1] = canvas.shape[0] - points[:, 1]
        cv.polylines(canvas, [points], False, (255, 0, 0), 12)

        # apply anti aliasing
        canvas = cv.resize(src=canvas, dsize=None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)

        # show canvas
        cv.imshow('HEX', canvas)
        cv.waitKey(1000)
