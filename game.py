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

    def eval(self):
        return random()

    def alphabeta(self, n, a, b):
        # n % 2 = 0 -> max
        # n % 2 = 1 -> min
        # self.render()

        if n == 3 or self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):
            return self.eval()

        g = -np.inf if n % 2 else np.inf
        best_move, best_g = None, g

        for move in self.possible_moves():
            self.place(move, HexBoard.BLUE if n % 2 else HexBoard.RED)
            g = (max if n % 2 else min)(g, self.alphabeta(n + 1, a, b))
            self.place(move, HexBoard.EMPTY)
            if n % 2:
                a = max(a, g)
                if g > best_g:
                    best_move, best_g = move, g
                if g >= b:
                    break
            else:
                b = min(b, g)
                if g < best_g:
                    best_move, best_g = move, g
                if a >= g:
                    break

        if n == 0:
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