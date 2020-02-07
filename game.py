import cv2 as cv
import numpy as np
from math import sin, radians


class HexBoard:
    BLUE = 1
    RED = 2
    EMPTY = 3

    def __init__(self, board_size):
        self.board = {}
        self.size = board_size
        self.game_over = False
        for x in range(board_size):
            for y in range(board_size):
                self.board[x, y] = HexBoard.EMPTY

    def is_game_over(self):
        return self.game_over

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
        if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
            self.board[coordinates] = color
            if self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):
                self.game_over = True

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

    def print(self):
        print("   ", end="")
        for y in range(self.size):
            print(chr(y + ord('a')), "", end="")
        print("")
        print(" -----------------------")
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
        print("   -----------------------")

    def render(self):
        # calculate all relevant lengths
        hex_long = int(round(
            500 / (self.size * 3 - 1)
        ))
        hex_short = int(round(
            hex_long * sin(radians(30)) / sin(radians(60))
        ))
        hex_diag = int(round(
            hex_long * sin(radians(90)) / sin(radians(60))
        ))

        # create canvas
        canvas = np.ones((hex_diag * self.size + hex_short * (self.size + 1), hex_long * (self.size * 3 - 1), 3), np.uint8) * 255

        # render hexes
        for i in range(self.size):
            for j in range(self.size):
                h = i * (hex_diag + hex_short)
                w = i * hex_long + j * hex_long * 2

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
                cv.polylines(canvas, [points], True, (0, 0, 0), 4)
                cv.putText(
                    canvas,
                    str(i) + chr(ord('a') + j),
                    (w + int(hex_long / 1.75), h + hex_short + int(hex_diag / 1.5)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                )

        # show canvas
        cv.imshow('HEX', canvas)
        cv.waitKey(1000)
