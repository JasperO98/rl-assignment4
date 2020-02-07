from hex_skeleton import HexBoard

if __name__ == '__main__':
    colour = HexBoard.RED
    board = HexBoard(4)
    print('Red Starts')

    while not board.game_over:
        board.print()

        row = int(input("Row: "))
        column = ord(input("Column: ")) - ord('a')
        while not board.exists((column, row)) or not board.is_empty((column, row)):
            print('Invalid Move')
            row = int(input("Rij: "))
            column = ord(input("Column: ")) - ord('a')

        board.place((column, row), colour)
        colour = board.get_opposite_color(colour)

    board.print()
    if board.check_win(HexBoard.RED):
        print('Red Won')
    if board.check_win(HexBoard.BLUE):
        print('Blue Won')
