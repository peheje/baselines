import tabulate
from namedlist import namedlist


# Position = namedlist("Position", ["row", "col"])


class Position:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


class Gridworld:
    def __init__(self, rows, cols):
        self.goal = Position(rows - 10, cols - 1)    
        self.rows = rows
        self.cols = cols
        self.world = None
        self.pos = None
        self.col_wind = None
        self.reset()

    def reset(self):
        self.world = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.pos = Position(0, 0)
        self.world[self.goal.row][self.goal.col] = "G"
        self.world[self.pos.row][self.pos.col] = "S"
        # self.col_wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.__create_wind__()
        self.__wall__()

    def __create_wind__(self):
        self.col_wind = []
        q1 = self.cols // 4
        q2 = self.cols // 2
        q3 = (self.cols // 4) * 3
        for i in range(self.cols):
            if 0 < i <= q1:
                self.col_wind.append(0)
            elif q1 < i <= q2:
                self.col_wind.append(0)
            elif q2 < i <= q3:
                self.col_wind.append(0)
            else:
                self.col_wind.append(0)

    def __wall__(self):
        q1 = self.cols // 4
        q2 = self.cols // 2
        q3 = (self.cols // 4) * 3
        for i in range(0, self.rows-1):
            self.world[i][q2] = "W"

    def wind(self):
        for i in range(self.col_wind[self.pos.col]):
            self.world[self.pos.row][self.pos.col] = "x"
            self.move(Position(-1, 0))
        self.world[self.pos.row][self.pos.col] = "S"
        self.world[self.goal.row][self.goal.col] = "G"

    def move(self, move):
        """ Returns reward = 1 if we moved into goal """
        self.world[self.pos.row][self.pos.col] = "x"
        new_row = self.pos.row + move.row
        new_col = self.pos.col + move.col
        if -1 < new_row < self.rows and -1 < new_col < self.cols:
            if self.world[new_row][new_col] != "W":
                self.pos.row = new_row
                self.pos.col = new_col
        self.world[self.pos.row][self.pos.col] = "S"
        self.world[self.goal.row][self.goal.col] = "G"
        return 1 if self.pos == self.goal else 0

    def __str__(self):
        print(self.col_wind)
        return tabulate.tabulate(self.world, tablefmt="fancy_grid")
