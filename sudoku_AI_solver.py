import copy

class Sudoku_AI_solver:
    def __init__(self, board):
        self.board = board  # 9x9 grid
        self.domains = self.initialize_domains()

    def initialize_domains(self):
        domains = {}
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    domains[(i, j)] = set(range(1, 10))
                else:
                    domains[(i, j)] = {self.board[i][j]}
        return domains

    def enforce_node_consistency(self):
        for (i, j), domain in self.domains.items():
            if len(domain) == 1:
                self.domains[(i, j)] = set(domain)

    def revise(self, x, y):
        revised = False
        if len(self.domains[y]) == 1:
            val = next(iter(self.domains[y]))
            if val in self.domains[x] and len(self.domains[x]) > 1:
                self.domains[x].remove(val)
                revised = True
        return revised

    def ac3(self):
        queue = [(x, y) for x in self.domains for y in self.get_neighbors(x)]
        while queue:
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.get_neighbors(x):
                    if z != y:
                        queue.append((z, x))
        return True

    def get_neighbors(self, cell):
        i, j = cell
        neighbors = set()

        # Row and column
        for k in range(9):
            if k != j:
                neighbors.add((i, k))
            if k != i:
                neighbors.add((k, j))

        # Box
        box_i = 3 * (i // 3)
        box_j = 3 * (j // 3)
        for x in range(box_i, box_i + 3):
            for y in range(box_j, box_j + 3):
                if (x, y) != (i, j):
                    neighbors.add((x, y))

        return neighbors

    def assignment_complete(self, assignment):
        return all(len(values) == 1 for values in assignment.values())

    def consistent(self, assignment):
        for cell, val_set in assignment.items():
            if len(val_set) != 1:
                continue
            val = next(iter(val_set))
            for neighbor in self.get_neighbors(cell):
                if val in assignment[neighbor] and len(assignment[neighbor]) == 1:
                    return False
        return True

    def select_unassigned_variable(self, assignment):
        unassigned = [v for v in assignment if len(assignment[v]) > 1]
        if not unassigned:
            return None
        return min(unassigned, key=lambda x: len(assignment[x]))

    def order_domain_values(self, var, assignment):
        def count_constraints(val):
            count = 0
            for neighbor in self.get_neighbors(var):
                if val in assignment[neighbor]:
                    count += 1
            return count
        return sorted(assignment[var], key=count_constraints)

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = copy.deepcopy(assignment)
            new_assignment[var] = {value}

            solver = Sudoku_AI_solver(self.board)
            solver.domains = new_assignment
            if solver.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result:
                    return result
        return None

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        result = self.backtrack(copy.deepcopy(self.domains))
        if result:
            self.domains = result
            self.update_board()
        return self.board

    def update_board(self):
        for (i, j), values in self.domains.items():
            if len(values) == 1:
                self.board[i][j] = next(iter(values))

    def print_board(self):
        for i, row in enumerate(self.board):
            print(" ".join(str(num) if num != 0 else "." for num in row))
            if i % 3 == 2 and i != 8:
                print("-" * 21)

import sys

def load_puzzle_from_file(filename):
    board = []
    with open(filename, "r") as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            if len(row) != 9:
                raise ValueError("Each row must have 9 numbers.")
            board.append(row)
    if len(board) != 9:
        raise ValueError("Puzzle must have 9 rows.")
    return board


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sudoku_AI_solver.py puzzle.txt")
        sys.exit(1)

    puzzle_file = sys.argv[1]
    try:
        puzzle = load_puzzle_from_file(puzzle_file)
    except Exception as e:
        print(f"Error loading puzzle: {e}")
        sys.exit(1)

    solver = Sudoku_AI_solver(puzzle)
    solved = solver.solve()

    if solved:
        print("\nPuzzle solved successfully:\n")
        solver.print_board()
    else:
        print("\nNo solution could be found for this puzzle.")