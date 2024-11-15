import heapq

class PuzzleState:
    def __init__(self, board, zero_pos, moves=0):
        self.board = board
        self.zero_pos = zero_pos  # (row, col) of the blank space
        self.moves = moves

    def __lt__(self, other):
        # This is required by heapq to compare PuzzleState objects
        return True  # or any other logic, but here we simply return True to allow heapq to work

    def is_goal(self):
        return self.board == [1, 2, 3, 4, 5, 6, 7, 8, 0]

    def get_neighbors(self):
        # Possible movements: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        x, y = self.zero_pos
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_board = self.board[:]
                # Swap the zero with the neighbor
                new_board[x * 3 + y], new_board[new_x * 3 + new_y] = new_board[new_x * 3 + new_y], new_board[x * 3 + y]
                neighbors.append(PuzzleState(new_board, (new_x, new_y), self.moves + 1))
        
        return neighbors

def manhattan_distance(board):
    distance = 0
    for idx, value in enumerate(board):
        if value != 0:  # Skip the blank space
            target_x = (value - 1) // 3
            target_y = (value - 1) % 3
            current_x = idx // 3
            current_y = idx % 3
            distance += abs(current_x - target_x) + abs(current_y - target_y)
    return distance

def misplaced_tiles(board):
    return sum(1 for idx, value in enumerate(board) if value != 0 and value != idx + 1)

def a_star(start_board, heuristic_func=manhattan_distance):
    start_zero_pos = start_board.index(0)
    start_state = PuzzleState(start_board, (start_zero_pos // 3, start_zero_pos % 3))
    
    open_set = []
    heapq.heappush(open_set, (heuristic_func(start_board), start_state))
    came_from = {}
    g_score = {tuple(start_board): 0}
    f_score = {tuple(start_board): heuristic_func(start_board)}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current.is_goal():
            return current.moves
        
        for neighbor in current.get_neighbors():
            tentative_g_score = g_score[tuple(current.board)] + 1

            if tuple(neighbor.board) not in g_score or tentative_g_score < g_score[tuple(neighbor.board)]:
                came_from[tuple(neighbor.board)] = current
                g_score[tuple(neighbor.board)] = tentative_g_score
                f_score[tuple(neighbor.board)] = tentative_g_score + heuristic_func(neighbor.board)

                if (f_score[tuple(neighbor.board)], neighbor) not in open_set:
                    heapq.heappush(open_set, (f_score[tuple(neighbor.board)], neighbor))
    
    return -1  # Return -1 if no solution is found

# Example usage
if __name__ == "__main__":
    initial_board = [1, 2, 3, 6, 5, 4, 0, 7, 8]  # Example initial state
    print("Using Manhattan Distance:")
    moves = a_star(initial_board, heuristic_func=manhattan_distance)
    if moves != -1:
        print(f"Solution found in {moves} moves.")
    else:
        print("No solution exists.")

    print("\nUsing Misplaced Tiles:")
    moves = a_star(initial_board, heuristic_func=misplaced_tiles)
    if moves != -1:
        print(f"Solution found in {moves} moves.")
    else:
        print("No solution exists.")