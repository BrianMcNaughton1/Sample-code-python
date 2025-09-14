import heapq  # Import the heapq module for implementing a priority queue (min-heap)
import time   # Import the time module for measuring execution time and delays

# Constants representing different elements in the Sokoban grid
EMPTY = 0        # Empty space
WALL = 1         # Wall
BOX = 2          # Box
KEEPER = 3       # Keeper (player)
GOAL = 4         # Goal
BOX_ON_GOAL = 5  # Box on goal
KEEPER_ON_GOAL = 6  # Keeper on goal

# Possible directions the keeper can move, represented as (delta_row, delta_column)
directions = {
    'up': (-1, 0),     # Move up by decreasing the row index
    'down': (1, 0),    # Move down by increasing the row index
    'left': (0, -1),   # Move left by decreasing the column index
    'right': (0, 1)    # Move right by increasing the column index
}

class State:
    """
    Class representing the state of the Sokoban game.
    """
    def __init__(self, grid):
        # Initialize the State with a grid representing the game board
        # The grid is a tuple of tuples, where each inner tuple represents a row
        self.grid = grid

    def __hash__(self):
        # Define the hash function for the State, allowing it to be used in sets and as dictionary keys
        # Hashing is based on the grid (which is a tuple of tuples, so it's hashable)
        return hash(self.grid)

    def __eq__(self, other):
        # Define equality comparison between two State objects
        # Two states are equal if their grids are equal
        return self.grid == other.grid

    def __str__(self):
        # Return a string representation of the state for visualization
        # Converts each cell in the grid to its corresponding character
        return '\n'.join(''.join(self.char_map[cell] for cell in row) for row in self.grid)

    # Mapping from cell values to characters for visualization
    char_map = {
        EMPTY: ' ',            # Empty space
        WALL: '#',             # Wall
        BOX: '$',              # Box
        KEEPER: '@',           # Keeper (player)
        GOAL: '.',             # Goal
        BOX_ON_GOAL: '*',      # Box on goal
        KEEPER_ON_GOAL: '+',   # Keeper on goal
    }

    @staticmethod
    def from_list_of_lists(grid_list):
        # Static method to create a State from a list of lists (e.g., from a 2D array)
        # Converts the list of lists to a tuple of tuples for immutability and hashability
        return State(tuple(tuple(row) for row in grid_list))

def goal_p(state):
    """
    Goal test function: returns True if the state is a goal state (all boxes are on goals).
    """
    for row in state.grid:
        for cell in row:
            if cell == BOX:
                # If any cell contains a box not on a goal, it's not a goal state
                return False
    # If no boxes are found (all boxes are on goals), return True
    return True

def find_keeper(state):
    """
    Function to find the position of the keeper (player) in the grid.
    """
    for i, row in enumerate(state.grid):
        for j, cell in enumerate(row):
            if cell == KEEPER or cell == KEEPER_ON_GOAL:
                # Return the coordinates (row index, column index) of the keeper
                return (i, j)
    return None  # Should not happen in valid states

def try_move(state, keeper_pos, move):
    """
    Function to attempt to move the keeper in a specified direction.
    Returns a new State if the move is valid, or None if the move is invalid.
    """
    i, j = keeper_pos            # Current position of the keeper
    di, dj = move                # Direction to move
    grid = [list(row) for row in state.grid]  # Create a mutable copy of the grid

    rows = len(grid)             # Number of rows in the grid
    cols = len(grid[0])          # Number of columns in the grid

    i1, j1 = i + di, j + dj      # New position the keeper wants to move into

    # Check if the new position is within bounds
    if i1 < 0 or i1 >= rows or j1 < 0 or j1 >= cols:
        return None  # Move is invalid (out of bounds)

    cell1 = grid[i1][j1]         # Content of the cell the keeper wants to move into

    if cell1 == WALL:
        # Can't move into a wall
        return None

    if cell1 == EMPTY or cell1 == GOAL:
        # If the cell is empty or a goal, move the keeper into it
        cell_keeper_from = grid[i][j]  # Current cell where the keeper is

        # Update the cell the keeper is moving into
        if cell1 == EMPTY:
            grid[i1][j1] = KEEPER
        elif cell1 == GOAL:
            grid[i1][j1] = KEEPER_ON_GOAL

        # Update the cell the keeper is moving from
        if cell_keeper_from == KEEPER:
            grid[i][j] = EMPTY
        elif cell_keeper_from == KEEPER_ON_GOAL:
            grid[i][j] = GOAL

        # Create a new State with the updated grid
        new_state = State(tuple(tuple(row) for row in grid))
        return new_state

    if cell1 == BOX or cell1 == BOX_ON_GOAL:
        # If the cell contains a box or a box on a goal, attempt to push the box
        i2, j2 = i1 + di, j1 + dj  # Position beyond the box

        # Check if the position beyond the box is within bounds
        if i2 < 0 or i2 >= rows or j2 < 0 or j2 >= cols:
            return None  # Can't push box out of bounds

        cell2 = grid[i2][j2]       # Content of the cell beyond the box

        if cell2 == EMPTY or cell2 == GOAL:
            # If the cell beyond the box is empty or a goal, we can push the box

            # Move the box into the cell beyond
            if cell2 == EMPTY:
                grid[i2][j2] = BOX
            elif cell2 == GOAL:
                grid[i2][j2] = BOX_ON_GOAL

            # Move the keeper into the box's former position
            if cell1 == BOX:
                grid[i1][j1] = KEEPER
            elif cell1 == BOX_ON_GOAL:
                grid[i1][j1] = KEEPER_ON_GOAL

            cell_keeper_from = grid[i][j]  # Current cell where the keeper is

            # Update the cell the keeper is moving from
            if cell_keeper_from == KEEPER:
                grid[i][j] = EMPTY
            elif cell_keeper_from == KEEPER_ON_GOAL:
                grid[i][j] = GOAL

            # Create a new State with the updated grid
            new_state = State(tuple(tuple(row) for row in grid))
            return new_state

    # If none of the above conditions are met, the move is invalid
    return None

def successors(state):
    """
    Function to generate all valid successor states from the current state.
    """
    keeper_pos = find_keeper(state)  # Find the current position of the keeper
    successors = []
    for dir_name, (di, dj) in directions.items():
        # Attempt to move the keeper in each possible direction
        new_state = try_move(state, keeper_pos, (di, dj))
        if new_state:
            # If the move is valid and results in a new state, add it to the list of successors
            successors.append(new_state)
    return successors  # Return the list of successor states

def cost_fn(state1, state2):
    """
    Cost function between two states; in this case, each move has a uniform cost of 1.
    """
    return 1  # Uniform cost for each move

def get_positions(state, cell_values):
    """
    Function to find all positions of cells with specified values in the grid.
    """
    positions = []
    for i, row in enumerate(state.grid):
        for j, cell in enumerate(row):
            if cell in cell_values:
                # If the cell's value is in the list of specified cell values, add its position
                positions.append((i, j))
    return positions  # Return the list of positions

def remaining_cost_fn(state):
    """
    Heuristic function estimating the remaining cost to reach the goal from the current state.
    Sum of the minimum distances from each box to any goal.
    """
    box_positions = get_positions(state, [BOX])  # Positions of boxes not on goals
    goal_positions = get_positions(state, [GOAL, KEEPER_ON_GOAL])  # Positions of goals

    if not box_positions or not goal_positions:
        # If there are no boxes or no goals, the remaining cost is zero
        return 0

    total_distance = 0
    for box in box_positions:
        # For each box, find the minimum Manhattan distance to any goal
        min_distance = min(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in goal_positions)
        total_distance += min_distance  # Add the minimum distance to the total

    return total_distance  # Return the total heuristic cost

def print_state(state):
    """
    Function to print the state grid in a human-readable format.
    """
    for row in state.grid:
        # Convert each cell in the row to its corresponding character and join them into a string
        print(''.join(State.char_map[cell] for cell in row))
    print()  # Print a blank line for separation

def astar(start_state, goal_p, successors, cost_fn, remaining_cost_fn):
    """
    A* search algorithm to find the shortest path from start_state to a goal state.
    """
    open_list = []  # Priority queue (min-heap) of states to explore, ordered by estimated total cost
    closed_set = set()  # Set of states already evaluated
    came_from = {}  # Dictionary mapping a state to its parent state in the search path
    counter = 0     # Counter to act as a tie-breaker in the priority queue when costs are equal

    estimated_cost = remaining_cost_fn(start_state)  # Heuristic estimate for the start state
    # Push the start state onto the open list with its estimated total cost
    heapq.heappush(open_list, (estimated_cost, 0, counter, start_state, None))
    counter += 1  # Increment counter

    while open_list:
        # Pop the state with the lowest estimated total cost from the open list
        total_estimated_cost, cost_so_far, _, current_state, parent = heapq.heappop(open_list)

        if current_state in closed_set:
            # If we've already evaluated this state, skip it
            continue

        closed_set.add(current_state)  # Mark the current state as evaluated
        came_from[current_state] = parent  # Record how we reached this state

        if goal_p(current_state):
            # If the current state is a goal state, reconstruct the path to it
            path = []
            state = current_state
            while state:
                path.append(state)  # Add the state to the path
                state = came_from[state]  # Move to the parent state
            path.reverse()  # Reverse the path to start from the initial state
            return path  # Return the solution path

        for neighbor in successors(current_state):
            if neighbor in closed_set:
                # If we've already evaluated this neighbor, skip it
                continue
            new_cost = cost_so_far + cost_fn(current_state, neighbor)  # Cost to reach neighbor
            estimated_total_cost = new_cost + remaining_cost_fn(neighbor)  # Estimated total cost
            # Push the neighbor onto the open list with its estimated total cost
            heapq.heappush(open_list, (estimated_total_cost, new_cost, counter, neighbor, current_state))
            counter += 1  # Increment counter

    # If the open list is empty and no solution was found, return None
    return None

# Define the initial puzzle grid
p1 = [
    [0, 0, 1, 1, 1, 1, 0, 0, 0],   # Row 0
    [1, 1, 1, 0, 0, 1, 1, 1, 1],   # Row 1
    [1, 0, 0, 0, 0, 0, 2, 0, 1],   # Row 2
    [1, 0, 1, 0, 0, 1, 2, 0, 1],   # Row 3
    [1, 0, 4, 0, 4, 1, 3, 0, 1],   # Row 4
    [1, 1, 1, 1, 1, 1, 1, 1, 1]    # Row 5
]

# Create the start state from the puzzle grid
start_state = State.from_list_of_lists(p1)

# Run the A* algorithm to find the solution
start_time = time.time()  # Record the start time
solution = astar(start_state, goal_p, successors, cost_fn, remaining_cost_fn)
end_time = time.time()    # Record the end time

# Display the solution
if solution:
    # If a solution was found, display it
    print(f"Solution found in {len(solution) - 1} moves.")  # Number of moves is one less than the number of states
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    for state in solution:
        print_state(state)  # Print each state in the solution path
        time.sleep(0.2)     # Pause briefly to visualize the moves
else:
    # If no solution was found, inform the user
    print("No solution found.")
