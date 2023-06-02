import numpy as np
from socketIO_client import SocketIO

server_url = "http://192.168.56.1"
server_port = 4000

socketIO = SocketIO(server_url, server_port)

transposition_table = {}  # Transposition table to store evaluated positions

def on_connect():
    print("Connected to server")
    socketIO.emit('signin', {
        'user_name': "Ale",
        'tournament_id': 5555,
        'user_role': 'player'
    })

def on_ok_signin():
    print("Login")

def on_finish(data):
    game_id = data['game_id']
    player_turn_id = data['player_turn_id']
    winner_turn_id = data.get('winner_turn_id')  # Use get() method to retrieve the value safely
    board = data['board']
    
    # Your cleaning board logic here
    
    if winner_turn_id is not None:
        print("Winner:", winner_turn_id)
    else:
        print("It's a tie!")
    print(board)
    socketIO.emit('player_ready', {
        'tournament_id': 5555,
        'player_turn_id': player_turn_id,
        'game_id': game_id
    })
    
def evaluate_board(board, player_id):
    # Check if the board position is already evaluated in the transposition table
    key = tuple(map(tuple, board))
    if key in transposition_table:
        return transposition_table[key]
    
    # Define heuristic weights for different features
    weights = {
        'open_threes': 100,
        'half_threes': 10,
        'open_twos': 5,
        'half_twos': 1,
        'three_in_row': 50,  # New heuristic for three in a row
        'four_in_row': 1000,  # New heuristic for four in a row
        'open_fours': 10000,  # New heuristic for open fours
        'half_fours': 1000     # New heuristic for half fours
    }
    
    # Calculate the heuristic score
    score = 0
    
    # Evaluate open threes
    open_threes = np.sum(board[:, :-3] == player_id) + np.sum(board[:, 3:] == player_id)
    score += weights['open_threes'] * open_threes
    
    # Evaluate half threes
    half_threes = np.sum(board[:, 1:-3] == player_id) + np.sum(board[:, 3:-1] == player_id)
    score += weights['half_threes'] * half_threes
    
    # Evaluate open twos
    open_twos = np.sum(board[:, :-2] == player_id) + np.sum(board[:, 2:] == player_id)
    score += weights['open_twos'] * open_twos
    
    # Evaluate half twos
    half_twos = np.sum(board[:, 1:-2] == player_id) + np.sum(board[:, 2:-1] == player_id)
    score += weights['half_twos'] * half_twos
    
    # Evaluate three in a row
    three_in_row = np.sum(board[:, :-2] == player_id) + np.sum(board[:, 2:] == player_id)
    score += weights['three_in_row'] * three_in_row
    
    # Evaluate four in a row
    four_in_row = np.sum(board[:, :-3] == player_id) + np.sum(board[:, 3:] == player_id)
    score += weights['four_in_row'] * four_in_row
    
    # Evaluate open fours
    open_fours = np.sum(board[:, :-4] == player_id) + np.sum(board[:, 4:] == player_id)
    score += weights['open_fours'] * open_fours
    
    # Evaluate half fours
    half_fours = np.sum(board[:, 1:-4] == player_id) + np.sum(board[:, 4:-1] == player_id)
    score += weights['half_fours'] * half_fours
    
    # Store the evaluated position in the transposition table
    transposition_table[key] = score
    
    return score

def extends_vertical_three(board, move, player_id):
    # Check if the move extends a three-in-a-row vertically
    col = move
    row = np.argmax(board[:, col] == 0)
    board[row, col] = player_id
    
    extended = False
    
    # Check for an extension downwards
    if row > 2:
        if np.all(board[row-3:row, col] == player_id):
            extended = True
    
    # Check for an extension upwards
    if row < board.shape[0] - 3:
        if np.all(board[row+1:row+4, col] == player_id):
            extended = True
    
    board[row, col] = 0
    
    return extended

def extends_horizontal_four(board, move, player_id):
    # Check if the move extends an open four horizontally
    row = np.argmax(board[:, move] == 0)
    board[row, move] = player_id
    
    extended = False
    
    # Check for an extension to the left
    if move > 3:
        if np.all(board[row, move-4:move] == player_id):
            extended = True
    
    # Check for an extension to the right
    if move < board.shape[1] - 4:
        if np.all(board[row, move+1:move+5] == player_id):
            extended = True
    
    board[row, move] = 0
    
    return extended

def extends_vertical_four(board, move, player_id):
    # Check if the move extends an open four vertically
    col = move
    row = np.argmax(board[:, col] == 0)
    board[row, col] = player_id
    
    extended = False
    
    # Check for an extension downwards
    if row > 3:
        if np.all(board[row-4:row, col] == player_id):
            extended = True
    
    # Check for an extension upwards
    if row < board.shape[0] - 4:
        if np.all(board[row+1:row+5, col] == player_id):
            extended = True
    
    board[row, col] = 0
    
    return extended

def extends_horizontal_half_four(board, move, player_id):
    # Check if the move extends a half four horizontally
    row = np.argmax(board[:, move] == 0)
    board[row, move] = player_id
    
    extended = False
    
    # Check for an extension to the left
    if move > 3:
        if np.all(board[row, move-4:move-1] == player_id) and np.all(board[row, move+1:move+2] == 0):
            extended = True
    
    # Check for an extension to the right
    if move < board.shape[1] - 4:
        if np.all(board[row, move-1:move] == 0) and np.all(board[row, move+1:move+5] == player_id):
            extended = True
    
    board[row, move] = 0
    
    return extended

def extends_vertical_half_four(board, move, player_id):
    # Check if the move extends a half four vertically
    col = move
    row = np.argmax(board[:, col] == 0)
    board[row, col] = player_id
    
    extended = False
    
    # Check for an extension downwards
    if row > 3:
        if np.all(board[row-4:row-1, col] == player_id) and np.all(board[row+1:row+2, col] == 0):
            extended = True
    
    # Check for an extension upwards
    if row < board.shape[0] - 4:
        if np.all(board[row-1:row, col] == 0) and np.all(board[row+1:row+5, col] == player_id):
            extended = True
    
    board[row, col] = 0
    
    return extended

def order_moves(board, player_id, moves):
    # Sort the moves based on their potential impact
    sorted_moves = []
    for move in moves:
        # Make a copy of the board and apply the move
        new_board = board.copy()
        new_board = make_move(new_board, move, player_id)

        # Evaluate the board after the move
        score = evaluate_board(new_board, player_id)

        # Check if the move creates a win, extends a three-in-a-row, open four, or half four
        if is_winning_move(new_board, player_id) or extends_horizontal_three(new_board, move, player_id) or extends_vertical_three(new_board, move, player_id):
            score += 1000
        elif extends_horizontal_four(new_board, move, player_id) or extends_vertical_four(new_board, move, player_id):
            score += 10000
        elif extends_horizontal_half_four(new_board, move, player_id) or extends_vertical_half_four(new_board, move, player_id):
            score += 1000

        sorted_moves.append((move, score))

    # Sort the moves in descending order of score
    sorted_moves.sort(key=lambda x: x[1], reverse=True)

    # Extract the moves from the sorted list
    ordered_moves = [move for move, _ in sorted_moves]

    return ordered_moves

def minimax(board, player_id, depth, alpha, beta, maximizing_player):
    # Base cases: leaf node or depth limit reached
    if depth == 0 or is_game_over(board):
        return None, evaluate_board(board, player_id)
    
    # Determine the opponent's player ID
    opponent_id = 1 if player_id == 2 else 2
    
    if maximizing_player:
        best_score = float('-inf')
        best_move = None
        
        # Get the valid moves and order them based on their potential impact
        moves = get_valid_moves(board)
        ordered_moves = order_moves(board, player_id, moves)
        
        for move in ordered_moves:
            # Make a copy of the board and apply the move
            new_board = board.copy()
            new_board = make_move(new_board, move, player_id)
            
            # Recursively evaluate the child node
            _, score = minimax(new_board, player_id, depth - 1, alpha, beta, False)
            
            # Update the best score and move
            if score > best_score:
                best_score = score
                best_move = move
            
            # Update alpha
            alpha = max(alpha, best_score)
            
            # Alpha-beta pruning
            if alpha >= beta:
                break
        
        return best_move, best_score
    
    else:
        best_score = float('inf')
        best_move = None
        
        # Get the valid moves and order them based on their potential impact
        moves = get_valid_moves(board)
        ordered_moves = order_moves(board, opponent_id, moves)
        
        for move in ordered_moves:
            # Make a copy of the board and apply the move
            new_board = board.copy()
            new_board = make_move(new_board, move, opponent_id)
            
            # Recursively evaluate the child node
            _, score = minimax(new_board, player_id, depth - 1, alpha, beta, True)
            
            # Update the best score and move
            if score < best_score:
                best_score = score
                best_move = move
            
            # Update beta
            beta = min(beta, best_score)
            
            # Alpha-beta pruning
            if alpha >= beta:
                break
        
        return best_move, best_score

def is_game_over(board):
    # Check for a win in rows
    for row in board:
        for col in range(len(row) - 3):
            if np.all(row[col:col+4] == 1) or np.all(row[col:col+4] == 2):
                return True
    
    # Check for a win in columns
    for col in range(board.shape[1]):
        for row in range(len(board) - 3):
            if np.all(board[row:row+4, col] == 1) or np.all(board[row:row+4, col] == 2):
                return True
    
    # Check for a win in diagonals (positive slope)
    for row in range(len(board) - 3):
        for col in range(board.shape[1] - 3):
            if np.all(np.diag(board[row:row+4, col:col+4]) == 1) or np.all(np.diag(board[row:row+4, col:col+4]) == 2):
                return True
    
    # Check for a win in diagonals (negative slope)
    for row in range(len(board) - 3):
        for col in range(board.shape[1] - 3):
            if np.all(np.diag(np.fliplr(board[row:row+4, col:col+4])) == 1) or np.all(np.diag(np.fliplr(board[row:row+4, col:col+4])) == 2):
                return True
    
    # Check for a tie
    if np.count_nonzero(board == 0) == 0:
        return True
    
    return False

def is_winning_move(board, player_id):
    # Check for a win in rows
    for row in board:
        for col in range(len(row) - 3):
            if np.all(row[col:col+4] == player_id):
                return True
    
    # Check for a win in columns
    for col in range(board.shape[1]):
        for row in range(len(board) - 3):
            if np.all(board[row:row+4, col] == player_id):
                return True
    
    # Check for a win in diagonals (positive slope)
    for row in range(len(board) - 3):
        for col in range(board.shape[1] - 3):
            if np.all(np.diag(board[row:row+4, col:col+4]) == player_id):
                return True
    
    # Check for a win in diagonals (negative slope)
    for row in range(len(board) - 3):
        for col in range(board.shape[1] - 3):
            if np.all(np.diag(np.fliplr(board[row:row+4, col:col+4])) == player_id):
                return True
    
    return False

def extends_horizontal_three(board, move, player_id):
    # Check if the move extends a three-in-a-row horizontally
    row = np.argmax(board[:, move] == 0)
    board[row, move] = player_id
    
    extended = False
    
    # Check for an extension to the left
    if move > 2:
        if np.all(board[row, move-3:move] == player_id):
            extended = True
    
    # Check for an extension to the right
    if move < board.shape[1] - 3:
        if np.all(board[row, move+1:move+4] == player_id):
            extended = True
    
    board[row, move] = 0
    
    return extended

def get_valid_moves(board):
    return np.where(board[0] == 0)[0].tolist()

def make_move(board, move, player_id):
    for row in reversed(range(len(board))):
        if board[row, move] == 0:
            board[row, move] = player_id
            break
    
    return board

def on_ready(data):
    game_id = data['game_id']
    player_turn_id = data['player_turn_id']
    board = np.array(data['board'])
    print("I'm player:", player_turn_id)
    print(board)
    
    # Define the depth of the search tree
    depth = 5
    
    # Clear the transposition table for a new game
    transposition_table.clear()
    
    if np.count_nonzero(board) == 0:
        # If the board is empty, play in the center
        best_move = board.shape[1] // 2
    else:
        # Find the best move using the minimax algorithm with alpha-beta pruning
        best_move, _ = minimax(board, player_turn_id, depth, float('-inf'), float('inf'), True)
    
    print("Move in:", best_move)
    
    socketIO.emit('play', {
        'tournament_id': 5555,
        'player_turn_id': player_turn_id,
        'game_id': game_id,
        'movement': best_move
    })

socketIO.on('connect', on_connect)
socketIO.on('ok_signin', on_ok_signin)
socketIO.on('finish', on_finish)
socketIO.on('ready', on_ready)
socketIO.on('finish', on_finish)

socketIO.wait()
