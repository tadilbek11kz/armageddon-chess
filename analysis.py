import io
import chess
import chess.pgn
import chess.engine
import math
import csv
import os
import sys

# Set the path to your Stockfish engine
ENGINE_PATH = "stockfish"  # Adjust this path as necessary

# Maximum depth for analysis
MAX_DEPTH = 23

# Configuration for the chess engine
# Adjust the number of threads and hash size as needed
# Threads - Number of threads to use for analysis
# Hash - Size of the hash table in MB
CONFIG = {
    "Threads": 8,
    "Hash": 2048
}


def get_task_batch(file_path, task_id, num_tasks):
    # This function used for splitting the workload across multiple tasks
    # It reads the CSV file and returns a specific batch of games based on the task ID and total number of tasks.
    """Get the specific batch of games for this task"""
    total_games = 115
    batch_size = math.ceil(total_games / num_tasks)
    start_idx = task_id * batch_size
    end_idx = min((task_id + 1) * batch_size, total_games)

    games = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            if start_idx <= idx < end_idx:
                games.append(row)

    return games


def get_games(file_path):
    # This function reads the CSV file and returns all games.
    # It is used when running the script without SLURM.
    """Get the specific batch of games for this task"""
    games = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            games.append(row)

    return games


def get_score(board, engine):
    """Get the current centipawn and mate score of the board using the chess engine."""
    # returns a dictionary with centipawn_score and mate_score
    info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
    # 1500 is the mate score threshold, adjust as necessary
    return {
        "centipawn_score": info['score'].white().score(mate_score=1500),
        "mate_score": info['score'].white().mate(),
    }


def analyze_game(game, engine):
    """Analyze a single game and return the move data."""
    # game is a tuple of (game_id, armageddon, pgn)
    # where game_id is a string, armageddon is an integer (1 or 0), and pgn is the game in PGN format.
    # The function reads the PGN, analyzes each move, and returns a list of dictionaries with move data.
    game_id, armageddon, pgn = game
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    game_result = game.headers['Result']
    rating = {True: game.headers.get("WhiteElo", None), False: game.headers.get("BlackElo", None)}
    prev_clock = {True: game.clock(), False: game.clock()}
    time_control = game.headers.get("TimeControl", None)

    # If time control is not specified, default to "0+0" (no time control)
    if time_control is None:
        time_control = "0+0"

    move_data = []
    move_count = 0

    # Iterate through the mainline of the game and analyze each move
    # The mainline is the sequence of moves played in the game.
    # For each move, we analyze the position, get the score, and calculate various metrics.
    # The move_count is used to keep track of the number of moves played in the game.
    for node in game.mainline():
        data, current_clock = analyze_move(node, engine, board, game_id, armageddon, move_count, game_result, rating, prev_clock, time_control)
        print(f"Analyzing move {move_count + 1} for game {game_id}")
        move_data.append(data)
        prev_clock[node.turn()] = current_clock
        move_count += 1

    return move_data


def get_move_time(current_clock, prev_clock):
    """Calculate the time taken for the current move based on the current and previous clock values."""
    if prev_clock is None:
        return 0

    return prev_clock - current_clock


def get_alternative_complexity(board, engine):
    """Calculate the complexity of the position based on the difference in scores of the top 5 moves."""
    # This function analyzes the top 5 moves and calculates the average difference in centipawn scores
    # between the best move and the second to fifth best moves.
    # It returns the average difference in centipawn scores.
    top_5_moves = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH), multipv=5)
    best_move = top_5_moves[0]['pv'][0]
    best_move_score = get_after_move_score(board, engine, best_move, MAX_DEPTH)
    average_difference = 0

    # Calculate the average difference in centipawn scores between the best move and the next 4 moves
    # top_5_moves may contain less than 5 moves if the position is not complex enough
    for i in range(1, len(top_5_moves)):
        second_best_move = top_5_moves[i]['pv'][0]
        second_best_move_score = get_after_move_score(board, engine, second_best_move, MAX_DEPTH)
        average_difference += abs(best_move_score - second_best_move_score)

    # Avoid division by zero if there are no alternative moves
    try:
        average_difference /= len(top_5_moves) - 1
    except ZeroDivisionError:
        average_difference = 0

    return average_difference


def get_complexity(board, engine):
    """Calculate the complexity of the position based on the difference in scores of the best and second-best moves."""
    # This function analyzes the position and calculates the complexity based on the difference in scores
    # between the best move and the second-best move at increasing depths.
    # It returns the total complexity, the depth at which the best move is found, and the scores of the best and second-best moves.
    complexity = 0

    previous_best_move = None
    best_move_overall = engine.analyse(board, chess.engine.Limit(depth=1))['pv'][0]
    depth_measure = None

    # Analyze the position at increasing depths from 2 to MAX_DEPTH
    # For each depth, we get the best move and the second-best move, and calculate the complexity based on their scores.
    # The best move is the move with the highest score, and the second-best move is the move with the second-highest score.
    for depth in range(2, MAX_DEPTH + 1):
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
        best_move = info[0]['pv'][0]
        best_move_score = get_after_move_score(board, engine, best_move, depth)

        if depth_measure is None and best_move == best_move_overall:
            depth_measure = depth

        # If there is no second-best move, we use the best move score for comparison
        if len(info) > 1:
            second_best_move = info[1]['pv'][0]
            second_best_move_score = get_after_move_score(board, engine, second_best_move, depth)
        else:
            second_best_move_score = best_move_score

        if previous_best_move != best_move:
            complexity += abs(best_move_score - second_best_move_score)

        previous_best_move = best_move

    return complexity, depth_measure


def get_cp_ralative_change(turn, before_score, after_score):
    """Calculate the relative change in centipawn score."""
    cp_change = after_score - before_score
    if not turn:
        cp_change = -cp_change

    return cp_change


def get_after_move_score(board, engine, move, depth=12):
    """Get the centipawn score after making a move."""
    # This function temporarily pushes the move onto the board, analyzes the position,
    # and then pops the move off the board to restore the original position.
    # It returns the centipawn score of the position after the move.
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    board.pop()
    return info['score'].white().score(mate_score=1500)


def get_win_percentage(centipawn_score):
    """Convert centipawn score to win percentage."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * centipawn_score)) - 1)


def get_accuracy(before, after_centipawn_score):
    """Calculate the accuracy of a move based on the centipawn score before and after the move."""
    after = get_win_percentage(after_centipawn_score)
    if after >= before:
        return 100.0
    else:
        win_diff = before - after
        raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) - 3.166924740191411
        accuracy = raw + 1
        accuracy = max(0, min(100, accuracy))
        return accuracy


def get_best_moves(board, engine):
    """Get the best move for the current position using the chess engine."""
    # This function analyzes the position and returns the best move and its score.
    # It uses multipv=2 to get the best move and the second-best move.
    # The best move is the move with the highest score, and the second-best move is used for comparison.
    # It returns the best move and its score.
    info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH), multipv=2)
    best_move_score = get_after_move_score(board, engine, info[0]['pv'][0], MAX_DEPTH)
    best_move_uci = info[0]['pv'][0].uci()
    if len(info) < 2:
        return best_move_score, best_move_score, best_move_uci, best_move_uci

    second_best_move_score = get_after_move_score(board, engine, info[1]['pv'][0], MAX_DEPTH)
    second_best_move_uci = info[1]['pv'][0].uci()
    return best_move_score, second_best_move_score, best_move_uci, second_best_move_uci


def analyze_move(node, engine, board, game_id, armageddon, move_count, game_result, rating, prev_clock, time_control):
    """Analyze a single move in the game and return the move data."""
    move = node.move

    # Configure the engine based on the current turn
    # White prefers to avoid draws, while Black seeks draws.
    if node.turn() == chess.WHITE:
        engine.configure({"Contempt": 100})  # White avoids draws
    else:
        engine.configure({"Contempt": -50})  # Black seeks draws

    # Get the current score of the board
    # and the best move for the current position.
    current_score = get_score(board, engine)

    best_move_score, second_best_move_score, best_move_uci, second_best_move_uci = get_best_moves(board, engine)

    # Calculate the win percentage based on the centipawn score
    win_percentage = get_win_percentage(current_score["centipawn_score"])

    current_clock = node.clock()
    move_time = get_move_time(current_clock, prev_clock[node.turn()])

    # Calculate the complexity of the position.
    complexity, depth_measure = get_complexity(board, engine)
    alternative_complexity = get_alternative_complexity(board, engine)

    # If + is in time_control, split it into time and increment
    # Add the increment multiplied by the move count to the time
    if "+" in time_control:
        time, inc = time_control.split("+")
        time = int(time) + (int(inc) * move_count)
    else:
        time = int(time_control)

    # Add the move to the board
    # This is the actual move being played in the game.
    board.push(move)

    # Get the score after the move is played
    # This is used to calculate the change in centipawn score after the move.
    # The after_score is used to calculate the accuracy of the move.
    after_score = get_score(board, engine)
    accuracy = get_accuracy(win_percentage, after_score["centipawn_score"])
    cp_difference = after_score["centipawn_score"] - current_score["centipawn_score"]

    if not node.turn():
        cp_difference = -cp_difference

    data = {
        "game_id": game_id,
        "armageddon": armageddon,
        "move_number": move_count + 1,
        "move": move.uci(),
        "centipawn_score": current_score["centipawn_score"],
        "mate_score": current_score["mate_score"],
        "move_time": move_time,
        "complexity": complexity,
        "alternative_complexity": alternative_complexity,
        "game_result": game_result,
        "clock": current_clock,
        "rating": rating[node.turn()],
        "depth_measure": depth_measure,
        "complexity_average": complexity / MAX_DEPTH,
        "time_control": time,
        "time_control_type": time_control,
        "best_move": 1 if best_move_uci == move.uci() else 0,
        "win_percentage": win_percentage,
        "accuracy": accuracy,
        "centipawn_score_change": get_cp_ralative_change(node.turn(), current_score["centipawn_score"], after_score["centipawn_score"]),
        "best_move_cp_change": get_cp_ralative_change(node.turn(), current_score["centipawn_score"], best_move_score),
        "second_best_move_cp_change": get_cp_ralative_change(node.turn(), current_score["centipawn_score"], second_best_move_score),
    }

    return data, current_clock


def main():
    """Main function with Slurm to run the analysis."""

    # This section is used to run the script with SLURM on HPC.
    # It retrieves the task ID, number of tasks, and job ID from the environment variables set by SLURM.
    task_id = int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
    job_id = os.environ.get('SLURM_JOB_ID', '0')
    file_path = sys.argv[1]
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine.configure(CONFIG)

    print(f"Task {task_id} starting processing")

    # Get the specific batch of games for this task
    games = get_task_batch(f"{file_path}", task_id, num_tasks)

    print(f"Task {task_id} processing {len(games)} games")

    # Filter games to only include those with armageddon set to 1
    games = [(row[0], row[1], row[2]) for row in games if int(row[1]) == 1]

    # Iterate through the games and analyze each one
    for index, game in enumerate(games):
        try:
            results = analyze_game(game, engine)
        except Exception as e:
            print(f"Error analyzing game {game[0]}: {e}")
            continue

        print(f"Game {index + 1}/{len(games)}: {game[0]}")
        print(f"Task {task_id} processed {index + 1} games out of {len(games)}")

        output_file = f"game_moves_job{job_id}_task{task_id}.csv"
        with open(output_file, "a") as file:
            writer = csv.writer(file)
            for game_moves in results:
                writer.writerow(game_moves.values())
    engine.close()
    print(f"Task {task_id} completed processing")


if __name__ == "__main__":
    main()
