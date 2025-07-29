# Chess Armageddon Game Analysis

This repository contains a comprehensive chess analysis system designed to analyze Armageddon games using the Stockfish chess engine. The project is optimized for both local execution and high-performance computing (HPC) environments using SLURM.

## Overview

Armageddon games are decisive chess matches where White receives more time but must win (a draw counts as a win for Black). This analysis tool evaluates the quality of moves, complexity of positions, and various chess metrics to understand playing patterns in these high-pressure scenarios.

## Features

- **Deep Position Analysis**: Uses Stockfish engine with configurable depth (default: 23 ply)
- **Move Quality Assessment**: Calculates accuracy, centipawn changes, and identifies best moves
- **Position Complexity**: Measures position complexity using multiple algorithms
- **Time Analysis**: Analyzes move timing and time control effects
- **Parallel Processing**: Supports SLURM for distributed analysis on HPC systems
- **Multiple Engine Versions**: Compatible with different Stockfish versions

## Project Structure

```
├── analysis.py           # Main analysis script for SLURM/HPC
```

## Requirements

### Dependencies
- Python 3.12
- `python-chess` library
- `csv` (built-in)
- `math` (built-in)
- `io` (built-in)
- `os` (built-in)
- `sys` (built-in)

### Chess Engine
- Stockfish chess engine
- Compatible with macOS, Linux, and Windows

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd armageddon
   ```

2. **Install Python dependencies**:
   ```bash
   pip install python-chess
   ```

3. **Set up Stockfish engine**:
   - For macOS: Use the included `stockfish/Mac/stockfish`
   - For Linux: Use `stockfish/stockfish-ubuntu-20.04-x86-64`
   - Or install globally and use `stockfish` command

## Configuration

### Engine Settings
Modify the configuration in the analysis scripts:

```python
ENGINE_PATH = "stockfish"  # Path to Stockfish executable
MAX_DEPTH = 23            # Analysis depth (higher = more accurate, slower)
CONFIG = {
    "Threads": 8,         # Number of CPU threads
    "Hash": 2048          # Hash table size in MB
}
```

### Analysis Parameters
- **Contempt Settings**: 
  - White: +100 (avoids draws)
  - Black: -50 (seeks draws)
- **Mate Score Threshold**: 1500 centipawns
- **Multipv**: Up to 5 best moves analyzed for complexity

## Usage

### HPC/SLURM Analysis
For distributed analysis on HPC systems:

```bash
srun --cpus-per-task=M --ntasks=N --mem-per-cpu=3750M python analysis.py {file_path}
```

Where M is the number of CPU cores per task and N is the number of parallel tasks.

### Data Format

Input CSV format:
```
game_id,armageddon,pgn
974,1,"[Event ""Division III""] [Site ""Chess.com""] ..."
```

- `game_id`: Unique identifier for the game
- `armageddon`: 1 for Armageddon games, 0 for regular games
- `pgn`: Complete PGN notation of the game

## Output Metrics

The analysis generates the following metrics for each move:

### Basic Information
- `game_id`: Game identifier
- `armageddon`: Armageddon flag (1/0)
- `move_number`: Move number in the game
- `move`: Move in UCI notation
- `game_result`: Final game result

### Position Evaluation
- `centipawn_score`: Position evaluation in centipawns
- `mate_score`: Mate in N moves (if applicable)
- `win_percentage`: Win probability percentage

### Move Quality
- `accuracy`: Move accuracy (0-100%)
- `best_move`: 1 if move is engine's top choice, 0 otherwise
- `centipawn_score_change`: Change in evaluation after move
- `best_move_cp_change`: Change if best move was played
- `second_best_move_cp_change`: Change if second-best move was played

### Position Complexity
- `complexity`: Total complexity across all depths
- `alternative_complexity`: Average difference between top 5 moves
- `complexity_average`: Normalized complexity per depth
- `depth_measure`: Depth at which best move stabilizes

### Time Analysis
- `move_time`: Time spent on the move
- `clock`: Remaining time on clock
- `time_control`: Effective time control
- `time_control_type`: Original time control format
- `rating`: Player's rating

## Analysis Algorithms

### Complexity Calculation
1. **Standard Complexity**: Measures score differences between best moves at increasing depths
2. **Alternative Complexity**: Analyzes score gaps between top 5 moves
3. **Depth Stability**: Determines when the best move choice stabilizes

### Accuracy Formula
Based on win percentage changes:
```python
accuracy = 103.1668100711649 * exp(-0.04354415386753951 * win_diff) - 3.166924740191411 + 1
```

### Win Percentage Conversion
```python
win_percentage = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawn_score)) - 1)
```

## Performance Optimization

### Local Machine
- Adjust `Threads` based on CPU cores
- Increase `Hash` size for more RAM
- Reduce `MAX_DEPTH` for faster analysis

### HPC Environment
- Use SLURM array jobs for parallel processing
- Filters to analyze only Armageddon games (armageddon=1)

## Troubleshooting

### Common Issues
1. **Engine Path Error**: Ensure Stockfish is correctly installed and path is accurate
2. **Memory Issues**: Reduce hash size or depth for systems with limited RAM
3. **CSV Format**: Ensure input CSV matches expected format with proper PGN notation

### Performance Tips
- Use SSD storage for faster file I/O
- Monitor memory usage with large hash tables
- Consider move time limits for very long games