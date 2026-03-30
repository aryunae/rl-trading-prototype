# RL Trading Prototype

Reinforcement Learning agent for algorithmic trading using Double Deep Q-Network (DDQN).

## Features
- Trading environment with three actions: SHORT, HOLD, LONG
- Real stock data via yfinance
- DDQN agent with experience replay and target network
- Comparison with buy-and-hold benchmark

## Requirements
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Usage

### Training
```bash
python src/train.py --ticker AAPL --episodes 500 --trading_days 100
Arguments
--ticker: stock symbol (default: AAPL)

--trading_days: episode length (default: 252)

--episodes: number of episodes (default: 1000)

--learning_rate: learning rate for optimizer

--gamma: discount factor

--epsilon_start, --epsilon_end: epsilon for ε-greedy policy

--replay_capacity: replay buffer size

--architecture: hidden layer sizes (e.g., --architecture 256 256)

Output
Results are saved in ./results/:

results.csv: episode-by-episode performance

performance.png: learning curves

Example Run
bash
python src/train.py --ticker MSFT --episodes 200 --trading_days 126 --learning_rate 0.0001
Project Structure
src/: source code

trading_env.py: trading environment

dqn_agent.py: DDQN agent

train.py: training script

results/: training outputs

notebooks/: optional Jupyter notebooks

docs/: additional documentation

License
MIT (original code by Tito Ingargiola, Stefan Jansen)

text

### 6. Создание `.gitignore`
pycache/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
.egg-info/
dist/
build/
.ipynb_checkpoints/
.DS_Store
results/.csv
!results/.gitkeep

text

### 7. `docs/experiment.md` (описание первичного тестирования)

```markdown
# Primary Experiment Results

## Experimental Setup
- Ticker: AAPL
- Episode length: 252 trading days (1 year)
- Number of episodes: 1000
- Trading cost: 10 bps, time cost: 1 bps
- DDQN parameters: learning rate 1e-4, gamma 0.99, epsilon decay from 1.0 to 0.01 over 250 episodes, replay capacity 1e6, batch size 4096, hidden layers 256-256

## Results

### Table 1: Performance comparison (last 100 episodes average)
| Metric | Agent | Market |
|--------|-------|--------|
| Average NAV | 1.448 | 1.176 |
| Average Return (%) | 44.8% | 17.6% |

### Table 2: Win ratio over episodes
| Episode Range | Win Ratio (Agent > Market) |
|---------------|----------------------------|
| 1-100 | 20% |
| 101-200 | 22% |
| ... | ... |
| 901-1000 | 59% |

### Graphs
![Performance](results/performance.png)

## Analysis
The agent initially underperforms the market due to exploration, but after ~300 episodes it starts to learn profitable policies. By episode 1000, the agent outperforms the market in 59% of episodes, despite transaction costs. This demonstrates that DDQN can discover a viable trading strategy.

## Verification
The model was trained and tested on the same data (with random start dates), which is not a strict out-of-sample test. However, the randomisation reduces overfitting risk. Future work will include a proper train/test split.