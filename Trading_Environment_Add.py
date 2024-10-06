#!/usr/bin/env python
# coding: utf-8

# # Trading Environment for Reinforcement Learning
# 
# We build an environment that adheres to the OpenAI Gym architecture.
# 
# The trading environment consists of three core classes that interact to facilitate the agent's activities:
# 
# 1. **`DataSource` Class**: 
#    - Loads historical price data and generates several technical indicators for each time step, providing the agent with observations.
#    - The data can be optionally normalized to facilitate model training.
# 
# 2. **`TradingSimulator` Class**: 
#    - Tracks the agent's positions, trades, costs, and computes the Net Asset Value (NAV) over time.
#    - Implements a buy-and-hold benchmark strategy for performance comparison.
# 
# 3. **`TradingEnvironment` Class**: 
#    - Orchestrates the interaction between the agent, `DataSource`, and `TradingSimulator`.
#    - Provides market observations, receives actions, and computes rewards.
# 

# ### DataSource: Managing Market Data and Technical Indicators
# 
# #### 1. **Data Acquisition**:
# The `DataSource` class fetches historical market data (Close, Volume, High, Low prices) from Yahoo Finance for a specified stock ticker (e.g., `^GSPC` for S&P 500).
# 
# #### 2. **Technical Indicators**:
# The class calculates multiple technical indicators using the TA-Lib library to describe the market state at each time step:
# - **Returns**: Percentage change in prices over 2, 5, 10, and 21 days.
# - **Stochastic RSI (STOCHRSI)**: Measures momentum relative to the RSI range, identifying overbought/oversold conditions.
# - **MACD**: Shows the difference between short-term and long-term exponential moving averages to capture trend.
# - **Average True Range (ATR)**: Measures market volatility based on the range of price movements.
# - **Bollinger Bands (BBANDS)**: Plots upper and lower volatility bands around a moving average.
# - **On-Balance Volume (OBV)**: Tracks volume in relation to price movement to capture buying/selling pressure.
# - **Stochastic Oscillator**: Compares the close price to a price range over a given period to measure momentum.
# - **Ultimate Oscillator (ULTOSC)**: Combines multiple timeframes to assess price momentum.
# 
# #### 3. **Normalization**:
# The data is optionally normalized to scale features, helping reinforcement learning models converge faster by keeping all inputs on a similar scale.
# 
# #### 4. **Step and Reset Methods**:
# The `step` method provides the next observation from the dataset for the agent to act on, while the `reset` method starts a new episode by randomizing the starting point in the dataset.
# 

# In[2]:


import logging
import gym as gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import yfinance as yf
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import talib

# Set up logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

# DataSource Class
class DataSource:
    """
    Data source for TradingEnvironment.

    Loads and preprocesses daily price and volume data.
    Provides technical indicators and returns for each new episode.
    """

    def __init__(self, trading_days=252, ticker='^GSPC', start_date='2004-01-01', normalize=True):
        """
        :param trading_days: Number of trading days per episode (default: 252, equivalent to a year)
        :param ticker: Stock symbol to fetch (default: '^GSPC' for S&P 500 index)
        :param start_date: Start date for fetching historical data
        :param normalize: Whether to normalize the data (default: True)
        """
        self.ticker = ticker
        self.trading_days = trading_days
        self.start_date = start_date
        self.normalize = normalize
        self.data = self._load_data()

        # Save initial (raw) data to CSV
        self.data.to_csv(f'{self.ticker}_raw_data.csv')
        print(f"Raw data saved to {self.ticker}_raw_data.csv")

        # Check the raw data
        print("Raw data:\n", self.data.head())
        
        self.preprocess_data()

        # Save preprocessed data to CSV
        self.data.to_csv(f'{self.ticker}_preprocessed_data.csv')
        print(f"Preprocessed data saved to {self.ticker}_preprocessed_data.csv")

        # Check preprocessed data
        print("Preprocessed data with technical indicators:\n", self.data.head())

        # Check basic statistics of the data
        print("Statistics of preprocessed data:\n", self.data.describe())

        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0

    def _load_data(self):
        """
        Load historical price data from Yahoo Finance. This function retrieves close prices,
        volumes, and high/low prices for the given ticker.

        Returns:
            DataFrame: Cleaned data with columns: ['Close', 'Volume', 'Low', 'High'].
        """
        log.info(f'Loading data for {self.ticker} starting from October 2004...')

        # Hardcoding the start date to October 1, 2004
        start_date = '2004-10-01'

        try:
            # Download historical data from Yahoo Finance
            df = yf.download(self.ticker, start=start_date)
            
            # Check if required columns exist
            required_columns = ['Close', 'Volume', 'Low', 'High']
            if not all(col in df.columns for col in required_columns):
                log.error(f'Missing required columns in data for {self.ticker}.')
                return None

            # Selecting the adjusted close, volume, low, and high columns and dropping rows with missing values
            df = df[required_columns].dropna()
            
            log.info(f'Successfully retrieved data for {self.ticker}.')
            return df

        except Exception as e:
            log.error(f'Error loading data for {self.ticker}: {e}')
            return None

    def preprocess_data(self):
        """Calculate returns and technical indicators, then remove missing values."""
        
        # Calculate percentage returns over different periods
        self.data['returns'] = self.data['Close'].pct_change()
        self.data['ret_2'] = self.data['Close'].pct_change(2)
        self.data['ret_5'] = self.data['Close'].pct_change(5)
        self.data['ret_10'] = self.data['Close'].pct_change(10)
        self.data['ret_21'] = self.data['Close'].pct_change(21)
        self.data['rsi'] = talib.STOCHRSI(self.data['Close'])[1]
        self.data['macd'] = talib.MACD(self.data['Close'])[1]
        self.data['atr'] = talib.ATR(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['bb_upper'], self.data['bb_middle'], self.data['bb_lower'] = talib.BBANDS(self.data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)#new
        self.data['obv'] = talib.OBV(self.data['Close'], self.data['Volume']) #new
        
        slowk, slowd = talib.STOCH(self.data.High, self.data.Low, self.data.Close)
        self.data['stoch'] = slowd - slowk
        self.data['ultosc'] = talib.ULTOSC(self.data.High, self.data.Low, self.data.Close)
        
        # Remove infinite values and drop unnecessary columns
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['High', 'Low', 'Close', 'Volume'], axis=1)
                     .dropna())
        
        r = self.data.returns.copy()
        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)
        features = self.data.columns.drop('returns')
        self.data['returns'] = r  # don't scale returns
        self.data = self.data.loc[:, ['returns'] + list(features)]
        log.info(self.data.info())

    def reset(self):
        """
        Resets the data to start from a random point for each episode.
        """
        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """
        Returns the data for the current trading day and checks if the episode is done.
        """
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step >= self.trading_days
        return obs, done

# Test the DataSource class
data_source = DataSource(trading_days=252, ticker='^GSPC')
data_source.reset()
obs, done = data_source.take_step()

print("First observation:", obs)
print("Done:", done)


# ### Trading Simulator: Calculating Positions, NAV, and Rewards
# 
# #### 1. **Positions**:
# At each time step, the agent can take one of the following actions:
# - **Long (Buy)**: Increase the position in the asset.
# - **Short (Sell)**: Bet against the asset, benefiting if the price goes down.
# - **Hold**: Keep the current position unchanged.
# 
# #### 2. **Net Asset Value (NAV)**:
# NAV represents the total value of the agent's portfolio over time. It is updated at each step based on the market return and the agent’s position:
# 
# $$
# \text{NAV}_{t+1} = \text{NAV}_t \times (1 + \text{Strategy Return}_t)
# $$
# 
# - $\text{NAV}_t $ is the Net Asset Value at time  $t$,
# - $\text{Strategy Return}_t$ is the return generated by the agent’s position at time $t$.
# 
# #### 3. **Reward Calculation**:
# The reward for the agent is based on the change in the agent's portfolio value (NAV) after accounting for trading costs:
# 
# 
# $\text{Reward}_t = \text{Position}_t \times \text{Market Return}_t - \text{Costs}_t$
# 
# - $ \text{Position}_t $ is the agent’s current position (long, short, or hold),
# - $ \text{Market Return}_t $ is the market’s return for that time step,
# - $ \text{Costs}_t $ are the costs incurred due to trading (e.g., trading costs, time decay costs).

# In[3]:


import logging
import numpy as np
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# TradingSimulator Class
class TradingSimulator:
    """
    Core trading simulator for single-instrument trading.
    Tracks actions (long, short, hold), calculates rewards, applies trading costs, 
    and maintains Net Asset Value (NAV).
    """
    #trading_cost_bps=1e-3, time_cost_bps=1e-4
    def __init__(self, steps, trading_cost_bps=0, time_cost_bps=0):
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        log.debug(f"Initializing simulator with {steps} steps, trading cost: {trading_cost_bps}, time cost: {time_cost_bps}")
        self.reset()

    def reset(self):
        """
        Resets the simulation to its initial state.
        """
        log.debug("Resetting the simulator.")
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def take_step(self, action, market_return):
        """
        Simulates a single trading step based on the action taken by the agent.
        """
        log.debug(f"Step {self.step}: Taking action {action} with market return {market_return}.")
        
        # Retrieve the position from the previous step
        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return

        # Action is converted to position: 0 -> short, 1 -> hold, 2 -> long
        end_position = action - 1
        n_trades = end_position - start_position

        # Calculate trading and time decay costs
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        log.debug(f"Trade costs: {trade_costs}, Time cost: {time_cost}, Total cost: {self.costs[self.step]}")

        # Calculate reward as the market return adjusted by position and costs
        reward = start_position * market_return - self.costs[self.step]
        self.strategy_returns[self.step] = reward
        log.debug(f"Reward: {reward}, Strategy return: {self.strategy_returns[self.step]}")

        # Update NAV based on strategy returns
        self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
        self.market_navs[self.step] = start_market_nav * (1 + market_return)
        log.debug(f"NAV: {self.navs[self.step]}, Market NAV: {self.market_navs[self.step]}")

        # Update positions, actions, and trade information
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades
        self.actions[self.step] = action  # Store the action

        self.step += 1
        info = {'reward': reward, 'nav': self.navs[self.step - 1], 'costs': self.costs[self.step - 1]}
        log.debug(f"Step info: {info}")
        return reward, info

    def results(self):
        """
        Returns the current state of the simulator as a DataFrame.
        """
        log.debug("Returning results as DataFrame.")
        return pd.DataFrame({
            'action': self.actions,
            'nav': self.navs,
            'market_nav': self.market_navs,
            'market_return': self.market_returns,
            'strategy_return': self.strategy_returns,
            'position': self.positions,
            'cost': self.costs,
            'trade': self.trades
        })

# Test the TradingSimulator
simulator = TradingSimulator(steps=10)
for i in range(10):
    action = np.random.randint(0, 3)  # Random action: 0 (short), 1 (hold), 2 (long)
    market_return = np.random.uniform(-0.01, 0.01)  # Simulate random market return
    simulator.take_step(action, market_return)

# Get results as a DataFrame
results = simulator.results()

# Print formatted results
print(f"{'Step':>4} | {'Action':>6} | {'NAV':>10} | {'Market NAV':>12} | {'Market Return':>13} | "
      f"{'Strategy Return':>16} | {'Position':>9} | {'Cost':>6} | {'Trade':>6}")

# Print each row of the DataFrame in a formatted manner
for idx, row in results.iterrows():
    print(f"{idx:>4} | {int(row['action']):>6} | {row['nav']:>10.4f} | {row['market_nav']:>12.4f} | "
          f"{row['market_return']:>13.4%} | {row['strategy_return']:>16.4%} | "
          f"{int(row['position']):>9} | {row['cost']:>6.4f} | {int(row['trade']):>6}")


# # TradingEnvironment Class
# 
# The `TradingEnvironment` class is a custom trading environment built for reinforcement learning. It follows the OpenAI Gym architecture and allows agents to interact with a simulated financial market through trading actions. The agent's primary goal is to maximize its **Net Asset Value (NAV)** by taking one of three possible actions: **short**, **hold**, or **long**.
# 
# ### Key Features:
# 
# - **Actions**: The agent can take three possible actions at each time step:
#   - **0**: Short the asset
#   - **1**: Hold the current position
#   - **2**: Go long on the asset
# 
# - **Data Source**: The environment retrieves historical financial data (e.g., stock prices) through the `DataSource` class, which preprocesses the data and computes technical indicators.
# 
# - **Trading Simulator**: The `TradingSimulator` class tracks the agent's trades and portfolio value, considering both trading costs (e.g., commissions) and time decay costs.
# 
# - **Observation Space**: The environment provides the agent with market data (e.g., returns, technical indicators) at each step, allowing it to make informed trading decisions.
# 
# - **Reward Calculation**: The reward is calculated based on the agent's action and the market return, adjusted for trading costs. The agent aims to maximize its rewards over time by effectively managing its portfolio.
# 
# - **Environment Reset**: At the start of each episode, the environment resets the data and simulator to simulate a new trading period.
# 
# This environment serves as the foundation for training reinforcement learning agents to develop trading strategies based on historical financial data.
# 

# In[4]:


# TradingEnvironment Class
class TradingEnvironment(gym.Env):
    """
    A custom trading environment for reinforcement learning.

    Allows the agent to take actions (short, hold, long) and tracks the performance via NAV.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, trading_days=252, trading_cost_bps=1e-3, time_cost_bps=1e-4, ticker='MSFT'):
        super(TradingEnvironment, self).__init__()
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.ticker = ticker

        # Initialize data source and simulator
        self.data_source = DataSource(trading_days=trading_days, ticker=ticker)
        self.simulator = TradingSimulator(steps=trading_days, trading_cost_bps=trading_cost_bps, time_cost_bps=time_cost_bps)

        # Define action space (0: short, 1: hold, 2: long) and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=self.data_source.min_values.values,  # Convert pandas Series to numpy array
            high=self.data_source.max_values.values,  # Convert pandas Series to numpy array
            dtype=np.float32
        )

        self.reset()

    def seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Executes a step in the environment by applying the chosen action.
        """
        assert self.action_space.contains(action), f"{action} is an invalid action."
        observation, done = self.data_source.take_step()  # Get next day's data
        reward, info = self.simulator.take_step(action, market_return=observation[0])
        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to start a new episode.
        """
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]  # Return the first observation

    def render(self, mode='human'):
        """Rendering is not implemented in this environment."""
        pass

    def close(self):
        """Cleans up resources when the environment is closed."""
        pass


# In[5]:


get_ipython().system('jupyter nbconvert --to script Trading_Environment_Add.ipynb')


# In[6]:


# Get DataFrame with results
result = simulator.results()
final = result.iloc[-1]  # Get the last entry (final state)
print(f"Final NAV: {final.nav:.6f}")
print(f"Final Market Return: {final.market_return:.6f}")
print(results)

