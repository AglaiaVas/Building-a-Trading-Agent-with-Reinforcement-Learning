
# DDQN Trading Agent Model Card

## Model Details

- **Model Name**: Double Deep Q-Network (DDQN) Trading Agent
- **Model Type**: Reinforcement Learning (Q-learning variant)
- **Framework**: OpenAI and Torch
- **Input**: State vector including technical indicators such as close prices, high prices, low prices, volume, RSI, MACD, ATR, etc.
- **Output**: Actions – "buy," "sell," or "hold" signals for trading.

## Model Architecture

- **Neural Network Structure**: Two fully connected hidden layers with 256 units each.
- **Activation Function**: ReLU for each layer.
- **Optimiser**: Adam optimiser for training the network.

### Hyperparameters:
- **Gamma (Discount Factor)**: 0.99 – Determines how future rewards are weighted relative to immediate rewards.
- **Tau (Target Network Update Frequency)**: 100 steps.
- **Learning Rate**: 0.0001.
- **Batch Size**: 4096.
- **Replay Buffer Capacity**: 1e6 experiences.
- **Epsilon**: Starts at 1.0 and decays to 0.01 over 250 steps using an exponential decay factor of 0.99.
- **Exploration Strategy**: Epsilon-Greedy policy to balance exploration and exploitation.

## Intended Use

The DDQN model is designed to create a trading agent that can operate autonomously on stock market data. The goal is to maximise cumulative returns by taking advantage of market fluctuations and capitalising on short-term trading opportunities. The agent is trained on historical data from the S&P 500 index, using technical indicators to make informed decisions.

## Training Data

The dataset consists of daily historical stock data for the S&P 500 index, from 1 October 2004 to 4 October 2024, sourced from Yahoo Finance. The key fields include:
- Close prices
- High prices
- Low prices
- Volume

### Data Visualisation
Below is a graph of the raw S&P 500 closing prices over time:

![Formatted Raw Data Graph](https://github.com/AglaiaVas/Building-a-Trading-Agent-with-Reinforcement-Learning/blob/e535cc0ee79dcd1a21a9fd1c981ef26c60b68def/formatted_raw_data_graph.png)

## Hyperparameter Optimisation

The hyperparameters of the DDQN model were optimised using a simple grid search for a small number of episodes (10) due to computational challenges.

### Optimised Hyperparameters:
- **Learning Rate**: A grid search tested 3 values (0.0001, 0.001, 0.01) across 10 episodes.
- **Gamma (Discount Factor)**: Values tested were 0.9, 0.95, and 0.99, controlling how future rewards are weighted relative to immediate rewards.
- **Epsilon Decay**: The decay strategies tested were 0.99, 0.995, and 0.999. A trial-and-error approach for epsilon decay found that 250 steps were optimal, as 1000 steps caused excessive exploration.

## Performance Metrics

The agent’s performance improves significantly after 300 episodes, indicating better learning and adaptation.

![Agent vs Market Performance](https://github.com/AglaiaVas/Building-a-Trading-Agent-with-Reinforcement-Learning/blob/ec09fd30b190b80ab91e623467bb9b966635791c/agent_vs_market_rolling_means_final.png)

### Agent vs Market NAV:
- The rolling mean of the agent's Net Asset Value (NAV) compared to the market NAV provides insight into the agent’s overall performance. While the agent can track the market's performance, it experiences periods of outperformance and underperformance, highlighting its ability to exploit short-term opportunities.

![Agent Average Rolling Win Mean](https://github.com/AglaiaVas/Building-a-Trading-Agent-with-Reinforcement-Learning/blob/e535cc0ee79dcd1a21a9fd1c981ef26c60b68def/agent_average_rolling_win_mean.png)

### Rolling Win Ratio:
- The rolling win ratio illustrates how often the agent outperforms the market across a 100-episode window. While the agent achieves some wins, it faces difficulties in maintaining consistent performance, particularly in bullish market phases.

## Key Insights:

- **Challenges**: The S&P 500 index consists of top-performing, diversified stocks, making it difficult for the agent to consistently outperform during bullish markets, which benefit from overall market growth and stability.
  
- **Opportunities**: The agent shows potential during volatile or corrective phases, leveraging technical indicators like RSI, MACD, and ATR to identify market reversals and optimise trading decisions.

## Ethical Considerations

- **Bias**: The model may not generalise well to other markets or time periods, as it has been specifically trained on historical S&P 500 data. Its performance may vary significantly in different market conditions.
- **Fairness**: The model could favour certain stock types or market conditions, potentially amplifying risks during highly volatile or bearish periods.

## Limitations

- **Overfitting**: The model is sensitive to hyperparameters and may overfit to the training data, leading to poor generalisation.
- **Scalability**: Longer training periods or additional episodes significantly increase computational time, with the current run taking around 3 hours for 800 episodes.

## Conclusion

The DDQN trading agent has demonstrated potential in outperforming the market in specific scenarios, particularly during volatile or corrective phases. However, it struggles to consistently outperform a stable and diversified index like the S&P 500. Further improvements, including more episodes, additional data inputs, and optimisation techniques, could enhance the agent’s overall performance and competitiveness in varied market conditions.

## Contact

If you have any questions or suggestions, feel free to contact me via [LinkedIn](https://www.linkedin.com/in/aglaia-vasileiou-3888626/).
