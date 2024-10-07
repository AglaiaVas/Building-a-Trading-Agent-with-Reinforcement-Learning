
# Building a Trading Agent with Reinforcement Learning


This project involves developing a trading agent using Double Deep Q-Network (DDQN) reinforcement learning. The agent learns from stock market data and suggests actions such as "buy," "sell," or "hold" to maximize cumulative returns. The model is trained using daily historical data from the SNP 500 index and utilises several technical indicators to inform its trading decisions.

## Data
The dataset consists of daily historical stock data for the SNP 500 index, covering the period from October 1, 2004, to October 4, 2024. The dataset includes fields like close prices, high prices, low prices, and volume. The data is sourced from Yahoo Finance.

### Data Visualisation
Here’s a graph of the raw SNP 500 closing prices over time:

![Formatted Raw Data Graph](https://github.com/AglaiaVas/Building-a-Trading-Agent-with-Reinforcement-Learning/blob/e535cc0ee79dcd1a21a9fd1c981ef26c60b68def/formatted_raw_data_graph.png)


This graph shows the historical closing prices of the SNP 500 index over the specified period.

## Technical Explanation of the Model

The model is built on the **Double Deep Q-Network (DDQN)**, which is an enhancement of the basic DQN algorithm. DDQN helps reduce the overestimation of Q-values by decoupling the action selection from action evaluation.

### Key Features of DDQN:
1. **Online Network**: This network selects the best action to take for the given state.
2. **Target Network**: This network evaluates the Q-value of the action selected by the online network to stabilise training.
3. **Experience Replay**: Stores past experiences (state, action, reward, next state) in a buffer and samples mini-batches for training, which breaks the correlation between consecutive experiences.
4. **Epsilon-Greedy Policy**: Balances exploration (trying new actions) and exploitation (using the best-known actions). The agent starts with high exploration (high epsilon) and gradually reduces it to focus more on exploitation as it learns.

### Q-learning Update Rule:
The Q-value update for a given state-action pair \( (s, a) \) in DDQN is:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{online}}(s', a')) - Q(s, a) \right)
$$


Where:

- $Q(s, a)$: Q-value for action $a$ in state $s$,
- $r$: Reward,
- $\gamma$: Discount factor for future rewards,
- $Q_{\text{target}}$: Q-value evaluated by the target network,
- $Q_{\text{online}}$: Q-value from the online network,
- $\alpha$: Learning rate.

## Model Architecture

The model architecture consists of two fully connected hidden layers with 256 units each. The agent uses the **ReLU activation function** and **Adam optimizer** to train the neural network.

### Key Model Hyperparameters:
- **Gamma**: 0.99 (discount factor for future rewards)
- **Tau**: 100 (target network update frequency)
- **Learning Rate**: 0.0001 (learning rate for gradient descent)
- **Batch Size**: 4096 (number of experiences per training step)
- **Replay Capacity**: 1e6 (size of the experience replay buffer)
- **Epsilon**: Starts at 1.0 and decays to 0.01 over 250 steps using an exponential decay factor of 0.99.

## Hyperparameter Optimisation

The hyperparameters of the DDQN model were optimised using a simple grid search approach, where multiple combinations of hyperparameters were tested to find the optimal configuration. This method systematically explores various values across predefined ranges of key hyperparameters.

### Optimised Hyperparameters:
- **Learning Rate**: A grid search tested 3 values (0.0001, 0.001, 0.01) across 10 episodes due to computational limits
- **Gamma (Discount Factor)**: Values tested were 0.9, 0.95, and 0.99, controlling how future rewards are weighted relative to immediate rewards.
- **Epsilon Decay**: The decay strategies tested were 0.99, 0.995, and 0.999, impacting how the exploration rate decreases over time, balancing exploration and exploitation.
- Additionally trial-and-error approach for epsilon decay found that 250 steps were optimal, when 800 steps were used, a bigger number led to excessive exploration


## Results

![Agent vs Market Performance](https://github.com/AglaiaVas/Building-a-Trading-Agent-with-Reinforcement-Learning/blob/ec09fd30b190b80ab91e623467bb9b966635791c/agent_vs_market_rolling_means_final.png)


### Agent vs. Market NAV:
- The agent's performance improves noticeably after approximately 300 steps, indicating that it starts to learn more effectively and better adjusts its strategy based on the environment. This is reflected in the agent’s ability to track the market closely and occasionally outperform it.
- 
- The rolling mean of the agent's NAV performance, compared to the market NAV, provides valuable insight into the agent's overall performance. The agent often closely tracks the market but experiences occasional periods of outperformance and underperformance. This reflects its ability to follow market movements while trying to capitalise on short-term opportunities.

- The S&P 500 typically grows by approximately 10% annually, presenting a challenge for the agent to consistently outperform such a robust and diverse index over time.

![Agent Average Rolling Win Mean](https://github.com/AglaiaVas/Building-a-Trading-Agent-with-Reinforcement-Learning/blob/e535cc0ee79dcd1a21a9fd1c981ef26c60b68def/agent_average_rolling_win_mean.png)

- The agent's performance improves noticeably after approximately 300 steps, indicating that it starts to learn more effectively and better adjusts its strategy based on the environment. This is reflected in the agent’s ability to track the market closely and occasionally outperform it.
- The rolling win ratio shows how often the agent outperforms the market over a 100-episode window. While the agent successfully outperforms in certain windows, it struggles to maintain consistency over all episodes.

- The S&P 500 index is composed of top-performing stocks, giving the market a natural advantage in bullish conditions, making it harder for the agent to outperform, especially during sustained growth phases.

- The S&P 500 is a diversified index, and its inherent stability makes it challenging for a trading agent to outperform without taking significant risks. The agent is competing against a benchmark that benefits from overall market growth and stability.

- During strong bull markets, the agent is at a disadvantage, as the index tends to follow steady upward trends. This limits the agent’s ability to find short-term opportunities.

- The agent's ability to adapt to market fluctuations and take advantage of short-term deviations is a positive indicator of its potential. The use of technical indicators like RSI, MACD, and ATR aids in identifying market reversals or entry/exit points.

- Reinforcement learning allows the agent to learn from experience and optimize its strategy over time, which is valuable in dynamic or volatile market conditions where passive strategies may underperform.

### Additional Considerations:
**More Episodes for Better Results:**
Running additional episodes would enhance the agent's ability to learn and refine its strategy, improving overall performance. However, this comes at the expense of longer computational time. Reinforcement learning models typically require extensive iterations to converge on optimal strategies, which presents a trade-off between model accuracy and resource use. For example, the current run took 3 hours, and increasing the number of episodes would proportionally extend this time.

### Conclusion:
The DDQN-based trading agent demonstrates potential in outperforming the S&P 500 in specific market conditions, particularly during volatile or corrective phases. However, it faces significant challenges in consistently outperforming a diversified and upward-trending index like the S&P 500. Despite these challenges, the agent’s ability to learn and adapt over time highlights the power of reinforcement learning in trading applications. Further improvements to the model, including more episodes, additional data inputs, and optimization techniques, could enhance the agent’s overall performance and make it more competitive in varied market conditions.

## Reference

This project is based on the book **"Machine Learning for Algorithmic Trading"** by Stefan Jansen and has been expanded to fit the SNP 500 dataset using reinforcement learning techniques.

## Contact Details
If you have any questions or suggestions, feel free to contact me via [LinkedIn](https://www.linkedin.com/in/aglaia-vasileiou-3888626/).
