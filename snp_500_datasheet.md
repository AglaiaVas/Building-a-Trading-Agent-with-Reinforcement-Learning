
# Datasheet for SNP 500 Data

## Motivation

- **Purpose**: The dataset was created to develop and train a trading agent using Double Deep Q-Network (DDQN) reinforcement learning. The data will be used to model stock market performance and test trading strategies.
- **Creators**: The data was sourced from Yahoo Finance using an add-in tool.
- **Funding**: No specific funding was involved in the creation of this dataset; it was acquired for personal use in building a reinforcement learning model for stock trading.

## Composition

- **Instances**: The dataset contains daily historical data for the SNP 500 from **October 1, 2004**, to **October 4, 2024**.
- **Data Types**: Each instance represents a trading day and includes the following features: close prices, volume, low prices, and high prices.
- **Missing Data**: There may be missing data due to non-trading days (e.g., holidays or weekends), but no missing data is expected for trading days.
- **Confidentiality**: The dataset does not contain confidential information, as it is publicly available market data.

## Collection Process

- **Acquisition**: The data was acquired using the Yahoo Finance add-in tool, which retrieves historical market data.
- **Sampling**: The dataset represents a complete sample of SNP 500 data from October 1, 2024, to October 4, 2024.
- **Time Frame**: The data spans from October 1, 2024, to October 4, 2024, and includes daily information for each trading day.

## Preprocessing/Cleaning/Labeling

- **Preprocessing**: Preprocessing involved calculating financial metrics (e.g., returns, technical indicators such as moving averages, RSI) necessary for reinforcement learning tasks. Additional features may include custom state vectors for the trading agent.
- **Raw Data**: The raw data was saved as downloaded from Yahoo Finance.

## Uses

- **Primary Use**: The dataset is primarily used to develop and train a trading agent using DDQN reinforcement learning to simulate and improve trading strategies in the SNP 500 index.
- **Potential Uses**: It can also be used for time series analysis, financial forecasting, and algorithmic trading.
- **Risks**: The dataset represents historical market performance, which may not reflect future market conditions. Overfitting to historical data could result in suboptimal trading strategies. Additionally, financial models based on this dataset could inadvertently introduce bias.
- **Limitations**: The dataset only includes daily data for a limited period, which may limit the granularity of certain trading strategies or analyses, such as intraday trading.

## Distribution

- **Distribution**: The data was downloaded from Yahoo Finance and is publicly available under Yahoo Finance’s terms of use.
- **Licensing**: Subject to Yahoo Finance's copyright and terms of use, the dataset cannot be redistributed without proper attribution or for commercial purposes without permission.

## Maintenance

- **Maintainer**: The dataset is maintained by Yahoo Finance. Updates to the data can be retrieved by accessing the Yahoo Finance API or using their add-in tool for the most current information.
