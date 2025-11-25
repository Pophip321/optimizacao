# Overview

This is a **Portfolio Optimization application** built with Streamlit that helps users analyze and optimize stock portfolios using Modern Portfolio Theory. The application downloads historical stock price data from Yahoo Finance for Brazilian stocks (B3 exchange), calculates the efficient frontier, and provides three core optimal asset allocation strategies plus optional user portfolio analysis. Users can visualize risk-return tradeoffs and explore different portfolio configurations.

# User Preferences

Preferred communication style: Simple, everyday language.

# Portfolio Strategies

## Implemented Strategies

The application implements three core portfolio optimization strategies:

1. **Maximum Sharpe Ratio**: Maximizes the risk-adjusted return (return/volatility ratio)
   - Best for investors seeking optimal risk-return balance
   - Calculated using constrained optimization to maximize Sharpe ratio
   
2. **Minimum Variance**: Minimizes portfolio volatility/risk
   - Best for conservative investors prioritizing capital preservation
   - Finds the lowest-risk combination of assets
   
3. **Target Return**: Minimizes risk while achieving a specific target return
   - User-defined monthly return target
   - Finds minimum-risk portfolio that meets the return objective

4. **User Custom Portfolio** (optional): Analyzes user-provided portfolio weights
   - Allows benchmarking personal allocations against optimal strategies
   - Shows how current portfolio compares in terms of risk, return, and Sharpe ratio

## Removed Features (Optimized for Simplicity)

The following strategies and features were removed to streamline the application:
- Monte Carlo simulation (previously used for visualization)
- DR1 (Diversification Ratio maximization)
- CR1 (Correlation minimization)
- CR1-RET (Return/Correlation ratio maximization)
- DR1-RET (Return/Diversification ratio maximization)
- ERC (Equal Risk Contribution)
- Maximum/Minimum Return portfolios

**Rationale**: Focus on the three most practical and widely-used strategies (Sharpe, Variance, Target) for cleaner user experience and faster calculations.

# System Architecture

## Frontend Architecture

**Technology**: Streamlit web framework
- **Rationale**: Streamlit provides rapid development of data-driven web applications with minimal frontend code, perfect for financial analysis tools
- **Key Features**:
  - Interactive sidebar for user inputs (date ranges, stock tickers, calculation preferences)
  - Wide layout configuration for data visualization
  - Real-time parameter adjustment without page reloads
- **Pros**: Fast prototyping, Python-native, built-in widgets
- **Cons**: Limited customization compared to traditional web frameworks

## Visualization Layer

**Technology**: Plotly (graph_objects and express modules)
- **Rationale**: Chosen over Matplotlib for interactive, publication-quality financial charts
- **Purpose**: Rendering efficient frontier plots, risk-return scatter plots, and portfolio composition visualizations
- **Alternatives Considered**: Matplotlib (static only), Altair (less feature-rich for financial data)

## Data Processing Pipeline

**Architecture Pattern**: Sequential data transformation pipeline

1. **Data Acquisition**: Yahoo Finance API (yfinance library)
   - Fetches historical OHLC data for Brazilian stocks (B3 exchange with .SA suffix)
   - Supports adjusted prices (dividends, splits, bonuses)
   
2. **Returns Calculation**: NumPy/Pandas
   - Choice between simple returns or log returns (user preference)
   - Handles missing data and market holidays
   
3. **Portfolio Optimization**: SciPy optimization engine
   - Implements Markowitz Mean-Variance Optimization
   - Uses `scipy.optimize.minimize` for constrained optimization
   - Calculates efficient frontier through multiple portfolio combinations

**Design Decision**: In-memory processing only
- **Rationale**: Portfolio analysis is ephemeral - no need to persist historical calculations
- **Pros**: Simplicity, no database overhead
- **Cons**: Re-fetches data on each session (mitigated by yfinance caching)

## Asset Validation

**Input Processing**: Manual ticker entry with validation
- Users input tickers without .SA suffix for usability
- System validates ticker existence via test API calls to Yahoo Finance
- Filters invalid/delisted tickers before analysis

# External Dependencies

## Financial Data Provider

**Yahoo Finance API** (via yfinance Python library)
- **Purpose**: Primary source for Brazilian stock market data (B3 exchange)
- **Data Retrieved**: Historical OHLC prices, adjusted prices, volume
- **Integration Method**: Python library wrapper around Yahoo Finance API
- **Rate Limits**: Subject to Yahoo Finance fair use policy
- **Ticker Format**: Requires .SA suffix for Brazilian stocks (e.g., PETR4.SA)

## Computational Libraries

**SciPy** - Scientific computing and optimization
- **Usage**: `scipy.optimize.minimize` for portfolio weight optimization
- **Algorithm**: Constrained optimization for efficient frontier calculation

**NumPy** - Numerical computing
- **Usage**: Matrix operations, statistical calculations, returns computation

**Pandas** - Data manipulation
- **Usage**: Time series handling, data cleaning, covariance matrix calculation

## Deployment Platform

**Streamlit Cloud** (implied)
- **Purpose**: Hosting and serving the web application
- **Configuration**: `streamlit` as primary framework dependency

## Python Version Requirements

- Compatible with Python 3.7+ (required for type hints and modern syntax)
- No database system required (stateless application)