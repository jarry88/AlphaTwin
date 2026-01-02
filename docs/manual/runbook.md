# Operations Manual (Runbook)

**Version**: 1.0
**Last Updated**: January 2, 2026
**Environment**: Development

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Data Pipeline Operations](#data-pipeline-operations)
4. [Strategy Development](#strategy-development)
5. [Backtesting Procedures](#backtesting-procedures)
6. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
7. [Maintenance Tasks](#maintenance-tasks)
8. [Emergency Procedures](#emergency-procedures)

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git repository cloned
- Python 3.13 environment (optional, Docker handles this)

### Launch Sequence

1. **Start Documentation Portal**:
   ```bash
   docker-compose up -d docs
   ```
   Access at: http://localhost:8001

2. **Start Development Environment**:
   ```bash
   docker-compose up -d app
   ```
   Jupyter Lab at: http://localhost:8888

3. **Verify Services**:
   ```bash
   docker-compose ps
   ```

### First Backtest Execution

```bash
# Enter the app container
docker-compose exec app bash

# Run a sample backtest
cd /app
python -c "
from src.data_loader import DataLoader
from src.signals import SignalManager, MovingAverageCrossover
from src.backtest_engine import BacktestEngine

# Download sample data
loader = DataLoader()
data = loader.download_stock_data('EURUSD=X', '2022-01-01', '2024-01-01')
processed = loader.process_data(data)

# Generate signals
manager = SignalManager()
manager.add_generator('MA_Crossover', MovingAverageCrossover())
signals = manager.generate_all_signals(processed)['MA_Crossover']

# Run backtest
engine = BacktestEngine(initial_capital=10000)
result = engine.run_backtest(processed, signals)
engine.print_summary(result)
"
```

## System Architecture

### Container Layout

```
AlphaTwin System
├── docs (MkDocs Portal - Port 8001)
│   ├── Material Theme + Dark Mode
│   ├── Mermaid.js Diagrams
│   └── Comprehensive Documentation
│
├── app (Quantitative Engine - Port 8888)
│   ├── Jupyter Lab Environment
│   ├── Core Python Modules
│   └── Data Processing Pipeline
│
└── data (Persistent Storage)
    ├── raw/ (Original Market Data)
    └── processed/ (Clean Datasets)
```

### Component Dependencies

- **docs container**: Depends on MkDocs configuration and markdown files
- **app container**: Depends on Python modules and data directories
- **Data persistence**: Mounted volumes ensure data survives container restarts

## Data Pipeline Operations

### Daily Data Update Procedure

#### Automated Update
```bash
# Run data update script (to be implemented)
docker-compose exec app python -c "
from src.data_loader import DataLoader
loader = DataLoader()
data = loader.download_stock_data('EURUSD=X')
processed = loader.process_data(data)
loader.save_raw_data(data, 'EURUSD=X')
loader.save_processed_data(processed, 'EURUSD=X')
print('Data update completed')
"
```

#### Manual Data Loading
```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader()

# Download specific date range
data = loader.download_stock_data(
    symbol='EURUSD=X',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Process and save
processed = loader.process_data(data)
loader.save_raw_data(data, 'EURUSD=X')
loader.save_processed_data(processed, 'EURUSD=X')
```

### Data Quality Checks

#### Completeness Verification
```python
import pandas as pd

# Load processed data
data = pd.read_csv('data/processed/EURUSD=X_processed.csv', index_col=0, parse_dates=True)

# Check for missing values
missing_count = data.isnull().sum().sum()
print(f'Missing values: {missing_count}')

# Check date continuity
date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
missing_dates = date_range.difference(data.index)
print(f'Missing trading days: {len(missing_dates)}')
```

#### Data Integrity Validation
```python
# Verify OHLC relationships
invalid_bars = data[(data['High'] < data['Low']) |
                   (data['High'] < data['Open']) |
                   (data['High'] < data['Close']) |
                   (data['Low'] > data['Open']) |
                   (data['Low'] > data['Close'])]

if len(invalid_bars) > 0:
    print(f'Invalid price bars: {len(invalid_bars)}')
else:
    print('Data integrity check passed')
```

## Strategy Development

### Adding New Strategies

1. **Create Strategy Class**:
```python
from src.signals import SignalGenerator

class MyCustomStrategy(SignalGenerator):
    def __init__(self, parameter1=20, parameter2=50):
        self.parameter1 = parameter1
        self.parameter2 = parameter2

    def generate_signals(self, data):
        # Implement your strategy logic
        signals = pd.Series(0, index=data.index)

        # Your signal generation code here
        # signals[condition] = 1  # Buy
        # signals[condition] = -1  # Sell

        return signals
```

2. **Register Strategy**:
```python
from src.signals import SignalManager
from my_custom_strategy import MyCustomStrategy

manager = SignalManager()
manager.add_generator('My_Strategy', MyCustomStrategy(param1=10, param2=30))
```

3. **Test Strategy**:
```python
# Generate signals
signals = manager.generate_all_signals(data)['My_Strategy']

# Quick validation
buy_signals = (signals == 1).sum()
sell_signals = (signals == -1).sum()
hold_signals = (signals == 0).sum()

print(f'Buy signals: {buy_signals}')
print(f'Sell signals: {sell_signals}')
print(f'Hold signals: {hold_signals}')
```

### Strategy Parameter Optimization

```python
from src.backtest_engine import BacktestEngine, StrategyComparator

# Define parameter ranges
short_windows = [10, 20, 30]
long_windows = [30, 50, 100]

results = []
for short in short_windows:
    for long in long_windows:
        if short >= long:
            continue

        # Create strategy
        strategy = MovingAverageCrossover(short_window=short, long_window=long)

        # Generate signals
        signals = strategy.generate_signals(data)

        # Run backtest
        engine = BacktestEngine()
        result = engine.run_backtest(data, signals)

        results.append({
            'short': short,
            'long': long,
            'sharpe': result.performance_metrics['sharpe_ratio'],
            'return': result.performance_metrics['total_return']
        })

# Find best parameters
best_result = max(results, key=lambda x: x['sharpe'])
print(f'Best parameters: Short={best_result["short"]}, Long={best_result["long"]}')
print(f'Sharpe Ratio: {best_result["sharpe"]:.2f}')
```

## Backtesting Procedures

### Standard Backtest Execution

```python
from src.backtest_engine import BacktestEngine

# Initialize engine
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001  # 0.1% per trade
)

# Run backtest
result = engine.run_backtest(data, signals, position_size=0.5)

# Display results
engine.print_summary(result)

# Generate charts
engine.plot_results(result, save_path='backtest_results.png')
```

### Multi-Strategy Comparison

```python
from src.backtest_engine import StrategyComparator

# Initialize comparator
comparator = StrategyComparator()

# Test multiple strategies
strategies = {
    'MA_Crossover': MovingAverageCrossover(),
    'RSI': RSI(),
    'Momentum': Momentum()
}

for name, strategy in strategies.items():
    signals = strategy.generate_signals(data)
    result = engine.run_backtest(data, signals)
    comparator.add_result(name, result)

# Compare performance
comparison_df = comparator.compare_performance()
print(comparison_df)

# Plot comparison
comparator.plot_comparison('sharpe_ratio', save_path='strategy_comparison.png')
```

### Walk-Forward Analysis

```python
# Define rolling window parameters
window_size = 252  # 1 year of trading days
step_size = 21     # 1 month step

results = []
for i in range(window_size, len(data), step_size):
    # Define training and testing periods
    train_start = i - window_size
    train_end = i - 1
    test_end = min(i + step_size - 1, len(data) - 1)

    # Training data
    train_data = data.iloc[train_start:train_end]

    # Optimize parameters on training data
    # (parameter optimization code here)

    # Test on out-of-sample data
    test_data = data.iloc[i:test_end+1]
    test_signals = strategy.generate_signals(test_data)
    test_result = engine.run_backtest(test_data, test_signals)

    results.append(test_result.performance_metrics)

# Aggregate walk-forward results
avg_sharpe = sum(r['sharpe_ratio'] for r in results) / len(results)
avg_return = sum(r['total_return'] for r in results) / len(results)

print(f'Walk-forward Sharpe: {avg_sharpe:.2f}')
print(f'Walk-forward Return: {avg_return:.2%}')
```

## Monitoring & Troubleshooting

### Health Checks

#### Container Status
```bash
# Check running containers
docker-compose ps

# View container logs
docker-compose logs docs
docker-compose logs app

# Check resource usage
docker stats
```

#### Application Health
```python
# Test data pipeline
try:
    from src.data_loader import DataLoader
    loader = DataLoader()
    test_data = loader.download_stock_data('EURUSD=X', '2024-01-01', '2024-01-02')
    print("✅ Data pipeline operational")
except Exception as e:
    print(f"❌ Data pipeline error: {e}")

# Test strategy generation
try:
    from src.signals import MovingAverageCrossover
    strategy = MovingAverageCrossover()
    signals = strategy.generate_signals(test_data)
    print("✅ Strategy generation operational")
except Exception as e:
    print(f"❌ Strategy error: {e}")

# Test backtesting
try:
    from src.backtest_engine import BacktestEngine
    engine = BacktestEngine(initial_capital=1000)
    result = engine.run_backtest(test_data, signals)
    print("✅ Backtesting operational")
except Exception as e:
    print(f"❌ Backtesting error: {e}")
```

### Common Issues & Solutions

#### Issue: Port Already in Use
**Symptoms**: Container fails to start with port binding error
**Solution**:
```bash
# Find process using port
lsof -i :8001

# Kill process or change port in docker-compose.yml
# Then restart
docker-compose down
docker-compose up -d
```

#### Issue: Out of Memory
**Symptoms**: Container crashes during large data processing
**Solution**:
```bash
# Increase Docker memory limit in Docker Desktop
# Or process data in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

#### Issue: API Rate Limiting
**Symptoms**: Yahoo Finance API returns errors
**Solution**:
```python
import time
import random

# Implement exponential backoff
def api_call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = (2 ** attempt) + random.random()
            time.sleep(wait_time)
```

#### Issue: Data Quality Problems
**Symptoms**: Invalid OHLC data or missing values
**Solution**:
```python
# Data validation function
def validate_ohlc_data(data):
    issues = []

    # Check OHLC relationships
    invalid_high = data[data['High'] < data[['Open', 'Close', 'Low']].max(axis=1)]
    if len(invalid_high) > 0:
        issues.append(f"Invalid High prices: {len(invalid_high)} bars")

    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")

    return issues

issues = validate_ohlc_data(data)
if issues:
    print("Data quality issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Data quality check passed")
```

### Performance Monitoring

#### System Resources
```bash
# Monitor container resources
docker stats alphatwin-docs alphatwin-app

# Check disk usage
df -h data/
du -sh data/raw/ data/processed/
```

#### Application Performance
```python
import time
import psutil

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory

        print(f"Execution time: {execution_time:.2f}s")
        print(f"Memory usage: {memory_usage / 1024 / 1024:.1f} MB")

        return result
    return wrapper

@monitor_performance
def run_backtest(data, signals):
    engine = BacktestEngine()
    return engine.run_backtest(data, signals)
```

## Maintenance Tasks

### Weekly Tasks

#### Data Backup
```bash
# Create timestamped backup
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_data_${timestamp}.tar.gz" data/

# Clean old backups (keep last 4 weeks)
ls backup_data_*.tar.gz | head -n -4 | xargs rm -f
```

#### Log Rotation
```bash
# Rotate application logs
find . -name "*.log" -mtime +7 -delete

# Archive old data (optional)
find data/raw/ -name "*.csv" -mtime +90 -exec mv {} data/archive/ \;
```

#### Dependency Updates
```bash
# Update Python packages
pip list --outdated
pip install --upgrade -r requirements.txt

# Rebuild containers with updates
docker-compose build --no-cache
```

### Monthly Tasks

#### Performance Review
- Analyze backtest results across different market conditions
- Review strategy performance metrics
- Identify underperforming strategies for optimization

#### Code Quality Check
```bash
# Run static analysis
python -m pylint src/ --reports=y

# Check test coverage
python -m pytest --cov=src/ --cov-report=html
```

#### Documentation Update
- Update API documentation
- Review and update runbook procedures
- Archive completed development logs

## Emergency Procedures

### System Down
1. **Check container status**: `docker-compose ps`
2. **View error logs**: `docker-compose logs`
3. **Restart services**: `docker-compose restart`
4. **Full rebuild if needed**: `docker-compose down && docker-compose up -d`

### Data Corruption
1. **Stop all services**: `docker-compose down`
2. **Restore from backup**: `tar -xzf backup_data_latest.tar.gz`
3. **Validate restored data**: Run data quality checks
4. **Restart services**: `docker-compose up -d`

### API Outage
1. **Check API status**: Visit Yahoo Finance website
2. **Implement fallback**: Use cached data if available
3. **Wait for resolution**: Monitor API status
4. **Resume normal operations**: When API is restored

### Security Incident
1. **Isolate system**: Disconnect from network if compromised
2. **Assess damage**: Check for unauthorized access/data exfiltration
3. **Restore from clean backup**: Use known-good backup
4. **Update security**: Patch vulnerabilities, change credentials
5. **Resume operations**: With enhanced monitoring

## Contact Information

### Development Team
- **Lead Developer**: AlphaTwin Team
- **Documentation**: docs/index.md
- **Repository**: https://github.com/jarry88/AlphaTwin

### Emergency Contacts
- **System Issues**: Check logs first, then escalate to development team
- **Data Issues**: Validate data integrity, restore from backup if needed
- **Performance Issues**: Monitor resources, optimize code as needed

---

*This runbook provides comprehensive operational procedures for the AlphaTwin system. Regular review and updates are recommended to maintain operational excellence.*
