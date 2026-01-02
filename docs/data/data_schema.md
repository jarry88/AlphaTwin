# Data Schema & Standards

**Version**: 1.0
**Date**: January 2, 2026
**Status**: Active

## Overview

This document defines the complete data schema, standards, and quality requirements for AlphaTwin's quantitative trading platform. It serves as the authoritative reference for all data structures, validation rules, and processing standards.

## Schema Architecture

### Data Layers

```
Raw Data Layer (Immutable)
├── Source: Yahoo Finance API, CSV files
├── Format: Original API responses, raw CSV
├── Purpose: Preserve original data integrity
└── Storage: data/raw/ directory

Processed Data Layer (Derived)
├── Source: Raw data transformations
├── Format: Enhanced OHLCV with indicators
├── Purpose: Analysis-ready datasets
└── Storage: data/processed/ directory

Analysis Data Layer (Computed)
├── Source: Processed data + calculations
├── Format: Strategy-specific datasets
├── Purpose: Strategy optimization and backtesting
└── Storage: data/analysis/ directory (future)
```

## Core Data Schema

### Raw Data Schema (Immutable)

#### OHLCV Base Structure
All financial time series data follows the OHLCV (Open, High, Low, Close, Volume) standard with additional metadata fields.

```python
# Required fields for all financial instruments
raw_data_schema = {
    # Primary identifiers
    "symbol": {
        "type": "string",
        "description": "Trading symbol identifier",
        "examples": ["EURUSD=X", "AAPL", "BTC-USD"],
        "constraints": {
            "required": True,
            "min_length": 1,
            "max_length": 20,
            "pattern": r"^[A-Z0-9\-=\.]+$"
        }
    },

    # Temporal fields
    "timestamp": {
        "type": "datetime64[ns, UTC]",
        "description": "UTC timestamp of data point",
        "constraints": {
            "required": True,
            "timezone": "UTC",
            "resolution": "1 minute minimum"
        }
    },

    # Price fields (decimal precision varies by asset class)
    "open": {
        "type": "float64",
        "description": "Opening price for the period",
        "constraints": {
            "required": True,
            "min_value": 0.000001,  # Minimum tick size
            "decimal_places": {
                "forex": 5,      # EURUSD: 1.23456
                "stocks": 2,     # AAPL: 123.45
                "crypto": 8      # BTC: 12345.12345678
            }
        }
    },

    "high": {
        "type": "float64",
        "description": "Highest price during the period",
        "constraints": {
            "required": True,
            "min_value": "open",
            "relationship": "high >= max(open, close)"
        }
    },

    "low": {
        "type": "float64",
        "description": "Lowest price during the period",
        "constraints": {
            "required": True,
            "max_value": "open",
            "relationship": "low <= min(open, close)"
        }
    },

    "close": {
        "type": "float64",
        "description": "Closing price for the period",
        "constraints": {
            "required": True,
            "bounds": ["low", "high"]
        }
    },

    "adj_close": {
        "type": "float64",
        "description": "Adjusted closing price (dividends/splits)",
        "constraints": {
            "required": False,  # May not be available for all sources
            "relationship": "Similar to close, adjusted for corporate actions"
        }
    },

    # Volume field
    "volume": {
        "type": "int64",
        "description": "Trading volume for the period",
        "constraints": {
            "required": True,
            "min_value": 0,
            "max_value": {
                "forex": 1000000000,   # 1B max reasonable
                "stocks": 100000000,  # 100M max reasonable
                "crypto": None         # No upper limit for crypto
            }
        }
    }
}
```

#### Metadata Fields (Automatically Added)
```python
metadata_schema = {
    "data_source": {
        "type": "string",
        "description": "Data provider identifier",
        "examples": ["yahoo_finance", "csv_import", "api_feed"],
        "constraints": {
            "required": True,
            "enum": ["yahoo_finance", "bloomberg", "refinitiv", "csv_import"]
        }
    },

    "ingestion_timestamp": {
        "type": "datetime64[ns, UTC]",
        "description": "When data was ingested into system",
        "constraints": {
            "required": True,
            "auto_generated": True
        }
    },

    "data_quality_flags": {
        "type": "object",
        "description": "Quality assessment results",
        "properties": {
            "completeness_score": {"type": "float64", "range": [0, 1]},
            "consistency_check": {"type": "boolean"},
            "outlier_detected": {"type": "boolean"},
            "validation_errors": {"type": "array", "items": {"type": "string"}}
        }
    },

    "processing_version": {
        "type": "string",
        "description": "Data processing pipeline version",
        "examples": ["v1.0.0", "v1.1.2"],
        "constraints": {
            "required": True,
            "pattern": r"^v\d+\.\d+\.\d+$"
        }
    }
}
```

### Processed Data Schema (Analysis-Ready)

#### Return Calculations
```python
returns_schema = {
    "returns": {
        "type": "float64",
        "description": "Arithmetic daily returns: (close - prev_close) / prev_close",
        "constraints": {
            "range": [-0.5, 0.5],  # Reasonable daily return bounds
            "allow_null": True     # First period will be null
        }
    },

    "log_returns": {
        "type": "float64",
        "description": "Logarithmic returns: ln(close / prev_close)",
        "constraints": {
            "range": [-1.0, 1.0],  # Reasonable log return bounds
            "allow_null": True
        }
    },

    "cumulative_returns": {
        "type": "float64",
        "description": "Cumulative returns from start date",
        "constraints": {
            "min_value": -1.0,    # Can't lose more than 100%
            "calculation": "(1 + returns).cumprod() - 1"
        }
    }
}
```

#### Technical Indicators
```python
indicators_schema = {
    # Moving Averages
    "sma_20": {
        "type": "float64",
        "description": "20-period Simple Moving Average",
        "formula": "close.rolling(window=20).mean()"
    },

    "sma_50": {
        "type": "float64",
        "description": "50-period Simple Moving Average",
        "formula": "close.rolling(window=50).mean()"
    },

    "ema_20": {
        "type": "float64",
        "description": "20-period Exponential Moving Average",
        "formula": "close.ewm(span=20).mean()"
    },

    # Momentum Indicators
    "rsi_14": {
        "type": "float64",
        "description": "14-period Relative Strength Index",
        "range": [0, 100],
        "formula": "Complex RSI calculation with gains/losses"
    },

    "macd": {
        "type": "float64",
        "description": "MACD line: EMA12 - EMA26",
        "formula": "ema_12 - ema_26"
    },

    "macd_signal": {
        "type": "float64",
        "description": "MACD signal line: EMA9 of MACD",
        "formula": "macd.ewm(span=9).mean()"
    },

    "macd_histogram": {
        "type": "float64",
        "description": "MACD histogram: MACD - signal",
        "formula": "macd - macd_signal"
    },

    # Volatility Indicators
    "bb_upper": {
        "type": "float64",
        "description": "Bollinger Band upper: SMA20 + (2 * std20)",
        "formula": "sma_20 + (close.rolling(20).std() * 2)"
    },

    "bb_lower": {
        "type": "float64",
        "description": "Bollinger Band lower: SMA20 - (2 * std20)",
        "formula": "sma_20 - (close.rolling(20).std() * 2)"
    },

    "bb_middle": {
        "type": "float64",
        "description": "Bollinger Band middle (same as SMA20)",
        "formula": "sma_20"
    },

    "bb_width": {
        "type": "float64",
        "description": "Bollinger Band width: (upper - lower) / middle",
        "formula": "(bb_upper - bb_lower) / bb_middle"
    },

    # Volatility Measures
    "volatility_20": {
        "type": "float64",
        "description": "20-day rolling volatility (standard deviation of returns)",
        "formula": "returns.rolling(20).std() * sqrt(252)"
    },

    "parkinson_vol": {
        "type": "float64",
        "description": "Parkinson volatility estimator using OHLC",
        "formula": "sqrt(1/(4*N*ln(2)) * sum(ln(H/L)^2))"
    },

    # Volume Indicators
    "volume_sma_20": {
        "type": "float64",
        "description": "20-period volume moving average",
        "formula": "volume.rolling(20).mean()"
    },

    "volume_ratio": {
        "type": "float64",
        "description": "Volume ratio: volume / volume_sma_20",
        "formula": "volume / volume_sma_20"
    },

    "obv": {
        "type": "float64",
        "description": "On Balance Volume cumulative indicator",
        "formula": "Complex OBV calculation based on price direction"
    }
}
```

#### Quality and Validation Fields
```python
quality_schema = {
    "data_quality_score": {
        "type": "float64",
        "description": "Overall data quality score (0-1)",
        "calculation": "Weighted average of multiple quality metrics",
        "range": [0, 1]
    },

    "completeness_score": {
        "type": "float64",
        "description": "Data completeness percentage",
        "calculation": "Percentage of non-null values",
        "range": [0, 1]
    },

    "has_missing_values": {
        "type": "boolean",
        "description": "Flag indicating missing values in row",
        "calculation": "Any null values in critical fields"
    },

    "has_outliers": {
        "type": "boolean",
        "description": "Flag indicating statistical outliers detected",
        "calculation": "Beyond 3 standard deviations from mean"
    },

    "validation_errors": {
        "type": "array",
        "description": "List of validation error messages",
        "items": {"type": "string"},
        "max_items": 10
    },

    "processing_timestamp": {
        "type": "datetime64[ns, UTC]",
        "description": "When this data row was processed",
        "auto_generated": True
    }
}
```

## Data Quality Standards

### Completeness Requirements

| Field Category | Required Completeness | Notes |
|----------------|----------------------|-------|
| OHLC Prices | >99.8% | Critical for all analysis |
| Volume | >95% | Can be estimated if missing |
| Adjusted Close | >90% | May not be available for all assets |
| Technical Indicators | >95% | Calculated fields, should be complete |
| Quality Flags | 100% | System-generated, always present |

### Accuracy Standards

#### Price Precision Requirements
- **Forex**: Minimum 4 decimal places (e.g., 1.2345)
- **Stocks**: Minimum 2 decimal places (e.g., 123.45)
- **Crypto**: Minimum 2 decimal places, up to 8 for high-precision

#### Time Accuracy Requirements
- **Resolution**: Minimum 1-minute intervals
- **Timezone**: All timestamps must be UTC
- **Continuity**: No gaps > 5 trading days without documentation

#### Value Range Validation
```python
# Automatic validation rules
validation_rules = {
    "price_positive": "all_prices > 0",
    "high_low_relationship": "high >= low",
    "ohlc_bounds": "low <= open <= high and low <= close <= high",
    "volume_non_negative": "volume >= 0",
    "returns_reasonable": "abs(returns) < 0.5",  # Max 50% daily change
    "volatility_bounds": "0 <= volatility <= 1.0"  # 0-100% annual vol
}
```

### Consistency Rules

#### OHLC Relationship Validation
```python
def validate_ohlc_relationships(data):
    """Validate OHLC price relationships"""
    violations = []

    # High should be >= max(open, close)
    high_violations = data[data['high'] < data[['open', 'close']].max(axis=1)]
    if len(high_violations) > 0:
        violations.append(f"High price violations: {len(high_violations)}")

    # Low should be <= min(open, close)
    low_violations = data[data['low'] > data[['open', 'close']].min(axis=1)]
    if len(low_violations) > 0:
        violations.append(f"Low price violations: {len(low_violations)}")

    return violations
```

#### Time Series Continuity
```python
def validate_time_series_continuity(data, max_gap_days=5):
    """Validate time series continuity"""
    data = data.sort_index()
    time_diffs = data.index.to_series().diff()

    # Find gaps larger than threshold
    large_gaps = time_diffs[time_diffs > pd.Timedelta(days=max_gap_days)]

    if len(large_gaps) > 0:
        return f"Found {len(large_gaps)} gaps > {max_gap_days} days"
    else:
        return "Time series continuity validated"
```

## Data Storage Standards

### File Formats

#### Raw Data Storage
- **Format**: CSV with metadata headers
- **Compression**: gzip (.csv.gz)
- **Naming**: `{symbol}_raw_{date}.csv.gz`
- **Retention**: Keep indefinitely (immutable)

#### Processed Data Storage
- **Format**: Parquet for efficient columnar access
- **Compression**: Snappy (balance speed/size)
- **Naming**: `{symbol}_processed_{date}.parquet`
- **Partitioning**: By symbol and date ranges

#### Analysis Data Storage
- **Format**: HDF5 for complex datasets
- **Compression**: Enabled for storage efficiency
- **Naming**: `{strategy}_{symbol}_analysis_{date}.h5`
- **Indexing**: Multi-level indexing for fast queries

### Directory Structure
```
data/
├── raw/                    # Immutable raw data
│   ├── forex/             # EURUSD=X_raw_20240101.csv.gz
│   ├── stocks/            # AAPL_raw_20240101.csv.gz
│   └── crypto/            # BTC_raw_20240101.csv.gz
├── processed/             # Analysis-ready data
│   ├── forex/             # EURUSD=X_processed_20240101.parquet
│   ├── stocks/            # AAPL_processed_20240101.parquet
│   └── crypto/            # BTC_processed_20240101.parquet
└── analysis/              # Strategy-specific datasets (future)
    ├── backtests/         # Backtesting results
    ├── features/          # ML feature sets
    └── models/            # Model artifacts
```

## Schema Evolution

### Version Control
- **Schema Versions**: Semantic versioning (major.minor.patch)
- **Backward Compatibility**: New fields are additive only
- **Breaking Changes**: Require major version bump
- **Migration Scripts**: Automated data migration between versions

### Change Management
```python
# Example schema change tracking
schema_changes = {
    "v1.0.0": {
        "description": "Initial OHLCV schema",
        "fields_added": ["symbol", "timestamp", "open", "high", "low", "close", "volume"],
        "migration_required": False
    },

    "v1.1.0": {
        "description": "Added technical indicators",
        "fields_added": ["sma_20", "rsi_14", "macd", "bb_upper", "bb_lower"],
        "migration_required": True,
        "migration_script": "migrate_v1_0_to_v1_1.py"
    },

    "v1.2.0": {
        "description": "Added quality metrics",
        "fields_added": ["data_quality_score", "validation_errors"],
        "migration_required": False  # New fields with defaults
    }
}
```

### Validation Rules Evolution
- **New Rules**: Can be added without breaking existing data
- **Rule Changes**: Require careful testing and potential reprocessing
- **Deprecation**: Old rules marked deprecated before removal

## Quality Monitoring

### Automated Quality Checks

#### Daily Quality Report
```python
def generate_quality_report(data):
    """Generate comprehensive data quality report"""
    report = {
        "summary": {
            "total_records": len(data),
            "date_range": f"{data.index.min()} to {data.index.max()}",
            "symbols_covered": data['symbol'].nunique()
        },

        "completeness": {
            "overall_score": data['data_quality_score'].mean(),
            "fields_completeness": (1 - data.isnull().mean()).to_dict(),
            "records_with_issues": (data['data_quality_score'] < 0.95).sum()
        },

        "accuracy": {
            "ohlc_violations": validate_ohlc_relationships(data),
            "time_series_gaps": validate_time_series_continuity(data),
            "outlier_percentage": (data['has_outliers'].sum() / len(data)) * 100
        },

        "consistency": {
            "duplicate_timestamps": data.index.duplicated().sum(),
            "non_chronological": (~data.index.is_monotonic_increasing).sum(),
            "timezone_consistency": (data.index.tz.zone == 'UTC').all()
        }
    }

    return report
```

#### Quality Thresholds
```python
quality_thresholds = {
    "critical": {
        "completeness": 0.998,      # 99.8% complete
        "accuracy": 0.999,          # 99.9% accurate
        "consistency": 1.0          # 100% consistent
    },

    "warning": {
        "completeness": 0.995,      # 99.5% complete
        "accuracy": 0.995,          # 99.5% accurate
        "consistency": 0.999        # 99.9% consistent
    },

    "acceptable": {
        "completeness": 0.99,       # 99% complete
        "accuracy": 0.99,           # 99% accurate
        "consistency": 0.995        # 99.5% consistent
    }
}
```

## Implementation Guidelines

### Data Processing Pipeline

#### Standard Processing Flow
1. **Ingest**: Load raw data from sources
2. **Validate**: Check data integrity and completeness
3. **Clean**: Handle missing values and outliers
4. **Enrich**: Add technical indicators and derived fields
5. **Quality Check**: Validate final dataset quality
6. **Store**: Save in appropriate format with metadata

#### Error Handling Strategy
- **Validation Errors**: Log and attempt correction
- **Processing Errors**: Fail fast with detailed error messages
- **Storage Errors**: Retry with fallback mechanisms
- **Quality Failures**: Flag data but don't prevent storage

### Performance Considerations

#### Memory Management
- **Chunked Processing**: For large datasets (>100MB)
- **Garbage Collection**: Explicit cleanup after processing
- **Data Types**: Use appropriate dtypes to minimize memory usage

#### Computational Efficiency
- **Vectorized Operations**: Prefer pandas/numpy over loops
- **Parallel Processing**: Use dask for CPU-intensive operations
- **Caching**: Cache frequently used calculations

---

*This schema provides the foundation for reliable, consistent, and high-quality financial data processing across all AlphaTwin components.*
