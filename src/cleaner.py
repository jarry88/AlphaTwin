"""
Data Cleaning Module for AlphaTwin

Provides comprehensive data cleaning, validation, and quality assurance
for financial time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality assessment results."""
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    overall_score: float
    issues_found: List[str]
    recommendations: List[str]
    metadata: Dict[str, any]


class DataValidator:
    """Validates raw financial data integrity and consistency."""

    def __init__(self):
        self.validation_rules = {
            'price_positive': self._validate_positive_prices,
            'ohlc_relationships': self._validate_ohlc_relationships,
            'volume_non_negative': self._validate_volume_non_negative,
            'timestamp_chronological': self._validate_timestamp_order,
            'price_reasonable_range': self._validate_price_ranges,
            'volume_reasonable_range': self._validate_volume_ranges
        }

    def validate_dataset(self, data: pd.DataFrame, symbol: str = None) -> DataQualityReport:
        """
        Perform comprehensive validation on financial dataset.

        Args:
            data: OHLCV DataFrame to validate
            symbol: Trading symbol for context

        Returns:
            DataQualityReport with validation results
        """
        issues = []
        scores = {}

        # Run all validation checks
        for rule_name, validator_func in self.validation_rules.items():
            try:
                score, rule_issues = validator_func(data)
                scores[rule_name] = score
                issues.extend(rule_issues)
            except Exception as e:
                logger.warning(f"Validation rule {rule_name} failed: {e}")
                scores[rule_name] = 0.0
                issues.append(f"Validation error in {rule_name}: {str(e)}")

        # Calculate composite scores
        completeness_score = self._calculate_completeness_score(data)
        accuracy_score = np.mean(list(scores.values()))
        consistency_score = self._calculate_consistency_score(data)

        # Overall quality score (weighted average)
        weights = {'completeness': 0.4, 'accuracy': 0.4, 'consistency': 0.2}
        overall_score = (
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency']
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(scores, issues)

        return DataQualityReport(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
            issues_found=issues,
            recommendations=recommendations,
            metadata={
                'symbol': symbol,
                'record_count': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}" if len(data) > 0 else None,
                'validation_timestamp': datetime.utcnow().isoformat(),
                'validation_rules_applied': list(scores.keys())
            }
        )

    def _validate_positive_prices(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate that all price fields are positive."""
        issues = []
        price_cols = ['open', 'high', 'low', 'close']

        if not all(col in data.columns for col in price_cols):
            return 0.0, ["Required price columns missing"]

        negative_prices = 0
        for col in price_cols:
            if col in data.columns:
                neg_count = (data[col] <= 0).sum()
                negative_prices += neg_count
                if neg_count > 0:
                    issues.append(f"Found {neg_count} negative/zero values in {col}")

        score = 1.0 - (negative_prices / len(data)) if len(data) > 0 else 0.0
        return max(0.0, score), issues

    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate OHLC price relationships."""
        issues = []
        violations = 0

        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return 0.0, ["OHLC columns missing for relationship validation"]

        # High should be >= max(open, close)
        high_violations = data[data['high'] < data[['open', 'close']].max(axis=1)]
        if len(high_violations) > 0:
            violations += len(high_violations)
            issues.append(f"High price violations: {len(high_violations)} records")

        # Low should be <= min(open, close)
        low_violations = data[data['low'] > data[['open', 'close']].min(axis=1)]
        if len(low_violations) > 0:
            violations += len(low_violations)
            issues.append(f"Low price violations: {len(low_violations)} records")

        # Open and Close should be within [low, high] bounds
        bounds_violations = data[
            (data['open'] < data['low']) | (data['open'] > data['high']) |
            (data['close'] < data['low']) | (data['close'] > data['high'])
        ]
        if len(bounds_violations) > 0:
            violations += len(bounds_violations)
            issues.append(f"OHLC bounds violations: {len(bounds_violations)} records")

        score = 1.0 - (violations / len(data)) if len(data) > 0 else 0.0
        return max(0.0, score), issues

    def _validate_volume_non_negative(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate that volume is non-negative."""
        issues = []

        if 'volume' not in data.columns:
            return 0.0, ["Volume column missing"]

        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"Found {negative_volume} negative volume values")

        score = 1.0 - (negative_volume / len(data)) if len(data) > 0 else 0.0
        return max(0.0, score), issues

    def _validate_timestamp_order(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate that timestamps are in chronological order."""
        issues = []

        if len(data) < 2:
            return 1.0, []

        # Check for chronological order
        non_chronological = (~data.index.is_monotonic_increasing).sum()
        if non_chronological > 0:
            issues.append(f"Found {non_chronological} non-chronological timestamps")

        # Check for duplicates
        duplicates = data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")

        score = 1.0 - ((non_chronological + duplicates) / len(data)) if len(data) > 0 else 0.0
        return max(0.0, score), issues

    def _validate_price_ranges(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate that prices are within reasonable ranges."""
        issues = []
        outliers = 0

        if 'close' not in data.columns:
            return 0.0, ["Close price column missing for range validation"]

        # Define reasonable bounds (can be made configurable per asset class)
        # Using very broad bounds to avoid false positives
        min_price = 0.000001  # $0.000001 (crypto dust)
        max_price = 10000000  # $10M (reasonable upper bound)

        out_of_range = ((data['close'] < min_price) | (data['close'] > max_price)).sum()
        if out_of_range > 0:
            outliers += out_of_range
            issues.append(f"Found {out_of_range} prices outside reasonable range")

        # Check for extreme daily changes (>50% in a day)
        if len(data) > 1:
            returns = data['close'].pct_change()
            extreme_changes = (abs(returns) > 0.5).sum()
            if extreme_changes > 0:
                outliers += extreme_changes
                issues.append(f"Found {extreme_changes} extreme daily price changes (>50%)")

        score = 1.0 - (outliers / len(data)) if len(data) > 0 else 0.0
        return max(0.0, score), issues

    def _validate_volume_ranges(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Validate that volume values are reasonable."""
        issues = []

        if 'volume' not in data.columns:
            return 0.0, ["Volume column missing for range validation"]

        # Check for unrealistic volumes (can be asset-specific)
        # For now, using broad bounds
        max_reasonable_volume = 1000000000  # 1 billion

        excessive_volume = (data['volume'] > max_reasonable_volume).sum()
        if excessive_volume > 0:
            issues.append(f"Found {excessive_volume} records with excessive volume")

        score = 1.0 - (excessive_volume / len(data)) if len(data) > 0 else 0.0
        return max(0.0, score), issues

    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in critical_cols if col in data.columns]

        if not available_cols:
            return 0.0

        # Calculate completeness for each critical column
        completeness_scores = []
        for col in available_cols:
            null_count = data[col].isnull().sum()
            completeness = 1.0 - (null_count / len(data)) if len(data) > 0 else 0.0
            completeness_scores.append(completeness)

        return np.mean(completeness_scores)

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        if len(data) < 2:
            return 1.0

        scores = []

        # Check time series continuity (gaps)
        time_diffs = data.index.to_series().diff().dropna()
        max_gap = pd.Timedelta(days=5)  # Allow up to 5-day gaps
        large_gaps = (time_diffs > max_gap).sum()
        continuity_score = 1.0 - (large_gaps / len(time_diffs)) if len(time_diffs) > 0 else 1.0
        scores.append(continuity_score)

        # Check for reasonable data density
        expected_records = len(pd.date_range(data.index.min(), data.index.max(), freq='D'))
        actual_records = len(data)
        density_score = min(1.0, actual_records / expected_records) if expected_records > 0 else 1.0
        scores.append(density_score)

        return np.mean(scores)

    def _generate_recommendations(self, scores: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate improvement recommendations based on validation results."""
        recommendations = []

        # Low completeness recommendations
        if scores.get('completeness', 1.0) < 0.95:
            recommendations.append("Consider implementing data imputation strategies for missing values")
            recommendations.append("Review data source reliability and consider backup providers")

        # OHLC relationship issues
        if scores.get('ohlc_relationships', 1.0) < 0.99:
            recommendations.append("Review OHLC data integrity - check data source or parsing logic")
            recommendations.append("Implement automated OHLC validation in data ingestion pipeline")

        # Volume issues
        if scores.get('volume_non_negative', 1.0) < 1.0:
            recommendations.append("Validate volume data at source - implement negative value filtering")

        # Timestamp issues
        if scores.get('timestamp_chronological', 1.0) < 0.99:
            recommendations.append("Implement timestamp sorting and deduplication in data pipeline")
            recommendations.append("Consider timezone normalization to UTC for all data sources")

        # General recommendations
        if len(issues) > 10:
            recommendations.append("Consider implementing automated data quality monitoring and alerting")

        if not recommendations:
            recommendations.append("Data quality is within acceptable parameters")

        return recommendations


class DataCleaner:
    """Handles data cleaning and preprocessing operations."""

    def __init__(self):
        self.cleaning_stats = {}

    def clean_dataset(self, data: pd.DataFrame, strategy: str = 'conservative') -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Apply comprehensive data cleaning to financial dataset.

        Args:
            data: Raw OHLCV DataFrame
            strategy: Cleaning strategy ('conservative', 'moderate', 'aggressive')

        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        original_shape = data.shape
        cleaning_report = {
            'original_records': len(data),
            'strategy_used': strategy,
            'operations_performed': [],
            'records_removed': 0,
            'values_imputed': 0,
            'outliers_handled': 0
        }

        cleaned_data = data.copy()

        # Apply cleaning operations based on strategy
        if strategy == 'conservative':
            cleaned_data = self._conservative_cleaning(cleaned_data, cleaning_report)
        elif strategy == 'moderate':
            cleaned_data = self._moderate_cleaning(cleaned_data, cleaning_report)
        elif strategy == 'aggressive':
            cleaned_data = self._aggressive_cleaning(cleaned_data, cleaning_report)

        # Update final statistics
        cleaning_report['final_records'] = len(cleaned_data)
        cleaning_report['records_removed'] = original_shape[0] - len(cleaned_data)
        cleaning_report['data_loss_percentage'] = (cleaning_report['records_removed'] / original_shape[0]) * 100

        return cleaned_data, cleaning_report

    def _conservative_cleaning(self, data: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Conservative cleaning - minimal data removal, focus on obvious errors."""
        # Remove completely null records
        null_records = data.isnull().all(axis=1)
        if null_records.any():
            data = data[~null_records]
            report['operations_performed'].append(f"Removed {null_records.sum()} completely null records")

        # Fix OHLC relationship violations (swap high/low if obviously wrong)
        data = self._fix_ohlc_relationships(data, report)

        # Remove negative prices and volumes
        negative_prices = (data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        negative_volume = data['volume'] <= 0

        invalid_records = negative_prices | negative_volume
        if invalid_records.any():
            data = data[~invalid_records]
            report['operations_performed'].append(f"Removed {invalid_records.sum()} records with negative values")

        return data

    def _moderate_cleaning(self, data: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Moderate cleaning - balance between data preservation and quality."""
        # Start with conservative cleaning
        data = self._conservative_cleaning(data, report)

        # Remove statistical outliers (beyond 5 standard deviations)
        data = self._remove_extreme_outliers(data, report, threshold=5.0)

        # Interpolate small gaps in time series
        data = self._interpolate_time_gaps(data, report, max_gap_days=3)

        return data

    def _aggressive_cleaning(self, data: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Aggressive cleaning - prioritize data quality over completeness."""
        # Start with moderate cleaning
        data = self._moderate_cleaning(data, report)

        # Remove all records with any missing values
        missing_data = data.isnull().any(axis=1)
        if missing_data.any():
            data = data[~missing_data]
            report['operations_performed'].append(f"Removed {missing_data.sum()} records with missing values")

        # Remove extreme outliers (beyond 3 standard deviations)
        data = self._remove_extreme_outliers(data, report, threshold=3.0)

        return data

    def _fix_ohlc_relationships(self, data: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Fix obvious OHLC relationship errors."""
        fixed_count = 0

        # Check for high/low swaps (high < low)
        swap_mask = data['high'] < data['low']
        if swap_mask.any():
            # Swap high and low values
            data.loc[swap_mask, ['high', 'low']] = data.loc[swap_mask, ['low', 'high']].values
            fixed_count += swap_mask.sum()
            report['operations_performed'].append(f"Fixed high/low swaps in {swap_mask.sum()} records")

        # Ensure high >= max(open, close) and low <= min(open, close)
        # For minor violations, adjust bounds
        high_fixes = data['high'] < data[['open', 'close']].max(axis=1)
        if high_fixes.any():
            data.loc[high_fixes, 'high'] = data.loc[high_fixes, ['open', 'close']].max(axis=1)
            fixed_count += high_fixes.sum()

        low_fixes = data['low'] > data[['open', 'close']].min(axis=1)
        if low_fixes.any():
            data.loc[low_fixes, 'low'] = data.loc[low_fixes, ['open', 'close']].min(axis=1)
            fixed_count += low_fixes.sum()

        if high_fixes.any() or low_fixes.any():
            report['operations_performed'].append(f"Fixed OHLC bounds in {high_fixes.sum() + low_fixes.sum()} records")

        return data

    def _remove_extreme_outliers(self, data: pd.DataFrame, report: Dict, threshold: float = 3.0) -> pd.DataFrame:
        """Remove statistical outliers based on price movements."""
        if len(data) < 10:  # Need minimum data for statistical analysis
            return data

        # Calculate returns
        returns = data['close'].pct_change()

        # Identify outliers using z-score
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return > 0:
            z_scores = abs((returns - mean_return) / std_return)
            outlier_mask = z_scores > threshold

            if outlier_mask.any():
                data = data[~outlier_mask]
                report['operations_performed'].append(f"Removed {outlier_mask.sum()} statistical outliers (z > {threshold})")
                report['outliers_handled'] += outlier_mask.sum()

        return data

    def _interpolate_time_gaps(self, data: pd.DataFrame, report: Dict, max_gap_days: int = 3) -> pd.DataFrame:
        """Interpolate small gaps in time series data."""
        if len(data) < 2:
            return data

        # Find gaps smaller than threshold
        time_diffs = data.index.to_series().diff()
        small_gaps = (time_diffs > pd.Timedelta(days=1)) & (time_diffs <= pd.Timedelta(days=max_gap_days))

        if small_gaps.any():
            # For now, just log the gaps - full interpolation would require more complex logic
            gap_count = small_gaps.sum()
            report['operations_performed'].append(f"Found {gap_count} small time gaps (≤{max_gap_days} days) suitable for interpolation")

        return data


class QualityChecker:
    """Provides statistical quality analysis and reporting."""

    def __init__(self):
        self.quality_metrics = {}

    def assess_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive quality assessment on processed dataset.

        Args:
            data: Processed financial dataset

        Returns:
            Dictionary with quality metrics and analysis
        """
        quality_report = {
            'summary': {
                'total_records': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}" if len(data) > 0 else None,
                'columns_present': list(data.columns),
                'data_types': data.dtypes.to_dict()
            },

            'completeness_analysis': self._analyze_completeness(data),
            'distribution_analysis': self._analyze_distributions(data),
            'consistency_analysis': self._analyze_consistency(data),
            'quality_score': self._calculate_overall_quality_score(data)
        }

        return quality_report

    def _analyze_completeness(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze data completeness across all columns."""
        completeness = {}

        for column in data.columns:
            null_count = data[column].isnull().sum()
            completeness_pct = (1 - null_count / len(data)) * 100 if len(data) > 0 else 0

            completeness[column] = {
                'null_count': null_count,
                'completeness_percentage': completeness_pct,
                'status': 'Good' if completeness_pct > 95 else 'Poor' if completeness_pct < 80 else 'Fair'
            }

        return completeness

    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze statistical distributions of key fields."""
        distributions = {}

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            series = data[column].dropna()

            if len(series) > 0:
                distributions[column] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'quartiles': {
                        '25%': series.quantile(0.25),
                        '50%': series.quantile(0.50),
                        '75%': series.quantile(0.75)
                    }
                }

        return distributions

    def _analyze_consistency(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze data consistency and relationships."""
        consistency = {}

        # Time series analysis
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            consistency['time_series'] = {
                'is_monotonic': data.index.is_monotonic_increasing,
                'has_duplicates': data.index.duplicated().any(),
                'avg_time_gap': time_diffs.mean(),
                'max_time_gap': time_diffs.max(),
                'missing_dates': self._count_missing_dates(data)
            }

        # OHLC relationships
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_consistent = (
                (data['high'] >= data['low']) &
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            ).all()

            consistency['ohlc_relationships'] = {
                'consistent': ohlc_consistent,
                'violations': (~ohlc_consistent).sum() if not ohlc_consistent else 0
            }

        return consistency

    def _calculate_overall_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        scores = []

        # Completeness score (40% weight)
        completeness_scores = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                completeness = (1 - data[col].isnull().mean())
                completeness_scores.append(completeness)

        if completeness_scores:
            scores.append(np.mean(completeness_scores) * 0.4)

        # Consistency score (30% weight)
        consistency_score = 1.0
        if len(data) > 1:
            # Time series consistency
            if not data.index.is_monotonic_increasing:
                consistency_score *= 0.8
            if data.index.duplicated().any():
                consistency_score *= 0.9

            # OHLC consistency
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_valid = (
                    (data['high'] >= data['low']) &
                    (data['high'] >= data['open']) &
                    (data['high'] >= data['close']) &
                    (data['low'] <= data['open']) &
                    (data['low'] <= data['close'])
                ).mean()
                consistency_score *= ohlc_valid

        scores.append(consistency_score * 0.3)

        # Distribution score (30% weight)
        distribution_score = 1.0
        if 'close' in data.columns and len(data) > 10:
            returns = data['close'].pct_change().dropna()
            if len(returns) > 0:
                # Check for reasonable volatility
                volatility = returns.std()
                if volatility > 0.1:  # Very high volatility might indicate issues
                    distribution_score *= 0.9

                # Check for extreme outliers
                z_scores = abs((returns - returns.mean()) / returns.std())
                outlier_pct = (z_scores > 5).mean()
                distribution_score *= (1 - outlier_pct)

        scores.append(distribution_score * 0.3)

        return np.mean(scores) if scores else 0.0

    def _count_missing_dates(self, data: pd.DataFrame) -> int:
        """Count missing dates in time series."""
        if len(data) < 2:
            return 0

        expected_range = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq='D'
        )

        actual_dates = set(data.index.date)
        expected_dates = set(expected_range.date)

        missing_dates = expected_dates - actual_dates
        return len(missing_dates)


class DataProcessingPipeline:
    """Orchestrates the complete data processing pipeline."""

    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.quality_checker = QualityChecker()

    def process_dataset(self, raw_data: pd.DataFrame, symbol: str = None,
                       cleaning_strategy: str = 'moderate') -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Execute complete data processing pipeline.

        Args:
            raw_data: Raw OHLCV DataFrame
            symbol: Trading symbol for context
            cleaning_strategy: Data cleaning approach

        Returns:
            Tuple of (processed_data, processing_report)
        """
        processing_report = {
            'symbol': symbol,
            'processing_timestamp': datetime.utcnow().isoformat(),
            'stages': {}
        }

        # Stage 1: Validation
        logger.info("Stage 1: Data validation")
        validation_report = self.validator.validate_dataset(raw_data, symbol)
        processing_report['stages']['validation'] = {
            'quality_report': validation_report.__dict__,
            'passed': validation_report.overall_score > 0.8
        }

        # Stage 2: Cleaning
        logger.info("Stage 2: Data cleaning")
        cleaned_data, cleaning_report = self.cleaner.clean_dataset(raw_data, cleaning_strategy)
        processing_report['stages']['cleaning'] = cleaning_report

        # Stage 3: Feature Engineering
        logger.info("Stage 3: Feature engineering")
        processed_data = self._add_technical_features(cleaned_data)
        processing_report['stages']['feature_engineering'] = {
            'features_added': len(processed_data.columns) - len(cleaned_data.columns),
            'final_columns': list(processed_data.columns)
        }

        # Stage 4: Quality Assessment
        logger.info("Stage 4: Quality assessment")
        quality_report = self.quality_checker.assess_quality(processed_data)
        processing_report['stages']['quality_assessment'] = quality_report

        # Add processing metadata
        processed_data = self._add_processing_metadata(processed_data, processing_report)

        logger.info(f"Data processing completed. {len(processed_data)} records processed.")
        return processed_data, processing_report

    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and derived features."""
        df = data.copy()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        # Volatility (20-day rolling)
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        return df

    def _add_processing_metadata(self, data: pd.DataFrame, processing_report: Dict) -> pd.DataFrame:
        """Add processing metadata to the dataset."""
        df = data.copy()

        # Add quality score
        quality_score = processing_report['stages']['quality_assessment']['quality_score']
        df['data_quality_score'] = quality_score

        # Add processing timestamp
        df['processing_timestamp'] = pd.Timestamp(processing_report['processing_timestamp'])

        return df


# Convenience functions for quick use
def validate_data(data: pd.DataFrame, symbol: str = None) -> DataQualityReport:
    """Quick validation function."""
    validator = DataValidator()
    return validator.validate_dataset(data, symbol)


def clean_data(data: pd.DataFrame, strategy: str = 'moderate') -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Quick cleaning function."""
    cleaner = DataCleaner()
    return cleaner.clean_dataset(data, strategy)


def assess_quality(data: pd.DataFrame) -> Dict[str, any]:
    """Quick quality assessment function."""
    checker = QualityChecker()
    return checker.assess_quality(data)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('.')

    from src.data_loader import DataLoader

    # Load sample data
    loader = DataLoader()
    try:
        sample_data = loader.download_stock_data('EURUSD=X', '2024-01-01', '2024-01-15')
        print(f"Downloaded {len(sample_data)} records")

        # Process through pipeline
        pipeline = DataProcessingPipeline()
        processed_data, report = pipeline.process_dataset(sample_data, 'EURUSD=X')

        print(f"Processing completed: {len(processed_data)} records")
        print(".2f")
        print(f"Cleaning removed {report['stages']['cleaning']['records_removed']} records")

    except Exception as e:
        print(f"Error: {e}")
