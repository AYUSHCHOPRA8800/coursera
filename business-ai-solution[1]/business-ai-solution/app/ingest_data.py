import pandas as pd
import numpy as np
import requests
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

class DataIngestion:
    """Data ingestion and preprocessing for economic indicators"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Economic indicators to fetch
        self.indicators = {
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'inflation': 'FP.CPI.TOTL.ZG',      # Inflation, consumer prices (annual %)
            'unemployment': 'SL.UEM.TOTL.ZS',   # Unemployment, total (% of total labor force)
            'interest_rate': 'FR.INR.RINR',     # Real interest rate (%)
            'trade_balance': 'NE.RSB.GNFS.ZS',  # Trade balance (% of GDP)
            'government_debt': 'GC.DOD.TOTL.GD.ZS',  # Central government debt (% of GDP)
            'foreign_investment': 'BX.KLT.DINV.WD.GD.ZS',  # Foreign direct investment (% of GDP)
            'population_growth': 'SP.POP.GROW'  # Population growth (annual %)
        }
        
        # Countries to analyze
        self.countries = [
            'USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'ITA', 'CAN', 'BRA', 'IND',
            'RUS', 'KOR', 'AUS', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'CHE'
        ]
    
    def fetch_world_bank_data(self, country_code: str, indicator_code: str, 
                            start_year: int = 2010, end_year: int = 2023) -> pd.DataFrame:
        """Fetch data from World Bank API"""
        try:
            url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
            params = {
                'format': 'json',
                'date': f"{start_year}:{end_year}",
                'per_page': 100
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) < 2 or not data[1]:
                logger.warning(f"No data found for {country_code} - {indicator_code}")
                return pd.DataFrame()
            
            # Extract data
            records = []
            for item in data[1]:
                if item['value'] is not None:
                    records.append({
                        'year': int(item['date']),
                        'value': float(item['value']),
                        'country_code': country_code,
                        'indicator_code': indicator_code
                    })
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Error fetching data for {country_code} - {indicator_code}: {e}")
            return pd.DataFrame()
    
    def create_synthetic_data(self, country_code: str, start_year: int = 2010, 
                            end_year: int = 2023) -> pd.DataFrame:
        """Create synthetic economic data for testing when API is unavailable"""
        logger.info(f"Creating synthetic data for {country_code}")
        
        years = list(range(start_year, end_year + 1))
        np.random.seed(hash(country_code) % 2**32)  # Consistent data per country
        
        data = {
            'date': [f"{year}-12-31" for year in years],
            'gdp_growth': np.random.normal(2.5, 1.5, len(years)),
            'inflation': np.random.normal(2.0, 1.0, len(years)),
            'unemployment': np.random.normal(5.0, 2.0, len(years)),
            'interest_rate': np.random.normal(3.0, 1.5, len(years)),
            'trade_balance': np.random.normal(0.0, 3.0, len(years)),
            'government_debt': np.random.normal(60.0, 15.0, len(years)),
            'foreign_investment': np.random.normal(2.0, 1.5, len(years)),
            'population_growth': np.random.normal(1.0, 0.5, len(years)),
            'target': np.random.normal(100.0, 10.0, len(years))  # Economic health index
        }
        
        # Add some realistic trends
        for i, year in enumerate(years):
            trend_factor = (year - start_year) / (end_year - start_year)
            data['gdp_growth'][i] += trend_factor * np.random.normal(0.5, 0.3)
            data['target'][i] += trend_factor * np.random.normal(5.0, 2.0)
        
        df = pd.DataFrame(data)
        df['country_code'] = country_code
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean the data"""
        if df.empty:
            return df
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Add lagged features
        for col in numeric_columns:
            if col != 'target':
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)
        
        # Add rolling averages
        for col in numeric_columns:
            if col != 'target':
                df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std()
        
        # Drop rows with NaN values after feature engineering
        df = df.dropna()
        
        return df
    
    def ingest_country_data(self, country_code: str, use_synthetic: bool = True) -> bool:
        """Ingest data for a specific country"""
        try:
            logger.info(f"Starting data ingestion for {country_code}")
            
            if use_synthetic:
                df = self.create_synthetic_data(country_code)
            else:
                # Try to fetch real data first
                all_data = []
                for indicator_name, indicator_code in self.indicators.items():
                    indicator_df = self.fetch_world_bank_data(country_code, indicator_code)
                    if not indicator_df.empty:
                        all_data.append(indicator_df)
                
                if not all_data:
                    logger.warning(f"No real data available for {country_code}, using synthetic data")
                    df = self.create_synthetic_data(country_code)
                else:
                    # Combine all indicators
                    df = pd.concat(all_data, ignore_index=True)
                    df = df.pivot_table(index='year', columns='indicator_code', values='value', aggfunc='first')
                    df.reset_index(inplace=True)
                    df.rename(columns={'year': 'date'})
                    df['date'] = df['date'].astype(str) + '-12-31'
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            if df.empty:
                logger.error(f"No data available for {country_code}")
                return False
            
            # Save to file
            output_path = self.data_dir / f"{country_code.lower()}_data.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Data saved for {country_code}: {len(df)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting data for {country_code}: {e}")
            return False
    
    def ingest_all_countries(self, use_synthetic: bool = True) -> Dict[str, bool]:
        """Ingest data for all countries"""
        results = {}
        
        logger.info(f"Starting data ingestion for {len(self.countries)} countries")
        
        for country in self.countries:
            success = self.ingest_country_data(country, use_synthetic)
            results[country] = success
        
        # Create summary statistics
        successful_countries = sum(results.values())
        logger.info(f"Data ingestion completed: {successful_countries}/{len(self.countries)} countries successful")
        
        return results
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """Create a combined dataset with all countries"""
        combined_data = []
        
        for country in self.countries:
            file_path = self.data_dir / f"{country.lower()}_data.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['country'] = country
                combined_data.append(df)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df.to_csv(self.data_dir / "combined_data.csv", index=False)
            logger.info(f"Combined dataset created with {len(combined_df)} records")
            return combined_df
        else:
            logger.error("No country data files found")
            return pd.DataFrame()
    
    def validate_data_quality(self) -> Dict[str, Dict]:
        """Validate data quality for all countries"""
        quality_report = {}
        
        for country in self.countries:
            file_path = self.data_dir / f"{country.lower()}_data.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                quality_report[country] = {
                    'record_count': len(df),
                    'missing_values': df.isnull().sum().to_dict(),
                    'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A",
                    'features': list(df.columns)
                }
            else:
                quality_report[country] = {'error': 'File not found'}
        
        # Save quality report
        with open(self.data_dir / "data_quality_report.json", 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info("Data quality report generated")
        return quality_report

def main():
    """Main function for data ingestion"""
    logger.info("Starting data ingestion process")
    
    # Initialize data ingestion
    ingestion = DataIngestion()
    
    # Ingest data for all countries (using synthetic data for demo)
    results = ingestion.ingest_all_countries(use_synthetic=True)
    
    # Create combined dataset
    combined_df = ingestion.create_combined_dataset()
    
    # Validate data quality
    quality_report = ingestion.validate_data_quality()
    
    logger.info("Data ingestion process completed")
    
    return {
        'ingestion_results': results,
        'combined_dataset_size': len(combined_df),
        'quality_report': quality_report
    }

if __name__ == "__main__":
    main()
