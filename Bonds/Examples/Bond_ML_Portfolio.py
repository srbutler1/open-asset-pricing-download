#!/usr/bin/env python
# coding: utf-8

"""
Bond Machine Learning Portfolio Example

This script implements machine learning portfolios for corporate bonds,
similar to the equity ML portfolio example but adapted for fixed income.
It uses WRDS bond data and the OpenAP package.
"""

import pandas as pd
import polars as pl
import numpy as np
import wrds
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime as dt
import time
import warnings
warnings.filterwarnings('ignore')

print("Connecting to WRDS...")
wrds_conn = wrds.Connection()

# Set parameters
training_start_year = 1980  # Start of training data
training_end_year = 1999    # End of initial training period
test_start_year = 2000      # Start of testing/prediction period
refit_period = 2            # Refit model every 2 years
n_portfolios = 5            # Number of portfolios to form

print(f"Training period: {training_start_year}-{training_end_year}")
print(f"Testing period: {test_start_year}-present")
print(f"Refitting every {refit_period} years")

# Step 1: Get bond returns from WRDS
print("\nRetrieving bond returns from WRDS...")
start_time = time.time()

bond_returns = wrds_conn.raw_sql("""
    SELECT cusip_id, date, ret_eom as ret, price_eom as price,
           t_volume, t_spread, t_yld_pt
    FROM wrds_bond.bondret
    WHERE date >= '1980-01-01'
    AND ret_eom IS NOT NULL
""", date_cols=["date"])

print(f"Retrieved {len(bond_returns)} bond return observations")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Step 2: Get bond characteristics from FISD
print("\nRetrieving bond characteristics from WRDS...")
start_time = time.time()

bond_chars = wrds_conn.raw_sql("""
    SELECT cusip_id, offering_date, maturity, 
           coupon, offering_amt, 
           security_level, bond_type, 
           convertible, callable, putable,
           rule_144a, rating_class
    FROM fisd.fisd_mergedissue
""", date_cols=["offering_date", "maturity"])

print(f"Retrieved characteristics for {len(bond_chars)} bonds")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Step 3: Get the bond-equity linking table
print("\nRetrieving bond-equity linking table...")
start_time = time.time()

bond_equity_link = wrds_conn.raw_sql("""
    SELECT cusip, permno, link_startdt as start_date, 
           link_enddt as end_date
    FROM wrds_bond.bondcrsp_link
""", date_cols=["start_date", "end_date"])

print(f"Retrieved {len(bond_equity_link)} bond-equity links")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Convert to polars for better memory management
print("\nConverting data to polars format...")
bond_returns = pl.from_pandas(bond_returns)
bond_chars = pl.from_pandas(bond_chars)
bond_equity_link = pl.from_pandas(bond_equity_link)

# Step 4: Calculate bond features
print("\nCalculating bond features...")

# Create rating numeric feature (higher = better rating)
rating_map = {
    "AAA": 10, "AA+": 9.7, "AA": 9.3, "AA-": 9, 
    "A+": 8.7, "A": 8.3, "A-": 8, 
    "BBB+": 7.7, "BBB": 7.3, "BBB-": 7, 
    "BB+": 6.7, "BB": 6.3, "BB-": 6, 
    "B+": 5.7, "B": 5.3, "B-": 5, 
    "CCC+": 4.7, "CCC": 4.3, "CCC-": 4, 
    "CC": 3, "C": 2, "D": 1
}

# Join bond returns with characteristics
bond_data = bond_returns.join(
    bond_chars, 
    left_on="cusip_id", 
    right_on="cusip_id",
    how="left"
)

# Calculate time-varying bond features
bond_data = bond_data.with_columns([
    # Time to maturity (in years)
    (pl.col("maturity").dt.epoch_days() - 
     pl.col("date").dt.epoch_days()).div(365).alias("time_to_maturity"),
    
    # Bond age (in years)
    (pl.col("date").dt.epoch_days() - 
     pl.col("offering_date").dt.epoch_days()).div(365).alias("bond_age"),
    
    # Create dummy variables for bond characteristics
    pl.col("convertible").cast(pl.Int32).alias("convertible_dummy"),
    pl.col("callable").cast(pl.Int32).alias("callable_dummy"),
    pl.col("putable").cast(pl.Int32).alias("putable_dummy"),
    pl.col("rule_144a").cast(pl.Int32).alias("rule_144a_dummy"),
    
    # Create rating numeric feature
    pl.col("rating_class").map_dict(rating_map).alias("rating_numeric")
])

print(f"Calculated features for {len(bond_data)} bond-month observations")

# --- Save bond data and predictor samples ---
# Save a sample of the bond data
try:
    bond_data_sample = bond_data.sample(1000) if hasattr(bond_data, 'sample') else bond_data.head(1000)
    bond_data_sample = bond_data_sample.to_pandas() if hasattr(bond_data_sample, 'to_pandas') else bond_data_sample
    bond_data_sample.to_csv("../Data/samples/bond_data_sample.csv", index=False)
    print("Saved bond data sample to ../Data/samples/bond_data_sample.csv")
except Exception as e:
    print(f"Could not save bond data sample: {e}")

# Step 5: Add equity-based predictors
print("\nDownloading equity predictors from Open Asset Pricing...")
start_time = time.time()

# Import OpenAP
import openassetpricing as oap
openap = oap.OpenAP()

# Choose number of signals based on memory constraints
# Adjust this number if needed
nsignals_for_ml = 20  # Start with a smaller number for testing

# Download equity predictors
try:
    print("Downloading equity predictors...")
    equity_predictors = openap.dl_all_signals("polars")
    
    # Convert date format to match bond data
    equity_predictors = equity_predictors.with_columns([
        pl.col("yyyymm").cast(pl.Utf8).str.strptime(
            pl.Date, "%Y%m"
        ).dt.end_of_month().alias("date")
    ])
    
    print(f"Downloaded {len(equity_predictors.columns) - 3} equity predictors")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Get list of equity predictors
    equity_predictor_list = [col for col in equity_predictors.columns 
                            if col not in ["permno", "yyyymm", "date"]]
    
    # Limit to top N equity predictors if needed
    equity_predictor_list = equity_predictor_list[:nsignals_for_ml]
    print(f"Using {len(equity_predictor_list)} equity predictors")
    
    # Save a sample of the equity predictors (if available)
    try:
        equity_predictors_sample = equity_predictors.sample(1000) if hasattr(equity_predictors, 'sample') else equity_predictors.head(1000)
        equity_predictors_sample = equity_predictors_sample.to_pandas() if hasattr(equity_predictors_sample, 'to_pandas') else equity_predictors_sample
        equity_predictors_sample.to_csv("../Data/samples/equity_predictors_sample.csv", index=False)
        print("Saved equity predictors sample to ../Data/samples/equity_predictors_sample.csv")
    except Exception as e:
        print(f"Could not save equity predictors sample: {e}")
    
    # Join bond data with equity predictors through the linking table
    print("\nMerging bond data with equity predictors...")
    start_time = time.time()
    
    # First join bond data with the linking table
    bond_with_permno = bond_data.join(
        bond_equity_link.select("cusip", "permno", "start_date", "end_date"),
        left_on="cusip_id",
        right_on="cusip",
        how="left"
    )
    
    # Filter to keep only valid links (date within link period)
    bond_with_permno = bond_with_permno.filter(
        (pl.col("date") >= pl.col("start_date")) & 
        (pl.col("date") <= pl.col("end_date"))
    )
    
    # Now join with equity predictors
    combined_data = bond_with_permno.join(
        equity_predictors.select(["permno", "date"] + equity_predictor_list),
        left_on=["permno", "date"],
        right_on=["permno", "date"],
        how="left"
    )
    
    print(f"Combined data has {len(combined_data)} observations")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
except Exception as e:
    print(f"Error downloading equity predictors: {e}")
    print("Continuing with bond data only...")
    combined_data = bond_data
    equity_predictor_list = []

# Step 6: Prepare data for machine learning
print("\nPreparing data for machine learning...")

# Define bond predictors
bond_predictors = [
    "time_to_maturity", "bond_age", "coupon", 
    "convertible_dummy", "callable_dummy", "putable_dummy",
    "rating_numeric", "t_spread", "t_yld_pt"
]

# Filter to keep only columns with sufficient non-null values
bond_predictors = [col for col in bond_predictors 
                  if combined_data[col].null_count() < len(combined_data) * 0.5]

# Combine bond and equity predictors
all_predictors = bond_predictors + equity_predictor_list
print(f"Using {len(all_predictors)} total predictors")

# Create forward returns (1-month ahead)
combined_data = combined_data.sort(["cusip_id", "date"])
combined_data = combined_data.with_columns([
    pl.col("ret").shift(-1).over("cusip_id").alias("forward_ret")
])

# Drop rows with missing forward returns
ml_data = combined_data.filter(pl.col("forward_ret").is_not_null())

# Select only needed columns
ml_data = ml_data.select(
    ["cusip_id", "permno", "date", "forward_ret"] + all_predictors
)

# Fill missing values with column medians
for col in all_predictors:
    median_val = ml_data[col].median()
    if median_val is not None:
        ml_data = ml_data.with_columns([
            pl.col(col).fill_null(median_val)
        ])
    else:
        ml_data = ml_data.with_columns([
            pl.col(col).fill_null(0)
        ])

print(f"ML data has {len(ml_data)} observations")

# Convert to pandas for sklearn compatibility
ml_data_pd = ml_data.to_pandas()

# Step 7: Define training and testing periods
print("\nDefining training and testing periods...")

# Initial training data
train_data = ml_data_pd[ml_data_pd["date"].dt.year <= training_end_year]
print(f"Initial training data: {len(train_data)} observations")

# Test data
test_data = ml_data_pd[ml_data_pd["date"].dt.year >= test_start_year]
print(f"Test data: {len(test_data)} observations")

# Step 8: Train initial models
print("\nTraining initial models...")
start_time = time.time()

# Prepare features and target for initial training
X_train = train_data[all_predictors].fillna(0)
y_train = train_data["forward_ret"]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a neural network model
nn_model = MLPRegressor(
    hidden_layer_sizes=(32, 16), 
    activation="relu",
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)

# Also train a simple OLS model for comparison
ols_formula = "forward_ret ~ " + " + ".join(all_predictors)
try:
    ols_model = smf.ols(
        formula=ols_formula, 
        data=train_data.fillna(0)
    ).fit()
except Exception as e:
    print(f"Error fitting OLS model: {e}")
    print("Using a simpler OLS model...")
    # Use a subset of predictors if there's an issue
    simple_predictors = bond_predictors[:5]  # Use fewer predictors
    ols_formula = "forward_ret ~ " + " + ".join(simple_predictors)
    ols_model = smf.ols(
        formula=ols_formula, 
        data=train_data.fillna(0)
    ).fit()

print(f"Models trained in {time.time() - start_time:.2f} seconds")

# Step 9: Form portfolios based on predictions with expanding window
print("\nForming portfolios with expanding window approach...")

# Function to form portfolios for a given time period
def form_portfolios(data, nn_model, ols_model, scaler, predictors, n_portfolios=5):
    # Prepare features
    X = data[predictors].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    data["nn_pred"] = nn_model.predict(X_scaled)
    
    try:
        data["ols_pred"] = ols_model.predict(data[predictors].fillna(0))
    except Exception as e:
        print(f"Error in OLS prediction: {e}")
        # Use a subset of predictors if there's an issue
        simple_predictors = [p for p in predictors if p in ols_model.params.index]
        if len(simple_predictors) > 1:
            data["ols_pred"] = ols_model.predict(data[simple_predictors].fillna(0))
        else:
            data["ols_pred"] = 0  # Default if prediction fails
    
    # Form portfolios based on predictions
    for pred_col in ["nn_pred", "ols_pred"]:
        # Assign portfolio ranks (1 = lowest predicted return, 5 = highest)
        try:
            data[f"{pred_col}_port"] = pd.qcut(
                data[pred_col], 
                n_portfolios, 
                labels=range(1, n_portfolios+1)
            )
        except ValueError:
            # Handle case where all predictions are the same
            data[f"{pred_col}_port"] = 3  # Middle portfolio
    
    return data

# Process test data in expanding window approach
all_results = []
unique_years = sorted(test_data["date"].dt.year.unique())

for year in range(test_start_year, unique_years[-1] + 1, refit_period):
    print(f"\nProcessing year {year}...")
    
    # Refit models using all data up to the current year
    train_window = ml_data_pd[ml_data_pd["date"].dt.year < year]
    
    if len(train_window) > 0:
        print(f"Refitting models with {len(train_window)} observations...")
        
        # Refit models
        X_train = train_window[all_predictors].fillna(0)
        y_train = train_window["forward_ret"]
        X_train_scaled = scaler.fit_transform(X_train)
        
        nn_model.fit(X_train_scaled, y_train)
        
        try:
            ols_model = smf.ols(formula=ols_formula, data=train_window.fillna(0)).fit()
        except Exception as e:
            print(f"Error refitting OLS model: {e}")
            # Continue with previous OLS model
    
    # Get data for current period (next refit_period years)
    current_data = test_data[
        (test_data["date"].dt.year >= year) & 
        (test_data["date"].dt.year < year + refit_period)
    ]
    
    if len(current_data) > 0:
        print(f"Forming portfolios for {len(current_data)} observations...")
        
        # Form portfolios
        portfolio_data = form_portfolios(
            current_data, nn_model, ols_model, scaler, all_predictors, n_portfolios
        )
        all_results.append(portfolio_data)
    else:
        print(f"No data available for years {year}-{year+refit_period-1}")

# Step 10: Analyze portfolio performance
if len(all_results) > 0:
    print("\nAnalyzing portfolio performance...")
    
    # Combine all results
    portfolio_results = pd.concat(all_results)
    
    # Calculate portfolio returns
    nn_returns = portfolio_results.groupby(
        ["date", "nn_pred_port"]
    )["forward_ret"].mean().reset_index()
    
    ols_returns = portfolio_results.groupby(
        ["date", "ols_pred_port"]
    )["forward_ret"].mean().reset_index()
    
    # Rename for clarity
    nn_returns = nn_returns.rename(
        columns={"nn_pred_port": "port", "forward_ret": "ret"}
    )
    nn_returns["model"] = "Neural Network"
    
    ols_returns = ols_returns.rename(
        columns={"ols_pred_port": "port", "forward_ret": "ret"}
    )
    ols_returns["model"] = "OLS"
    
    # Combine
    all_returns = pd.concat([nn_returns, ols_returns])
    
    # Pivot to get returns by portfolio
    port_pivot = all_returns.pivot_table(
        index=["date", "model"], 
        columns="port", 
        values="ret"
    ).reset_index()
    
    # Calculate long-short (5 minus 1)
    try:
        port_pivot["long_short"] = port_pivot[5] - port_pivot[1]
    except KeyError:
        # Handle case where not all portfolios exist
        available_ports = sorted([col for col in port_pivot.columns if isinstance(col, int)])
        if len(available_ports) >= 2:
            port_pivot["long_short"] = port_pivot[available_ports[-1]] - port_pivot[available_ports[0]]
        else:
            port_pivot["long_short"] = 0
    
    # Calculate performance by year
    port_pivot["year"] = port_pivot["date"].dt.year
    annual_performance = port_pivot.groupby(
        ["year", "model"]
    )["long_short"].agg(["mean", "std"]).reset_index()
    
    # Calculate Sharpe ratio (assuming 0 risk-free rate for simplicity)
    annual_performance["sharpe"] = annual_performance["mean"] / annual_performance["std"].replace(0, np.nan)
    annual_performance["sharpe"] = annual_performance["sharpe"].fillna(0)
    
    # Calculate performance by decade
    port_pivot["decade"] = (port_pivot["date"].dt.year // 10) * 10
    decade_performance = port_pivot.groupby(
        ["decade", "model"]
    )["long_short"].agg(["mean", "std", "count"]).reset_index()
    
    decade_performance["sharpe"] = decade_performance["mean"] / decade_performance["std"].replace(0, np.nan)
    decade_performance["sharpe"] = decade_performance["sharpe"].fillna(0)
    
    # Print results
    print("\nBond Portfolio Performance by Decade:")
    print(decade_performance)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    
    for model in port_pivot["model"].unique():
        model_data = port_pivot[port_pivot["model"] == model].sort_values("date")
        plt.plot(
            model_data["date"], 
            (1 + model_data["long_short"]/100).cumprod(), 
            label=model
        )
    
    plt.title("Cumulative Returns of Long-Short Bond Portfolios")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Starting at $1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../Data/results/Bond_ML_Portfolio_Returns.png")
    print("\nSaved cumulative returns plot to ../Data/results/Bond_ML_Portfolio_Returns.png")
    
    # Save results to CSV
    decade_performance.to_csv("../Data/results/Bond_ML_Portfolio_Performance.csv", index=False)
    print("Saved performance metrics to ../Data/results/Bond_ML_Portfolio_Performance.csv")
    
    # Print summary statistics
    print("\nSummary of Long-Short Portfolio Returns:")
    summary = port_pivot.groupby("model")["long_short"].agg(
        ["mean", "std", "min", "max", "count"]
    ).reset_index()
    summary["sharpe"] = summary["mean"] / summary["std"]
    print(summary)
    
    print("\nBond ML Portfolio analysis complete!")
else:
    print("\nNo portfolio results to analyze. Check your data and parameters.")
