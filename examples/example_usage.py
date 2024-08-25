import openassetpricing as oap


# Initialize OpenAP
openap = oap.OpenAP(2024)

# ==========
# List available datasets
# ==========
openap.list_datasets()



# ==========
# Download SignalDoc.csv in pandas dataframe
# ==========
df = openap.dl('signal_doc', 'pandas')



# ==========
# Portfolios -> Full Sets OP -> PredictorPortsFull.csv
# ==========

# Download entier file in polars dataframe
df = openap.dl('port_op', 'polars')

# Download specific predictors (can be single or multiple predictors)
df = openap.dl('port_op', 'polars', predictor=['AM'])
df = openap.dl('port_op', 'polars', predictor=['AM', 'Mom12m'])



# ==========
# Portfolios -> Full Sets Alt -> PredictorAltPorts_Deciles.zip
# Portfolios -> Full Sets Alt -> PredictorAltPorts_DecilesVW.zip
# ==========

# Download entier file
df = openap.dl('port_deciles_ew', 'polars')
df = openap.dl('port_deciles_vw', 'polars')

# Download specific predictors (can be single or multiple predictors)
df = openap.dl('port_deciles_ew', 'polars', predictor=['Accruals'])
df = openap.dl('port_deciles_ew', 'polars', predictor=['BM', 'Mom6m'])
