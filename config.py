import os
import datetime

start = "2017-01-01"  # year-month-day. Start date to extract data
start = datetime.datetime.strptime(start, "%Y-%m-%d")
end = "2019-12-31"  # End date
end = datetime.datetime.strptime(end, "%Y-%m-%d")
type_price = "Adj Close"  # Type of price to work on


# Creating dates for factors study
initial_start = "2010-01-01"
initial_end = "2019-12-31"
datetime_initial_start = datetime.datetime.strptime(initial_start, "%Y-%m-%d")
datetime_initial_end = datetime.datetime.strptime(initial_end, "%Y-%m-%d")

data_folder = "data"  # name of the folder that will contain our dataset

factors_file = os.path.join(data_folder, 'factors.xlsx')

# Stocks to load. For CAC40, see : https://en.wikipedia.org/wiki/CAC_40

symbols = ["AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "ATO.PA", "CS.PA", "BNP.PA",
           "EN.PA", "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA",
           "EL.PA", "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA", "ML.PA",
           "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA",
           "SU.PA", "GLE.PA", "STLA", "STM.PA", "TEP.PA", "HO.PA", "FP.PA",
           "URW.AS", "VIE.PA", "DG.PA", "VIV.PA", "WLN.PA"]


"""
symbols = ["AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "ATO.PA", "CS.PA", "BNP.PA",
           "EN.PA", "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA",
           "EL.PA", "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA", "ML.PA",
           "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA"]
"""

factors = ["VALUE", "GROWTH", "QUALITY", "MOMENTUM", "SIZE"]

# Choose optimization method : "mean_variance_optimizing", "equal_weights", "sharpe_ratio", "min_Vol", "max_diversification", "min_correlation", "equal_risk", "russel"
optimization_method = ["mean_variance_optimizing", "equal_weights", "sharpe_ratio",
                       "max_diversification", "min_correlation", "min_Vol", "equal_risk", "russel",
                       "risk_parity", "nasdaq"]

# Show investment parameters
start_simulation = "2020-01-01"  # start of the simulation
start_simulation = datetime.datetime.strptime(start_simulation, "%Y-%m-%d")
end_simulation = "2020-12-31"  # end of the simulation
end_simulation = datetime.datetime.strptime(end_simulation, "%Y-%m-%d")

lbd = 2  # Aversion to risk
min_return = 0.001
risk_free = 0
