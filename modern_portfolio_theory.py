import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gmean
from config import *
from dataset_creation import *


def plot_modern_portfolio_theory(symbols, start, end, type_price, data_folder):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    df = pd.DataFrame()
    for file in os.listdir(data_folder):
        symbol = file[:-4][:-22]
        date = file[:-4][-21:]

        if (symbol in symbols) and (date == (start_str + "_" + end_str)):  # correct date for the symbol
            # Load the data
            path = os.path.join(data_folder, file)
            df_symb = pd.read_csv(path)

            df_symb["date"] = pd.to_datetime(df_symb["date"])
            other_columns = list(df_symb.columns)
            other_columns.remove("date")
            df_symb[other_columns] = df_symb[other_columns].astype("float32")

            # Calculate return
            df[symbol] = df_symb[type_price].pct_change()  # percentage change

    df = df.dropna() # remove NaN values
    df.reset_index(drop=True, inplace=True)

    # Calculate final values
    X = 100 * np.sqrt(252) * df.std()  # standart deviation, one way to calculate volatility
    X = X.tolist()
    Y = gmean(1 + df).tolist()  # geometric mean
    for k in range(len(Y)):
        Y[k] = 100 * ((Y[k])**(252) - 1)
    symbols_name = df.columns

    # Plot points with labels
    plt.scatter(X, Y)
    for index, symbol in enumerate(symbols_name):
        plt.annotate(symbol, (X[index], Y[index]))

    plt.xlabel("Risk (annualized volatility) (%)")
    plt.ylabel("Expected return (geometric mean) (%)")
    plt.title("Modern Portfolio Theory (stocks)")
    plt.tight_layout()
    plt.show()


plot_modern_portfolio_theory(symbols, start, end, type_price, data_folder)
