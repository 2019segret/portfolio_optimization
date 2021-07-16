import pandas_datareader as pdr
import datetime
import os
import pandas as pd
import numpy as np
from config import *

if not os.path.exists(data_folder):
    os.mkdir(data_folder)


def create_dateset(symbols, start, end, type_price, data_folder):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    # Download the data
    for symbol in symbols:
        path = os.path.join(data_folder, symbol + "_" + start_str + "_" + end_str + ".csv")
        if not os.path.exists(path):
            print("Process {} ...".format(symbol))
            df = pdr.DataReader(symbol, start=start_str, end=end_str, data_source="yahoo")
            df["date"] = list(df.index)
            df["date"] = pd.to_datetime(df["date"]) # good format
            df.reset_index(drop=True, inplace=True)
            df.to_csv(path, index=False)

    # Import the data and create features

    df = pd.DataFrame()
    sigma = []
    for file in os.listdir(data_folder):
        symbol = file[:-4][:-22]
        date = file[:-4][-21:]

        if (symbol in symbols) and (date == (start_str + "_" + end_str)): # correct date for the symbol
            # Load the data
            path = os.path.join(data_folder, file)
            df_symb = pd.read_csv(path)

            df_symb["date"] = pd.to_datetime(df_symb["date"])
            other_columns = list(df_symb.columns)
            other_columns.remove("date")
            df_symb[other_columns] = df_symb[other_columns].astype("float32")

            # Add price and calculate return
            df[symbol] = df_symb[type_price]
            df[symbol + "_return"] = df[symbol].pct_change()  # percentage change
            sigma.append(df[symbol + "_return"][1:].std())

    df["date"] = df_symb["date"]  # add at lease one date column
    df = df.dropna()  # remove NaN values
    df.reset_index(drop=True, inplace=True)

    return df, sigma


def create_factors(start=datetime_initial_start, end=datetime_initial_end):
    # Download the data
    df = pd.read_excel(factors_file, engine="openpyxl")
    sigma = []

    for factor in list(df.columns)[1:]:
        df[factor] = df[factor].astype("float32")
        # Add price and calculate return
        df[factor + "_return"] = df[factor].pct_change()  # percentage change
        sigma.append(df[factor + "_return"].std())

    df = df.dropna()  # remove NaN values
    df["date"] = pd.to_datetime(df["Date"])
    df.drop(columns=["Date"], inplace=True)
    df = df[df["date"] >= start]
    end_shift = end + datetime.timedelta(days=1)
    df = df[df["date"] < end_shift]
    df.reset_index(drop=True, inplace=True)
    return df, sigma[1:]
