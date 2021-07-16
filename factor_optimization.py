"""
Optimization of the 5 factors
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import gmean
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from dataset_creation import *
from optimization import *
from portfolio import portfolio_simulation
from config import *

initial_start = "2014-01-01"
initial_end = "2020-12-31"

datetime_initial_start = datetime.datetime.strptime(initial_start, "%Y-%m-%d")
datetime_initial_end = datetime.datetime.strptime(initial_end, "%Y-%m-%d")
window = 6 # in months

# Initialize the moving window
datetime_start = datetime.datetime.strptime(initial_start, "%Y-%m-%d")  # these two quantities will increase in the for loop
datetime_end = datetime_start + relativedelta(months=window)

initial_datetime_start_simu = datetime_end + datetime.timedelta(days=1) # simulation start here

datetime_start_simu = initial_datetime_start_simu  # these two quantities will increase in the for loop
datetime_end_simu = datetime_start_simu + relativedelta(months=1)

total_time = [[] for _ in range(len(optimization_method))]
total_return = [[] for _ in range(len(optimization_method))]

rolling_window = (int(initial_end[2:4]) - int(initial_start[2:4]) + 1 )*12 - window 
for _ in tqdm(range(rolling_window)):  # launch the simulations and move the rolling window
    names = []  # name of the method (to plot the result later)
    opt_methods = portfolio_simulation(data_folder,
                                       optimization_method, datetime_start,
                                       datetime_end, datetime_start_simu,
                                       datetime_end_simu, lbd, min_return,
                                       symbols=factors, low_bound=0.1,
                                       high_bound=0.3, factor=True)
    i = 0
    for (name, [time, return_list]) in zip(opt_methods.keys(), opt_methods.values()):
        total_time[i] = total_time[i] + time
        total_return[i] = total_return[i] + return_list
        names.append(name)
        i += 1

    # roll the window
    datetime_end = datetime_end_simu
    datetime_start = datetime_end - relativedelta(months=window)

    datetime_start_simu = datetime_end_simu + datetime.timedelta(days=1)
    datetime_end_simu = datetime_start_simu + relativedelta(months=1)

# Extract benchmark_price for comparison
benchmark = create_factors(start=initial_datetime_start_simu, end=datetime_end_simu)
benchmark_price = benchmark[0]["BENCHMARK"].tolist()

# List of prices calculated with the returns
initial_price = benchmark[0]["BENCHMARK"].iloc[0]  # price of benchmark when we start the simulations
price = [[initial_price] for opt in range(len(optimization_method))]
for i in range(len(total_time[0])):
    for opt in range(len(optimization_method)):
        price[opt].append(price[opt][-1] * (1 + total_return[opt][i]))
price = [price[opt][1:] for opt in range(len(optimization_method))]  # remove the initial price

# Extract the price for the equal weights strategy
for opt in range(len(names)):
    if names[opt] == "equal_weights":
        ind_equal_weights = opt
equal_weights_price = price[opt]


# Plot the result
def plot_comparison(total_time, price, names, benchmark_price):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)

    best_strat_name = ""  # store the best strategy
    best_return = -np.inf

    for opt in range(len(optimization_method)):  # plot the price for the different methods
        strat_name = names[opt]
        axs.plot(total_time[opt], price[opt], label="{strat_name}".format(strat_name=strat_name))

        return_opt = (price[opt][-1] - price[opt][0]) / price[opt][0]
        if return_opt > best_return:  # it's the best strategy
            best_return = return_opt
            best_strat_name = strat_name

    axs.plot(total_time[0], benchmark_price, label="{strat_name}".format(strat_name="Benchmark"))
    bench_return = (benchmark_price[-1] - benchmark_price[0]) / benchmark_price[0]  # return of the benchmark

    axs.set_title("Evolution of the portfolio for different optimization strategies.")
    axs.set_xlabel("Date")
    axs.set_ylabel("Price")

    fig.suptitle("Comparison of different portfolio optimization and a benchmark (factors). Best strategy : {best_strat_name}. Return : {best_return}. Benchmark return : {bench_return}".format(best_strat_name=best_strat_name, best_return=round(best_return, 5), bench_return=round(bench_return, 5)), fontsize=12)
    plt.legend(loc="best")
    plt.show()
    return best_strat_name, best_return, bench_return


def plot_comparison_last_year(total_time, price, names, benchmark_price, best_strat_name, best_return, bench_return):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)

    for opt in range(len(optimization_method)):  # plot the price for the different methods
        strat_name = names[opt]
        axs.plot(total_time[opt][-365:], price[opt][-365:], label="{strat_name}".format(strat_name=strat_name))

    axs.plot(total_time[0][-365:], benchmark_price[-365:], label="{strat_name}".format(strat_name="Benchmark"))

    axs.set_title("Evolution of the portfolio for different optimization strategies (for the last month).")
    axs.set_xlabel("Date")
    axs.set_ylabel("Price")

    fig.suptitle("Comparison of different portfolio optimization and a benchmark (factors). Best strategy : {best_strat_name}. Return : {best_return}. Benchmark return : {bench_return}".format(best_strat_name=best_strat_name, best_return=round(best_return, 5), bench_return=round(bench_return, 5)), fontsize=12)
    plt.legend(loc="best")
    plt.show()


def plot_modern_portfolio_theory_factor(price, names, benchmark_price):
    df = pd.DataFrame()
    for i in range(len(names)):
        df[names[i]] = price[i]
    df["BENCHMARK"] = benchmark_price
    for i in range(len(names)):
        df[names[i]] = df[names[i]].pct_change()
    df["BENCHMARK"] = df["BENCHMARK"].pct_change()
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
    plt.title("Modern Portfolio Theory (factors)")
    plt.tight_layout()
    plt.show()


# Number of year each strategy beat the benchmark
def beat_the_benchmark(price, names, benchmark_price):
    print("\n########\nCheck if the price is higher than the benchmark for a given strategy each year :\n")
    nb_beat_benchmark = [0 for _ in range(len(names))]
    nb_year = 0
    for i in range(251, len(price[0]), 252):
        for opt in range(len(names)):
            if price[opt][i] >= benchmark_price[i]:
                nb_beat_benchmark[opt] += 1
        nb_year += 1
    
    for opt in range(len(names)):
        print("{strat_name} beat the benchmark {nb_beat}/{nb_year} of the time\n".format(strat_name=names[opt], nb_beat=nb_beat_benchmark[opt], nb_year=nb_year))

    print("\n########\nCheck if the return each year is higher than the one of the benchmark for a given strategy :\n")
    nb_beat_benchmark = [0 for _ in range(len(names))]
    nb_year = 0
    for i in range(251, len(price[0]), 252):
        for opt in range(len(names)):
            if ((price[opt][i] - price[opt][i-251]) / price[opt][i-251]) >= ((benchmark_price[i] - benchmark_price[i-251]) / benchmark_price[i-251]):
                nb_beat_benchmark[opt] += 1
        nb_year += 1
    
    for opt in range(len(names)):
        print("{strat_name} beat the benchmark {nb_beat}/{nb_year} of the time\n".format(strat_name=names[opt], nb_beat=nb_beat_benchmark[opt], nb_year=nb_year))


# Number of year each strategy beat the equal weights strategy
def beat_the_equal_weights(price, names, equal_weights_price):
    print("\n########\nCheck if the price is higher than the equal weights price for a given strategy each year :\n")
    nb_beat_equal_weights = [0 for _ in range(len(names))]
    nb_year = 0
    for i in range(251, len(price[0]), 252):
        for opt in range(len(names)):
            if price[opt][i] >= equal_weights_price[i]:
                nb_beat_equal_weights[opt] += 1
        nb_year += 1
    
    for opt in range(len(names)):
        if names[opt] != "equal_weights":
            print("{strat_name} beat the equal weights {nb_beat}/{nb_year} of the time\n".format(strat_name=names[opt], nb_beat=nb_beat_equal_weights[opt], nb_year=nb_year))

    print("\n########\nCheck if the return each year is higher than the one of the equal weights for a given strategy :\n")
    nb_beat_equal_weights = [0 for _ in range(len(names))]
    nb_year = 0
    for i in range(251, len(price[0]), 252):
        for opt in range(len(names)):
            if ((price[opt][i] - price[opt][i-251]) / price[opt][i-251]) >= ((equal_weights_price[i] - equal_weights_price[i-251]) / equal_weights_price[i-251]):
                nb_beat_equal_weights[opt] += 1
        nb_year += 1
    
    for opt in range(len(names)):
        if names[opt] != "equal_weights":
            print("{strat_name} beat the equal weights {nb_beat}/{nb_year} of the time\n".format(strat_name=names[opt], nb_beat=nb_beat_equal_weights[opt], nb_year=nb_year))




best_strat_name, best_return, bench_return = plot_comparison(total_time, price, names, benchmark_price)
plot_comparison_last_year(total_time, price, names, benchmark_price, best_strat_name, best_return, bench_return)
plot_modern_portfolio_theory_factor(price, names, benchmark_price)
beat_the_benchmark(price, names, benchmark_price)
beat_the_equal_weights(price, names, equal_weights_price)
