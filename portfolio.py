import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from config import *
from optimization import *
from sklearn.linear_model import LinearRegression


def portfolio_simulation(data_folder,
                         optimization_method, start, end, start_simulation,
                         end_simulation, lbd, min_return,
                         symbols=symbols, low_bound=0, high_bound=1,
                         factor=False):
    d = {}
    if not factor:
        df, sigma = create_dateset(symbols, start, end, type_price,
                                   data_folder)
        Sigma, returns, P = create_return_and_sigma_matrix(df, sigma, False)
        df_simulation, sigma = create_dateset(symbols, start_simulation,
                                              end_simulation, type_price,
                                              data_folder)
    else:
        df, sigma = create_factors(start, end)
        Sigma, returns, P = create_return_and_sigma_matrix(df, sigma)
        df_simulation, sigma = create_factors(start_simulation,
                                              end_simulation)

    for meth in optimization_method:
        if meth == 'mean_variance_optimizing':
            mvo = mean_variance_optimizing(Sigma, returns, lbd, low_bound,
                                           high_bound, factor)
            
            nb_try = 1  # we try 10 times or less to seek for convergence
            while not(mvo.success) and nb_try < 10:
                mvo = mean_variance_optimizing(Sigma, returns, lbd, low_bound,
                                               high_bound, factor)
                nb_try += 1
            mvo = mvo.x
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return our portfolio

            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)
        elif meth == "equal_weights":

            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio

            for i in range(len(symbols)):  # return of each asset ponterated by equal coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = 1/len(Sigma)
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'min_Vol':
            mvo = min_Vol(Sigma, returns, min_return, low_bound,
                          high_bound, factor)
            nb_try = 1  # we try 10 times or less to seek for convergence
            while not(mvo.success) and nb_try < 10:
                mvo = min_Vol(Sigma, returns, min_return, low_bound, high_bound, factor)
                nb_try += 1
            mvo = mvo.x
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio

            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'sharpe_ratio':
            mvo = sharp_ratio_optimizing(Sigma, returns, low_bound,
                                         high_bound, factor)
            
            nb_try = 1  # we try 10 times or less to seek for convergence
            while not(mvo.success) and nb_try < 10:
                mvo = sharp_ratio_optimizing(Sigma, returns, low_bound,
                                             high_bound, factor)
                nb_try += 1
            mvo = mvo.x
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return our portfolio

            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'max_diversification':
            mvo = max_diversification(P, returns, low_bound,
                                      high_bound, factor)
            
            nb_try = 1  # we try 10 times or less to seek for convergence
            while not(mvo.success) and nb_try < 10:
                mvo = max_diversification(P, returns, low_bound,
                                          high_bound, factor)
                nb_try += 1
            mvo = mvo.x
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio

            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)
        elif meth == 'min_correlation':
            mvo = min_correlation(Sigma, returns, low_bound,
                                  high_bound, factor)
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio
            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'equal_risk':
            mvo = equal_risk_contribution(Sigma, returns, low_bound, high_bound,
                                          factor)
            
            nb_try = 1  # we try 10 times or less to seek for convergence
            while not(mvo.success) and nb_try < 10:
                mvo = equal_risk_contribution(Sigma, returns, low_bound, high_bound,
                                              factor)
                nb_try += 1
            mvo = mvo.x
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio
            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'russel' and factor:
            mvo = russel(df)
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio
            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'risk_parity':
            mvo = risk_parity(df, Sigma, returns)
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio
            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

        elif meth == 'nasdaq':
            mvo = ["VALUE", "GROWTH", "QUALITY", "MOMENTUM", "SIZE"]
            alpha= []
            for factor in factors:
                y = np.array(df["BENCHMARK_return"])
                x = np.array(df[factor + "_return"]).reshape((-1, 1))
                model = LinearRegression().fit(x, y)
                alpha += [model.intercept_]
            minalpha=min(alpha)
            for i in range(5):
                if minalpha == alpha[i]:
                    mvo[i] = 0
                else:
                    mvo[i] = 0.25
            return_list = [0 for _ in range(len(df_simulation))]  # contain the evolution of the total return of our portfolio
            for i in range(len(symbols)):  # return of each asset ponterated by optimized coefficients
                symbol = symbols[i]
                symbol_proportion_in_portfolio = mvo[i]
                symbol_price = df_simulation[symbol].tolist()
                for k in range(len(df_simulation)):

                    if not factor:
                        return_of_symbol = (symbol_price[k] - symbol_price[0]) / symbol_price[0]
                    else:
                        if k == 0:
                            return_of_symbol = 0
                        else:
                            return_of_symbol = (symbol_price[k] - symbol_price[k-1]) / symbol_price[k-1]
                    return_list[k] += symbol_proportion_in_portfolio * return_of_symbol

            d[meth] = (list(df_simulation["date"]), return_list)

    return d


def plot_portfolio_simulation(opt_methods):  # for stock
    fig, axs = plt.subplots(1, 1, constrained_layout=True)

    best_return_name = ""  # store the best return
    best_return = - np.inf

    for (name, [time, return_list]) in zip(opt_methods.keys(), opt_methods.values()):
        axs.plot(time, return_list, label="Return {0}".format(name))
        axs.set_title("Evolution of total return")
        axs.set_xlabel("Date")
        axs.set_ylabel("Return")

        if return_list[-1] > best_return:
            best_return = return_list[-1]
            best_return_name = name

    fig.suptitle("Evolution of total return (stocks). Best return the method {method} with a total return of {best_return}".format(method=best_return_name, best_return=round(best_return, 5)), fontsize=12)
    plt.legend(loc="best")
    plt.show()


def plot_efficient_frontier(opt_methods):  # for stock
    fig, axs = plt.subplots(1, 1, constrained_layout=True)

    for (name, [time, return_list]) in zip(opt_methods.keys(), opt_methods.values()):

        transfo_total_return = np.array(return_list) + 1
        return_list_daily = []  # first value is a nan theoricaly
        for k in range(1, len(transfo_total_return)):
            return_list_daily.append((transfo_total_return[k] / transfo_total_return[k-1]) - 1)

        df_std = pd.DataFrame()
        df_std["return"] = return_list_daily

        Y = gmean(1 + df_std).tolist()  # geometric mean
        Y[0] = 100 * ((Y[0])**(252) - 1)

        axs.scatter((100 * np.sqrt(252) * df_std.std()).tolist()[0], Y[0])
        axs.annotate(name, ((100 * np.sqrt(252) * df_std.std()).tolist()[0], Y[0]))

    axs.set_title("Efficient Frontier (Absolute space, in-sample)")
    axs.set_xlabel("Risk (annualized volatility) (%)")
    axs.set_ylabel("Expected return (geometric mean) (%)")

    fig.suptitle("Efficient frontier (stocks).", fontsize=12)
    plt.legend(loc="best")
    plt.show()

"""
# For stocks
opt_methods = portfolio_simulation(data_folder, optimization_method,
                                   start, end,
                                   start_simulation, end_simulation, lbd,
                                   min_return, symbols,
                                   low_bound=0, high_bound=1,
                                   factor=False)

# Plot
plot_portfolio_simulation(opt_methods)
plot_efficient_frontier(opt_methods)
"""
