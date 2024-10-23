############################################################
# By Santiago Mora cruz and Gabriel Reynoso Escamilla
# 
# Inspiration from Stock Portfolio Optimization - Ryan O'Connell
# Special thanks to Professor Salvador Hinojosa
############################################################

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st

############# Data ##############
# Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ',
       'V', 'UNH', 'JPM', 'PG', 'HD', 'MA', 'BAC', 'XOM', 'PFE', 'KO', 'MRK',
       'PEP', 'ABBV', 'CSCO', 'AVGO', 'TMO', 'LLY', 'NFLX', 'COST', 'ORCL',
       'NKE', 'DIS', 'ABT', 'CRM', 'ACN', 'TXN', 'VZ', 'ADBE', 'MCD', 'HON',
       'WMT', 'INTC', 'QCOM', 'UPS', 'MS', 'LIN', 'CVX', 'T', 'UNP', 'LOW',
       'SBUX', 'NEE', 'PM', 'AMGN', 'DHR', 'BA', 'BMY', 'SPGI', 'CAT', 'GS',
       'RTX', 'BLK', 'SCHW', 'AMT', 'IBM', 'MDLZ', 'ISRG', 'NOW', 'PLD',
       'INTU', 'DE', 'BKNG', 'MO', 'MDT', 'CI', 'CB', 'LMT', 'MMC', 'TGT',
       'GE', 'PYPL', 'ELV', 'ZTS', 'ADI', 'SYK', 'DUK', 'EW', 'SO', 'CL',
       'VRTX', 'FIS', 'ICE', 'APD', 'ETN', 'SHW', 'CCI', 'GM']

############ Data functions ##############3
def obtain_from_yahoo(years_backwards, tickers = tickers):
    end_date = datetime.today()
    start_date = end_date - timedelta(days = years_backwards*365)
    df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start = start_date, end = end_date)
        df[ticker] = data['Adj Close']
    return df

def log_returns_covariance_matrix(df):
    log_returns = np.log(df / df.shift(1)) # ln (precio de día n - precio del día n-1)
    log_returns = log_returns.dropna()

    cov_matrix = log_returns.cov() * 252 # 252 para valores anuales 
    return log_returns, cov_matrix

########### Portfolio Metrics ###############
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights) * 252 # 252 para anual

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate = 0.02):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

########### Genetic Algorithm #####################

# Crear Individuo, normalizando para asegurar que la suma de los pesos sea igual a 1 
def createIndividual(nbAssets):
    weights = np.random.rand(nbAssets)
    return weights / np.sum(weights)

# Two-point crossover
def combine(parentA, parentB, cRate):
    if random.random() <= cRate:
        # Seleccionar dos crossover points
        cPoint1 = np.random.randint(1, len(parentA) - 1)
        cPoint2 = np.random.randint(cPoint1 + 1, len(parentA))

        # Crear offsprings usando two-point crossover
        offspringA = np.concatenate((parentA[:cPoint1], parentB[cPoint1:cPoint2], parentA[cPoint2:]))
        offspringB = np.concatenate((parentB[:cPoint1], parentA[cPoint1:cPoint2], parentB[cPoint2:]))
    else:
        offspringA = np.copy(parentA)
        offspringB = np.copy(parentB)
    
    # Normalizar para asegurar que la suma de los pesos sea igual a 1 
    offspringA = offspringA / np.sum(offspringA)
    offspringB = offspringB / np.sum(offspringB)
    
    return offspringA, offspringB


# Mutación asegurando que la suma de los pesos sea igual a 1 y que no haya valores negativos
 
def mutate(individual, mRate):
    for i in range(len(individual)):
        if random.random() <= mRate:
            individual[i] += np.random.uniform(-0.2, 0.2)
    # Ensure no negative weights
    individual = np.clip(individual, 0, None)
    # Ensure weights sum to 1
    individual = individual / np.sum(individual)
    return individual

# Fitness function (Sharpe Ratio)
def evaluate(individual, log_returns, cov_matrix, risk_free_rate=0.02):
    return sharpe_ratio(individual, log_returns, cov_matrix, risk_free_rate)

# Selección (torneo)
def select(population, evaluation, tournamentSize):
    winner = np.random.randint(0, len(population))
    for _ in range(tournamentSize - 1):
        rival = np.random.randint(0, len(population))
        if evaluation[rival] > evaluation[winner]:
            winner = rival
    return population[winner]

# Dominancia: True si a domina a b
def dominates(a, b):
    return (a[0] > b[0] and a[1] <= b[1]) or (a[0] >= b[0] and a[1] < b[1])

# evaluación Multi-objetivo: Regresa ambos objetivos
def evaluate_multiobjective(individual, log_returns, cov_matrix):
    return expected_return(individual, log_returns), standard_deviation(individual, cov_matrix)

# Aproximación al frente de Pareto: soluciones no dominadas
def non_dominated_sorting(population_evaluations):
    pareto_front = []
    for i, eval_i in enumerate(population_evaluations):
        dominated = False
        for j, eval_j in enumerate(population_evaluations):
            if i != j and dominates(eval_j, eval_i):
                dominated = True
                break
        if not dominated:
            pareto_front.append(i)
    return pareto_front

# Algoritmo genético multiobjetivo (Maximizar retorno, minimizar riesgo)
def geneticAlgorithm_multiobjective(nbAssets, populationSize, cRate, mRate, generations, log_returns, cov_matrix):
    population = [createIndividual(nbAssets) for _ in range(populationSize)]
    evaluations = [evaluate_multiobjective(ind, log_returns, cov_matrix) for ind in population]
    
    # Finding the initial Pareto front
    pareto_indices = non_dominated_sorting(evaluations)
    pareto_population = [population[i] for i in pareto_indices]
    pareto_evaluations = [evaluations[i] for i in pareto_indices]

    for gen in range(generations):
        new_population = []
        for _ in range(populationSize // 2):
            parentA = select(population, [ev[0] - ev[1] for ev in evaluations], 3)
            parentB = select(population, [ev[0] - ev[1] for ev in evaluations], 3)
            offspringA, offspringB = combine(parentA, parentB, cRate)
            new_population.extend([offspringA, offspringB])
        
        population = [mutate(ind, mRate) for ind in new_population]
        evaluations = [evaluate_multiobjective(ind, log_returns, cov_matrix) for ind in population]

        # Actualizar frente de Pareto
        all_population = pareto_population + population
        all_evaluations = pareto_evaluations + evaluations
        pareto_indices = non_dominated_sorting(all_evaluations)
        pareto_population = [all_population[i] for i in pareto_indices]
        pareto_evaluations = [all_evaluations[i] for i in pareto_indices]

    return pareto_population, pareto_evaluations

##################### GA Results ##########################

def mean_sharpe_ratio(solutions, values):
    sr = []
    # Sharpe ratio medio
    for sol, val in zip(solutions, values):
        sr.append((val[0] - 0.02) / val[1])
    return np.mean(sr)

def get_returns_risks(solutions, values):
    returns = []
    risks = []
    for sol, val in zip(solutions, values):
        returns.append(val[0])
        risks.append(val[1])
    return returns, risks

def plot_pareto_front(returns, risks):
    # Crear la figura
    retris = pd.DataFrame({
        'Return': returns,
        'Risk': risks
    })
    return(retris)

def plot_single_solution(solution):
    # Graficar solo los stocks cuyos pesos son mayores que 0
    filtered_tickers = [tickers[i] for i in range(len(solution)) if solution[i] > 0.002]
    filtered_weights = [solution[i] for i in range(len(solution)) if solution[i] > 0.002]

    single_sol = pd.DataFrame({
        'Asset': filtered_tickers,
        'Weight': filtered_weights
    })

    return single_sol

def sort_pareto_values_and_solutions(pareto_values, solutions):
    # Zip pareto_values y solutions para mantener el vínculo
    combined = list(zip(pareto_values, solutions))
    
    # Ordenar en base al segundo elemento de pareto_values
    sorted_combined = sorted(combined, key=lambda x: x[0][1])
    
    # Separar los valores ordenados y las soluciones
    sorted_values = [x[0] for x in sorted_combined]
    sorted_solutions = [x[1] for x in sorted_combined]
    
    return sorted_values, sorted_solutions

################## Main #####################

def main():
    st.header('Multi-Objective Genetic Algorithm for Portfolio Optimization')
    st.markdown("This website showcases a **Multi-Objective Genetic Algorithm** for **portfolio optimization**, focusing on maximizing expected returns while minimizing risk (covariance). Users can **customize** the genetic algorithm's **parameters**, including population size, crossover rate, mutation rate, and the number of generations. The algorithm generates a **set of optimal solutions**, displayed on a **Pareto front**, allowing users to visualize the **trade-offs** between **risk and return**. The platform is designed for **educational purposes**, enabling users to experiment with **real market data** to explore **efficient portfolio allocations**.")
    
    st.write('\n')
    st.write('\n')

    if 'play' not in st.session_state:
        st.session_state.play = False
        st.session_state.solutions = None
        st.session_state.valuess = None

    with st.sidebar:
        yearsback = st.slider("How far back do you want the analysis to go? (in years)", 1, 5, 3)
        modify = st.checkbox("Default Parameters for GA", value=True)

        if not modify:
            pop_size = st.slider("Population Size", 10, 50, 30)
            crossover_rate = st.slider("Crossover Rate", 0.01, 1.00, 0.80)
            mutation_rate = st.slider("Mutation Rate", 0.01, 1.00, 0.05)
            generations = st.slider("Number of Generations", 50, 500, 200)
        else:
            pop_size = 30
            crossover_rate = 0.8
            mutation_rate = 0.05
            generations = 200
        
        agree = st.checkbox('I have read the disclaimer and agree to proceed')
        if agree:
            if st.button('See Results!'):
                st.session_state.play = True

        with st.expander('Disclaimer'):
            st.write('This project is intended for educational purposes only and is not for profit. The information and tools provided should be used with caution, as the results may not always be accurate or precise. Users are encouraged to verify the outputs and apply their own judgement before making any decisions based on the results. The creators of this tool cannot, in any way, be held liable for any losses users may suffer. This tool is not intended to replace professional financial advice.')

        with st.expander('About'):
            st.markdown("Created by [Santiago Mora](https://github.com/santiagomora2) in association with [Gabriel Reynoso](https://devpost.com/gre20003). Special thanks to Professor Salvador Hinojosa, with some inspiration form Ryan O'Conell's Stock Portfolio Optimization")


    if st.session_state.play and st.session_state.solutions is None:
        # Only retrieve data and run the GA if solutions are not already stored
        with st.spinner('Retrieving information from Yahoo! Finance'):
            df = obtain_from_yahoo(yearsback)

        with st.spinner('Computing returns and covariance'):
            log_returns, cov_matrix = log_returns_covariance_matrix(df)

        with st.spinner('Running Genetic Algorithm (Almost there!)'):
            pareto_solutions, pareto_values = geneticAlgorithm_multiobjective(len(tickers), pop_size, crossover_rate, mutation_rate, generations, log_returns, cov_matrix)
            
            values, solutions = sort_pareto_values_and_solutions(pareto_values, pareto_solutions)

            # Save the results in session state so they don't need to be recomputed
            st.session_state.solutions = solutions
            st.session_state.valuess = values

    if st.session_state.solutions is not None:
        container = st.container()
        with container:
            st.subheader('Solution')
            # Calculate and display the mean Sharpe ratio
            msr = mean_sharpe_ratio(st.session_state.solutions, st.session_state.valuess)
            msrs = f'Estimated mean Sharpe Ratio of solutions in Pareto front: {msr:.4f}'
            st.write(msrs + '\n')

            # Get returns and risks for plotting the Pareto front
            returns, risks = get_returns_risks(st.session_state.solutions, st.session_state.valuess)

            st.write('To view a single solution from the Pareto front, select a solution number and click Show Solution! (ordered from least to most risk):')
            
            # Allow the user to select a specific solution based on risk
            solution_index = st.slider("Select a solution:", 1, len(st.session_state.solutions), 1)
            
            
            inf = f'Estimated annual return of the selected solution: {100*st.session_state.valuess[solution_index-1][0]:.2f} %'
            inf2 = f'Estimated annual risk of the selected solution: {100*st.session_state.valuess[solution_index-1][1]:.2f} %'
            st.write(inf)
            st.write(inf2)
            # Simply plot the selected solution without re-running the algorithm
            single_sol = plot_single_solution(st.session_state.solutions[solution_index - 1])

            st.write('This barplot shows how much each individual asset represents of the total budget, for this individual solution. \n\n')

            st.bar_chart(single_sol, x = 'Asset', y='Weight', horizontal = True)
            
            # Plot and display the Pareto front
            retris = plot_pareto_front(returns, risks)
            st.subheader('\nVisualization of solutions in Pareto Front:')

            st.write("In this graph, every dot is a solution in the estimated Pareto front, with it's own estimated risk and retun\n\n")

            st.scatter_chart(retris, x = 'Return', y = 'Risk')
            
            st.info('If you wish to try other parameters for the GA, please refresh the page', icon="ℹ️")


if __name__ == "__main__":
    main()