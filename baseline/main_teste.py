import os
import datetime

import logging
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import utils as uls
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.hill_climbing import HillClimbing
from algorithms.random_search import RandomSearch
from problems.continuous import Continuous
from algorithms.pso import PSO



# setup logger
file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "LogFiles/" + (str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv"))
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

# ++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
# ++++++++++++++++++++++++++
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33, random_state=0)

# setup benchmarks
n_runs = 5
n_gen = 100
validation_p = .2
validation_threshold = .07

# Genetic Algorithm setup
ps = 50
p_c = .8
radius = .2
pressure = .2

# Simulated Annealing setup
ns = ps
control = 2
update_rate = 0.9

# setup PSO
social = 1.
cognitve = 1.
intertia = .1
swarm_size = 10
n_iter = 100

# data frame with results ##################################

#results = pd.DataFrame(columns=['op9', 'op10'])


for seed in range(n_runs):
    random_state = uls.get_random_state(seed)

    #++++++++++++++++++++++++++
    # THE ANN
    # restrictions:
    # - 2 h.l. with Sigmoid a.f.
    # - Softmax a.f. at output
    # - 20%, out of remaining 67%, for validation
    #++++++++++++++++++++++++++
    # ann's architecture
    hidden_architecture = np.array([[10, sigmoid], [10, sigmoid]])
    n_weights = X_train.shape[1]*10*10*len(digits.target_names)
    # create ann
    ann_i = ANN(hidden_architecture, softmax, accuracy_score, (X_train, y_train), random_state, validation_p, digits.target_names)

    #++++++++++++++++++++++++++
    # THE PROBLEM INSTANCE
    # - optimization of ANN's weights is a COP
    #++++++++++++++++++++++++++
    ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                     minimization=False, validation_threshold=validation_threshold)

    #++++++++++++++++++++++++++
    # THE SEARCH
    # restrictions:
    # - 5000 offsprings/run max*
    # - 50 offsprings/generation max*
    # - use at least 5 runs for your benchmarks
    # * including reproduction
    #++++++++++++++++++++++++++

    # genetic alg.
    ga1 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.2)

    ga2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.4)

    ga3 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.4)

    ga4 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.2)

    ga5 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)

    ga6 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, 0.9, uls.parametrized_ball_mutation(radius), 0.2)

    ga7 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, 0.9, uls.parametrized_ball_mutation(radius), 0.1)

    ga8 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.1),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)

    ga9 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.3),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)

    ###

    ga10 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)

    '''
    ga11 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.random_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)
    '''
    ga12 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.shuffle_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)

    ##

    ga13 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.two_point_crossover, p_c, uls.shuffle_mutation, 0.4)
    ga14 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.two_point_crossover, p_c, uls.scramble_mutation, 0.4)

    # sim. ann.
    sa1 = SimulatedAnnealing(ann_op_i, random_state, ps, uls.parametrized_ball_mutation(radius), control, update_rate)

    sa2 = SimulatedAnnealing(ann_op_i, random_state, ps, uls.parametrized_ball_mutation(radius), control, 0.95)

    ## one point

    op1 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.2)

    op2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.4)

    op3 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                          uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.4)

    ## two point
                        # ball mutation
    tp1 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                     uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.1)

    tp2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.2)

    tp3 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.4)

                        # scramble
    tp4 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.scramble_mutation, 0.1)

    tp5 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.scramble_mutation, 0.2)

    tp6 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.two_point_crossover, p_c, uls.scramble_mutation, 0.4)

                        # ball mutation 2
    tp7 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                     uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(0.1), 0.2)
    tp8 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                     uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.2)
    tp9 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                     uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(0.5), 0.2)

    #onepoint 2
    op4 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.5), 0.4)
    op5 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.4), 0.4)
    op6 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.5)
    # op 6 é o melhor, fit = 0.38, corremos outra vez e dá resultado diferente
    op7 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.scramble_mutation, 0.4)
    op8 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.shuffle_mutation, 0.4)




    # one point -> increase Mutation
    op9 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.5)
    op10 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.6)
    op11 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.7)
    op12 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.8)

    ##

    op13 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.3), 0.8)
    op14 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.35), 0.85)
    # tp
    tp10 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), 0.7)


    ##

    op15 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.1),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.35), 0.85)
    op16 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.3),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.4), 0.85)
    op17 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.one_point_crossover, p_c, uls.shuffle_mutation, 0.8)


    ##
    op17 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.4),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.4), 0.85)
    op18 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.3),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.5), 0.85)
    op19 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.4),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.4), 0.8)
    op20 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.4),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.2), 0.8)


    # Three point crossover and reverse mutations

    thp1 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                           uls.three_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.6)
    thp2 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.three_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.7)
    thp3 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(pressure),
                            uls.three_point_crossover, p_c, uls.parametrized_ball_mutation(0.3), 0.8)


    ## changing mutation one point

    op21 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.1),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.35), 0.7)
    op22 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.05),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.35), 0.8)
    op23 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.5),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.45), 0.8)

    # changing mutation >60 iterations
    op24 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.1),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.35), 0.85)


    # two point crossover
    tp11 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.1),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.35), 0.8)

    tp12 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.3),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.35), 0.8)

    tp13 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.4),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.35), 0.8)

    tp14 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.5),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.35), 0.8)


    # one point 15 tunning

    op25 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.15),
                            uls.one_point_crossover, p_c, uls.parametrized_ball_mutation(0.37), 0.85)

    op26 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.2),
                            uls.one_point_crossover, 0.85, uls.parametrized_ball_mutation(0.37), 0.85)

    op27 = GeneticAlgorithm(ann_op_i, random_state, ps, uls.parametrized_tournament_selection(0.23),
                            uls.one_point_crossover, 0.9, uls.parametrized_ball_mutation(0.37), 0.85)

    # sa -> bad results


    # hill climbing
    hc1 = HillClimbing(ann_op_i, random_state, ps, uls.parametrized_ball_mutation(radius))

    # PSO #################### NOT WORKING
    pso1 = PSO(ann_op_i, random_state, swarm_size, social, cognitve, inertia=0.5)

    # random search
    rs1 = RandomSearch(ann_op_i, random_state)

    search_algorithms = [op25,op26,op27]

    # initialize search algorithms
    [algorithm.initialize() for algorithm in search_algorithms]

    # execute search
    [algorithm.search(n_iterations=n_gen, report=True) for algorithm in search_algorithms]


    # data frame with results ##################################
    if seed == 0:
        results = pd.DataFrame(columns=['op25','op26','op27'])
    results.loc[seed] = [algorithm.best_solution.fitness for algorithm in search_algorithms]
    #results.loc[seed] = [op9.best_solution.fitness]
    ''',
                         op10.best_solution.fitness,
                         op11.best_solution.fitness,
                         op12.best_solution.fitness]'''


# analyze data-frame
print("Testing Data means:\n" + str(results))
print("Descriptive statistics:\n" + str(results.describe()))



file_name = "_results.txt"
full_path = os.path.realpath(__file__)
file_path = os.path.dirname(full_path)+'\\LogFiles\\'
file_path+=str(datetime.datetime.now().date()) + \
           "-" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) + file_name

results.to_csv(file_path)


# boxplot
boxplot_name = "boxplot.png"
full_path = os.path.realpath(__file__)
file_path = os.path.dirname(full_path)+'\\LogFiles\\'
file_path+=str(datetime.datetime.now().date()) + \
           "-" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) + boxplot_name
results.boxplot()
plt.savefig(file_path)


#++++++++++++++++++++++++++
# TEST
# - test algorithms on unseen data
#++++++++++++++++++++++++++
make_plots = False

for algorithm in search_algorithms:
    ann_i._set_weights(algorithm.best_solution.representation)
    y_pred = ann_i.stimulate_with(X_test, False)
    accuracy = accuracy_score(y_test, y_pred)
    print("Unseen Accuracy of %s: %.2f" % (algorithm.__class__, accuracy_score(y_test, y_pred)))


    if make_plots:
        n_images = 25
        images = X_test[0:n_images].reshape((n_images, 8, 8))
        f = plt.figure(figsize=(10, 10))
        for i in range(n_images):
            sub = f.add_subplot(5, 5, i + 1)
            sub.imshow(images[i], cmap=plt.get_cmap("Greens") if y_pred[i] == y_test[i] else plt.get_cmap("Reds"))
            plt.xticks([])
            plt.yticks([])
            sub.set_title('y^: %i, y: %i' % (y_pred[i], y_test[i]))
        f.suptitle('Testing classifier on unseen data')
        plt.show()


