import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle

from functools import reduce

def get_random_state(seed):
    return np.random.RandomState(seed)


def random_boolean_1D_array(length, random_state):
    return random_state.choice([True, False], length)


def bit_flip(bit_string, random_state):
    neighbour = bit_string.copy()
    index = random_state.randint(0, len(neighbour))
    neighbour[index] = not neighbour[index]

    return neighbour


def parametrized_iterative_bit_flip(prob):
    def iterative_bit_flip(bit_string, random_state):
        neighbor = bit_string.copy()
        for index in range(len(neighbor)):
            if random_state.uniform() < prob:
                neighbor[index] = not neighbor[index]
        return neighbor

    return iterative_bit_flip


def random_float_1D_array(hypercube, random_state):
    return np.array([random_state.uniform(tuple_[0], tuple_[1])
                     for tuple_ in hypercube])


def random_float_cbound_1D_array(dimensions, l_cbound, u_cbound, random_state):
    return random_state.uniform(lower=l_cbound, upper=u_cbound, size=dimensions)

############################################# MUTATION ################################################################

def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array(
            [random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])

    return ball_mutation

def shuffle_mutation(offsp, random_state):
    len_ = len(offsp)
    off_r = offsp.copy()
    point = random_state.randint(len_)
    point2 = random_state.randint(len_)
    off_r[point] = offsp[point2]
    off_r[point2] = offsp[point]
    return off_r

def scramble_mutation(offsp, random_state):
    len_ = len(offsp)
    off_r = offsp.copy()
    point = random_state.randint(len_)
    point2 = random_state.randint(low=point, high=len_)
    off_r = np.concatenate((offsp[0:point], shuffle(offsp[point:point2], random_state=random_state), offsp[point2:len_]))
    return off_r

def reverse_mutation(offsp, random_state):
    len_ = len(offsp)
    off_r = offsp.copy()
    reversed_offsp = offsp[::-1].copy()
    point = random_state.randint(len_)
    point2 = random_state.randint(low=point, high=len_)
    off_r = np.concatenate((offsp[0:point],reversed_offsp[point:point2], offsp[point2:len_]))
    return off_r


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=0)#len(point.shape) % 2 - 1)


def rastrigin(point):
    a = len(point) * 10 if len(point.shape) <= 2 else point.shape[0] * 10
    return a + np.sum(point ** 2 - 10. * np.cos(np.pi * 2. * point), axis=0)

############################################# Crossover ################################################################

def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

def two_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    point2 = random_state.randint(low=point, high=len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:point2], p1_r[point2:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:point2], p2_r[point2:len_]))
    return off1_r, off2_r

def three_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    point2 = random_state.randint(low=point, high=len_)
    point3 = random_state.randint(low=point2, high=len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:point2], p1_r[point2:point3], p2_r[point3:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:point2], p2_r[point2:point3], p1_r[point3:len_]))
    return off1_r, off2_r

'''
def uniform_crossover(p1_r, p2_r, random_state, p_c = p_c):
    len_ = len(p1_r)
    off1_r, off2_r = p1_r,p2_r
    for point in range(len_):
        if random_state.uniform() < p_c:
            off1_r[point] = p2_r[point]
            off2_r[point] = p1_r[point]
    return off1_r, off2_r
'''
def random_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    off1_r, off2_r = p1_r, p2_r
    n_points = random_state.randint(len_)
    for n in range(n_points):
        point = random_state.randint(len_)
        point2 = random_state.randint(len_)
        off1_r[point] = p2_r[point2]
        off2_r[point2] = p1_r[point]
    return off1_r, off2_r

def shuffle_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    p1_r = shuffle(p1_r, random_state=random_state)
    p2_r = shuffle(p2_r, random_state=random_state)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r

#############################################################################################################

def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
    return [(l_cbound, u_cbound) for _ in range(dimensions)]


def parametrized_ann(ann_i):
    def ann_ff(weights):
        return ann_i.stimulate(weights)

    return ann_ff


def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population) * pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection


class Dplot():

    def background_plot(self, hypercube, function_):
        dim1_min = hypercube[0][0]
        dim1_max = hypercube[0][1]
        dim2_min = hypercube[1][0]
        dim2_max = hypercube[1][1]

        x0 = np.arange(dim1_min, dim1_max, 0.1)
        x1 = np.arange(dim2_min, dim2_max, 0.1)
        x0_grid, x1_grid = np.meshgrid(x0, x1)

        x = np.array([x0_grid, x1_grid])
        y_grid = function_(x)

        # plot
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlim(dim1_min, dim1_max)
        self.ax.set_ylim(dim2_min, dim2_max)
        self.ax.plot_surface(x0_grid, x1_grid, y_grid, rstride=1, cstride=1, color="green", alpha=0.15) # cmap=cm.coolwarm,

    def iterative_plot(self, points, z, best=None):
        col = "k" if best is None else np.where(z == best, 'r', 'k')
        size = 75 if best is None else np.where(z == best, 150, 75)
        self.scatter = self.ax.scatter(points[0], points[1], z, s=size, alpha=0.75, c=col)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.scatter.remove()