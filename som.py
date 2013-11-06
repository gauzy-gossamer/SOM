import sys

from PIL import Image

import random
import math
import numpy as np

import theano
import theano.tensor as T

# Self Organizing Map
#   as explained here - http://www.ai-junkie.com/ann/som/som1.html
# Parameters:
#   input_w - number of input parameters
#   width, height - dimensions of the resulting map
#   iterations  - number of iterations to run the algorithm
#   start_lr - initial learning rate
#   print_fr - print an image every n iterations

class SOM(object):
    def __init__(self, input_w = 3, width = 10, height = 10, iterations = 800, start_lr = 0.1, print_fr = 100):
        coordinates = [[i,j] for i in range(height) for j in range(width)]
        self.lattice = theano.shared(value=np.asarray(coordinates,
            dtype=theano.config.floatX),
            name='lattice', borrow=True)

        self.radius = max(width, height)/2
        self.iterations = iterations
        self.start_lr = start_lr
        self.time_const = iterations/math.log(self.radius)

        self.W = theano.shared(value=np.asarray(np.random.rand(width*height, input_w), 
            dtype=theano.config.floatX),
            name='W', borrow=True)

        self.input_w = input_w
        self.width = width
        self.height = height
        self.print_fr = print_fr

        self.it = theano.shared(np.asarray(1,
            dtype=theano.config.floatX))

    def min_dist(self, data):
        return T.argmin(T.sqrt(T.sum((data - self.W)**2, 1)))

    def in_neighbourhood(self, point):
        return T.sqrt(T.sum((point - self.lattice)**2,1))

    def compute_influence(self, dists):
        return T.shape_padright(T.exp(-dists**2/(2*self.nhood_radius**2)),1)

    def ret_w(self):
        return self.W.astype(theano.config.floatX)

    def print_image(self, iteration):
        l = len(self.last_W[0])/3

        # convert high dimensional data to RGB representation

        tmp_rgb = [[ k for k in [sum(self.last_W[i*self.height + j][l*c:l*c + l]) for c in range(3)] ] 
            for j in range(self.width) for i in range(self.height)]
        max_k = np.max(tmp_rgb)
        arr = [[[ tmp_rgb[i*self.height + j][c]/max_k*255 for c in range(3)]
            for j in range(self.width)] for i in range(self.height)]

      #  RGB colors
      #  arr = [[[ self.last_W[i*self.width + j][c]*256 for c in range(3)] for j in range(self.width)] for i in range(self.height)]

      #  grayscale
      #  arr = [[ (self.last_W[i*self.height + j][0]*65536) + (self.last_W[i*self.height + j][1]*256) + self.last_W[i*self.height + j][2] 
      #     for j in range(self.width)] for i in range(self.height)]
        im = Image.fromarray(np.uint8(arr))

        im = im.resize((self.width*16, self.height*16))

        im.save('som_images/output_%i.gif' %(iteration))

    def train(self, data):
        data = np.asarray(data, dtype=theano.config.floatX)
        val = T.vector('val')

        min_idx = self.min_dist(val)

        self.nhood_radius = self.radius*T.exp(-self.it/self.time_const)

        dists = self.in_neighbourhood(self.lattice[min_idx])

        in_nhood = dists < self.nhood_radius

        lr = self.start_lr*T.exp(-self.it/self.iterations)

        updates = [(self.W, self.W + self.compute_influence(dists)*T.shape_padright(in_nhood,1)*lr*(val - self.W))]

        epoch = theano.function(inputs=[val], outputs=self.ret_w(), updates=updates)

        update_iteration = theano.function(inputs=[], outputs=self.it,
            updates={self.it: self.it + 1})

        self.last_W = None

        for i in range(self.iterations):
            index = np.random.random_integers(0, len(data) - 1)
            self.last_W = epoch(data[index])

            #index += 1
            #if index >= len(data):
            #    index = 0

            self.it = update_iteration()

            print i

            # print an image every 100 epochs
            if (i + 1) % self.print_fr == 0:
                self.print_image(i)


if __name__ == '__main__':
    # test dataset
    dataset = [ 
        [1, 0,   0], 
        [0, 1,   0], 
        [0, 0.5, 0.25],
        [0, 0,   1], 
        [0, 0,   0.5],
        [1, 1,   0.2],
        [1, 0.4, 0.25],
        [1, 0,   1]  
    ]

    # run with additional colors
    #for j in range(3):
    #    for i in range(25):
    #        point = [0,0,0]
    #        point[j] = random.random()
    #        dataset.append(point)

    print repr(dataset)

    width = 30
    height = 30

    org_map = SOM(iterations = 800, width = width, height = height, input_w = 3)

    org_map.train(dataset)
