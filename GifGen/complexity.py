'''
Idea based on Shangnan (2019) [http://inspirehep.net/record/1722270] regarding classical complexity and it's relation to entropy. This script considers a string of K n-bits (x1 x2 ... xK) in an "all-i" configuration. It then randomly updates the configuration to illustrate the long term behaviour.
'''
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import sys
plt.style.use('ja')

def update_configuration(state, n):
    K = len(state) - 1
    idx = randint(0, K)
    up = bool(randint(0, 1))
    if up:
        if state[idx] != n - 1:
            state[idx] = state[idx] + 1
        else:
            state[idx] = 0
    else:
        if state[idx] != 0:
            state[idx] = state[idx] - 1
        else:
            state[idx] = n - 1
    return state

bar_color = '#477890'
line_color = '#990D35'
text_color = '#153243'

if __name__ == '__main__':
    K = int(sys.argv[1])
    n = int(sys.argv[2])
    iters = int(sys.argv[3])
    state = np.zeros(K) + randint(0, n - 1)
    iter = 0
    plt.ion()
    plt.figure(figsize=(7, 7))
    while iter < iters:
        print(str(iter) + '/' + str(iters), end='\r')
        unique, counts = np.unique(state, return_counts=True)
        plt.bar(unique, counts, width=0.5, align='center', alpha=1.0, color=bar_color)
        axis = plt.axis()
        axis = (-0.1*n, 1.1*n - 1, 0, K)
        vals = np.linspace(-n, 2*n, 100)
        avg = np.zeros(100) + (K/n)
        plt.plot(vals, avg, lw = 4, color=line_color, alpha = 0.8)
        plt.axis(axis)
        style = dict(size=15, color=text_color)
        plt.text(0.3*n - 1, 0.9*K, r'$K = {} \,$'.format(K) + r', ' + '$n = {} \,$'.format(n) + r', ' + r'Iteration: ${} / {} \,$'.format(iter, iters), **style, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.xticks([])
        plt.draw()
        #plt.savefig('frames/fig_' + '{0:04d}'.format(iter), transparent=True)
        plt.pause(0.0001)
        plt.clf()
        state = update_configuration(state, n)
        iter += 1
