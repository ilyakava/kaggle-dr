import sys
import cPickle
import numpy
import matplotlib.pyplot as plt

import pdb

def plot_results(result_file):
    f = open(result_file)
    historical_train_losses, historical_val_losses, historical_val_kappas = cPickle.load(f)

    plt.plot(numpy.array(historical_train_losses)[:,0], numpy.array(historical_train_losses)[:,1], 'g', label="Training error")
    plt.plot(numpy.array(historical_val_losses)[:,0], numpy.array(historical_val_losses)[:,1], 'r', label="Validation error")
    plt.plot(numpy.array(historical_val_kappas)[:,0], numpy.array(historical_val_kappas)[:,1], 'b', label="Validation Kappa")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    plot_results(sys.argv[1])
