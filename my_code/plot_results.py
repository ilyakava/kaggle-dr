import sys
import cPickle
import numpy
import matplotlib.pyplot as plt

import pdb

def plot_results(result_file):
    f = open(result_file)
    historical_train_losses, historical_val_losses, historical_val_kappas, n_iter_per_epoch = cPickle.load(f)
    n_iter_per_epoch = float(n_iter_per_epoch)

    train = numpy.array(historical_train_losses)
    valid = numpy.array(historical_val_losses)
    kappa = numpy.array(historical_val_kappas)
    # scale iter to epoch
    train[:,0] = train[:,0] / n_iter_per_epoch
    valid[:,0] = valid[:,0] / n_iter_per_epoch
    kappa[:,0] = kappa[:,0] / n_iter_per_epoch

    if sum(numpy.isnan(train[:,1])):
        first_nan = numpy.isnan(train[:,1]).tolist().index(True) / n_iter_per_epoch
        last_nan = len(train[:,1]) - numpy.isnan(train[:,1]).tolist()[::-1].index(True) / n_iter_per_epoch
        plt.axvspan(first_nan, last_nan, facecolor='r', alpha=0.5)
        plt.text(first_nan - 7, -0.5, "nan Zone ->", color='r', size=18)

    plt.plot(train[:,0], train[:,1], 'g', label="Training error")
    plt.plot(valid[:,0], valid[:,1], 'r', label="Validation error")
    plt.plot(kappa[:,0], kappa[:,1], 'b', label="Validation Kappa")
    plt.xlabel("Epoch")
    plt.ylabel("Best Val MSE: %.3f and Kappa: %.3f" % (min(valid[:,1]), max(kappa[:,1])))
    plt.title(result_file)
    plt.ylim((-1,1.5))
    plt.xlim((0,len(train[:,1]) / n_iter_per_epoch))
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    plot_results(sys.argv[1])
