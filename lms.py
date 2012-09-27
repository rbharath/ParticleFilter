import math
import random
import subprocess
from optparse import OptionParser
import matplotlib.pyplot as plot
import matplotlib.mlab as mlab

""" A basic testing ground for supervised machine learning ideas. The goal
    is to gather basic implementations (with pictures of course) that will
    let me build intuition on how basic learning algorithms behave in
    practice.

    In Progress:
        1) Gradient Descent
"""

def parse_args():
    parser = OptionParser()
    #parser.add_option("--file", dest="file",
    #                  help="Specify the file with training data",
    #                  action="store", type="string")
    #parser.add_option("--learner", dest="file",
    #                  help="Specify learning algorithm: Gradient Descent",
    #                  action="store", type="string")
    (options, args) = parser.parse_args()
    return options

class DataGenerator:
    """ A problem in machine learning seems to be obtaining data that
        models the problem in question. For basic explorations in ML, it
        will probably be easier to generate our data according to some
        basic probabilistic model rather than by trawling the
        web for suitable models.
    """

    def __init__(self):
        pass

    def noisy_linear_data(self, d, thetas, sigma, N, sep):
        """ The underlying model is

            f(thetas; x) = theta_0 + \sum_{j=1}^d theta_j * x_j

            For x1,...,xd in range 0..N, we calculate the value
            of f(thetas; x). Then, we shift according to noise sigma.
            Return noisy signal.
        """
        if len(thetas) != d + 1:
            print "Wrong number of parameters provided!"
        if d != 1:
            print "Higher dimensions not supported yet!"
        y = []
        for i in range(N/sep):
            f = thetas[0] + thetas[1] * (i * sep)
            n = (.25 + random.random()) * random.gauss(0, sigma)
            y.append(f + n)
        return y

class GradientDescent:
    """ Implements a simple gradient descent algorithm that minimizes
        sum of squares error. Allows us to do linear regression
        to find parameters for linear function:

        f(x) = theta_0 + \sum_{j=1}^m theta_n x_n
    """
    def __init__(self):
        pass

    def h(self, thetas, xs):
        if len(thetas) != len(xs) + 1:
            print "Improper sizes passed to hypothesis h"
        val = thetas[0]
        for i in range(len(xs)):
            val += thetas[i+1] * xs[i]
        return val

    def linear_regression(self, xs, ys, alpha, d, N):
        if len(xs) != len(ys):
            print "Improper Sizes Passed to linear regression"
            return None
        thetas = [0] * d
        # Need to Fix this for data with more than one dimension
        for n in range(N):
            print "n: " + str(n)
            print "thetas: " + str(thetas)
            for j in range(d):
                diff = 0
                for i in range(len(xs)):
                    diff += (ys[i] - self.h(thetas, [xs[i]])) * xs[i]
                thetas[j] += diff
        vals = []
        for x in xs:
            vals.append(self.h(thetas, [x]))
        return vals

if __name__ == "__main__":
    options = parse_args()
    gen = DataGenerator()
    grad = GradientDescent()
    out = "regression.png"
    # Trying to duplicate the portland housing data given elsewhere
    xs = range(0, 5000, 50)
    ys = gen.noisy_linear_data(1, [71.27, .1345], 60, 5000, 50)
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xs, ys)

    # Save and open the plotted figure
    plot.savefig(out)
    subprocess.call(["eog", out, "&"])
