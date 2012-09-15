import random
import matplotlib.pyplot as plot
import matplotlib.mlab as mlab
import subprocess
import math
from optparse import OptionParser

""" A basic testing ground for particle filter ideas. The goal is to gather
    a number of interesting random models and implement some standard
    particle filtering algorithms. The options allow me to run the various
    filters and gain an idea of how the various algorithms and models behave.

    Models:
        1) Gaussian Noise
        2) Noisy Linear
    Filters:
        1) BootstrapFilter

    InProgress:
        1) StochasticVolatility

    @author: rbharath@stanford
    @date: 9/7/2012
"""

out = "rand.png"

def parse_args():
    parser = OptionParser()
    parser.add_option("-T", "--time", dest="T",
                      help="Time", action="store",
                      type="int", default=100)
    parser.add_option("-N", dest="N",
                      help="Num Particle", action="store",
                      type="int", default=1000)
    # These options provide choice of parameters
    parser.add_option("--low", dest="low",
                      help="Param: lower limit", action="store",
                      type="int", default=0)
    parser.add_option("--high", dest="high",
                      help="Param: higher limit", action="store",
                      type="int", default=100)
    parser.add_option("--beta", dest="beta",
                      help="Param: velocity", action="store",
                      type="float", default=5)
    parser.add_option("--sigma", dest="sigma",
                      help="Param: Variance", action="store",
                      type="float", default=1)
    # These options provide choice of model
    parser.add_option("--model", dest="model",
                      help="noisy_linear, gaussian",
                      type="string", action="store")
    parser.add_option("--plot_theta", dest="plot_theta",
                      help="Plot: graph the hidden true state theta",
                      default=False, action="store_true")
    parser.add_option("--plot_z", dest="plot_z",
                      help="Plot: graph the noisy evidence z",
                      default=False, action="store_true")
    parser.add_option("--plot_thetahat", dest="plot_thetahat",
                      help="Plot: graph the inferred true state thetahat",
                      default=False, action="store_true")
    # The following options control various features of graphed particles
    parser.add_option("--filter", dest="filter",
                      help="Particles: show last time step particle estimate",
                      default=False, action="store_true")
    parser.add_option("--expectation", dest="expectation",
                      help="Particles: show the expectation of particles",
                      default=False, action="store_true")

    (options, args) = parser.parse_args()
    return options

def path_expectation(particles, options):
    """ Given a list of particles, each of which represents a path
        in R^2, compute the expectation path.
    """
    e = []
    if len(particles) == 0:
        return None
    for i in range(options.T):
        xavg = 0
        yavg = 0
        for p,logw in particles:
            x,y = p[i]
            xavg += x
            yavg += y
        xavg = float(xavg) / len(particles)
        yavg = float(yavg) / len(particles)
        e.append((xavg, yavg))
    return e

class GaussianNoise:
    """ Models a static point with Gaussian Noise coming
        in over time.

        Dom(x_i) = R
        Dom(y_i) = R

        x = Uniform(low, high)
        y = Uniform(low, high)
        x_n = N(x, sigma)
        y_n = N(y, sigma)

        Particle Format
        (x,y)
    """
    def __init__(self, options):
        self.thetas = []
        self.zs = []
        self.options = options
        x = random.uniform(self.options.low, self.options.high)
        y = random.uniform(self.options.low, self.options.high)
        self.thetas.append((x,y))

        for i in range(options.T):
            xnoise = random.gauss(x, self.options.sigma)
            ynoise = random.gauss(y, self.options.sigma)
            self.zs.append((xnoise, ynoise))

    def prior(self):
        """ The prior distribution on particles for particle filtering
            algorithms.
        """
        X = random.uniform(self.options.low, self.options.high)
        Y = random.uniform(self.options.low, self.options.high)
        return (X, Y)

    def transition(self, theta):
        """ The Gaussian Noise model is memory less, so we just sample from
            the known static state.
        """
        Xnoise = random.gauss(self.thetas[0][0], self.options.sigma)
        Ynoise = random.gauss(self.thetas[0][1], self.options.sigma)
        return (Xnoise, Ynoise)

    def logEvidenceWeight(self, theta, t):
        """ Evaluate the probability of sampling z from a gaussian centered
            at theta.
        """
        (x,y) = theta
        (zx, zy) = self.zs[t]
        ret = math.log(mlab.normpdf(zx, x, self.options.sigma))
        ret += math.log(mlab.normpdf(zy, y, self.options.sigma))
        return ret

    def plotHidden(self, color):
        plot_graph(self.thetas, color)

    def plotNoisy(self, color):
        plot_graph(self.zs, color)

    def plotParticles(self, particles, color):
        if self.options.filter:
            for (particle, logW) in particles:
                plot_graph(particle[-1:], color)
        elif self.options.expectation:
            plot_graph(path_expectation(particles, self.options), color)
        else:
            for (particle, logW) in particles:
                plot_graph(particle, color)

class NoisyLinear:
    """ Models a linearly moving particle with gaussian noise in
        observations.

        Dom(x_i) = R
        Dom(y_i) = R

        x_0 = Uniform(low, high)
        x_0 = Uniform(low, high)
        vx = Uniform(0,beta)
        vy = Uniform(0,beta)

        x_n = x_{n-1} + vx
        y_n = y_{n-1} + vy
        Z_n = (x_n + N(0,sigma), y_n + N(0, sigma))

        Particle Format:
        (x,y)

        This class assumes the evolution parameters (velocities) are known
        to the particle filter. The tracking problem becomes much trickier
        when parameters are unknown.
    """
    def __init__(self, options):
        self.options = options
        self.thetas = []
        self.zs = []
        self.vx = random.random() * options.beta
        self.vy = random.random() * options.beta
        prev = 0
        for i in range(options.T):
            if i == 0:
                x = random.uniform(options.low, options.high)
                y = random.uniform(options.low, options.high)
            else:
                x = self.thetas[-1:][0][0]
                y = self.thetas[-1:][0][1]
            x = x + self.vx
            y = y + self.vy
            self.thetas.append((x,y))
            self.zs.append((x + random.gauss(0, options.sigma),
                            y + random.gauss(0, options.sigma)))

    def prior(self):
        """ The prior distribution on particles for particle filtering
            algorithms.
        """
        x = random.uniform(self.options.low, self.options.high)
        y = random.uniform(self.options.low, self.options.high)
        return (x,y)

    def transition(self, theta):
        """ In This model, X_n and Y_n are determined completed by X_{n-1}, Y_{n-1}
        and by self.vx and self.vy.
        """
        (x,y) = theta
        x = x + self.vx
        y = y + self.vy
        return (x, y)

    def logEvidenceWeight(self, theta, t):
        """ The log evidence weight is a product of Gaussians as in the
            GaussianNoise Example.
        """
        (x,y) = theta
        (zx, zy) = self.zs[t]
        ret = math.log(mlab.normpdf(zx, x, self.options.sigma))
        ret += math.log(mlab.normpdf(zy, y, self.options.sigma))
        return ret

    def plotHidden(self, color):
        plot_graph(self.thetas, color)

    def plotNoisy(self, color):
        plot_graph(self.zs, color)

    def plotParticles(self, particles, color):
        if self.options.filter:
            for (particle, logW) in particles:
                plot_graph(particle[-1:], color)
        elif self.options.expectation:
            plot_graph(path_expectation(particles, self.options), color)
        else:
            for (particle, logW) in particles:
                plot_graph(particle, color)

class StochasticVolatility:
    """ Stochastic Volatility model of the type used in financial
        econometrics.

        Dom(x_i) = R
        Dom(y_i) = R

        x_1 = N(0, sigma^2 / (1 - alpha^2))
        v_n = N(0,1)
        w_n = N(0,1)
        x_n = alpha * x_{n-1} + sigma * v_n
        y_n = beta * exp(x_n/2) * w_n
    """
    pass

class BootstrapFilter:
    """ Standard Bootstrap Particle Filter with No Extra Frills Added.
        This implementation is very basic. The goal is for me to gain intuition
        in coding particle filters.
    """
    def __init__(self, model, options):
        self.model = model
        self.T = len(model.zs)
        self.particles = []

    def getParticles(self):
        return self.particles

    def run(self):
        # Initialize the particles at time 0
        for i in range(options.N):
            # Generate Initial Data Uniformly throughout grid.
            particle = [self.model.prior()]
            logWeight = model.logEvidenceWeight(particle[0], 0)
            self.particles.append((particle, logWeight))
        # For each time step
        for t in range(self.T-1):
            for j in range(options.N):
                (particle, logWeight) = self.particles[j]
                # transition draws theta_t from the proposal distribution given
                # theta_0,.., theta_{t-1} and z_0, ..., z_t
                # Typically, we just choose theta_t | theta_{t-1}
                theta = model.transition(particle[t])
                particle.append(theta)
                # Assume here that the proposal distribution is the transition
                # p(theta_t | theta_{t-1}), so we get a simple form for the weights
                # w = p(z_t | theta_t)
                logW = model.logEvidenceWeight(particle[t+1], t+1)
                logWeight += logW
                self.particles[j] = (particle, logWeight)
            # This causes underflow especially when parameters are unknown.
            Sum = sum([math.exp(logW) for (p, logW) in self.particles])
            norm_weights = [logW - math.log(Sum) for (p, logW) in self.particles]
            # Resample particles according to weights
            # Todo: Explore better resampling methods
            new_particles = []
            for i in range(options.N):
                r = random.random()
                pos = 0
                while True:
                    r -= math.exp(norm_weights[pos])
                    if r <= 0:
                        break
                    pos += 1
                (p, w) = self.particles[pos]
                p_copy = p[:]
                new_particles.append((p_copy, math.log(1.0/options.N)))
            self.particles = new_particles

def plot_graph(zs, color):
    xs = [z[0] for z in zs]
    ys = [z[1] for z in zs]
    plot.scatter(xs, ys, c=color)
    return

def display_figure():
    global out
    plot.savefig(out)
    subprocess.call(["eog", out, "&"])
    return

if __name__ == "__main__":
    options = parse_args()
    # Generate Hidden and Evidence State
    if options.model == "noisy_linear":
        model = NoisyLinear(options)
    elif options.model == "gaussian":
        model = GaussianNoise(options)
    # Perform Inference to Gain Hidden State
    infer = BootstrapFilter(model, options)
    infer.run()
    # Plot state
    if options.plot_theta:
        model.plotHidden('b')
    if options.plot_z:
        model.plotNoisy('g')
    if options.plot_thetahat:
        model.plotParticles(infer.getParticles(), 'r')
    if options.plot_theta or options.plot_z or options.plot_thetahat:
        display_figure()
