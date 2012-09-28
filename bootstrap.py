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
        3) Clock Hands
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
    # These options let us set the hidden state to make experimentation
    # easier
    parser.add_option("--x", dest="x",
                      help="State: set value of x", action="store",
                      type="float")
    parser.add_option("--y", dest="y",
                      help="State: set value of y", action="store",
                      type="float")
    # These options provide choice of parameters
    parser.add_option("--low", dest="low",
                      help="Param: lower limit", action="store",
                      type="int", default=0)
    parser.add_option("--high", dest="high",
                      help="Param: higher limit", action="store",
                      type="int", default=100)
    parser.add_option("--alpha", dest="alpha",
                      help="Param: alpha", action="store",
                      type="float", default=5)
    parser.add_option("--beta", dest="beta",
                      help="Param: beta", action="store",
                      type="float", default=5)
    parser.add_option("--sigma", dest="sigma",
                      help="Param: Variance", action="store",
                      type="float", default=1)
    parser.add_option("--k", dest="k",
                      help="Param: Number Hands", action="store",
                      type="int", default=1)
    # These options provide choice of model
    parser.add_option("--model", dest="model",
                      help="noisy_linear, gaussian, stochastic_volatility",
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

def path_expectation(particles, options, all_time=True):
    """ Given a list of particles, each of which represents a path
        in R^2, compute the expectation path.

        If we just wish for the expectation of the final time step,
        set all_time=False
    """
    e = []
    if len(particles) == 0:
        return None
    #num_dimensions = len(particles[0])
    p, logw = particles[0]
    # each particle contains a time history. Take the 0th timestep
    # data to undertand the dimensionality of the particle
    num_dimensions = len(p[0])
    if all_time:
        for i in range(options.T):
            avgs = [0] * num_dimensions
            for p,logw in particles:
                v = p[i]
                for j in range(num_dimensions):
                    avgs[j] += v[j]
            for j in range(num_dimensions):
                avgs[j] = float(avgs[j]) / len(particles)
            e.append(tuple(avgs[:]))
    else:
        avgs = [0] * num_dimensions
        for p,logw in particles:
            v = p[options.T - 1]
            for j in range(num_dimensions):
                avgs[j] += v[j]
        for j in range(num_dimensions):
            avgs[j] = float(avgs[j]) / len(particles)
        e.append(tuple(avgs[:]))
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
        if options.x is None:
            x = random.uniform(self.options.low, self.options.high)
        else:
            x = options.x
        if options.y is None:
            y = random.uniform(self.options.low, self.options.high)
        else:
            y = options.y
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

    def transition(self, thetahat):
        """ The Gaussian Noise model is memory less, but we want to
            keep information about whether our current location is good
            or bad.
        """
        (x, y) = thetahat
        Xnoise = random.gauss(x, self.options.sigma)
        Ynoise = random.gauss(y, self.options.sigma)
        return (Xnoise, Ynoise)

    def logEvidenceWeight(self, theta, t):
        """ Evaluate the probability of sampling z from a gaussian centered
            at theta.
        """
        (x,y) = theta
        (zx, zy) = self.zs[t]
        px = mlab.normpdf(zx, x, self.options.sigma)
        py = mlab.normpdf(zy, y, self.options.sigma)
        if px == 0 or py == 0:
            return True, 0
        else:
            ret = math.log(px) + math.log(py)
            return False, ret

    def plotHidden(self, color):
        plot_graph(self.thetas, color)

    def plotNoisy(self, color):
        plot_graph(self.zs, color)

    def plotParticles(self, particles, color):
        if self.options.filter:
            for (particle, logW) in particles:
                plot_graph(particle[-1:], color)
        elif self.options.expectation:
            plot_graph(path_expectation(particles, self.options,
                all_time=False), color)
        else:
            for (particle, logW) in particles:
                plot_graph(particle, color)

    def displayFigure(self, options):
        global out
        plot.xlim([options.low, options.high])
        plot.ylim([options.low, options.high])
        plot.savefig(out)
        subprocess.call(["eog", out, "&"])
        return

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
        px = mlab.normpdf(zx, x, self.options.sigma)
        py = mlab.normpdf(zy, y, self.options.sigma)
        if px == 0 or py == 0:
            return True, 0
        else:
            ret = math.log(px) + math.log(py)
            return False, ret

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

    def displayFigure(self, options):
        global out
        plot.xlim([options.low, options.high])
        plot.ylim([options.low, options.high])
        plot.savefig(out)
        subprocess.call(["eog", out, "&"])
        return

class ClockHands:
    """ A toy example of a model that evolves at different time scales.  There
    are k clock hands. The base of the k-th clock hand is at the tip of the
    (k-1)st clock hand. The k-th clock hand evolves at rate 10^k. That is, we
    expect the k-th hand to transition 10^k times per second.

    Each hand is stored as a list of k angles, the ith of which represents
    the angle of the ith clock hand.  The ith clock hand is of length 2^-i.
    """
    def __init__(self, options):
        self.options = options
        self.thetas = []
        self.zs = []
        for i in range(options.T):
            state = self.prior()
            noisy = self.observe(state)
            self.thetas.append(state)
            self.zs.append(noisy)

    def prior(self):
        """ Choose the collection of angles randomly.
        """
        state = [0] * options.k
        # Initialize the state to a random setting
        k = self.options.k
        for i in range(k):
            state[i] = random.uniform(0, 2 * math.pi)
        return state

    def observe(self, state):
        """ Given a state, get a noisy observation according to
            the observation model, which dictates gaussian noise
            on everything.
        """
        sigma = self.options.sigma
        k = self.options.k
        noisy = [0] * k
        for i in range(k):
            noisy[i] = random.gauss(state[i], sigma)
        return noisy

    def transition(self, theta):
        """ Clocks hands are progressively more likely to transition
            as we move further and further down the chain.
        """
        # Copy the state to
        state = theta[:]
        k = self.options.k
        for i in range(k):
            prob = 2 ** -(k - i)
            u = random.random()
            if u >= prob:
                continue
            state[i] = random.gauss(state[i], options.sigma)
        return state

    def logEvidenceWeight(self, theta, t):
        """ Return the log probability of the observation given
            the hidden state theta.
        """
        pass

    def plotHidden(self, color):
        """ Plot the time evolution of the state.
        """
        if len(self.thetas) == 0:
            return
        for state in self.thetas:
            self.plotArm(state)

    def plotArm(self, state):
        """ Plots the state as an arm in the manner described in
            the comments for the class.
        """
        fig = plot.figure()
        ax = fig.add_subplot(1,1,1)
        base = (0,0)
        for i in range(self.options.k):
            theta = state[i]
            r = 2 ** (-i)
            new_base = (base[0] + r * math.cos(theta),
                        base[1] + r * math.sin(theta))
            xs = [base[0], new_base[0]]
            ys = [base[1], new_base[1]]
            ax.plot(xs, ys)
            base = new_base

    def displayFigure(self, options):
        """ Set the limits to match the known size of the clock hand.
        """
        global out
        plot.xlim([-2, 2])
        plot.ylim([-2, 2])
        plot.savefig(out)
        subprocess.call(["eog", out, "&"])
        return

    def plotNoisy(self, color):
        """ Plot the sequence of noisy observations of the arm.
        """
        for noisy in self.zs:
            self.plotArm(noisy)
        pass

    def plotParticles(self, particles, color):
        """ For now, just plot the state of the particle at the
            last time step.
        """
        for p, logw in particles:
            self.plotArm(p[-1:])


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
    def __init__(self, options):
        self.thetas = []
        self.zs = []
        self.options = options
        x = (random.gauss(0, float(options.sigma ** 2) /
                float(1 + options.alpha **2)))
        w = random.gauss(0,1)
        y = options.beta * math.exp(float(x)/2) * w
        self.thetas.append(x)
        self.zs.append(y)

        for i in range(options.T - 1):
            x = self.thetas[-1:][0]
            v = random.gauss(0,1)
            w = random.gauss(0,1)
            x = options.alpha * x + options.sigma * v
            print "x: " + str(x)
            y = options.beta * math.exp(float(x)/2) * w
            self.thetas.append(x)
            self.zs.append(y)

    def prior(self):
        x = (random.gauss(0, float(options.sigma ** 2)/
                float(1 + options.alpha **2)))
        return x

    def transition(self, theta):
        x = theta
        v = random.gauss(0,1)
        x = options.alpha * x + options.sigma * v
        return x

    def logEvidenceWeight(self, theta, t):
        x = theta
        y = self.zs[t]
        py = mlab.normpdf(y,0, (options.beta**2) * math.exp(x))
        if py == 0:
            return True, 0
        else:
            return False, math.log(py)

    def plotHidden(self, color):
        pairs = zip(range(len(self.thetas)), self.thetas)
        plot_graph(pairs, color)

    def plotNoisy(self, color):
        pairs = zip(range(len(self.zs)), self.zs)
        plot_graph(pairs, color)

    def plotParticles(self, particles, color):
        if self.options.filter:
            for (particle, logW) in particles:
                plot_graph(particle[-1:], color)
        elif self.options.expectation:
            plot_graph(path_expectation(particles, self.options), color)
        else:
            for (particle, logW) in particles:
                plot_graph(particle, color)

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
        i = 0
        # Initialize the particles at time 0
        while i < options.N:
            # Generate Initial Data Uniformly throughout grid.
            particle = [self.model.prior()]
            underflow, logWeight = model.logEvidenceWeight(particle[0], 0)
            if not underflow:
                p = (particle, logWeight)
                self.particles.append(p)
                i += 1
        # For each extra time step
        for t in range(self.T-1):
            print "t: " + str(t)
            for j in range(options.N):
                (particle, logWeight) = self.particles[j]
                # transition draws theta_t from the proposal distribution given
                # theta_0,.., theta_{t-1} and z_0, ..., z_t
                # Typically, we just choose theta_t | theta_{t-1}
                if t >= len(particle):
                    print "j: " + str(j)
                    print "len(particle): " + str(len(particle))
                    print "t: " + str(t)
                    print "particle: " + str(particle)
                    print "self.particles: " + str([len(p[0]) for p in
                    self.particles])
                theta = model.transition(particle[t])
                pre = (len(particle))
                particle.append(theta)
                post = (len(particle))
                # Assume here that the proposal distribution is the transition
                # p(theta_t | theta_{t-1}), so we get a simple form for the weights
                # w = p(z_t | theta_t)
                underflow, logW = model.logEvidenceWeight(particle[t+1], t+1)
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


if __name__ == "__main__":
    options = parse_args()
    # Generate Hidden and Evidence State
    if options.model == "noisy_linear":
        model = NoisyLinear(options)
    elif options.model == "gaussian":
        model = GaussianNoise(options)
    elif options.model == "stochastic_volatility":
        model = StochasticVolatility(options)
    elif options.model == "clock_hands":
        model = ClockHands(options)
    # Perform Inference to Gain Hidden State
    if options.plot_thetahat:
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
        model.displayFigure(options)
