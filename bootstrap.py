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

    NEXT: Finish Bootstrap filter

    @author: rbharath@stanford
    @date: 9/7/2012
"""

out = "rand.png"

def parse_args():
    parser = OptionParser()
    parser.add_option("-T", dest="T",
                      help="Time", action="store",
                      type="int", default=100)
    parser.add_option("-N", dest="N",
                      help="Num Particle", action="store",
                      type="int", default=1000)
    parser.add_option("--low", dest="low",
                      help="lower limit", action="store",
                      type="int", default=0)
    parser.add_option("--high", dest="high",
                      help="higher limit", action="store",
                      type="int", default=100)
    parser.add_option("--beta", dest="beta",
                      help="velocity", action="store",
                      type="float", default=5)
    parser.add_option("--sigma", dest="sigma",
                      help="Variance", action="store",
                      type="float", default=1)
    parser.add_option("--noisy_linear", dest="noisy_linear",
                      help="noisy linear motion", action="store_true",
                      default=False)
    parser.add_option("--gaussian", dest="gaussian",
                      help="gaussian noise", action="store_true",
                      default=False)
    parser.add_option("--plot_theta", dest="plot_theta",
                      help="graph the hidden true state theta",
                      default=False, action="store_true")
    parser.add_option("--plot_z", dest="plot_z",
                      help="graph the noisy evidence z",
                      default=False, action="store_true")
    parser.add_option("--plot_thetahat", dest="plot_thetahat",
                      help="graph the inferred true state thetahat",
                      default=False, action="store_true")

    (options, args) = parser.parse_args()
    return options

class GaussianNoise:
    """ Models a static point with Gaussian Noise coming
        in over time.

        X = Uniform(low, high)
        Y = Uniform(low, high)
        X_n = N(X, sigma)
        Y_n = N(Y, sigma)

        Particle Format
        (X,Y)
    """
    def __init__(self, options):
        self.thetas = []
        self.zs = []
        self.options = options
        x = random.uniform(self.options.low, self.options.high)
        y = random.uniform(self.options.low, self.options.high)
        self.thetas.append((x,y))

        for i in range(options.T-1):
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
        for particle in particles:
            plot_graph(particle, color)

class NoisyLinear:
    """ Models a linearly moving particle with gaussian noise in
        observations.

        X_0 = Uniform(low, high)
        Y_0 = Uniform(low, high)
        V_0 = Uniform(0,beta)
        W_0 = Uniform(0,beta)

        X_n = X_{n-1} + v_{n-1}
        Y_n = Y_{n-1} + w_{n-1}
        Z_n = (X_n + N(0,sigma), Y_n + N(0, sigma))

        Particle Format:
        (V,W,X,Y)
    """
    def __init__(self, options):
        self.options = options
        self.thetas = []
        self.zs = []
        v = random.random() * options.beta
        w = random.random() * options.beta
        prev = 0
        for i in range(options.T):
            if i == 0:
                xprev = random.uniform(options.low, options.high)
                yprev = random.uniform(options.low, options.high)
            else:
                xprev = self.thetas[i-1][0]
                yprev = self.thetas[i-1][1]
            x = xprev + v
            y = yprev + w
            self.thetas.append((v,w, x,y))
            self.zs.append((x + random.gauss(0, options.sigma),
                            y + random.gauss(0, options.sigma)))

    def prior(self):
        """ The prior distribution on particles for particle filtering
            algorithms.
        """
        V = random.random() * self.options.beta
        W = random.random() * self.options.beta
        X = random.uniform(self.options.low, self.options.high)
        Y = random.uniform(self.options.low, self.options.high)
        return (V,W,X,Y)

    def transition(self, theta):
        """ In This model, X_n and Y_n are determined completed by X_{n-1}, Y_{n-1}
        and by the V and W.
        """
        (v,w,x,y) = theta
        x_new = x + v
        y_new = y + w
        return (v,w,x_new, y_new)

    def logEvidenceWeight(self, theta, t):
        """ The log evidence weight is a product of Gaussians as in the
            GaussianNoise Example.
        """
        (v,w,x,y) = theta
        (zx, zy) = self.zs[t]
        ret = math.log(mlab.normpdf(zx, x, self.options.sigma))
        ret += math.log(mlab.normpdf(zy, y, self.options.sigma))
        #print "zx: " + str(zx) + ", x: " + str(x)
        #print "zy: " + str(zy) + ", y: " + str(y)
        #print "logEvidenceWeight: " + str(ret)
        return ret

    def plotHidden(self, color):
        plot_graph([(x,y) for (v,w,x,y) in self.thetas], color)

    def plotNoisy(self, color):
        plot_graph(self.zs, color)

    def plotParticles(self, particles, color):
        for particle, weight in particles:
            plot_graph([(x,y) for (v,w,x,y) in particle])


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
            for pos in range(options.N):
                (particle, logWeight) = self.particles[pos]
                # transition draws theta_t from the proposal distribution given
                # theta_0,.., theta_{t-1} and z_0, ..., z_t
                # Typically, we just choose theta_t | theta_{t-1}
                theta = model.transition(particle[t])
                particle.append(theta)
                # Assume here that the proposal distribution is the transition
                # p(theta_t | theta_{t-1}), so we get a simple form for the weights
                # w = p(z_t | theta_t)
                logW = model.logEvidenceWeight(particle[t+1], t+1)
                #print "t: " + str(t) + ", pos: " + str(pos) + ", logW: " +str(logW)
                logWeight += logW
                self.particles[pos] = (particle, logWeight)
            # Not sure if this will cause underflow
            Sum = sum([math.exp(logW) for (p, logW) in self.particles])
            norm_weights = [logW - math.log(Sum) for (p, logW) in self.particles]
            #print "norm_weights: " + str(norm_weights)
            # Resample particles according to weights
            new_particles = []
            for i in range(options.N):
                r = random.random()
                pos = 0
                while True:
                    #print "i: " + str(i) + ", pos: " + str(pos)
                    r -= math.exp(norm_weights[pos])
                    if r <= 0:
                        break
                    pos += 1
                (p, w) = self.particles[pos]
                p_copy = p[:]
                new_particles.append((p_copy, math.log(1.0/options.N)))
            self.particles = new_particles

    def plotPrediction(self, color):
        model.plotParticles(self.particles)

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
    if options.noisy_linear:
        model = NoisyLinear(options)
    elif options.gaussian:
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
