import random
import math
import time

def poisson(lam):
    n = 0
    probOfN = math.exp(-lam)
    cumProb = probOfN
    u = random.random()

    while cumProb < u:
        n += 1
        probOfN *= (float(lam) / n)
        cumProb += probOfN
    return n


def poissonProb(lam, k):
    try:
        prob = math.exp(-lam)
        prob *= (lam ** k)
        prob /= math.factorial(k)
        return prob
    except OverflowError:
        return 0.0

def likelihoodCalculate(avgAircraft, avgBlipsPerAircraft, numBlips, N):
    counts = [0] * 5 * avgAircraft
    for i in range(N):
        A = poisson(avgAircraft)
        totalBlips = 0
        for a in range(A):
            b = poisson(avgBlipsPerAircraft)
            totalBlips += b
        if totalBlips == numBlips:
            counts[A] += 1
    s = sum(counts)
    print "sum: " + str(s)
    counts = [float(c)/s for c in counts]
    print "likelihoodCalculate: " + str(counts)

def openECSS(avgAircraft, avgBlipsPerAircraft, numBlips):
    table = []
    print "openECSS"
    l = 5 * avgAircraft
    init = [0] * numBlips
    for b in range(numBlips):
        init[b] = poissonProb(avgBlipsPerAircraft, b)
    result = [0.0]
    prev = init
    for i in range(l - 1):
        cur = [0] * numBlips
        for b in range(numBlips):
            for delta in range(numBlips):
                if delta + b >= numBlips:
                    break
                prob = poissonProb(avgBlipsPerAircraft, delta)
                cur[delta + b] += prob * prev[b]
        prob = poissonProb(avgAircraft, i)
        result.append(prev[numBlips - 1] * prob)
        prev = cur
    #for i in range(l):
    #    result.append(table[i][numBlips-1] * prob)
    s = sum(result)
    result = [float(r)/s for r in result]
    print "openECSS: " + str(result)


if __name__ == "__main__":
    avgAircraft = 1
    avgBlipsPerAircraft = 4
    numBlips = 3
    N = 1000000
    start = time.time()
    likelihoodCalculate(avgAircraft, avgBlipsPerAircraft, numBlips, N)
    elapsed = time.time() - start
    print "elapsed: " + str(elapsed)
    start = time.time()
    openECSS(avgAircraft, avgBlipsPerAircraft, numBlips)
    elapsed = time.time() - start
    print "elapsed: " + str(elapsed)
