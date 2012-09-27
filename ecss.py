import random
import math

def poisson(lam):
    n = 0
    probOfN = math.exp(-lam)
    cumProb = probOfN

    u = random.random()
    while cumProb < u:
        n += 1
        probOfN *= (lam /n)
        cumProb += probOfN
    return n


def poissonProb(lam, k):
    prob = math.exp(-lam)
    prob *= (lam ** k)
    prob /=

def likelihoodCalculate(avgAircraft, avgBlipsPerAircraft, numBlips, N):
    counts = [0] * 100
    for i in range(N):
        A = poisson(avgAircraft)
        totalBlips = 0
        for a in range(A):
            b = poisson(avgBlipsPerAircraft)
            totalBlips += b
        if totalBlips == numBlips:
            counts[A] += 1
    print "likelihoodCalculate: " + str(counts)

def openECSS(avgAircraft, avgBlipsPerAircraft, numBlips):
    table = []
    l = 2 * avgAircraft
    init = [0] * numBlips
    for b in range(numBlips):
        init[b] = poissonProb(avgBlipsPerAircraft, b)
    table.append(init)
    for i in range(l - 1):
        new = [0] * numBlips
        table.append(new)
        for b in range(numBlips):
            for delta in range(numBlips):
                if delta + b > numBlips:
                    break
                prob = poissonProb(avgBlipsPerAircraft, delta)
                table[i+1][delta + b] += prob * table[i][b]
    result = []
    for i in range(l):
        prob = poissonProb(avgAircraft, i)
        result.append(table[i][numBlips] * prob)
    print "openECSS: " + str(result)


if __name__ == "__main__":
    avgAircraft = 6
    avgBlipsPerAircraft = 4
    numBlips = 20
    N = 10000
    likelihoodCalculate(avgAircraft, avgBlipsPerAircraft, numBlips, N)
    openECSS(avgAircraft, avgBlipsPerAircraft, numBlips)
