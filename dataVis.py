from environment import Bandit
import matplotlib.pyplot as plt
import numpy as np

def plot(bandit):
    armCount = bandit.numArms
    armStats = []
    for i in range(armCount):
        currentArm = bandit.getArm(i+1)
        armStats.append(currentArm.returnStats())
    
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(1, armCount+1, 1), [armStats[0]['avgReward'], armStats[1]['avgReward'], armStats[2]['avgReward'], armStats[3]['avgReward']], 'ro')
    labels = range(1, len(armStats)+1, 1)
    plt.xticks(labels)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Avg. Reward')
    plt.xlabel('Agent ID')

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(1, armCount+1, 1), [armStats[0]['actualProbability'], armStats[1]['actualProbability'], armStats[2]['actualProbability'], armStats[3]['actualProbability']], 'bo')
    plt.xticks(labels)
    plt.yticks(np.arange(0, 1.01, .20))
    plt.ylabel('Agent Probability')
    plt.xlabel('Agent ID')

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(1, armCount+1, 1), [armStats[0]['winRatio'], armStats[1]['winRatio'], armStats[2]['winRatio'], armStats[3]['winRatio']], 'bo')
    plt.xticks(labels)
    plt.yticks(np.arange(0, 1.01, .20))
    plt.ylabel('Win Ratio')
    plt.xlabel('Agent ID')
    plt.tight_layout()

def avgGraph(bandits, numTests):
    # Create figure
    fig = plt.figure()
    avgGraph = fig.add_subplot(111)

    # Set the values for the x axis as 1...numTests
    xVals = [0]
    for i in range(numTests):
       xVals.append(i + 1)

    # For each bandit create a line of their rolling average
    for bandit in bandits:
        avgGraph.plot(xVals, bandit.testAvgs, label= bandit.name)

    # Create a line with a constant y value of the expected value of the arm with the highest expected value
    maxExpectedVal = max(bandits[0].expectedVals)
    maxExpectedValY = []
    for i in range (numTests + 1):
        maxExpectedValY.append(maxExpectedVal)
    avgGraph.plot(xVals, maxExpectedValY, label = 'Max Expected Value', color = 'black', ls = 'dashed')
    avgGraph.legend()
    # Set the names and intervals for the x and y axes
    #plt.xticks(np.arange(0, numTests+1))
    plt.xlabel('Number of Tests')
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Average Reward')

def percentOptimalGraph(bandits, numTests, numInstances):
    # Create figure
    fig = plt.figure()
    avgGraph = fig.add_subplot(111)

    # Set the values for the x axis as 1...numTests
    xVals = [0]
    for i in range(numTests):
        xVals.append(i + 1)

    # Create a line with a constant y value of the expected value of the arm with the highest expected value
    #maxExpectedVal = max(bandits[0].expectedVals)

    # For each bandit create a line of their rolling average
    for bandit in bandits:
        yVals = [0]
        for i in range(len(bandits[0].optimalWins)):
            yVals.append(((bandit.optimalWins[i] + 0.0)/numInstances) * 100)
        avgGraph.plot(xVals, yVals, label= bandit.name)

    avgGraph.legend()
    plt.xlabel('Number of Tests')
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel('% Optimal Action')
    plt.tight_layout()
    plt.show()
    