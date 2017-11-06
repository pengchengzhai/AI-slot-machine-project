from environment import *
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math


temperature = 5

def greedyEpsilon(bandit, playerCredits):
    bestArmIndex = 0
    armCount = bandit.numArms

    # epsilon, the ratio of exploring vs exploiting
    epsilon = .2

    for i in range(playerCredits):
        if np.random.random() > epsilon:
            # Exploit
            currentArm = bandit.getArm(bestArmIndex+1)
            currentArm.play()
        else:
            # Explore
            exploreIndex = randint(0, armCount-1)
            currentArm = bandit.getArm(exploreIndex+1)
            currentArm.play()
            # Change the best arm index when the winning ratio for the current arm is larger
            if exploreIndex != bestArmIndex and currentArm.returnStats()['winRatio'] > bandit.getArm(bestArmIndex+1).returnStats()['winRatio']:
                bestArmIndex = exploreIndex
    if bandit.getBestArm().getArmId() == bandit.getRealBestArm().getArmId():
        return True
    else:
        return False

def decGreedyEpsilon(bandit, playerCredits):
    bestArmIndex = 0
    armCount = bandit.numArms

    # epsilon, the ratio of exploring vs exploiting
    epsilon = .2
    # amount to decrease epsilon per trial
    decAmount = epsilon/playerCredits

    for i in range(playerCredits):
        if np.random.random() > epsilon:
            # Exploit
            currentArm = bandit.getArm(bestArmIndex + 1)
            #if currentArm.getArmId() == bandit.getRealBestArm().getArmId():
            #    bandit.timesBestPulled[i] += 1
            currentArm.play()
            epsilon -= decAmount
        else:
            # Explore
            exploreIndex = randint(0, armCount - 1)
            currentArm = bandit.getArm(exploreIndex + 1)
            #if currentArm.getArmId() == bandit.getRealBestArm().getArmId():
            #    bandit.timesBestPulled[i] += 1
            currentArm.play()
            epsilon -= decAmount
            # Change the best arm index when the winning ratio for the current arm is larger
            if exploreIndex != bestArmIndex and currentArm.returnStats()['winRatio'] > \
                    bandit.getArm(bestArmIndex + 1).returnStats()['winRatio']:
                bestArmIndex = exploreIndex
    if bandit.getBestArm().getArmId() == bandit.getRealBestArm().getArmId():
        return True
    else:
        return False

def softmax(bandit, playerCredits):
    armCount = bandit.numArms
    # Initialize avgRewards list with armCount 1's and pickProb list with armCount 0's
    avgRewards = []
    pickProb = []
    for i in range(armCount):
        avgRewards.append(1.0)
        pickProb.append(0)
    for play in range(playerCredits):
        # Find the probability that of choosing the arm for each arm using the softmax equation
        avgSum = sum(avgRewards)
        for i in range(armCount):
            pickProb[i] = avgRewards[i]/avgSum
        # Choose an arm and pull it
        result = np.random.random()
        totalChecked = 0
        for i in range(armCount):
            if result > totalChecked and result < totalChecked + pickProb[i]:
                bandit.arms[i].play()
                avgRewards[i] = math.exp(bandit.arms[i].avgReward/temperature)
                break
            totalChecked += pickProb[i]
    bestArmIndex = 0
    winRatio = 0.0
    for arm in bandit.arms:
        if arm.returnStats()['winRatio'] > winRatio:
            winRatio = arm.returnStats()['winRatio']
            bestArmIndex = arm.getArmId()
    if bandit.getBestArm().getArmId() == bandit.getRealBestArm().getArmId():
        return True
    else:
        return False

def ucb1(bandit, playerCredits):

    for i in range(playerCredits):
        for arm in bandit.arms:
            arm.play()
            if arm.avgReward > 0 and arm.numPulls > 0:
                arm.ucb = np.float64(arm.wins - (arm.numPulls-arm.wins)) / (arm.avgReward*arm.numPulls) if (arm.avgReward*arm.numPulls) > 0 else 0
                arm.ucb1Value= arm.ucb + math.sqrt(2.0 * math.log(playerCredits) / (arm.avgReward*arm.numPulls))
    bestArmIndex=-1
    bestUcb1Value=-1*float("inf")
    for i in range(len(bandit.arms)):
        if bandit.arms[i].ucb1Value > bestUcb1Value:
            bestUcb1Value = bandit.arms[i].ucb1Value
            bestArmIndex = i
    bandit.bestArm = bandit.arms[bestArmIndex]
    if bandit.arms[bestArmIndex].getArmId() == bandit.getRealBestArm().getArmId():
        return True
    else:
        return False
        
def dynam(bandit, playerCredits):
    # Initialize theta
    theta = .05
    continueEval = True
    while continueEval:
        # Initialize delta
        delta = 0
        # Get the value of the bellman equation for each arm and get the differnece between that and the current value
        for arm in bandit.arms:
            temp = arm.dyValue
            arm.play()
            arm.dyValue = arm.avgReward
            # Get the greatest difference between previous value and current
            delta = max(delta, abs(temp - arm.dyValue))
        # If delta is less than theta terminate loop
        if delta <= theta:
            continueEval = False

    # Set best arm and check if it's the true best arm
    bestArmIndex = -1
    bestValue = -1 * float("inf")
    for i in range(len(bandit.arms)):
        if bandit.arms[i].dyValue > bestValue:
            bestValue = bandit.arms[i].dyValue
            bestArmIndex = i
    bandit.bestArm = bandit.arms[bestArmIndex]
    if bandit.arms[bestArmIndex].getArmId() == bandit.getRealBestArm().getArmId():
        return True
    else:
        return False


def td (bandit, playerCredits):
    for i in range(playerCredits):
        # Update value for each arm
        for arm in bandit.arms:
            arm.tdValue += 1/(arm.numPulls + 1) * (arm.play() - arm.tdValue)
    
    # Set best arm and check if it's the true best arm
    bestArmIndex = -1
    bestTdValue = -1 * float("inf")
    for i in range(len(bandit.arms)):
        if bandit.arms[i].tdValue > bestTdValue:
            bestTdValue = bandit.arms[i].tdValue
            bestArmIndex = i
    bandit.bestArm = bandit.arms[bestArmIndex]
    if bandit.arms[bestArmIndex].getArmId() == bandit.getRealBestArm().getArmId():
        return True
    else:
        return False