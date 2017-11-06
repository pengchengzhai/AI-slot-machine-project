import random
from algorithms import *



class Bandit:
    """Initializes Bandit with an empty list of arms and the number of arms set to 0"""

    def __init__(self, name):
        self.name = name
        self.arms = []
        self.expectedVals = []
        self.testAvgs = [0]
        self.numPulls = 0.0
        self.avgReward = 0
        self.numArms = 0
        self.bestArm = None
        self.optimalWins = []
        
    """Creates a new arm. Gives it an id of 1 + the length of arms, so ids start at 1 rather than 0"""

    def addArm(self):
        newArm = Arm(len(self.arms) + 1)
        self.arms.append(newArm)
        self.expectedVals.append(newArm.expectedValue)
        self.numArms += 1
        newArm.parent = self
        return newArm

    """Returns an arm given an id"""

    def getArm(self, _id):
        return self.arms[_id - 1]


    """Returns the arm that has the highest average reward"""

    def getBestArm(self):
        bestVal = -1
        bestIndex = 0
        for i in range(self.numArms):
            if self.arms[i].avgReward > bestVal:
                bestVal = self.arms[i].avgReward
                bestIndex = i
        self.bestArm = self.getArm(bestIndex + 1)
        return self.bestArm

    def getRealBestArm(self):
        bestArm = 0
        bestProb = 0
        for i in range(self.numArms):
            tempProb = self.getArm(i).getProb()
            if tempProb > bestProb + 0.0:
                bestArm = i
                bestProb = tempProb

        return self.getArm(bestArm)

    """Trains a bandit using by using a specified algorithm"""

    def train(self, trials, algoId):

        # Choses an algorithm to train with based on the inputted id
        algos = {0 : greedyEpsilon,
                 1 : decGreedyEpsilon,
                 2 : softmax,
                 3 : ucb1,
                 4 : td,    
                 5 : dynam    }
        # Runs the chosen algorithm
        return algos[algoId](self, trials)

    """Tests the bandit by exploiting the best arm"""

    def test(self, trials):
        # Sets best arm to the best arm and initializes total to the total reward
        #self.bestArm = self.getBestArm()
        total = self.numPulls * self.avgReward
        # Pulls the arm trials amount of time and updates total and numPulls
        for i in range(trials):
            self.numPulls += 1
            result = random.uniform(.01, 1)
            if result <= self.bestArm.prob:
                total += self.bestArm.reward
        # Calculates avgReward and appends it to the list of test averages
        self.avgReward = total/self.numPulls
        self.testAvgs.append(self.avgReward)


class Arm:
    """Initializes Arm with a given id, a random win probability and a reward of 100. Everything else set to 0"""

    def __init__(self, _id):
        self._id = _id
        self.prob = random.uniform(.01, .5)
        self.reward = 100.0
        self.avgReward = 0
        self.numPulls = 0.0
        self.wins = 0.0
        self.expectedValue = self.prob * self.reward
        self.parent = None
        self.tdValue = 0.0
        self.ucb = 0.0
        self.ucb1Value = 0.0
        self.dyValue = 0.0

    def getArmId(self):
        return self._id

    def getProb(self):
        return self.prob

    """
    Simulates playing the arm by generating a random number and if it's less than or equal to the arm's probability it's a win, if it's greater then it's a loss.
    Then it increments numPulls and recalculates avgReward
    """

    def play(self):
        # print 'Playing arm {}'.format(self.id)
        armTotal = self.avgReward * self.numPulls
        self.numPulls += 1
        result = random.uniform(.01, 1)
        reward = 0.0;
        if result <= self.prob:
            reward = self.reward
            armTotal += self.reward
            self.wins += 1
        # print "You won!"
        # else:
        # print "You lost"
        self.avgReward = armTotal / self.numPulls
        return reward


    def returnStats(self):
        tempWinRatio = 0
        if self.numPulls != 0:
            tempWinRatio = self.wins / self.numPulls
        return {"_id": self._id, "winRatio": tempWinRatio, "actualProbability": self.prob, "avgReward": self.avgReward}

    """Prints the arm's winning percentage, probability and average reward"""

    def printStats(self):
        tempWinRatio = 0
        if self.numPulls != 0:
            tempWinRatio = self.wins / self.numPulls
        print 'Stats for arm {}:'.format(self._id)
        print '\tNumber of pulls for arm: ', self.numPulls
        print '\tWinning percentage: {0:.0f}%'.format(tempWinRatio * 100)
        print '\tActual probability: {0:.2f}'.format(self.prob)
        print '\tAverage reward: {0:.2f}'.format(self.avgReward)
