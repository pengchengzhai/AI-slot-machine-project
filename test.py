from environment import Bandit
import matplotlib.pyplot as plt
import numpy as np
from algorithms import *
from dataVis import *


"""Initialize n to the number of arms you want and create a Bandit"""
armCount = 10
# playerCredits, the number of pulls the player gets for a session
playerCredits = 50
numTests = 100
numInstances = 50
geBandit = Bandit('Greedy Epsilon (e=0.2)')
depBandit = Bandit('Dec Greedy Epsilon')
sfBandit = Bandit('Softmax')
ucBandit = Bandit('UCB1')
tdBandit = Bandit('TD')
dyBandit = Bandit('Dynamic')

geBandit2 = Bandit('Greedy Epsilon (e=0.2)')
depBandit2 = Bandit('Dec Greedy Epsilon')
sfBandit2 = Bandit('Softmax')
ucBandit2 = Bandit('UCB1')
tdBandit2 = Bandit('TD')
dyBandit2 = Bandit('Dynamic')

###################################################################
# This is for the % Optimal Graph

geBandit2.optimalWins = [0] * numTests
depBandit2.optimalWins = [0] * numTests
sfBandit2.optimalWins = [0] * numTests
ucBandit2.optimalWins = [0] * numTests
tdBandit2.optimalWins = [0] * numTests
dyBandit2.optimalWins = [0] * numTests

for outer in range(numInstances):
    geBandit2.arms = []
    geBandit2.numArms = 0
    depBandit2.arms = []
    depBandit2.numArms = 0
    sfBandit2.arms = []
    sfBandit2.numArms = 0
    ucBandit2.arms = []
    ucBandit2.numArms = 0
    tdBandit2.arms = []
    tdBandit2.numArms = 0
    dyBandit2.arms = []
    dyBandit2.numArms = 0

    for i in range(armCount):
        geBandit2.addArm()
        depBandit2.addArm()
        sfBandit2.addArm()
        ucBandit2.addArm()
        tdBandit2.addArm()
        dyBandit2.addArm()

        depBandit2.arms[i].prob = geBandit2.arms[i].prob
        depBandit2.arms[i].expectedValue = geBandit2.arms[i].expectedValue

        sfBandit2.arms[i].prob = geBandit2.arms[i].prob
        sfBandit2.arms[i].expectedValue = geBandit2.arms[i].expectedValue

        ucBandit2.arms[i].prob = geBandit2.arms[i].prob
        ucBandit2.arms[i].expectedValue = geBandit2.arms[i].expectedValue
        
        tdBandit2.arms[i].prob = geBandit2.arms[i].prob
        tdBandit2.arms[i].expectedValue = geBandit2.arms[i].expectedValue
        
        dyBandit2.arms[i].prob = geBandit2.arms[i].prob
        dyBandit2.arms[i].expectedValue = geBandit2.arms[i].expectedValue
        
    # Calls the an algorithm to train than test each bandit
    for i in range(numTests):
        if geBandit2.train(playerCredits, 0):
            geBandit2.optimalWins[i] += 1

        if depBandit2.train(playerCredits, 1):
            depBandit2.optimalWins[i] += 1

        if sfBandit2.train(playerCredits, 2):
            sfBandit2.optimalWins[i] += 1
        
        if ucBandit2.train(playerCredits, 3):
            ucBandit2.optimalWins[i] += 1
        
        if tdBandit2.train(playerCredits, 4):
            tdBandit2.optimalWins[i] += 1
        
        if dyBandit2.train(playerCredits, 5):
            dyBandit2.optimalWins[i] += 1

        geBandit2.test(playerCredits)
        
        depBandit2.test(playerCredits)
        
        sfBandit2.test(playerCredits)
        
        tdBandit2.test(playerCredits)
        
        ucBandit2.test(playerCredits)
        
        dyBandit2.test(playerCredits)
        
percentOptimalGraph([geBandit2, depBandit2, sfBandit2, ucBandit2, tdBandit2, dyBandit2], numTests, numInstances)
##################################################################

"""Add n arms to each Bandit"""
for i in range(armCount):
    geBandit.addArm()
    depBandit.addArm()
    sfBandit.addArm()
    ucBandit.addArm()
    tdBandit.addArm()

    depBandit.arms[i].prob = geBandit.arms[i].prob
    depBandit.arms[i].expectedValue = geBandit.arms[i].expectedValue

    sfBandit.arms[i].prob = geBandit.arms[i].prob
    sfBandit.arms[i].expectedValue = geBandit.arms[i].expectedValue

    ucBandit.arms[i].prob = geBandit.arms[i].prob
    ucBandit.arms[i].expectedValue = geBandit.arms[i].expectedValue
    
    tdBandit.arms[i].prob = geBandit.arms[i].prob
    tdBandit.arms[i].expectedValue = geBandit.arms[i].expectedValue

for i in range(numTests):
    geBandit.train(playerCredits, 0)
    depBandit.train(playerCredits, 1)
    sfBandit.train(playerCredits, 2)
    ucBandit.train(playerCredits, 3)
    tdBandit.train(playerCredits, 4)
    geBandit.test(playerCredits)
    depBandit.test(playerCredits)
    ucBandit.test(playerCredits)
    sfBandit.test(playerCredits)
    tdBandit.test(playerCredits)

'''
"""Prints the stats for all arms"""
print 'Stats for {}:'.format(geBandit.name)
for i in range(armCount):
    geBandit.arms[i].printStats()
print 'Stats for {}:'.format(sfBandit.name)
for i in range(armCount):
    sfBandit.arms[i].printStats()
print 'Stats for {}:'.format(ucBandit.name)
for i in range(armCount):
    ucBandit.arms[i].printStats()
'''
'''----------------------------------------------------------'''
"""---  Beyond here is the logic for plotting the graphs  ---"""
'''----------------------------------------------------------'''
#plot(geBandit)
avgGraph([geBandit, depBandit, sfBandit, ucBandit, tdBandit], numTests)
plt.tight_layout()
plt.show()
