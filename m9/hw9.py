# Assignment 9 
# Applied Machine Learning 
# Please refer to the Reinforcement Learning Jupyter notebook in course materials.  
 
# Answer questions 1-3 below considering any Nim game reinforcement learning model. 
 
# 1. [10 pts] Describe the environment in the Nim learning model. 
 
# 2. [10 pts] Describe the agent(s) in the Nim learning model (Hint, not just the Q-learner). Is 
# Guru an agent?  
 


# 3. [10 pts] Describe the reward and penalty in the Nim learning model. 
 
# 4. [10 pts] How many possible states there could be in the Nim game with a maximum of 10 
# items per pile and 3 piles total? (This problem requires a number for its answer, not merely 
# a closed-form expression.) 
 
# 5. [10 pts] How many possible unique actions are there for player 1 to take as their first action 
# in a Nim game with 10 items per pile and 3 piles total? (This problem also requires a 
# number for its answer, not merely a closed-form expression.) 
 
# 6. [10 pts] Do you think a Q-learner can beat the Guru player? Why or why not? Be thorough.  
 
# 7. [40 pts] Find a way to improve the provided Nim game learning model. (Hint: How about 
# penalizing the losses? Hint: It is indeed possible to find a better solution, which improves 
# the way Q-learning updates its Q-table). You must code a solution and also demonstrate 
# the improvement by reporting its performance against players (Random, Guru). 
# Do not put the Guru player’s operating code inside the learning module, as this would 
# defeat the purpose of reinforcement learning. However, you may train your improved Q-
# learner by having it playing against a Guru; using those games as experience is legitimate 
# reinforcement learning. 



## CODE FROM MODULE 9

import numpy as np
from random import randint, choice

# The number of piles is 3
# max number of items per pile
ITEMS_MX = 10

# Initialize starting position
def init_game()->list:
    return [randint(1,ITEMS_MX), randint(1,ITEMS_MX), randint(1,ITEMS_MX)]

# Based on X-oring the item counts in piles - mathematical solution
def nim_guru(_st:list):
    xored = _st[0] ^ _st[1] ^ _st[2]
    if xored == 0:
        return nim_random(_st)
    for pile in range(3):
        s = _st[pile] ^ xored
        if s <= _st[pile]:
            return _st[pile]-s, pile

# Random Nim player
def nim_random(_st:list):
    pile = choice([i for i in range(3) if _st[i]>0])  # find the non-empty piles
    return randint(1, _st[pile]), pile  # random move

def nim_qlearner(_st:list):
    global qtable
    # pick the best rewarding move, equation 1
    a = np.argmax(qtable[_st[0], _st[1], _st[2]])  # exploitation
    # index is based on move, pile
    move, pile = a%ITEMS_MX+1, a//ITEMS_MX
    # check if qtable has generated a random but game illegal move - we have not explored there yet
    if move <= 0 or _st[pile] < move:
        move, pile = nim_random(_st)  # exploration
    return move, pile  # action

Engines = {'Random':nim_random, 'Guru':nim_guru, 'Qlearner':nim_qlearner}

def game(_a:str, _b:str):
    state, side = init_game(), 'A'
    while True:
        engine = Engines[_a] if side == 'A' else Engines[_b]
        move, pile = engine(state)
        # print(state, move, pile)  # debug purposes
        state[pile] -= move
        if state == [0, 0, 0]:  # game ends
            return side  # winning side
        side = 'B' if side == 'A' else 'A'  # switch sides

def play_games(_n:int, _a:str, _b:str):
    from collections import defaultdict
    wins = defaultdict(int)
    for _ in range(_n):
        wins[game(_a, _b)] += 1
    # info
    print(f"{_n} games, {_a:>8s}{wins['A']:5d}  {_b:>8s}{wins['B']:5d}")
    return wins['A'], wins['B']

qtable, Alpha, Gamma, Reward = None, 1.0, 0.8, 100.0

# learn from _n games, randomly played to explore the possible states
def nim_qlearn(_n:int):
    global qtable
    # based on max items per pile
    qtable = np.zeros((ITEMS_MX+1, ITEMS_MX+1, ITEMS_MX+1, ITEMS_MX*3), dtype=np.float32)
    # play _n games
    for _ in range(_n):
        # first state is starting position
        st1 = init_game()
        while True:  # while game not finished
            # make a random move - exploration
            move, pile = nim_random(st1)
            st2 = list(st1)
            # make the move
            st2[pile] -= move  # --> last move I made
            if st2 == [0, 0, 0]:  # game ends
                qtable_update(Reward, st1, move, pile, 0)  # I won
                break  # new game

            qtable_update(0, st1, move, pile, np.max(qtable[st2[0], st2[1], st2[2]]))
            
            # Switch sides for play and learning
            st1 = st2

# Equation 3 - update the qtable
def qtable_update(r:float, _st1:list, move:int, pile:int, q_future_best:float):
    a = pile*ITEMS_MX+move-1
    qtable[_st1[0], _st1[1], _st1[2], a] = Alpha * (r + Gamma * q_future_best)

## LEARN
nim_qlearn(1000)

# Play games
print("BASELINE Q-Learner Performance")
play_games(1000, 'Random', 'Qlearner')
play_games(1000, 'Qlearner', 'Random')
play_games(1000, 'Guru', 'Qlearner')
play_games(1000, 'Qlearner', 'Guru')
play_games(1000, 'Guru', 'Guru')
play_games(1000, 'Qlearner', 'Qlearner')
