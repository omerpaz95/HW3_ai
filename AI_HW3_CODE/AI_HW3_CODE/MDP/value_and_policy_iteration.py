from copy import deepcopy
from time import sleep
import numpy as np

# Util = [[0] * 4] * 3
mdp_actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

Util = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

def state_to_reward(mdp, r, c):
    num = float(mdp.board[r][c]) if (mdp.board[r][c] != 'WALL') else -0.04
   # print("Reward for [", r, "][", c, "] is: ", num)
    return num


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    u_prev = U_init
    u_curr = U_init
    max_delta = epsilon*(1-mdp.gamma)/(mdp.gamma)
    while True:
        delta = 0
        u_prev = u_curr
        for r in range (mdp.num_row):
            for c in range (mdp.num_col):
                state = (int(r), int(c))
                # belman equation
                belman_results = []
                for a in mdp.actions:
                    temp_sum = 0
                    probs = mdp.transition_function[a]
                    for i, p in enumerate (probs):
                        new_state = mdp.step(state, mdp_actions[i])
                        # print("probability :  ", probs)
                        # print("action :  ", a)
                        # print("old state :  ", state)
                        # print("new state :  ", new_state)
                        temp_sum += p * u_prev[new_state[0]][new_state[1]]
                    belman_results.append(temp_sum)
                    # print("temp sum: ", temp_sum )
                reward = state_to_reward(mdp, r, c)
                u_curr[r][c] = reward + mdp.gamma * max(belman_results) 
                delta = max(delta, abs(u_curr[r][c] - u_prev[r][c]))
        if delta < max_delta or delta == 0:
            break
        else:
            print("delta = ", delta)
            sleep(1)
    return u_curr
                
            
        
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    # ====== YOUR CODE: ======
    policy = []
    for r in range(mdp.num_row):
        row = []
        for c in range(mdp.num_col):
            row.append(0)
        policy.append(row)
    for r in range (mdp.num_row):
            for c in range (mdp.num_col):
                state = (r, c)
                # belman equation
                belman_results = []
                for a in mdp.actions:
                    temp_sum = 0
                    probs = mdp.transition_function[a]
                    for i, p in enumerate (probs):
                        new_state = mdp.step(state, mdp_actions[i])
                        # print("probability :  ", probs)
                        # print("action :  ", a)
                        # print("old state :  ", state)
                        # print("new state :  ", new_state)
                        temp_sum += p * U[new_state[0]][new_state[1]]
                    belman_results.append(temp_sum)
                action= belman_results.index(max(belman_results))
                policy[r][c] = mdp_actions[action]
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            current_state = (r, c)
            sigma = 0
            if current_state in mdp.terminal_states:
                reward = state_to_reward(mdp, r, c)
                Util[r][c] = reward
            elif mdp.board[r][c] == 'WALL':
                Util[r][c] = -0.04 #TEMP
            else:
                probs = mdp.transition_function[policy[r][c]]
                for i, k in enumerate(probs):
                    a, b = mdp.step(current_state, mdp_actions[i])
                    sigma += k * Util[a][b]
                reward = state_to_reward(mdp, r, c)
                Util[r][c] = reward + mdp.gamma * sigma

    print("Printing utility...")
    mdp.print_utility(Util)
    return Util


def policy_iteration(mdp, policy_init):
    # Util = [[0]*mdp.num_col]*mdp.num_row
    unchanged = True
    u = 0
    for u in range(16): #while unchanged: #
        U = policy_evaluation(mdp, policy_init)
        unchanged = True
        print("Just to reiterate...")
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                current_state = (r, c)
                iteration_sum = -float("inf")
                argmax = 0 #Just to initialize it.
                for a in mdp.actions:
                    temp_sum = 0
                    probs = mdp.transition_function[a]
                    for i, k in enumerate(probs):
                        r1, c1 = mdp.step(current_state, mdp_actions[i])
                        temp_sum += k * U[r1][c1]
                    if temp_sum > iteration_sum:
                        iteration_sum = temp_sum
                        argmax = a
                if current_state not in mdp.terminal_states and mdp.board[r][c] != 'WALL':
                    probs = mdp.transition_function[policy_init[r][c]]
                    policy_sum = 0
                    for i, k in enumerate(probs):
                        r1, c1 = mdp.step(current_state, mdp_actions[i])
                        policy_sum += k * U[r1][c1]
                    if iteration_sum > policy_sum:
                        print("Changing policy in iteration number ", u)
                        policy_init[r][c] = argmax
                        unchanged = False
        u += 1
    return policy_init