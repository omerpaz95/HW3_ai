from copy import deepcopy
from time import sleep
import numpy as np

mdp_actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

def state_to_reward(mdp, r, c):
    num = float(mdp.board[r][c]) if (mdp.board[r][c] != 'WALL') else 0
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
    max_delta = epsilon*(1-mdp.gamma)/(mdp.gamma)
    for r in range (mdp.num_row):
            for c in range (mdp.num_col):
                u_prev[r][c] = state_to_reward(mdp, r, c)
    print("new initial utility: ", u_prev)
    u_curr = deepcopy(u_prev)
    iteration = 0

    while True:
        delta = 0
        u_prev = deepcopy(u_curr)
        for r in range (mdp.num_row):
            for c in range (mdp.num_col):
                state = (int(r), int(c))
                # belman equation
                if state in mdp.terminal_states:
                    reward = state_to_reward(mdp, r, c)
                    u_curr[r][c] = reward  
                elif mdp.board[r][c] == 'WALL':
                    u_curr[r][c] = 0
                else:
                    belman_results = []
                    for a in mdp.actions:
                        temp_sum = 0
                        probs = mdp.transition_function[a]
                        for i, p in enumerate (probs):
                            new_state = mdp.step(state, mdp_actions[i])
                            temp_sum += p * u_prev[new_state[0]][new_state[1]]
                        belman_results.append(temp_sum)
                        # print("temp sum: ", temp_sum )
                    reward = state_to_reward(mdp, r, c)
                    u_curr[r][c] = reward + mdp.gamma * max(belman_results) 
                    print("state: ", state)
                    print("reward = :", reward)
                    print("u_prev: ", u_prev[r][c])
                    print("u_curr: ", u_curr[r][c])
                    print("diff = ", abs(u_curr[r][c] - u_prev[r][c]))
                    print("belman_results: ", belman_results)
                    delta = max(delta, abs(u_curr[r][c] - u_prev[r][c]))
        if delta < max_delta or delta == 0 or delta < 0.01:
            print("")
            print("did ", iteration, " iterationsssss")
            break
        else:
            iteration+=1
            print("delta = ", delta)
    return u_curr
                
            
        
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    # ====== YOUR CODE: ======
    policy = [[0]*mdp.num_col]*mdp.num_row
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
    U = [[0]*mdp.num_col]*mdp.num_row
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            current_state = (r, c)
            sigma = 0
            if current_state in mdp.terminal_states:  
                reward = state_to_reward(mdp, r, c)
                U[r][c] = reward  
            elif mdp.board[r][c] == 'WALL':
                U[r][c] = 0 
            else:
                probs = mdp.transition_function[policy[r][c]]
                for i, k in enumerate(probs):
                    a, b = mdp.step(current_state, mdp_actions[i])
                    sigma += k * U[a][b]
                reward = state_to_reward(mdp, r, c)
                U[r][c] = reward + mdp.gamma * sigma 
    return U


def policy_iteration(mdp, policy_init):
    U = policy_evaluation(mdp, policy_init)
    unchanged = True
    while unchanged is True:
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                current_state = (r, c)
                iteration_sum = 0
                argmax = 0 #Just to initialize it.
                for a in mdp.actions:
                    temp_sum = 0
                    probs = mdp.transition_function[a]
                    for i, k in enumerate(probs):
                        a, b = mdp.step(current_state, mdp_actions[i])
                        temp_sum += k * U[a][b]
                    if temp_sum > iteration_sum:
                        iteration_sum = temp_sum
                        argmax = a
                if current_state not in mdp.terminal_states and mdp.board[r][c] != 'WALL':
                    probs = mdp.transition_function[policy_init[r][c]]
                    policy_sum = 0
                    for i, k in enumerate(probs):
                        a, b = mdp.step(current_state, mdp_actions[i])
                        policy_sum += k * U[a][b]
                    if iteration_sum > policy_sum:
                        policy_init[r][c] = mdp_actions[argmax]
                        unchanged = False
    return policy_init