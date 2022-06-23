from copy import deepcopy
import numpy as np


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
        for r in range (mdp.num_row):
            for c in range (mdp.num_col):
                state = (int(r), int(c))
                # belman equation
                belman_results = []
                for a in mdp.actions:
                    temp_sum = 0
                    probs = mdp.transition_function[a]
                    for p in probs:
                        new_state = mdp.step(state, a)
                        print(new_state)
                        temp_sum += probs[p] * u_prev[new_state[0]][new_state[1]]
                    belman_results.append(temp_sum)
                u_curr[r][c] = mdp.board[r][c] + max(belman_results)
                delta = max(delta, abs(u_curr[r][c] - u_prev[r][c]))
        if delta < max_delta:
            break
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
                    for k in probs:
                        temp_sum += probs[k] * U[mdp.step(state, k)]
                    belman_results.append(temp_sum)
                action= belman_results.index(max(belman_results))
                policy[r][c] = action
    # ========================


def policy_evaluation(mdp, policy):
    U = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            current_state = (i, j)
            sigma = 0
            probs = mdp.transition_function[policy[i][j]]
            for k in probs:
                a, b = mdp.step(current_state, k)
                if a < len(U) and b < U[a]:
                    sigma += probs[k] * U[a][b]
            U[i][j] = mdp.board[i][j] + mdp.gamma * sigma
    return U


def policy_iteration(mdp, policy_init):
    U = policy_evaluation(mdp, policy_init)
    unchanged = True
    while unchanged is True:
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                current_state = (i, j)
                iteration_sum = 0
                argmax = 0 #Just to initialize it.
                for a in mdp.actions:
                    temp_sum = 0
                    probs = mdp.transition_function[a]
                    for k in probs:
                        a, b = mdp.step(current_state, k)
                        temp_sum += probs[k] * U[a][b]
                    if temp_sum > iteration_sum:
                        iteration_sum = temp_sum
                        argmax = a
                probs = mdp.transition_function[policy_init[i][j]]
                policy_sum = 0
                for k in probs:
                    a, b = mdp.step(current_state, k)
                    policy_sum += probs[k] * U[a][b]
                if iteration_sum > policy_sum:
                    policy_init[i][j] = argmax
                    unchanged = False                                        
    return policy_init