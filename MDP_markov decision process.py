import random
import numpy as np
import matplotlib.pyplot as plt


S = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# A = [-3, -1, 1, 3]
A = [0, 1, 2, 3]
r = [-10, 0, 10, 0, 0, 0, 0, 0, 0]

P = {}
for state in S:
    P[state] = {}
    for action in A:
        if state == 2:
            P[state][action] = {2: 1.0}
            continue
        next = state + action
        if next < 0 or next > 8:
            next = state
        elif action == -1 and state % 3 == 0:
            next = state
        elif action == 1 and state % 3 == 2:
            next = state
        if next == state:
            P[state][action] = {state: 1.0}
        else:
            P[state][action] = {next: 0.8,state: 0.2}


# T(s, a, s_bar)
T = [
    # State 0
    {0: [0.1, 0.8, 0, 0.1, 0, 0, 0, 0, 0],
     3: [0.1, 0.1, 0, 0.8, 0, 0, 0, 0, 0]},

    # State 1
    {0: [0, 0.1, 0.8, 0, 0.1, 0, 0, 0, 0],
    2: [0.8, 0.1, 0, 0, 0.1, 0, 0, 0, 0],
    3: [0.1, 0, 0.1, 0, 0.8, 0, 0, 0, 0]},

    # State 2: goal state
    {},

    # State 3
    {0: [0.1, 0, 0, 0, 0.8, 0, 0.1, 0, 0],
    1: [0.8, 0, 0, 0.1, 0.1, 0, 0, 0, 0],
    3: [0, 0, 0, 0.1, 0.1, 0, 0.8, 0, 0]},

    # State 4
    {0: [0, 0.1, 0, 0, 0,0.8, 0, 0.1, 0],
    1: [0, 0.8, 0, 0.1, 0, 0.1, 0, 0, 0],
    2: [0, 0.1, 0, 0.8, 0, 0, 0, 0.1, 0],
    3: [0, 0, 0, 0.1, 0, 0.1, 0, 0.8, 0]},

    # State 5
    {1: [0, 0, 0.8, 0, 0.1, 0.1, 0, 0, 0],
    2: [0, 0, 0.1, 0, 0.8, 0, 0, 0, 0.1],
    3: [0, 0, 0, 0, 0.1, 0.1, 0, 0, 0.8]},

    # State 6
    {0: [0, 0, 0, 0.1, 0, 0, 0.1, 0.8, 0],
    1: [0, 0, 0, 0.8, 0, 0, 0.1, 0.1, 0]},

    # State 7
    {0: [0, 0, 0, 0, 0.1, 0, 0, 0.1, 0.8],
    1: [0, 0, 0, 0, 0.8, 0, 0.1, 0, 0.1],
    2: [0, 0, 0, 0, 0.1, 0, 0.8, 0.1, 0]},

    # State 8
    {1: [0, 0, 0, 0, 0, 0.8, 0, 0.1, 0.1],
    2: [0, 0, 0, 0, 0, 0.1, 0, 0.8, 0.1]}]

R = [-10, 0, +10, 0, 0, -10, 0, 0, 0]
A_opt = np.ones(9)
S_absorbing = [0, 2, 5]


def transition(state, action):
    return random.choices(S,weights=T[state][action],k=1)[0]

def run(start=6, max_step=5000):
    state = start
    path = [state]
    total_reward = 10
    for _ in range(max_step):
        if state == S_absorbing[1]:
            return path, total_reward, "Success"
        elif (state == S_absorbing[0] or state == S_absorbing[2]):
            return path, total_reward, "Wasted"
        options = list(T[state].keys())
        if not options:
            return path, total_reward, "Wasted"
        expected_r = {}
        for action in T[state]:
            expected_r[action] = 0
            for s_bar in S:
                expected_r[action] += (T[state][action][s_bar]* R[s_bar])
        action = max(expected_r,key=expected_r.get)
        A_opt[state] = action
        state = transition(state, action)
        path.append(state)
        total_reward += R[state]
        if state == S_absorbing[1]:
            return path, total_reward, "Success"
        elif (state == S_absorbing[0] or state == S_absorbing[2]):
            return path, total_reward, "Wasted"
    return path, total_reward, "Time Out"

trials = 1000
success = np.zeros(trials)
best_path = None
best_length = np.inf
best_path_each_iteration = []
best_length_each_iteration = np.full(trials,np.nan)
for i in range(trials):
    path, reward, result = run()
    if result == "Success":
        success[i] = 1
        current_length = len(path) - 1
        if current_length < best_length:
            best_length = current_length
            best_path = path.copy()
            print("Iteration:", i + 1,"New best path:", best_path,"Length:", best_length,"Reward:", reward)
    if best_path is None:
        best_path_each_iteration.append(None)
    else:
        best_path_each_iteration.append(best_path.copy())
        best_length_each_iteration[i] = (best_length)
print("\nSuccess rate:",np.mean(success))
print("Final best path:",best_path)
print("Final best length:",best_length)
print("Optimal actions:",A_opt)

for state in sorted(S):
    if state == S_absorbing[1]:
        print(state,"Goal state")
        continue
    elif (state == S_absorbing[0] or state == S_absorbing[2]):
        print(state,"Wasted state")
        continue
    expected_r = {}
    for action in T[state]:
        expected_r[action] = 0
        for s_bar in S:
            expected_r[action] += (T[state][action][s_bar]* R[s_bar])
    action = max(expected_r,key=expected_r.get)
    A_opt[state] = action
    prob_option = {
        option: (1.0 if option == action
                 else 0.0)
        for option in sorted(T[state].keys())}
    print("State:",state,"Expected reward:",expected_r,"Action probability:",prob_option,"Selected action:",action)

#best iter nominating
iterations = np.arange(1, len(best_length_each_iteration) + 1)

#Do not plot unfinished iter
best_length_plot = [
    np.nan if length == float("inf") else length
    for length in best_length_each_iteration]

for state in S:
    if state in S_absorbing or not T[state]:
        continue
    expected_r = {}
    for action in T[state]:
        expected_r[action] = 0
        for s_bar in S:
            expected_r[action] += (T[state][action][s_bar] * R[s_bar])

    A_opt[state] = max(expected_r, key=expected_r.get)
direction = {0: (0.30, 0),1: (0, 0.30),2: (-0.30, 0),3: (0, -0.30)}
iterations = np.arange(1,len(best_length_each_iteration) + 1)
best_length_plot = [
    np.nan if length == float("inf") else length
    for length in best_length_each_iteration]
fig, (ax_grid, ax_total) = plt.subplots(1,2,figsize=(13, 6),gridspec_kw={"width_ratios": [1, 1.3]})

for state in S:
    x = state % 3
    y = 2 - state // 3
    if state == S_absorbing[1]:
        color = "lightgreen"
    elif state == S_absorbing[0] or state == S_absorbing[2]:
        color = "lightcoral"
    elif state == 6:
        color = "gold"
    else:
        color = "white"
    square = plt.Rectangle((x - 0.5, y - 0.5),1,1,facecolor=color,edgecolor="black",linewidth=2)
    ax_grid.add_patch(square)

    # State number
    ax_grid.text(x - 0.38,y + 0.37,str(state),ha="left",va="top",fontsize=14,fontweight="bold")
    # Terminal states do not have an action
    if state == S_absorbing[1]:
        ax_grid.text(x,y,"GOAL",ha="center",va="center",fontsize=14,fontweight="bold")
    elif state == S_absorbing[0] or state == S_absorbing[2]:

        ax_grid.text(x,y,"Wasted",ha="center",va="center",fontsize=14,fontweight="bold")
    else:
        action = int(A_opt[state])
        dx, dy = direction[action]
        ax_grid.annotate("",xy=(x + dx, y + dy),xytext=(x - dx, y - dy),arrowprops={"arrowstyle": "-|>","color": "blue","linewidth": 3,"mutation_scale": 20})
ax_grid.set_xlim(-0.5, 2.5)
ax_grid.set_ylim(-0.5, 2.5)
ax_grid.set_aspect("equal")
ax_grid.set_xticks([])
ax_grid.set_yticks([])
ax_grid.set_title("Optimal Action for Each State")
ax_total.plot(iterations,best_length_plot,color="blue",linewidth=2,drawstyle="steps-post")
ax_total.scatter(iterations,best_length_plot,color="blue",s=1)
ax_total.set_title("Best Route Length by Iteration")
ax_total.set_xlabel("Iteration")
ax_total.set_ylabel("Best Path Length")
ax_total.grid(True)
if np.all(np.isnan(best_length_plot)):
    ax_total.text(0.5,0.5,"No successful route found",ha="center",va="center",transform=ax_total.transAxes)
plt.tight_layout()
plt.show()
