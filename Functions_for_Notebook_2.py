import random
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_multiple(grids: list, cols: int):
    n = len(grids)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20 * cols, 15 * rows))
    fig.patch.set_facecolor('black')
    axes = axes.flatten()

    for i, grid in enumerate(grids):
        axes[i].imshow(grid, cmap='binary', interpolation='nearest')
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(hspace=0.5, wspace=0.001)  # vertical and horizontal spacing
    plt.show()



def dynamics_initialandfinal(rule_number, L, initial_conditions, timesteps):
    # Building up the lookup table
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number not appropiate")
    
    rule_bin = format(rule_number, '08b')
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]
    rule = {n: int(b) for n, b in zip(neighbourhoods, rule_bin)} #Previous function for the rule

    # Set up initial row
    row = np.zeros(L, dtype=int)
    for i in initial_conditions:
        row[i] = 1
    rows = [row.copy()] #So we take an independent "snapshot"

    # Evolve through timesteps
    for i in range(timesteps):
        new_row = np.zeros(L, dtype=int)
        for j in range(L):
            left   = str(int(rows[-1][j - 1])) if j > 0 else '0'       # zero boundary
            centre = str(int(rows[-1][j]))
            right  = str(int(rows[-1][j + 1])) if j < L - 1 else '0'   # zero boundary
            neighbourhood = left + centre + right
            new_row[j] = rule[neighbourhood]
        rows.append(new_row)

    
    return rows[0], "->", rows[-1]


def state_comparison(system_1, system_2):
    if system_1.shape != system_2.shape:
        raise ValueError("Systems must be of the same size")
    
    j = np.sum(system_1 != system_2)
    n = system_1.size
    
    return (j, j / n)

def state_comparison_slice(system_1, system_2, timestep, axis):
    if axis == 1:
        i = timestep
        slice_1 = system_1[:, i]
        slice_2 = system_2[:, i]
    else:
        i = system_1.shape[0]-1-timestep
        slice_1 = system_1[i, :]
        slice_2 = system_2[i, :]

    if len(slice_1) != len(slice_2):
        raise ValueError("Systems must be of the same size")

    j = np.sum(slice_1 != slice_2)
    return ( j, j / len(slice_1))


def plot_cell(rule, cell_i, timesteps, L):
    initial_conditions = np.random.randint(0, 2, size=L)
    row = list(np.where(initial_conditions == 1)[0])
    system = dynamics(rule, L, row, timesteps)
    
    cell = system[:, cell_i]  
    
    plt.figure(figsize=(10, 4))
    plt.plot(cell)
    plt.xlabel('timestep')
    plt.ylabel('state (0 or 1)')
    plt.title(f'time evolution of cell {cell_i} — rule {rule}')
    plt.yticks([0, 1])
    plt.show()





# Missing that column function







def rule_1(rule):
    initial_conditions_1 = np.random.randint(0, 2, size=1000)
    initial_conditions_2 = np.random.randint(0, 2, size=1000)
    system_1 = dynamics(rule, 1000, initial_conditions_1, 1000)
    system_2 = dynamics(rule, 1000, initial_conditions_2, 1000)

    if state_comparison_slice(system_1, system_2, 999, axis=0)[1] <= 0.1:
        return True
    else:
        return False

def rule_2(rule): 
    initial_conditions_1 = np.random.randint(0, 2, size=1000)
    initial_conditions_2 = initial_conditions_1.copy()
    i = random.randint(0, 999)
    if initial_conditions_1[i] == 1:
        initial_conditions_1[i] = 0
    else:
        initial_conditions_1[i] = 1

    row_1 = list(np.where(initial_conditions_1 == 1)[0])
    row_2 = list(np.where(initial_conditions_2 == 1)[0])

    system_1 = dynamics(rule, 1000, row_1, 1000)
    system_2 = dynamics(rule, 1000, row_2, 1000)

    if system_1.shape != system_2.shape:
        raise ValueError("Systems must be of the same size")
    
    j = np.sum(system_1 != system_2)
    n = system_1.size

    if j/n <=0.1:
        return True
    else:
        return False



def rule_3(rule):
    initial_conditions_1 = np.random.randint(0, 2, size=1000)
    initial_conditions_2 = initial_conditions_1.copy()
    i = random.randint(0, 999)
    if initial_conditions_1[i] == 1:
        initial_conditions_1[i] = 0
    else:
        initial_conditions_1[i] = 1

    row_1 = list(np.where(initial_conditions_1 == 1)[0])
    row_2 = list(np.where(initial_conditions_2 == 1)[0])

    system_1 = dynamics(rule, 1000, row_1, 1000)
    system_2 = dynamics(rule, 1000, row_2, 1000)
    
    # row 0 is the final timestep due to reversal in dynamics. Very important
    vals = [np.sum(system_1[i] != system_2[i]) / 1000 for i in range(0, 100)]
    val = np.mean(vals)

    return 0.25 <= val <= 0.75, val





def is_rule(rule):
    if rule_1 == True:
        return "Class I"
    elif rule_2 == True:
        return "Class II"
    elif rule_3 == True:
        return "Class III"
    else:
        return "Class IV"

def H_CA(rule): #Take an empirical probability approach, this is the entrpy of the system at each timestep
    initial_conditions = np.random.randint(0, 2, size=1000)
    row = list(np.where(initial_conditions == 1)[0])
    system = dynamics(rule, 1000, row, 1000)
    entropies_at_each_timestep = []
    for t in range(1000):
        p1=np.mean(system[t])
        p0 = 1 - p1
    
        if p0 == 0 or p1 == 0:
            entropies_at_each_timestep.append(0)
        else:
            h = -p0 * np.log2(p0) - p1 * np.log2(p1)
            entropies_at_each_timestep.append(h)

    return entropies_at_each_timestep 

def H_cell(rule, location): #Evolution of the single cell across the timeseries
    initial_conditions = np.random.randint(0, 2, size=1000)
    row = list(np.where(initial_conditions == 1)[0])
    system = dynamics(rule, 1000, row, 1000)
    
    cell = system[:, location]

    p1 = np.mean(cell)
    p0 = 1 - p1
    
    if p0 == 0 or p1 == 0:
        return 0
    
    return -p0 * np.log2(p0) - p1 * np.log2(p)


def H_two_cells(rule, cell_i, cell_j):
    initial_conditions = np.random.randint(0, 2, size=L)
    row = list(np.where(initial_conditions == 1)[0])
    system = dynamics(rule, 1000, row, 1000)
    
    x = system[:, cell_i]  # time series of cell i
    y = system[:, cell_j]  # time series of cell j
    
    # joint probabilities
    p11 = np.mean((x == 1) & (y == 1))
    p00 = np.mean((x == 0) & (y == 0))
    p10 = np.mean((x == 1) & (y == 0))
    p01 = np.mean((x == 0) & (y == 1))
    
    # joint entropy
    h = 0
    for p in [p00, p01, p10, p11]:
        if p > 0:
            h -= p * np.log2(p)
    
    return h


def mutual_information(rule, cell_i, cell_j):
    try:
        cell_j = cell_i+1
    except:
        cell_j = cell_i-1
    
    ic = np.random.randint(0, 2, size=L)
    row = list(np.where(ic == 1)[0])
    system = dynamics(rule, 1000, row, 1000)
    
    x = system[:, cell_i]  # time series of cell i
    y = system[:, cell_j]  # time series of cell j
    
    H_x = H_cell(rule, cell_i)
    H_y = H_cell(rule, cell_j)
    H_xy = H_two_cells(rule, cell_i, cell_j)
    
    return H_x + H_y - H_xy

# Missing effective complexity and associated functions




def damage(rule):
    initial_conditions_1 = np.random.randint(0, 2, size=1000)
    initial_conditions_2 = initial_conditions_1.copy()
    i = random.randint(0, 999)
    if initial_conditions_1[i] == 1:
        initial_conditions_1[i] = 0
    else:
        initial_conditions_1[i] = 1

    row_1 = list(np.where(initial_conditions_1 == 1)[0])
    row_2 = list(np.where(initial_conditions_2 == 1)[0])

    system_1 = dynamics(rule, 1000, row_1, 1000)
    system_2 = dynamics(rule, 1000, row_2, 1000)
    
    dmg = np.array([np.sum(system_1[t] != system_2[t]) for t in range(1, 1000)])
    mean = np.mean(dmg)

    variance = np.var(dmg)
    return dmg, mean, variance

def lyapunov_exponent(rule):
    early = damage[:20] # Take early sample to avoid saturation
    early = early[early > 0] # Avoid log of 0
    t = np.arange(1, len(early) + 1)
    exponent = np.mean(np.log(early) / t)
    return exponent #Usually represented by a lambda symbol

def plot_damage(rule):
    dmg = damage(rule)[0]
    plt.figure(figsize=(10, 4))
    plt.plot(dmg)
    plt.axhline(y=500, color='r', linestyle='--', label='L/2 (max)')
    plt.xlabel('timestep')
    plt.ylabel('damage (different cells)')
    plt.title(f'damage spreading — rule {rule}')
    plt.legend()
    plt.show()


def langtons_lambda(rule):
    rule_bin = format(rule, '08b')
    return sum(int(b) for b in rule_bin) / 8

def classify_by_lambda(rule):
    λ = langtons_lambda(rule)
    
    if λ < 0.2 or λ > 0.8:
        return f"Class I (λ={λ:.3f})"
    elif 0.2 <= λ < 0.4 or 0.6 < λ <= 0.8:
        return f"Class II (λ={λ:.3f})"
    elif 0.45 <= λ <= 0.55:
        return f"Class IV (λ={λ:.3f})"
    else:
        return f"Class III (λ={λ:.3f})"

