
def vectorized_ic(L, initial_conditions):
    import itertools
    
    # build the lattice row — length L
    row = np.zeros(L, dtype=int)
    row[initial_conditions] = 1       
    
    # find which index in all_states this corresponds to
    all_states = list(itertools.product([0, 1], repeat=L))
    index      = all_states.index(tuple(row))  
    
    # return a vector of length 2^L with a 1 at that index
    v = np.zeros(2**L, dtype=int)
    v[index] = 1
    return v

def next_state(state, rule_number, L): #This code comes up in several functions so it is better to automate it
    rule   = make_rule(rule_number)
    result = []
    for i in range(L):
        left   = str(int(state[i - 1])) if i > 0 else '0'      
        center = str(int(state[i]))
        right  = str(int(state[i + 1])) if i < L - 1 else '0'    
        neighbourhood = left + center + right
        result.append(rule[neighbourhood])
    return tuple(result)



def markov_matrix_mapping(rule_number: int, L: int):
    import itertools

    rule       = make_rule(rule_number)
    all_states = list(itertools.product([0, 1], repeat=L))
    n          = len(all_states)

    M = np.zeros((n, n), dtype=int)

    state_labels = [''.join(str(b) for b in s) for s in all_states]
    print(f"\nTransition matrix — Rule {rule_number}, L={L}")
    print(f"Row = current state   Col = next state\n")
    print("        " + "  ".join(state_labels))
    print("       " + "---" * n)

    for j, state in enumerate(all_states):
        nxt = next_state(state, rule_number, L)   
        i   = all_states.index(nxt)
        label = ''.join(str(b) for b in state)
        row   = "  ".join(str(M[k][j]) for k in range(n))
        print(f"{label}  |  {row}")

    return M

def markov_matrix(rule_number: int, L: int):
    import itertools

    rule = make_rule(rule_number)
    all_states = list(itertools.product([0, 1], repeat=L))
    n  = len(all_states)

    M = np.zeros((n, n), dtype=int)

    state_labels = [''.join(str(b) for b in s) for s in all_states]

    for j, state in enumerate(all_states):
        nxt = next_state(state, rule_number, L)   
        i   = all_states.index(nxt)
        M[i][j] = 1
        label = ''.join(str(b) for b in state)
        row  = "  ".join(str(M[k][j]) for k in range(n))

    return M

def state_timestep(system, timestep):
    return system[:timestep]

def probabilistic_dynamics(rule_number, L, initial_conditions, probability_list, timesteps):
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number not appropriate")
    
    if probability_list is None:
        return dynamics(rule_number, L, initial_conditions, timesteps)
    
    elif isinstance(probability_list, float):
        new_list = []
        for i in range(len(initial_conditions)):
            new_list.append(probability_list)
        probability_list = new_list

    elif len(initial_conditions) != len(probability_list):
        raise ValueError("Each initial condition must have an associated probability")

    rule_bin = format(rule_number, '08b')
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]
    rule = {n: int(b) for n, b in zip(neighbourhoods, rule_bin)} #Previous function for the rule

    # Set up initial row
    row = np.zeros(L, dtype=int)

    for i, probability in zip(initial_conditions, probability_list):
        row[i] = 1 if random.random() < probability else 0
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

    grid = np.vstack(rows[::-1])
    return grid



def probabilistic_dynamics_periodic(rule_number, L, initial_conditions, probability_list, timesteps):
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number not appropriate")
    
    if probability_list is None:
        return dynamics(rule_number, L, initial_conditions, timesteps)
    
    elif isinstance(probability_list, float):
        new_list = []
        for i in range(len(initial_conditions)):
            new_list.append(probability_list)
        probability_list = new_list

    elif len(initial_conditions) != len(probability_list):
        raise ValueError("Each initial condition must have an associated probability")

    rule_bin = format(rule_number, '08b')
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]
    rule = {n: int(b) for n, b in zip(neighbourhoods, rule_bin)} #Previous function for the rule

    # Set up initial row
    row = np.zeros(L, dtype=int)

    for i, probability in zip(initial_conditions, probability_list):
        row[i] = 1 if random.random() < probability else 0
    rows = [row.copy()] #So we take an independent "snapshot"
    
    # Evolve through timesteps
    for i in range(timesteps):
        new_row = np.zeros(L, dtype=int)
        for j in range(L):
            left   = str(int(rows[-1][(j - 1) % L]))  # periodic boundary
            centre = str(int(rows[-1][j]))
            right  = str(int(rows[-1][(j + 1) % L]))   
            neighbourhood = left + centre + right
            new_row[j] = rule[neighbourhood]
        rows.append(new_row)

    grid = np.vstack(rows[::-1])
    return grid


def probabilistic_dynamics_reflection(rule_number, L, initial_conditions, probability_list, timesteps):
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number not appropriate")
    
    if probability_list is None:
        return dynamics(rule_number, L, initial_conditions, timesteps)
    
    elif isinstance(probability_list, float):
        new_list = []
        for i in range(len(initial_conditions)):
            new_list.append(probability_list)
        probability_list = new_list

    elif len(initial_conditions) != len(probability_list):
        raise ValueError("Each initial condition must have an associated probability")

    rule_bin = format(rule_number, '08b')
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]
    rule = {n: int(b) for n, b in zip(neighbourhoods, rule_bin)} #Previous function for the rule

    # Set up initial row
    row = np.zeros(L, dtype=int)

    for i, probability in zip(initial_conditions, probability_list):
        row[i] = 1 if random.random() < probability else 0
    rows = [row.copy()] #So we take an independent "snapshot"


    # Evolve through timesteps
    for i in range(timesteps):
        new_row = np.zeros(L, dtype=int)
        for j in range(L):
            left   = str(int(rows[-1][j + 1])) if j == 0 else str(int(rows[-1][j - 1]))  #mirror reflection
            centre = str(int(rows[-1][j]))
            right  = str(int(rows[-1][j - 1])) if j == L - 1 else str(int(rows[-1][j + 1]))
            neighbourhood = left + centre + right
            new_row[j] = rule[neighbourhood]
        rows.append(new_row)

    grid = np.vstack(rows[::-1])
    return grid


def probabilistic_dynamics_initialandfinal(rule_number, L, initial_conditions, probability_list, timesteps):
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number not appropriate")
    
    if probability_list is None:
        return dynamics(rule_number, L, initial_conditions, timesteps)
    
    elif isinstance(probability_list, float):
        new_list = []
        for i in range(len(initial_conditions)):
            new_list.append(probability_list)
        probability_list = new_list

    elif len(initial_conditions) != len(probability_list):
        raise ValueError("Each initial condition must have an associated probability")

    rule_bin = format(rule_number, '08b')
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]
    rule = {n: int(b) for n, b in zip(neighbourhoods, rule_bin)} #Previous function for the rule

    # Set up initial row
    row = np.zeros(L, dtype=int)

    for i, probability in zip(initial_conditions, probability_list):
        row[i] = 1 if random.random() < probability else 0
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

    grid = np.vstack(rows[::-1])
    return rows[0], "->", rows[-1]

