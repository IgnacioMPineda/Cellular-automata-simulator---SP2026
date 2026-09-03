import numpy as np

from Functions_for_Notebook_1 import make_rule


def make_rule(rule_number: int) -> dict:
    if not (0 <= rule_number <= 255):
        raise ValueError(f"Rule number must be between 0 and 255, got {rule_number}")
 
    rule_bin = format(rule_number, '08b')  # e.g. rule 110 -> '01101110 (encode it into byts)' The decimal -> binary process explained above
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]  # '111' down to '000'
 
    return {n: int(b) for n, b in zip(neighbourhoods, rule_bin)}

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



def dynamics_matrix(rule_number, L, initial_conditions, timesteps):

    def decode_state(index, L):
        """Inverse of vectorized_ic's binary indexing convention."""
        return np.array([(index >> (L - 1 - i)) & 1 for i in range(L)], dtype=int)

    M = markov_matrix(rule_number, L)
    vector = vectorized_ic(L, initial_conditions)

    rows = [decode_state(np.argmax(vector), L)]   # t=0
    for _ in range(timesteps):
        vector = M @ vector
        rows.append(decode_state(np.argmax(vector), L))

    grid = np.array(rows[::-1])   # earliest at bottom, matching dynamics()
    return grid




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

def vectorized_ic_probabilistic(L, initial_conditions, probability_list: list):
    import itertools
    
    # build the lattice row — length L
    row = np.zeros(L, dtype=int)
    for i, probability in zip(initial_conditions, probability_list):
        row[i] = 1 if random.random() < probability else 0       
    
    # find which index in all_states this corresponds to
    all_states = list(itertools.product([0, 1], repeat=L))
    index      = all_states.index(tuple(row))  
    
    # return a vector of length 2^L with a 1 at that index
    v = np.zeros(2**L, dtype=int)
    v[index] = 1
    return v

def markov_matrix_mapping_probabilistic(rule_number: int, L: int, p: float):
    all_states   = list(itertools.product([0, 1], repeat=L))
    n = len(all_states)
    M = np.zeros((n, n), dtype=float)

    state_labels = [''.join(str(b) for b in s) for s in all_states]
    print(f"\nProbabilistic Transition matrix — Rule {rule_number}, L={L}, p={p}")
    print(f"Row = next state   Col = current state\n")
    print("        " + "  ".join(state_labels))
    print("       " + "---" * n)

    for j, state in enumerate(all_states):
        nxt = next_state(state, rule_number, L)

        # bits that rule maps to 0 → always 0
        # bits that rule maps to 1 → 1 with probability p, else 0
        probabilistic_nxt = tuple(
            1 if (bit == 1 and random.random() < p) else 0
            for bit in nxt
        )

        i        = all_states.index(probabilistic_nxt)
        M[i][j]  = 1
        label    = ''.join(str(b) for b in state)
        row      = "  ".join(f"{M[k][j]:.0f}" for k in range(n))
        print(f"{label}  |  {row}")



def markov_matrix_probabilistic(rule_number: int, L: int, p: float):
    all_states = list(itertools.product([0, 1], repeat=L))
    n = len(all_states)
    M = np.zeros((n, n), dtype=float)

    for j, state in enumerate(all_states):
        nxt = next_state(state, rule_number, L)  # deterministic next state

        # build the probabilistic next state
        probabilistic_nxt = tuple(
            1 if (bit == 1 and random.random() < p) else 0
            for bit in nxt
        )
        # bit == 0 → always stays 0
        # bit == 1 → becomes 1 with probability p, else 0

        i = all_states.index(probabilistic_nxt)
        M[i][j] = 1

    return M

def markov_matrix_noise(rule_number: int, L: int, probability: float):
    """
    Normal CA transition matrix but every bit in the output
    has a small probability epsilon of flipping.
    
    probability = 0.0 → identical to normal deterministic markov_matrix
    probability = 0.5 → completely random, rule means nothing
    probability = 1 → like if the rule was inversed
    """

    all_states = list(itertools.product([0, 1], repeat=L))
    n  = len(all_states)
    M = np.zeros((n, n), dtype=float)

    for j, state in enumerate(all_states):
        nxt = next_state(state, rule_number, L)  # deterministic next state

        # apply bit-flip noise to every bit in nxt
        noisy_nxt = tuple(
            (1 - bit) if random.random() < probability else bit
            for bit in nxt
        )

        i = all_states.index(noisy_nxt)
        M[i][j] = 1

    return M

def is_reversible_matrix(M):
    if np.linalg.det(M) != 0:
        return True
    else:
        return False

def spectral_radius(M):
    eigenvalues = np.linalg.eig(M)[0]
    return np.max(np.abs(eigenvalues))

def spectral_gap(M):
    eigenvalues = np.abs(np.linalg.eig(M)[0])  
    eigenvalues = sorted(eigenvalues, reverse=True) 
    eigenvalue_max_1 = eigenvalues[0]  
    eigenvalue_max_2 = eigenvalues[1]   
    return eigenvalue_max_1 - eigenvalue_max_2


def algebraic_multiplicities(M):
    eigvals = np.linalg.eigvals(M) # Direct way to get eigenvalues without slicing
    
    unique_vals = []
    counts = []

    for v in eigvals:
        found = False
        for i, u in enumerate(unique_vals):
            if np.isclose(v, u):
                counts[i] += 1
                found = True
                break
        if not found:
            unique_vals.append(v)
            counts.append(1)

    return list(zip(unique_vals, counts)) # Dictionaries behave strangely with float numbers, so just use tuples in lists


def semiclassical_markov_matrix(rule_number: int, L: int, probability: float):
   
    all_states = list(itertools.product([0, 1], repeat=L))
    n = len(all_states)
    M = np.zeros((n, n), dtype=float)

    for j, state in enumerate(all_states):
        for noisy_state in all_states:
            # probability of this particular noisy version of `state`
            
            p_noise = 1.0
            for bit, noisy_bit in zip(state, noisy_state):
                p_noise *= probability if bit != noisy_bit else (1 - probability)
            if p_noise == 0.0:
                continue
            nxt = next_state(noisy_state, rule_number, L)  # deterministic rule on the noisy input
            i = all_states.index(nxt)
            M[i, j] += p_noise   # accumulate — several noisy states can map to the same next state

    return M


def semiclassical_dynamics(rule_number, L, initial_conditions, timesteps, probability):
    if not (0 <= rule_number <= 255):
        raise ValueError("Rule number not appropiate")
    rule = make_rule(rule_number)

    row = np.zeros(L, dtype=int)
    for i in initial_conditions:
        row[i] = 1
    rows = [row.copy()]  

    for _ in range(timesteps):
        prev = rows[-1]
        noisy_prev = np.array([
            1 - bit if random.random() < probability else bit
            for bit in prev
        ])
        new_row = np.zeros(L, dtype=int)
        for j in range(L):
            left   = str(int(noisy_prev[j-1])) if j > 0 else '0'
            centre = str(int(noisy_prev[j]))
            right  = str(int(noisy_prev[j+1])) if j < L-1 else '0'
            new_row[j] = rule[left+centre+right]
        rows.append(new_row)

    grid = np.array(rows[::-1])
    return grid

def det_vs_noise(rule_number, L, probabilities):
    dets = []
    for p in probabilities:
        M = markov_matrix_noise(rule_number, L, p)
        dets.append(abs(np.linalg.det(M)))
    return np.array(dets)

reversible_rules = [i for i in range(256) if is_reversible(i)]
print(f"Found {len(reversible_rules)} reversible rules: {reversible_rules}")

L = 6
probabilities = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5]

for rule in reversible_rules[:4]:
    dets = det_vs_noise(rule, L, probabilities)
    print(f"\nRule {rule} (deterministically reversible):")
    for p, det in zip(probabilities, dets):
        M = markov_matrix_noise(rule, L, p)
        print(f"  p={p:<6} det(M) = {det:.3e}   reversible? {is_reversible_matrix(M)}")

plt.figure(figsize=(8, 5))
for rule in reversible_rules[:4]:
    dets = det_vs_noise(rule, L, np.linspace(0, 0.3, 15))
    plt.semilogy(np.linspace(0, 0.3, 15), dets + 1e-300, marker='o', label=f'rule {rule}')
plt.xlabel('noise probability')
plt.ylabel('|det(M)|  (log scale)')
plt.title(f'Reversibility collapse under noise, L={L}')
plt.legend()
plt.tight_layout()
plt.show()


