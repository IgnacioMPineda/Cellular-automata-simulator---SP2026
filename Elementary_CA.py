import numpy as np
import random



def make_rule(rule_number: int) -> dict:
    if not (0 <= rule_number <= 255):
        raise ValueError(f"Rule number must be between 0 and 255, got {rule_number}")
 
    rule_bin = format(rule_number, '08b')  # e.g. rule 110 -> '01101110 (encode it into byts)' The decimal -> binary process explained above
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]  # '111' down to '000'
 
    return {n: int(b) for n, b in zip(neighbourhoods, rule_bin)}


def dynamics(rule_number, L, initial_conditions, timesteps):
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

    grid = np.vstack(rows[::-1])
    return grid


def dynamics_periodic(rule_number, L, initial_conditions, timesteps):
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
    rows = [row.copy()]

    # Evolve through timesteps
    for i in range(timesteps):
        new_row = np.zeros(L, dtype=int)
        for j in range(L):
            left   = str(int(rows[-1][(j - 1) % L]))  # periodic boundary
            center = str(int(rows[-1][j]))
            right  = str(int(rows[-1][(j + 1) % L]))
            neighbourhood = left + center + right
            new_row[j] = rule[neighbourhood]
        rows.append(new_row)

    grid = np.vstack(rows[::-1])
    return grid

def dynamics_reflection(rule_number, L, initial_conditions, timesteps):
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
    rows = [row.copy()]

    # Evolve through timesteps
    for i in range(timesteps):
        new_row = np.zeros(L, dtype=int)
        for j in range(L):
            left   = str(int(rows[-1][j + 1])) if j == 0 else str(int(rows[-1][j - 1]))  #mirror reflection
            center = str(int(rows[-1][j]))
            right  = str(int(rows[-1][j - 1])) if j == L - 1 else str(int(rows[-1][j + 1]))
            neighbourhood = left + center + right
            new_row[j] = rule[neighbourhood]
        rows.append(new_row)

    grid = np.vstack(rows[::-1])
    return grid

def plot_grid(grid):
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def make_rule_zero_condition(rule_number: int):
    if not (0 <= rule_number <= 255):
        raise ValueError(f"Rule number must be between 0 and 255, got {rule_number}")
 
    rule_bin = format(rule_number, '08b')  # e.g. rule 110 -> '01101110 (encode it into byts)' The decimal -> binary process explained above
    neighbourhoods = [format(i, '03b') for i in range(7, -1, -1)]  # '111' down to '000'
    
    results = {n: int(b) for n, b in zip(neighbourhoods, rule_bin)}
    results["000"] = 0
    return results

def make_rule_symmetry(rule_number: int) -> dict:
    rule = make_rule_zero_condition(rule_number)

    i = -1
    for key in rule:
        inverted_key = key[::-1]
        i += 1
        if i == 4:
            break
        
        else:
            rule[inverted_key] = rule[key]

        
    return(rule)
