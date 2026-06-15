
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
