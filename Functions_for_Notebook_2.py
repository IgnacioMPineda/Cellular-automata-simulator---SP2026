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



def plot_column(arr, col, cols=100):
    column = arr[:, col] if arr.ndim > 1 else arr
    n = len(column)

    def render_segment(seg, start_t):
        print(f'{"":7}┌─┐')
        for i, x in enumerate(seg):
            t = start_t + i
            dead = '█' if (t + col) % 2 == 0 else "□"  
            print(f't={t:<4} │{"□" if x else dead}│')
        print(f'{"":7}└─┘')

    segments = [column[i:i+cols] for i in range(0, n, cols)]
    for idx, seg in enumerate(segments):
        render_segment(seg, idx * cols)
        if idx < len(segments) - 1:
            print()


def plot_multiple_cells(*cells):
    fig, ax = plt.subplots(figsize=(10, 4))

    for rule, initial_conditions, timesteps, L, cell_i in cells:
        system = dynamics(rule, L, initial_conditions, timesteps)

        ax.plot(system[:, cell_i], label=f'rule {rule} — cell {cell_i}')

    ax.set_xlabel('timestep')
    ax.set_ylabel('state (0 or 1)')
    ax.set_yticks([0, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()

def fourier_cell(rule_number, L, initial_conditions, timesteps, cell_index):
    # run the CA
    grid = dynamics(rule_number, L, initial_conditions, timesteps)
    
    # extract the time series for one cell
    # grid is upside down so we reverse it to get time going forward
    time_series = grid[::-1, cell_index]   # shape (timesteps+1,)
    
    # fourier transform - Directly integrated in numpy
    fft    = np.fft.fft(time_series)
    power  = np.abs(fft)**2                # power spectrum
    freqs  = np.fft.fftfreq(len(time_series))  # frequencies
    
    # only plot positive frequencies, negative frequencies are mirrors of the positive and 
    # contain no additional information
    pos    = freqs > 0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # left — original time series
    axes[0].plot(time_series)
    axes[0].set_title(f"time evolution of cell {cell_index} — rule {rule_number}")
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("state (0 or 1)")
    
    # right — power spectrum
    axes[1].plot(freqs[pos], power[pos])
    axes[1].set_title(f"Fourier power spectrum — cell {cell_index} — rule {rule_number}")
    axes[1].set_xlabel("frequency (cycles per timestep)")
    axes[1].set_ylabel("power")
    
    # mark the dominant frequency
    dominant_freq = freqs[pos][np.argmax(power[pos])] #This is just the tallest spike
    axes[1].axvline(dominant_freq, color='red', linestyle='--', 
                    label=f"dominant freq = {dominant_freq:.4f}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Dominant frequency: {dominant_freq:.4f} cycles/timestep")
    print(f"Period: {1/dominant_freq:.1f} timesteps")

# run it
fourier_cell(67, 10, [1,7], 1000, cell_index=8)
fourier_cell(90, 10, [1,7], 1000, cell_index = 9)
        
def fourier_all_cells(rule_number, L, initial_conditions, timesteps):
    grid       = dynamics(rule_number, L, initial_conditions, timesteps)
    time_series = grid[::-1]   
    
    fig, axes = plt.subplots(L, 2, figsize=(14, 2*L))
    
    for cell in range(L):
        signal = time_series[:, cell]   # time series for this cell
        
        fft   = np.fft.fft(signal)
        power = np.abs(fft)**2
        freqs = np.fft.fftfreq(len(signal))
        pos   = freqs > 0
        
        dominant_freq = freqs[pos][np.argmax(power[pos])]
        period        = 1 / dominant_freq

        axes[cell, 0].plot(signal)
        axes[cell, 0].set_ylabel(f"cell {cell}")
        axes[cell, 0].set_ylim(-0.1, 1.1)
        if cell == 0:
            axes[cell, 0].set_title("time evolution")
        
        axes[cell, 1].plot(freqs[pos], power[pos])
        axes[cell, 1].axvline(dominant_freq, color='red', linestyle='--',
                              label=f"f={dominant_freq:.3f} T={period:.1f}")
        axes[cell, 1].legend(fontsize=7)
        if cell == 0:
            axes[cell, 1].set_title("power spectrum")

    axes[-1, 0].set_xlabel("timestep")
    axes[-1, 1].set_xlabel("frequency")
    fig.suptitle(f"Fourier analysis — Rule {rule_number}, all cells", fontsize=13)
    plt.tight_layout()
    plt.show()

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

def H_cell_EC(rule, cell_i, array=None): ### EC for effective complexity
    if array is not None:
        cell = array
    else:
        initial_conditions = np.random.randint(0, 2, size=1000)
        row = list(np.where(initial_conditions == 1)[0])
        system = dynamics(rule, 1000, row, 1000)
        cell = system[:, cell_i]
    
    p1 = np.mean(cell)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def H_two_cells_EC(rule, cell_i, cell_j, array_x=None, array_y=None):
    if array_x is not None and array_y is not None:
        x, y = array_x, array_y
    else:
        initial_conditions = np.random.randint(0, 2, size=1000)
        row = list(np.where(initial_conditions == 1)[0])
        system = dynamics(rule, 1000, row, 1000)
        x = system[:, cell_i]
        y = system[:, cell_j]
    
    p11 = np.mean((x == 1) & (y == 1))
    p00 = np.mean((x == 0) & (y == 0))
    p10 = np.mean((x == 1) & (y == 0))
    p01 = np.mean((x == 0) & (y == 1))
    
    h = 0
    for p in [p00, p01, p10, p11]:
        if p > 0:
            h -= p * np.log2(p)
    return h


def effective_complexity(rule):
    L = 500
    timesteps = 1000

    initial_conditions = np.random.randint(0, 2, size=L)
    row = list(np.where(initial_conditions == 1)[0])
    system = dynamics(rule, L, row, timesteps)

    initial = system[0, :]   # state of all cells at t=0
    final   = system[-1, :]  # state of all cells at t=999

    H_initial = H_cell_EC(rule, 0, array=initial)
    H_final   = H_cell_EC(rule, 0, array=final)
    H_joint   = H_two_cells_EC(rule, 0, 0, array_x=initial, array_y=final)

    return H_initial + H_final - H_joint




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




def apply_rule(rule, state):
    rule_bin = format(rule, '08b')[::-1]
    L = len(state)
    new_state = np.zeros(L, dtype=int)
    for i in range(L):
        left   = state[i - 1] if i > 0 else 0
        center = state[i]
        right  = state[i + 1] if i < L - 1 else 0
        neighbourhood = left * 4 + center * 2 + right * 1
        new_state[i] = int(rule_bin[neighbourhood])
    return new_state

def is_reversible(rule, L=10):
    seen = {}
    for i in range(2**L):
        initial = np.array([int(b) for b in format(i, f'0{L}b')])
        final = apply_rule(rule, initial)
        initial_key = tuple(initial)
        final_key   = tuple(final)
        if final_key in seen:
            if seen[final_key] != initial_key:
                return False
        else:
            seen[final_key] = initial_key
    return True


def gardens_of_eden(rule, L=10):
    all_states = set()
    reachable  = set()

    for i in range(2**L):
        state = tuple(int(b) for b in format(i, f'0{L}b'))
        all_states.add(state)
        next_state = tuple(apply_rule(rule, np.array(state)))
        reachable.add(next_state)

    gardens = all_states - reachable

    print(f"rule {rule}: {len(gardens)} gardens of Eden")
    for g in sorted(gardens):
        print(g)

    return gardens, len(gardens)


