from Functions_for_Notebook_1 import make_rule



def csr(M): # M is a matrix
    eigenvalues = np.linalg.eigvals(M)
    evals = np.asarray(eigenvalues)

    # Build the 2d complex plane with points
    points = np.column_stack((evals.real, evals.imag))

    tree = cKDTree(points)

    ratios = []

    for i, p in enumerate(points):

        # query 3 nearest neighbors (itself + 2 nearest)
        dist, idx = tree.query(p, k=3)

        # idx[0] is the point itself (distance 0)
        i1, i2, i3 = idx

        lam1 = evals[i1]
        lam2 = evals[i2]
        lam3 = evals[i3]

        # avoid division by zero (rare degeneracy case mentioned in literature)
        denom = (lam3 - lam1)
        if np.abs(denom) < 1e-12:
            continue

        z = (lam2 - lam1) / denom
        ratios.append(z)

    return np.array(ratios)


def phase_distribution(M):
    z = csr(M)
    
    
    r = np.abs(z)
    theta = np.angle(z)
    print(f"Distance: {r}", "\n", f"Angle: {theta}")

    plt.hist(r, bins=50, density=True)
    plt.title("|z| distribution")
    plt.show()  

    plt.hist(theta, bins=50, density=True)
    plt.title("arg(z) distribution")
    plt.show()


def plot_poisson_density(n_pts, box=None):
    if box is None:
        box = 5 * np.sqrt(n_pts)  # keeps mean spacing aprox. 0.2 regardless of n_pts

    pts = np.random.uniform(0, box, size=(n_pts, 2))
    levels = pts[:, 0] + 1j * pts[:, 1]

    points = np.column_stack((levels.real, levels.imag))
    tree = cKDTree(points, boxsize=box)
    _, idx = tree.query(points, k=3)

    lam1 = levels
    lam2 = levels[idx[:, 1]]
    lam3 = levels[idx[:, 2]]

    d_nn, d_nnn = lam2 - lam1, lam3 - lam1
    # minimum-image convention: fix displacements for periodic-wrapped pairs
    d_nn = (d_nn.real - box * np.round(d_nn.real / box)) + \
           1j * (d_nn.imag - box * np.round(d_nn.imag / box))
    d_nnn = (d_nnn.real - box * np.round(d_nnn.real / box)) + \
            1j * (d_nnn.imag - box * np.round(d_nnn.imag / box))

    mask = np.abs(d_nnn) > 1e-12
    z = d_nn[mask] / d_nnn[mask]

    plt.figure(figsize=(5, 5)) # Good settings for a nice output but they could be played with
    plt.hexbin(z.real, z.imag, gridsize=80, extent=(-1, 1, -1, 1), cmap="inferno")
    plt.gca().set_aspect("equal")
    plt.xlabel("Re z"); plt.ylabel("Im z")
    plt.title(f"(a) Poisson: ED density (n={n_pts})")
    plt.colorbar(label="count")
    plt.show()


def plot_ginue_density(N):
    G = (np.random.randn(N, N) + 1j*np.random.randn(N, N)) / np.sqrt(2*N)
    evals = np.linalg.eigvals(G)
    points = np.column_stack((evals.real, evals.imag))
    tree = cKDTree(points)

    ratios, origins = [], []
    for lam1, p in zip(evals, points):
        dist, idx = tree.query(p, k=3)
        lam2, lam3 = evals[idx[1]], evals[idx[2]]
        denom = lam3 - lam1
        if np.abs(denom) < 1e-12:
            continue
        ratios.append((lam2 - lam1) / denom)
        origins.append(lam1)

    z = np.array(ratios)
    origins = np.array(origins)
    z = z[np.abs(origins) < 0.8]  # keep bulk eigenvalues, drop edge of the disk

    plt.figure(figsize=(5, 5))
    plt.hexbin(z.real, z.imag, gridsize=80, extent=(-1, 1, -1, 1), cmap="inferno")
    plt.gca().set_aspect("equal")
    plt.xlabel("Re z"); plt.ylabel("Im z")
    plt.title(f"(e) GinUE ED (N={N}) density")
    plt.show()


def plot_poisson_theory():
    x = np.linspace(-1, 1, 300)
    y = np.linspace(-1, 1, 300)
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)
    density = np.where(R <= 1, 1/np.pi, np.nan)

    plt.figure(figsize=(5, 5))
    plt.imshow(density, origin="lower", extent=(-1, 1, -1, 1), cmap="inferno")
    plt.gca().set_aspect("equal")
    plt.xlabel("Re z"); plt.ylabel("Im z")
    plt.title("(b) Poisson: exact flat, Eq. (4)")
    plt.colorbar(label=r"$\varrho(z)$")
    plt.show()

from scipy.interpolate import griddata

def plot_tue3_surmise(n_quad=200, n_grid=150):
    s = np.linspace(-np.pi, np.pi, n_quad)
    S, T = np.meshgrid(s, s, indexing="ij")
    weight = (S**2 + T**2)**2 * (2 - np.cos(S) - np.cos(T))

    def tue3_xy(x, y):
        f2 = 2 - np.cos(S*x - T*y) - np.cos(T*x + S*y)
        f3 = 2 - np.cos(S*(x-1) - T*y) - np.cos(T*(x-1) + S*y)
        integrand = weight * f2 * f3
        return np.trapz(np.trapz(integrand, s, axis=1), s)

    # evaluate on a polar grid, then interpolate to Cartesian for plotting
    rr = np.linspace(1e-3, 0.999, n_grid)
    tt = np.linspace(-np.pi, np.pi, n_grid)
    vals = np.empty((n_grid, n_grid))
    for i, r in enumerate(rr):
        for j, th in enumerate(tt):
            vals[i, j] = tue3_xy(r*np.cos(th), r*np.sin(th))

    norm = np.sum(vals * rr[:, None]) * (rr[1]-rr[0]) * (tt[1]-tt[0])
    dens = vals / norm

    XX = rr[:, None] * np.cos(tt)[None, :]
    YY = rr[:, None] * np.sin(tt)[None, :]

    gx, gy = np.meshgrid(np.linspace(-1, 1, 220), np.linspace(-1, 1, 220))
    grid_vals = griddata((XX.ravel(), YY.ravel()), dens.ravel(), (gx, gy), method="linear")
    grid_vals[gx**2 + gy**2 > 1] = np.nan

    plt.figure(figsize=(5, 5))
    plt.imshow(grid_vals, origin="lower", extent=(-1, 1, -1, 1), cmap="inferno")
    plt.gca().set_aspect("equal")
    plt.xlabel("Re z"); plt.ylabel("Im z")
    plt.title("(f) TUE N=3 surmise, Eq. (7)")
    plt.show()
    return dens
