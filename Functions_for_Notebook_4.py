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
