import numpy as np
from numpy import linalg as lina

def diagonalise(M):
    """
    Diagonalises a given hermetian matrix, yielding eigenvalues and eigenvectors in ascending order

    :param M: Square Matrix to be diagonalised
    :return: Tuple of eigenvalues  (ascending order), and a 2d  array of all eigenvectors. The eigenvectors are the columns of this 2d array.
    """
    ew,ev=np.linalg.eigh(M) # EigenWerte und Vektoren
    sort_ind=ew.argsort() # Indexliste zum Sortieren (Aufsteigende EW)
    ew=ew[sort_ind]       # Sortiere Eigenwerte und Vektoren anhand dieser Liste
    ev=ev[:,sort_ind]
    return ew,ev

def spectral_width(M):
    """
    Calculates the spectral width of a given matrix, i.e. the difference between its largest and smallest eigenvalue
    :param M: Square matrix
    :return: Spectral width of given matrix
    """
    ew=diagonalise(M)[0]
    width=ew[-1]-ew[0]
    return width

def copy(M,N):
    """
    Copies the first N rows and N columns of given matrix M into a new submatrix.
    :param M: Square matrix M
    :param N: Amount of columns and rows to be copied.
    :return: NxN submatrix of the N first rows and columns of M
    """
    Res=np.zeros([N,N],dtype=complex)
    for i in range(N):
        for j in range(N):
            Res[i][j]=M[i][j]
    return Res


def lanczos(M, v0, Kmax, err=10 ** (-8), convergenceCheck="order1"):
    """
    Applies the lancos method to a hermetian matrix M and yields the first K approximate eigenvalues and eigenvectors of that matrix (in the Krylov space basis).
    Convergence is reached for this method once the relative absolute value between a previous  and a new Krylov Basis vector is smaller than the desired error.

    :param M: Square hermetian matrix M
    :param v0: Initial vector v0
    :param Kmax: Maximum number of iterations for the lancos algorithm
    :param err (optional): Desired inaccuracy to be achieved. Default: 10**(-8)
    :param convergenceCheck (optional): How to check for convergence.
                                        Possible parameters: "order1" - Assumes eigenvector absolute values are of order 1
                                                              For everything else: Checks spectral width of  given submatrix, calculating the approximate absolute value of the eigenvectors.

    :return: list of eigenvalues and eigenvectors, along the amount of iterations and  whether convergence has been reached.
             Careful: Eigenvectors are in the Krylov Space basis. To convert to standard basis, one must run Lanczos again.
    """
    Res = np.zeros([Kmax, Kmax], dtype=complex)  # Initialise result matrix in the Krylov space

    # Calculate the first lancos vector. All used vectors are immediately normed, as we are dealing with quantum states
    v_cur = np.array(v0, dtype=complex) / lina.norm(v0)
    m = M.dot(v_cur)
    a = np.vdot(v_cur, m)
    v_next = m - a * v_cur
    Res[0][0] = a
    K = 1
    for i in range(1, Kmax):
        # Calculate the next vector via given algorithm
        b = lina.norm(v_next)
        # First check the convergence
        if (convergenceCheck == "order1"):
            width = 1
        else:
            subM = copy(Res, i)
            width = spectral_width(subM)
        if (abs(b) / width < err):
            convergence = True
            Res = copy(Res, i)
            break
        if (i == Kmax - 1):
            convergence = False

        # Now update old Krylov vectors
        v_prev = v_cur
        v_cur = v_next / b
        # Calculate the new Krylov vector
        m = M.dot(v_cur)
        a = np.vdot(v_cur, m)
        Res[i][i] = a
        Res[i - 1][i] = b
        Res[i][i - 1] = b
        K = K + 1
        v_next = m - a * v_cur - b * v_prev
    diag = diagonalise(Res) # Diagonalise and output
    return np.real(diag[0]), np.real(np.transpose(diag[1])), K, convergence


def lanczos_ev(M, v0, Kmax, maxIters=1000, err=10 ** (-8), iters=0, convergenceCheck="order1"):
    """
    Applies the Lanczos method to a hermetian matrix M iteratively until convergence
    and yields the lowest eigenvalue and eigenvector in the standard basis, i.e. the ground state

    :param M: Square hermetian matrix M
    :param v0: Initial vector v0
    :param Kmax: Maximum number of iterations for each Lanczos run Lanczos algorithm
    :param MaxIters (optional): Maximum amount of Lanczos runs. Default: 1000
    :param err (optional): Desired inaccuracy to be achieved. Default: 10**(-8)
    :param convergenceCheck (optional): How to check for convergence.
                                        Possible parameters: "order1" - Assumes eigenvector absolute values are of order 1
                                                              For everything else: Checks spectral width of  given submatrix, calculating the approximate absolute value of the eigenvectors.
    :param iters (optional, should not be used): Amount of iterations already made.

    :return: Approximation for lowest eigenvalue and eigenvector of given hermetian matrix M in the normal, standard basis.
    """
    ew, ev, K, conv = lanczos(M, v0, Kmax, err = err , convergenceCheck=convergenceCheck)

    ew1 = ew[0]
    ev1 = ev[0]

    v_cur = np.array(v0, dtype=complex) / lina.norm(v0)
    true_ev = ev1[0] * v_cur

    m = M.dot(v_cur)
    a = np.vdot(v_cur, m)
    v_next = m - a * v_cur

    for i in range(1, K):
        # No longer checks for convergence, as this run is simply done to calculate the first eigenvector
        b = lina.norm(v_next)
        v_prev = v_cur
        v_cur = v_next / b
        true_ev = true_ev + ev1[i] * v_cur

        m = M.dot(v_cur)
        a = np.vdot(v_cur, m)
        v_next = m - a * v_cur - b * v_prev

    iters = iters + 1
    if conv == True:
        # print("Converged after {} iterations".format(iters))
        return ew1, true_ev
    elif (iters >= maxIters):
        print("Could not converge after {} Iterations".format(maxIters))
        return ew1, true_ev
    else:
        return lanczos_ev(M, true_ev, Kmax, maxIters, err, iters, convergenceCheck=convergenceCheck)