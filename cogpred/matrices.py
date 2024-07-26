import math

def compute_mat_size(l, with_diag=False):
    # Mat size is the positive root of :
    # n**2 - b*n - 2l = 0 
    # Where l is the length of pvalues array
    # and n is the square matrix size.
    if with_diag:
        b = 1
    else:
        b = -1
    n = (-b + math.sqrt(1 + 8 * l)) / 2
    if n != int(n):
        raise ValueError(f"Array of lenght {l} cannot be reshaped as a square matrix\
                         (with_diag is {with_diag})")
    return int(n)