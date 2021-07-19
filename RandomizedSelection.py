import random

def randomizedPartition(A, i, j):
    k = random.randint(i, j)
    t = A[k]
    A[k] = A[j]
    p, q = i, j
    while p != q:
        while p != q and A[p] <= t:
            p += 1
        A[q] = A[p]
        while p != q and A[q] > t:
            q -= 1
        A[p] = A[q]
    A[q] = t
    return q

def randomizedSelect(A, i, j, k):   # return the kth smallest elem. of A[i...j]
    if i == j:
        return A[i]
    q = randomizedPartition(A, i, j)
    n = q-i+1
    if k == n:
        return A[q]
    elif k < n:
        return randomizedSelect(A, i, q-1, k)
    else:
        return randomizedSelect(A, q+1, j, k-n)

def median(A):
    return randomizedSelect(A, 0, len(A)-1, int(len(A)/2)) if 1 == len(A) % 2 else 0.5 * (randomizedSelect(A, 0, len(A)-1, int(len(A)/2)) + randomizedSelect(A, 0, len(A)-1, int(len(A)/2)+1))

def weightedMedian(A, W):
    A_prime = []
    for a, w in zip(A, W):
        A_prime += [a]*w
    return median(A_prime)
    


