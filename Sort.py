

def createHeap(arr, small=True):
    for i in range(int(len(arr)/2-1)):
        adjustHeap(arr, i, small)

def adjustHeap(arr, i, small=True):
    tmp = arr[i]
    k = 2*i + 1
    while k < len(arr):
        if k+1 < len(arr) and arr[k+1]<arr[k]:
            k += 1
        if tmp <= arr[k]:
            break
        else:
            arr[i] = arr[k]
            i = k
            k = 2*k + 1
    arr[i] = tmp

def topN(arr, n):
    top_arr = arr[0:n]
    createHeap(top_arr)
    for i in range(n, len(arr)):
        if arr[i] > top_arr[0]:
            top_arr[0] = arr[i]
            adjustHeap(top_arr, 0)
    return top_arr

def topNidx(data, n):
    top_idx = createIdxHeap(data[0:n])
    for i in range(n, len(data)):
        if data[i] > data[top_idx[0]]:
            top_idx[0] = i
            adjustIdxHeap(top_idx, 0, data)
    return top_idx

def createIdxHeap(data, small=True):
    idx = [i for i in range(len(data))]
    for i in range(int(len(data)/2-1)):
        adjustIdxHeap(idx, i, data, small)
    return idx

def adjustIdxHeap(idx, i, data, small=True):
    tmp = idx[i]
    k = 2*i + 1
    while k < len(idx):
        if k+1 < len(idx) and data[idx[k+1]] < data[idx[k]]:
            k += 1
        if data[tmp] <= data[idx[k]]:
            break
        else:
            idx[i] = idx[k]
            i = k
            k = 2*k + 1
    idx[i] = tmp
