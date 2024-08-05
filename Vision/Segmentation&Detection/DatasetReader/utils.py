def contains_non_int(arr):
    return any(not isinstance(item, int) for item in arr)

def too_small_bbox(arr, treshold):
    if arr[2] < treshold:
        return True
    if  arr[3] < treshold:
        return True
    return False

# True if doesnt work
def check_bbox(arr):
    return contains_non_int(arr) or too_small_bbox(arr, 10)