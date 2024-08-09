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

def rescale_bboxes(bboxes, ratios):
    if isinstance(ratios, int):
        ratios = (ratios,ratios,ratios,ratios)
    # Ensure ratios are a tuple of four elements
    if len(ratios) != 4:
        raise ValueError("Ratios must be a tuple of four elements (x_ratio, y_ratio, w_ratio, h_ratio)")
    
    rescaled_bboxes = []
    for bbox in bboxes:
        # Ensure each bounding box has four elements
        if len(bbox) != 4:
            raise ValueError("Each bounding box must have four elements (x, y, w, h)")
        
        # Rescale each element of the bounding box by the respective ratio
        rescaled_bbox = [int(bbox[i] / ratios[i]) for i in range(4)]
        rescaled_bboxes.append(rescaled_bbox)
    
    return rescaled_bboxes

def compute_ratio(tuple1, tuple2):
    # Ensure both tuples have the same length
    if len(tuple1) != len(tuple2):
        raise ValueError("Tuples must be of the same length")

    # Compute the ratio for each element
    ratio = []
    for t1, t2 in zip(tuple1, tuple2):
        if t1 == 0:
            if t2 == 0:
                ratio.append(float('inf'))  # Handle the case where both elements are 0
            else:
                ratio.append(float('inf'))  # Handle division by zero
        else:
            ratio.append(t2 / t1)

    return tuple(ratio)