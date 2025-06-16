import numpy as np

arr = np.array([1, 3, 5, 5, 5, 7, 9])
values = [5, 6, 2]

result = np.searchsorted(arr, values, side='right')
print(result)


for iou_th in np.arange(0.50, 1.00, 0.05):
    print(f"{iou_th:.2f}")