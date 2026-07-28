import napypi as napy
import numpy as np

data = np.random.rand(5, 10)
result = napy.pearsonr(data)

print(result)