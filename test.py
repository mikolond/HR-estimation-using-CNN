import numpy as np




a1 = np.arange(10)
a2 = np.arange(5,15)
print("a1 before roll:",a1)
a1 = np.roll(a1, -5)
print("a1 after roll:",a1)