import numpy as np


x1 = 10
x2 = 100
y = 5
f = 35

f_min, f_max = 10,100
f_true = 35
delta = 5
f_wanted =list(range(f_true - delta, f_true + delta + 1))
f_unwanted = list(range(f_min, f_true - delta)) + list(range(f_true + delta + 1, f_max+1))

print("wanted",f_wanted)
print("unwanted",f_unwanted)