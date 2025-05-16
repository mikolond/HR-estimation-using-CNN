

path = "set.txt"
start = 96
end = 144

with open(path, 'w') as file:
    for i in range(start, end):
        file.write(f"{i}, ")    
        