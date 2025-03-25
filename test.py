import csv

file = open("test.csv","a")

for i in range(30):
    file.write(str(i) + "\n")

file.close()
