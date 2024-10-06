# example_script.py

f = open("test_videos/frequency.txt", "r")

frequencies = []
for line in f:
    frequencies.append(float(line))

print(frequencies)