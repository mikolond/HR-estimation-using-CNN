import csv


def create_csv():
    alphabet_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    numbers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random_numbers_list = [7,5,3,1,2,4,6,8,10,9]
    with open("test_file.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(10):
            data = [alphabet_list[i], numbers_list[i], random_numbers_list[i]]
            writer.writerow(data)

def read_csv():
    with open("test_file.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)

create_csv()
read_csv()
