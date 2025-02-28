import csv
import numpy as np

class csv_reader:
    def __init__(self):
        self.current_row = 0
        pass

    def load_file(self, file_path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            self.data = list(reader)

    def read_rows(self, n):
        rows = self.data[self.current_row:self.current_row + n]
        self.current_row += n
        return rows
    
    def read_row(self):
        row = self.data[self.current_row]
        self.current_row += 1
        return row

reader = csv_reader()
reader.load_file("evaluation_results.csv")
reader.read_row()
row = reader.read_rows(3)
row = np.array(row)[:,1].astype(np.float32)
print(row)
row = reader.read_row()
print(row)