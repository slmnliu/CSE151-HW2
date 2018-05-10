
import numpy as np

# Load the data from the file and return a list of lists, with each line as a list
def load_data(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    dataList = []
    
    for line in lines:
        lineList = line.split()
        fLineList = [float(i) for i in lineList]
        dataList.append(fLineList)
    return dataList

# Returns a set of unique values for a column in the data
def unique(rows, col):
    return set([row[col] for row in rows])

# Class that holds the column number and value of a feature_split
class Feature_Split:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    # Compares this feature split to an example
    def greater_than(self, example):
        val = example[self.column]
        return val >= self.value
    
    def __repr__(self):
        return "Is %s %s %s" % (str(self.column), ">=", str(self.value))

def partition(rows, split):
    true_rows, false_rows = [], []
    for row in rows:
        if split.greater_than(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def entropy():
    pass

# Find the best split
def find_best_split(rows):
    lowest_entropy = 1
    best_split =  None
    num_cols = len(rows[0]) - 1 # Subtract one to account for label

    for col in range(num_cols):
        values = unique(rows, col)
        