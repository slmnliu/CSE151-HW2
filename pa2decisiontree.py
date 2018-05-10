
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
    def compare(self, example):
        val = example[self.column]
        return val >= self.value
    
    def __repr__(self):
        return "Is %s %s %s" % (str(self.column), ">=", str(self.value))

def partition(rows, split):
    true_rows, false_rows = [], []
    for row in rows:
        if split.compare(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# Counts the number of labels for each label, and the total number of labels
def num_labels(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts, len(rows)

def entropy(true_rows, false_rows):
    
    # Calculate conditional entropy on true
    true_probs = {}
    true_counts, true_total = num_labels(true_rows)

    # Convert the counts into probabilities
    for label in true_counts:
        true_probs[label] = float(true_counts[label]) / true_total
    
    true_entropy = 0.0
    for label in true_probs:
        p = true_probs[label]
        if p != 0.0:
            true_entropy -= p * np.log(p)

    # Calculate conditional entropy on false
    false_probs = {}
    false_counts, false_total = num_labels(false_rows)

    # Convert the counts into probabilities
    for label in false_counts:
        false_probs[label] = float(false_counts[label]) / false_total
    
    false_entropy = 0.0
    for label in false_probs:
        p = false_probs[label]
        if p != 0.0:
            false_entropy -= p * np.log(p)

    total = true_total + false_total

    split_entropy = (true_total / total) * true_entropy + (false_total / total) * false_entropy

    return split_entropy

# Find the best split
def find_best_split(rows):
    lowest_entropy = 1
    best_split =  None
    num_cols = len(rows[0]) - 1 # Subtract one to account for label

    for col in range(num_cols):
        values = unique(rows, col)
        
        for val in values:
            split = Feature_Split(col, val)
            
            true_rows, false_rows = partition(rows, split)

            # If the split doesn't divide any of the dataset, skip it
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate entropy of the split
            ent = entropy(true_rows, false_rows)

            if ent < lowest_entropy:
                lowest_entropy = ent
                best_split = split
    return lowest_entropy, best_split
