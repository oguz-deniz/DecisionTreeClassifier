from typing import List
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, leftChild=None, rightChild=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.gain = gain
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    
    
    

class DecisionTreeClassifier:
    
    def __init__(self, max_depth: int):
        
        self.max_depth = max_depth
        self.root = None

    def build_tree(self,whole_data,current_depth=0):

        
        X = [row[:-1] for row in whole_data]
        y = [row[-1] for row in whole_data]
        
        feature_number = len(X[0])

        X = list(X)
        y = list(y)


        if current_depth <= self.max_depth:

            best_split = self.get_best_split(whole_data, feature_number)

            if "info_gain" in best_split and best_split["info_gain"] > 0:

                left_subtree = self.build_tree(best_split["dataset_left"], current_depth + 1)

                right_subtree = self.build_tree(best_split["dataset_right"], current_depth + 1)

                return Node(
                    feature=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    leftChild=left_subtree,
                    rightChild=right_subtree,
                    gain=best_split["info_gain"]
                )
            

        leaf_value = max(y, key=y.count)
       
        return Node(value=leaf_value)
    
    def get_best_split(self, whole_data, feature_number ):

        best_split = {}

        best_info_gain = -1

        for feature in range(feature_number):

            feature_values = [row[feature] for row in whole_data]

            candidate_thresholds = list(set(feature_values))
        

            for threshold in candidate_thresholds:

                data_left, data_right = self.split(whole_data, feature, threshold)

                if (len(data_left)>0 and len(data_right)>0):

                    y = [row[-1] for row in whole_data]
                    left_y = [row[-1] for row in data_left]
                    right_y = [row[-1] for row in data_right]

                    current_info_gain = self.info_gain(y, left_y, right_y)

                    if current_info_gain > best_info_gain:

                        best_split["feature_index"] = feature
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = data_left
                        best_split["dataset_right"] = data_right
                        best_split["info_gain"] = current_info_gain
                        best_info_gain = current_info_gain

        return best_split
    
    def split(self, whole_data, feature_number, threshold):
    
    
        for_left = [row for row in whole_data if row[feature_number] <= threshold]
        for_right = [row for row in whole_data if row[feature_number] > threshold]
    
        return for_left, for_right
    
    def info_gain(self, parent, left_child, right_child):
        
        ratio_l = len(left_child) / len(parent)
        ratio_r = len(right_child) / len(parent)
       
        gain = self.calculate_gini(parent) - (ratio_l*self.calculate_gini(left_child) + ratio_r*self.calculate_gini(right_child))
        return gain
    
    def calculate_gini(self,y):

        gini = 0
        label_number = Counter(y)
        total_labels = len(y)


        for label in label_number:
            prob_of_class = label_number[label] / total_labels
            gini += prob_of_class ** 2

        return 1-gini
    
    @staticmethod
    def concatenate_lists(X, y):
        combined = []

        for i in range(len(X)):
            combined.append(X[i] + [y[i]])

        return combined


    def fit(self, X: List[List[float]], y: List[int]):

        whole_data = self.concatenate_lists(X,y)

        self.num_classes = len(set(y))
       
        self.root = self.build_tree(whole_data,0)


    
 
    def predict(self, X: List[List[float]]):
       

        predictions = [self.classify(test_data, self.root) for test_data in X]
        return predictions
    
    def classify(self,test_data,tree):

        if tree.is_leaf():
            return tree.value
        
        feature_val = test_data[tree.feature]
        
        if feature_val <= tree.threshold:
            return self.classify(test_data, tree.leftChild)
        else:
            return self.classify(test_data, tree.rightChild)
        
        
    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            probabilities.append(self.classify_proba(sample, self.root))
        return probabilities

    def classify_proba(self, test_data, tree):
        if tree.is_leaf():
            class_probabilities = [0] * self.num_classes
            class_probabilities[tree.value] = 1  
            return class_probabilities

        feature_val = test_data[tree.feature]

        if feature_val <= tree.threshold:
            return self.classify_proba(test_data, tree.leftChild)
        else:
            return self.classify_proba(test_data, tree.rightChild)


        
        
        
        
    