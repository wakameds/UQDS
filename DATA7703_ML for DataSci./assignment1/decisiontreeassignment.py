#  Copyright (c) 2015-2021, Hideki WAKAYAMA, All rights reserved.
#  File: decisiontreeassignment.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 18/08/2021, 08:22

import sys
import random
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

def read_data(filename):
    """
    function to read data as dataframe
    :param filename(str): a file name
    :return: dataframe of the file
    """
    df = pd.read_table(filename)
    return df


def split_train_test(X, y, tr=0.7):
    """
    function for split dataset for train set and test set
    :param X(df): features dataframe
    :param y(series): label dataframe
    :param tr(float): training ratio
    :return: train set and test set of features X and label y
    """
    tr_size = int(X.shape[0]*tr)
    samples = list(range(X.shape[0]))
    random.shuffle(samples) #shuffle samples

    train_ids = samples[:tr_size]
    test_ids = samples[tr_size:]

    X_tr = X.iloc[train_ids]
    y_tr = y.iloc[train_ids]
    X_test = X.iloc[test_ids]
    y_test = y.iloc[test_ids]
    return X_tr, y_tr, X_test, y_test


def label_enchoder(df):
    """
    function to exchange categorical value into numerical value for using sklearn.tree.classifier
    :param df:dataframe
    """
    columns = df.columns
    lb_make = LabelEncoder()
    for str in columns:
        df[str] = lb_make.fit_transform(df[str])


def accuracy(y_pred, y):
    """
    function to compute accuracy
    :param y_pred(series):predicted labels
    :param y(series): true labels
    :return: accuracy
    """
    correct = 0
    for i in range(len(y)):
        if y_pred[i] == y.values[i]:
            correct += 1
    acc = correct/len(y) * 100
    return acc


class ID3:
    """
    Class for decision tree classifier with ID3
    """
    def __init__(self, max_depth):
        """
        :param max_depth: max depth of the tree
        """
        self.maxdepth = max_depth
        self.depth = 0 #depth of the tree from root
        self.extend = 0 #depth of the subset from the root


    def fit(self, X_train, y_train):
        """
        function to build tree model with ID3
        :param X_train: dataframe of the features
        :param y_train: series of the label
        """
        #make data combining X and y
        data = X_train.copy()
        data[y_train.name] = y_train

        #make tree structure
        self.tree = self.decision_tree(data, data, X_train.columns, y_train.name)


    def get_entropy(self, target_column):
        """
        function for compute entropy of the target column
        :param target_column(series):  input data in a column
        :return: entropy
        """
        #return values and counts of each value in a target column
        values, counts = np.unique(target_column, return_counts=True)

        # compute entropy: Sum(-prob * log_2(prob))
        entropy_list = []
        for i in range(len(values)):
            prob = counts[i]/np.sum(counts)
            entropy_list.append(-prob*np.log(prob))
        total_entropy = np.sum(entropy_list)
        return total_entropy


    def get_infomation_gain(self, data, feature_name, target_label_name):
        """
        function to compute information gain of target_label to the feature
        :param data(dataframe): dataframe
        :param feature_name(str): feature name
        :param target_label_name(str): label name
        :return: information gain: E(Y)-E(Y|X)
        """
        #E(Y)
        total_entropy = self.get_entropy(data[target_label_name])
        #return values and counts by values of the given feature
        values, counts = np.unique(data[feature_name], return_counts=True)

        #compute subset entropys
        subset_entropy_list = []
        for i in range(len(values)):
            subset_prob = counts[i]/np.sum(counts)
            subset_entropy = self.get_entropy(data.where(data[feature_name]==values[i]).dropna()[target_label_name])
            subset_entropy_list.append(subset_prob*subset_entropy)

        total_subset_entropy = np.sum(subset_entropy_list)
        #compute information gain with the given feature. Info gain: Entropy - Specific feature entropy
        info_gain = total_entropy - total_subset_entropy
        return info_gain

    def keys_count(self, tree):
        """
        function to count the max key number in a tree dictionary
        :param tree(dict): tree structure
        :return: the max count of the keys in a dictionary
        """
        return max(self.keys_count(v) if isinstance(v, dict) else 0 for v in tree.values()) + 1

    def max_depth(self, tree):
        """
        function to return the max depth of the tree
        :param tree(dict): tree structure
        :return: max depth of the current tree
        """
        depth = self.keys_count(tree)
        return depth/2

    def decision_tree(self, data, original_data, feature_names, target_label_name, parent_node_class=None):
        """
        function to build decision tree
        :param data(dataframe): data in the node
        :param original_data(dataframe): original data
        :param feature_names(list[str]):feature names
        :param target_label_name(str): label name
        :param parent_node_class: class of the node
        :return: class of the node class
        """
        #class of the label in a node
        unique_classes = np.unique(data[target_label_name]) #yes or no

        #1.the case that the node doesn't have over 1 class, the class of the node is decided
        if len(unique_classes) <= 1:
            return unique_classes[0]

        #2.the case that the subset is no instances, the majority class become the class of the node
        elif len(data) == 0:
            majority_class_idxs = np.argmax(np.unique(original_data[target_label_name], return_counts=True)[1])
            return np.unique(original_data[target_label_name])[majority_class_idxs]

        #3.the case that the dataset dosen't have any features, return the parent node class of the node
        elif len(feature_names) == 0:
            return parent_node_class

        #4. the case that the present depth of the tree reach over max depth of the tree, return None
        elif self.depth > self.maxdepth:
            return None

        #5.the case that 1, 2, 3, 4 is not sattisified, the node can generate the new child nodes
        else:
            self.extend += 1

            #decide the current node class
            majority_class_idxs = np.argmax(np.unique(data[target_label_name], return_counts=True)[1])
            parent_node_class = unique_classes[majority_class_idxs] #yes or no

            #get information gain, which is max, to decide split feature
            info_gain_values = [self.get_infomation_gain(data, feature, target_label_name) for feature in feature_names]
            best_feature_idx = np.argmax(info_gain_values)
            best_feature = feature_names[best_feature_idx] #the highest infomation gain feature name

            #build tree structure
            tree = {best_feature:{}}

            #build the child node under the parent node
            parent_feature_values = np.unique(data[best_feature])

            #print("***extend:{}, depth:{}, feature:{}, fvalues:{}***".format(self.extend, self.depth, best_feature, parent_feature_values))

            #1.search and build child nodes if depth is not reach max depth - 1
            if self.depth < self.maxdepth-1:
                if self.extend < self.maxdepth:
                    # remove the best feature from the feature name list
                    new_feature_names = [name for name in feature_names if name != best_feature]

                    for value in parent_feature_values:
                        sub_data = data.where(data[best_feature] == value).dropna()
                        #build subtree recursively
                        subtree = self.decision_tree(sub_data, original_data, new_feature_names, target_label_name, parent_node_class)
                        #add sbtree to original tree
                        tree[best_feature][value] = subtree
                    self.extend += 1

                else:
                    for value in parent_feature_values:
                        sub_data = data.where(data[best_feature] == value).dropna()
                        subtree = np.unique(sub_data.where(sub_data[best_feature] == value).dropna()[target_label_name], return_counts=True)
                        majority_label_idx = np.argmax(subtree[1])
                        tree[best_feature][value] = subtree[0][majority_label_idx]
                    self.extend = 0

            #2.make leaf node based ont the max depth parameter
            elif self.depth >= self.maxdepth - 1:
                for value in parent_feature_values:
                    sub_data = data.where(data[best_feature] == value).dropna()
                    subtree = np.unique(sub_data.where(sub_data[best_feature] == value).dropna()[target_label_name], return_counts=True)
                    majority_label_idx = np.argmax(subtree[1])
                    tree[best_feature][value] = subtree[0][majority_label_idx]
                self.extend = 0
            self.depth = self.max_depth(tree)
            return tree


    def predict(self, X):
        """
        function to make dictionary based on the train data
        :param X(dataframe): the feature data
        :return: list of the predicted label
        """
        instances = X.to_dict(orient = 'records') #pack data of each instance into dictionary
        predictions = []

        #predict label of the every instances in the input data
        for instance in instances:
            predictions.append(self.perform_prediction(instance, self.tree)) #self.tree is tree based on the trainset

        return predictions

    def perform_prediction(self, instance, model_tree, default=1):
        """
        function to decide predicted label based on the tree
        :param instance(dict): the data of the instance
        :param tree: trained tree
        :param default:
        :return: result
        """
        for attribute in list(instance.keys()):
            #confirm feature is included in the tree
            if attribute in list(model_tree.keys()):
                try:
                    result = model_tree[attribute][instance[attribute]]
                except:
                    return default

                result = model_tree[attribute][instance[attribute]]

                # if more attributes exist within result, recursively find best result
                if isinstance(result, dict):
                    return self.perform_prediction(instance, result)
                else:
                    return result



def main(args):

    if len(args) == 2:
        filename = args[1]
        df = read_data(filename)

        label_enchoder(df)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]

        depth = len(X.columns)+1

        # scikit-learn
        model_sk = tree.DecisionTreeClassifier(criterion='entropy')
        model_sk = model_sk.fit(X, y)
        y_pred_sk = model_sk.predict(X)
        acc_sk = accuracy(y_pred_sk, y)

        #ID3
        model = ID3(depth)
        model.fit(X, y)
        y_pred_id3 = model.predict(X)
        acc_id3 = accuracy(y_pred_id3, y)

        print("--Q1"+"--"*20)
        print("sklearn: Accuracy:{:.2f}%".format(acc_sk))
        print("ID3: Accuracy:{:.2f}%".format(acc_id3))


    elif len(args) == 3:
        # load data
        filename = args[1]
        depth = int(args[2])

        df = read_data(filename)

        label_enchoder(df)
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]

        # scikit-learn
        model_sk = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        model_sk = model_sk.fit(X, y)
        y_pred_sk = model_sk.predict(X)
        acc_sk = accuracy(y_pred_sk, y)

        # ID3
        model = ID3(depth)
        model.fit(X, y)
        y_pred_id3 = model.predict(X)
        acc_id3 = accuracy(y_pred_id3, y)

        print("--Q2" + "--" * 20)
        print("sklearn: Accuracy:{:.2f}%".format(acc_sk))
        print("ID3: Accuracy:{:.2f}%".format(acc_id3))


    elif len(args) == 4:
        #parameters
        train_filename = args[1]
        depth = int(args[2])
        test_filename = args[3]

        #train dataset
        df_tr = read_data(train_filename)
        label_enchoder(df_tr)
        X_tr = df_tr[df_tr.columns[:-1]]
        y_tr = df_tr[df_tr.columns[-1]]

        #test dataset
        df_test = read_data(test_filename)
        label_enchoder(df_test)
        X_test = df_test[df_test.columns[:-1]]
        y_test = df_test[df_test.columns[-1]]


        # scikit-learn
        model_sk = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        model_sk = model_sk.fit(X_tr, y_tr)
        y_pred_sk_tr = model_sk.predict(X_tr)
        acc_sk_tr = accuracy(y_pred_sk_tr, y_tr)

        y_pred_sk_test = model_sk.predict(X_test)
        acc_sk_test = accuracy(y_pred_sk_test, y_test)

        # ID3
        model = ID3(depth)
        model.fit(X_tr, y_tr)
        y_pred_id3_tr = model.predict(X_tr)
        acc_id3_tr = accuracy(y_pred_id3_tr, y_tr)

        y_pred_id3_test = model.predict(X_test)
        acc_id3_test = accuracy(y_pred_id3_test, y_test)

        print("--Q3" + "--" * 20)
        print("sklearn: Accuracy(train):{:.2f}%  Accuracy(test):{:.2f}%".format(acc_sk_tr, acc_sk_test))
        print("ID3: Accuracy(train):{:.2f}%   Accuracy(test):{:.2f}%".format(acc_id3_tr, acc_id3_test))

    else:
        print('please input correct format')


if __name__ == "__main__":
    args = sys.argv
    main(args)