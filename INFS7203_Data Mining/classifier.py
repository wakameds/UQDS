from __future__ import division
from math import log

from decisiontreec.dataset import copy_dataset
from decisiontreec.decision_tree import DecisionNode, EndNode

def _decision_tree(dataset):
    """
    Return a decision tree classifier, computed using the ID3 algorithm (DecisionNode or EndNode)
    Parameters:
        - dataset: the dataset used to train the classifier (Dataset)
    """
    if len(dataset.get_target_values()) == 1:
        return EndNode(dataset.get_instance(0).get_target_value())
    if len(dataset.get_attributes_names()) == 0:
        return EndNode(dataset.get_most_common_target())
    best_attribute = _get_best_attribute(dataset)
    decision_node = DecisionNode(best_attribute)
    for attribute_value in dataset.get_attribute_values(best_attribute):
        if dataset.count_instances(target=None, attribute={"name": best_attribute,
                                                           "value": attribute_value}) == 0:
            child = EndNode(dataset.get_most_common_target())
            decision_node.add_child(attribute_value, child)
        else:
            child = _decision_tree(copy_dataset(dataset, attribute={"name": best_attribute,
                                                         "value": attribute_value}))
            decision_node.add_child(attribute_value, child)
    return decision_node

# PRIVATE FUNCTIONS
# These functions should not be used outside the module
def _get_best_attribute(dataset):
    """
    Return the attribute of the dataset that best classifies examples of the dataset (String)
    Parameters:
        - dataset: the dataset on which the computation has to be done (Dataset)
    """
    attributes_names = dataset.get_attributes_names()
    max_information_gain = _information_gain(dataset, attributes_names[0])
    best_attribute = attributes_names[0]
    for index in range(1, len(attributes_names)):
        attribute_name = attributes_names[index]
        information_gain = _information_gain(dataset, attribute_name)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute_name
    return best_attribute

def _entropy(dataset):
    """
    Return the entropy that characterizes the given dataset (number)
    Parameters:
        - dataset: the dataset on which the entropy has to be computed (Dataset)
    """
    proportions = [dataset.count_instances(target=target)/dataset.count_instances()
                   for target in dataset.get_target_values()]
    return sum(-p*log(p, 2) for p in proportions)

def _information_gain(dataset, attribute_name):
    """
    Return the measure of the difference in entropy from before to after the dataset is split on the
    given attribute (in other words, how much uncertainty in the dataset was reduced after splitting
    the dataset on the given attribute) (number)
    Parameters:
        - dataset: the dataset on which the information gain has to be computed (Dataset)
        - attribute_name: the name of the attribute with which the information gain has to be
          computed (String)
    """
    T = set()
    attribute_values = dataset.get_attribute_values(attribute_name)
    for attribute_value in attribute_values:
        t = copy_dataset(dataset, attribute={"name": attribute_name, "value": attribute_value})
        T.add(t)
    return _entropy(dataset) - sum(_entropy(t)*t.count_instances()/dataset.count_instances()
                                   for t in T)

def _decision_tree_classify(decision_tree, instance):
    """
    Return the target value that the decision tree associates to the given instance (value)
    Parameters:
        - decision_tree: decision tree that determines which target is associated to the instance
          (DecisionNode or EndNode)
        - instance: the instance that has to be classified (DatasetInstance)
    """
    if isinstance(decision_tree, EndNode):
        return decision_tree.get_target_value()
    decision_attribute_name = decision_tree.get_decision_attribute()
    if instance.get_attribute_value(decision_attribute_name) not in decision_tree.children:
        return None
    return _decision_tree_classify(decision_tree.get_child(instance.get_attribute_value(decision_attribute_name)), instance)


def get_accuracy(dataset, predictions):
    """
    Return the accuracy of the predictions (number)
    Parameters:
        - dataset: the dataset containing all the instances correctly classified (Dataset)
        - predictions: a list of targets related to the instances of the dataset (List of values)
    """
    return sum([1 for index, instance in enumerate(dataset)
                if predictions[index] == dataset.get_instance(index).get_target_value()])/dataset.count_instances()
