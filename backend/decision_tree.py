from dataclasses import dataclass
import numpy as np
import json
import copy

from collections import Counter
from sklearn.metrics import accuracy_score

@dataclass
class Node:
    """Class representing a node in the decision tree."""
    def __init__(self, node_id, feature_index=None, threshold=None, num_samples=0, removed_features=[], left=None, right=None, output=None, depth=0, X=None, y=None):
        self.node_id = node_id
        self.feature_index = feature_index  
        self.threshold = threshold          
        self.num_samples = num_samples
        self.removed_features = removed_features
        self.left = left                    
        self.right = right                  
        self.output = output                
        self.depth = depth

class DecisionTreeClassifier:
    
    def __init__(self, id_counter=0, feature_names=None, feature_indices=None, class_names=None):
        self.id_counter = id_counter
        self.root = None
        self.feature_names = feature_names
        self.feature_indices = feature_indices
        if isinstance(class_names, np.ndarray):
            self.class_names = class_names.tolist()
        else:
            self.class_names = class_names

    def count_nodes(self):
        """Count the total number of nodes in the decision tree."""
        def _count_nodes_recursive(node):
            if node is None:
                return 0
            # Count the current node + recursively count left and right children
            return 1 + _count_nodes_recursive(node.left) + _count_nodes_recursive(node.right)

        return _count_nodes_recursive(self.root)

    def fit(self, X, y):
        """Builds the decision tree from the training data."""
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def score(self, X, y):
        """Evaluate the model using accuracy."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def visualize(self, print_features=False, print_classes=False):
        self._print_sub_tree(self.root, print_features=print_features, print_classes=print_classes)
    
    def _print_sub_tree(self, node, depth=0, indent="", print_features=False, print_classes=False):
        if print_features:
            removed_features_str = f", removed: {node.removed_features}" if node.removed_features else ""
        else:
            removed_features_str = ""
        if print_classes:
            collected_class_names_str = f", possible classes: {self.collect_events(node.node_id)}"
        else:
            collected_class_names_str = ""

        if node.output is not None:
            # Print the class name if it's a leaf node
            class_name = self.class_names[node.output] if self.class_names is not None else node.output
            print(f"{indent}|--- [{node.node_id}] class: {class_name} [{node.num_samples}]")
        else:
            feature_name = self.feature_names[node.feature_index] if self.feature_names is not None else f"Feature {node.feature_index}"
            # Print the current decision rule for the left child (<= threshold)
            print(f"{indent}|--- [{node.node_id}] ({feature_name}) <= {node.threshold:.2f}{removed_features_str}{collected_class_names_str} [{node.num_samples}]")
            self._print_sub_tree(node.left, depth + 1, indent + "|   ")
            # Print the current decision rule for the right child (> threshold)
            print(f"{indent}|--- [{node.node_id}] ({feature_name}) >  {node.threshold:.2f}{removed_features_str}{collected_class_names_str} [{node.num_samples}]")
            self._print_sub_tree(node.right, depth + 1, indent + "|   ")

    def _grow_tree(self, X, y, depth=0, max_depth=None, removed_features=[], recursive_removal=True):
        """Recursively grows the decision tree."""
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        node_id = self.id_counter
        self.id_counter += 1

        # Stop criteria
        if len(unique_classes) == 1 or (max_depth is not None and depth >= max_depth) or len(removed_features) == num_features:
            return Node(node_id, num_samples=num_samples, output=self._majority_class(y), depth=depth, removed_features=removed_features)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_features, removed_features)
        if best_feature is None:
            return Node(node_id, num_samples=num_samples, output=self._majority_class(y), depth=depth, removed_features=removed_features)

        # Create the left and right subtrees
        removed_features = removed_features if recursive_removal else []
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], max_depth=max_depth, depth=depth+1, removed_features=removed_features)
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], max_depth=max_depth, depth=depth+1, removed_features=removed_features)

        return Node(node_id, num_samples=num_samples, feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree, depth=depth, removed_features=removed_features)


    def _best_split(self, X, y, num_features, removed_features):
        """Find the best feature and threshold to split the data, ignoring removed features."""
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_index in range(num_features):
            # Skip removed features
            if feature_index in removed_features:
                continue

            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            num_left = Counter()
            num_right = Counter(classes)

            for i in range(1, len(classes)):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                if i == 0 or i == len(classes):
                    continue  # Skip invalid splits (empty set)

                gini_left = 1 - sum((num_left[x] / i) ** 2 for x in num_left)
                gini_right = 1 - sum((num_right[x] / (len(classes) - i)) ** 2 for x in num_right)

                gini = (i * gini_left + (len(classes) - i) * gini_right) / len(classes)

                if thresholds[i] == thresholds[i - 1]:
                    continue  # Skip duplicate threshold

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2  # Average for split

        return best_feature, best_threshold

    def _predict_sample(self, sample, node):
        """Predicts the class for a single sample."""
        if node.output is not None:
            return node.output
        if sample[node.feature_index] < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def _majority_class(self, y):
        """Returns the majority class from the labels."""
        return Counter(y).most_common(1)[0][0]
    
    def change_feature(self, node_id, feature_index, threshold, flip=False):
        _, node, _ = self._find_node(self.root, node_id)
        node.feature_index = feature_index
        node.threshold = threshold
        if flip:
            node.right, node.left = node.left, node.right

    def delete_branch(self, node_id, direction=None):
        """Delete the branch of the tree starting at the node with the given direction."""
        parent_node, node, parent_direction = self._find_node(self.root, node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found.")
        
        # if direction is given, set parent child to the other branch
        if direction == 'left':
            if parent_direction == 'left':
                parent_node.left = node.right
            elif parent_direction == 'right':
                parent_node.right = node.right
            elif parent_direction == None:
                self.root = node.right
        elif direction == 'right':
            if parent_direction == 'left':
                parent_node.left = node.left
            elif parent_direction == 'right':
                parent_node.right = node.left
            elif parent_direction == None:
                self.root = node.left
        # if no direction is given, compare num_samples and delete the one 
        elif direction is None:
            if node.left is not None and node.right is not None:
                if node.left.num_samples < node.right.num_samples:
                    if parent_direction == 'left':
                        parent_node.left = node.right
                    elif parent_direction == 'right':
                        parent_node.right = node.right
                    elif parent_direction == None:
                        self.root = node.right
                else:
                    if parent_direction == 'left':
                        parent_node.left = node.left
                    elif parent_direction == 'right':
                        parent_node.right = node.left
                    elif parent_direction == None:
                        self.root = node.left
            else:
                raise ValueError(f"Node {node_id} has no children to delete.")
        else:
            raise ValueError("Direction must be 'left', 'right', or None.")

    def delete_node(self, X, y, node_id, recursive_removal=True):
        """Delete the node with the specified node_id and regrow the subtree."""
        # Find the node and the parent node
        parent_node, node_to_delete, direction = self._find_node(self.root, node_id)
        
        if node_to_delete is None:
            print(f"Node with id {node_id} not found.")
            return

        if node_to_delete.output is not None:
            print(f"Node with id {node_id} is a leaf node.")
            return
        
        # Collect all features used by the subtree that includes the node to delete
        old_removed_features = node_to_delete.removed_features
        new_removed_features = self._collect_used_feature_indices(node_to_delete)
        removed_features = sorted(list(set(old_removed_features + new_removed_features)))
        depth = node_to_delete.depth
        max_depth = depth + get_max_depth(node_to_delete) - 1
        print(f"Depth: {depth}, Max Depth: {max_depth}")
        node_id = node_to_delete.node_id
        
        # Regrow the tree from the point where the node was removed
        X_sub, y_sub = self._get_data_for_subtree(X, y, node_id)
        new_sub_tree = self._grow_tree(X_sub, y_sub, depth=depth, max_depth=max_depth, removed_features=removed_features, recursive_removal=recursive_removal)
        
        if direction == 'left':
            parent_node.left = new_sub_tree
        elif direction == 'right':
            parent_node.right = new_sub_tree
        elif direction == None:
            self.root = new_sub_tree
    
    def print_tree_metrics(self, X, y, node):
        if node == None:
            return 

        for attribute in self.feature_indices:
            if node.feature_index in self.feature_indices[attribute]:
                X, y = self._get_data_for_subtree(X, y, node.node_id)
                feature_index = node.feature_index
                possible_events = self.collect_events(node_id=node.node_id)
                metrics = get_metrics(X, y, feature_index, possible_events, self.class_names, self.feature_names)
                print(f"Metrics for node {node.node_id}:")
                for metric in metrics:
                    print(metric)
                print("")

        self.print_tree_metrics(X, y, node.right)
        self.print_tree_metrics(X, y, node.left)
    
    def collect_events(self, node_id):
        """Collect all unique events in the subtree rooted at the node with node_id."""
        _, node, _ = self._find_node(self.root, node_id)
        if node is None:
            raise ValueError(f"Node with id {node_id} not found.")
        unique_classes = set()

        def _collect_events_recursive(node):
            if node is None:
                return
            if node.output is not None:
                unique_classes.add(node.output)
                return
            _collect_events_recursive(node.right)
            _collect_events_recursive(node.left)
        
        # start the recursion from the node
        _collect_events_recursive(node)
        
        # transform class indices into class names
        if self.class_names is None:
            raise ValueError("class_names attribute is not set in the tree.")
        class_names = [self.class_names[class_index] for class_index in unique_classes]
        
        return class_names

    # Find a node and return its parent, itself and the direction
    def find_node(self, node_id):
        parent, node, direction = self._find_node(self.root, node_id)
        return node

    def _find_node(self, node, node_id, parent=None, direction=None):
        if node.node_id == node_id:
            return parent, node, direction
        if node.left is not None:
            found_parent, found_node, found_direction = self._find_node(node.left, node_id, node, 'left')
            if found_node is not None:
                return found_parent, found_node, found_direction
        if node.right is not None:
            return self._find_node(node.right, node_id, node, 'right')
        return None, None, None

    def _collect_used_feature_indices(self, node):
        for attributes, indices in self.feature_indices.items():
            if node.feature_index in indices:
                return indices

    def collect_node_ids(self, node=None):
        if node is None:
            node = self.root
            
        output = [node.node_id]
        if node.output is None:
            left_result = self.collect_node_ids(node=node.left)
            right_result = self.collect_node_ids(node=node.right)
            output = output + right_result + left_result
        return output

    # Collect data for the subtree rooted at the node
    def _get_data_for_subtree(self, X, y, node_id, node=None):
        if node is None:
            node = self.root  # Start from the root node
        
        # If we found the node with the matching node_id, return the data
        if node.node_id == node_id:
            return X, y
        
        # If it's a leaf node and we haven't found the node_id, return None
        if node.output is not None:
            return None
        
        # Otherwise, traverse down the tree based on the conditions
        if node.feature_index is not None and node.threshold is not None:
            # Split data based on the feature and threshold at the current node
            left_indices = X[:, node.feature_index] < node.threshold
            right_indices = X[:, node.feature_index] >= node.threshold
            
            # Recursively check the left and right subtrees
            if node.left:
                left_result = self._get_data_for_subtree(X[left_indices], y[left_indices], node_id, node.left)
                if left_result is not None:
                    return left_result
            if node.right:
                right_result = self._get_data_for_subtree(X[right_indices], y[right_indices], node_id, node.right)
                if right_result is not None:
                    return right_result

        return None  # If the node_id is not found

    def _build_ancestor_map(self, node, parent_id=None, ancestor_map={}):
        if node is None:
            return

        # Initialize the ancestor list for this node
        if node.node_id not in ancestor_map:
            ancestor_map[node.node_id] = set()

        # If the node has a parent, add the parent and its ancestors to the current node's ancestor list
        if parent_id is not None:
            ancestor_map[node.node_id].add(parent_id)
            ancestor_map[node.node_id].update(ancestor_map[parent_id])

        # Recursively build the map for left and right children
        self._build_ancestor_map(node.left, node.node_id, ancestor_map)
        self._build_ancestor_map(node.right, node.node_id, ancestor_map)

        return ancestor_map

    def filter_nodes(self, node_list):
        # build the ancestor map for each node
        ancestor_map = self._build_ancestor_map(self.root)

        # filter out nodes whose ancestors are in the list
        filtered_nodes = []
        node_set = set(node_list)
        for node_id in node_list:
            if not any(ancestor in node_set for ancestor in ancestor_map.get(node_id, [])):
                filtered_nodes.append(node_id)

        return filtered_nodes

    def find_nodes_to_remove(self, critical_decisions, protective=False, node=None):
        if node is None:
            node = self.root

        if node.output is not None:
            return []
        
        left_result = self.find_nodes_to_remove(critical_decisions, node=node.left)
        right_result = self.find_nodes_to_remove(critical_decisions, node=node.right)

        output = left_result + right_result
        # removes every nodes that overlaps with a to_remove=True
        for decision in critical_decisions:
            if not decision.to_remove:
                continue
            for attribute in decision.attributes:
                if node.feature_index in self.feature_indices[attribute]:
                    possible_events = self.collect_events(node_id=node.node_id)
                    print(possible_events)
                    print(decision.possible_events)
                    if protective:
                        if set(possible_events) & set(decision.possible_events):
                            output.append(node.node_id)
                            return output
                    else:
                        if set(possible_events) >= set(decision.possible_events):
                            output.append(node.node_id)
                            return output

        return output

        

def save_tree_to_json(tree, file_path):
    """Saves the decision tree model along with its attributes to a JSON file."""

    def convert_to_python_type(value):
        """Convert numpy data types to native Python types for JSON serialization."""
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()  # Convert numpy arrays to lists
        return value

    def node_to_dict(node):
        """Recursively transforms a Node object into a serializable dictionary."""
        if node is None:
            return None
        return {
            'node_id': convert_to_python_type(node.node_id),
            'feature_index': convert_to_python_type(node.feature_index),
            'threshold': convert_to_python_type(node.threshold),
            'num_samples': convert_to_python_type(node.num_samples),
            'removed_features': convert_to_python_type(node.removed_features),
            'left': node_to_dict(node.left),
            'right': node_to_dict(node.right),
            'output': convert_to_python_type(node.output),
            'depth': convert_to_python_type(node.depth)
        }

    # Create a dictionary with both the tree and its classifier attributes
    tree_dict = {
        'root': node_to_dict(tree.root),
        'id_counter': convert_to_python_type(tree.id_counter),
        'feature_names': convert_to_python_type(tree.feature_names),
        'feature_indices': convert_to_python_type(tree.feature_indices),
        'class_names': convert_to_python_type(tree.class_names)
    }

    # Save to the specified JSON file
    with open(file_path, 'w') as json_file:
        json.dump(tree_dict, json_file, indent=4)

def load_tree_from_json(file_path):
    """Loads a decision tree from a JSON file and reconstructs the tree."""
    
    def dict_to_node(data):
        """Recursively transforms a dictionary back into a Node object."""
        if data is None:
            return None
        node = Node(
            node_id=data['node_id'],
            feature_index=data.get('feature_index'),
            threshold=data.get('threshold'),
            num_samples=data.get('num_samples'),
            removed_features=data.get('removed_features', []),
            left=dict_to_node(data.get('left')),
            right=dict_to_node(data.get('right')),
            output=data.get('output'),
            depth=data.get('depth')
        )
        return node
    
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        tree_dict = json.load(json_file)

    # Create a new DecisionTreeClassifier instance and set the attributes
    tree = DecisionTreeClassifier(
        feature_names=tree_dict.get('feature_names'),
        feature_indices=tree_dict.get('feature_indices'),
        class_names=tree_dict.get('class_names'),
        id_counter=tree_dict.get('id_counter')
    )
    
    # Reconstruct the root node from the dictionary
    tree.root = dict_to_node(tree_dict['root'])
    return tree

def get_max_depth(node):
    if node is None:
        return 0
    
    left_depth = get_max_depth(node.left)
    right_depth = get_max_depth(node.right)
    
    return 1 + max(left_depth, right_depth)

def get_deleted_nodes(dt_original, dt_modified):
    node_ids_original = set(dt_original.collect_node_ids())
    node_ids_modified = set(dt_modified.collect_node_ids())
    node_ids_missing = list(node_ids_original - node_ids_modified)
    return dt_original.filter_nodes(node_ids_missing)

def get_metrics(X, y, feature_index, possible_events, class_names, feature_names):
    output = []
    feature_name = feature_names[feature_index]
    protected_attribute = X[:, feature_index]
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    for event in possible_events:
        if event not in class_names:
            raise ValueError(f"Event '{event}' not found in class_names.")
        if isinstance(class_names, list):
            event_index = class_names.index(event)
        elif isinstance(class_names, np.ndarray):
            event_index = np.where(class_names == event)[0][0]
        y_binary = np.zeros_like(y)
        y_binary[y == event_index] = 1
        amount = np.sum((y_binary == 1) & (protected_attribute == 1))
        total_amount = np.sum(y_binary == 1)
        #print(len(y_binary))
        if total_amount < 1:
            output_string = f"Feature: {feature_name}, Event: {event} | INSUFFICIENT DATA"
            output.append(output_string)
            continue

        #print(f"metric for {feature_name} and {event} with length {len(y_binary)}")
        unprivileged_mask = protected_attribute == 0
        privileged_mask = protected_attribute == 1
        if np.any(unprivileged_mask):
            selection_rate_unprivileged = np.mean(y_binary[unprivileged_mask])
        else:
            selection_rate_unprivileged = 0
        if np.any(privileged_mask):
            selection_rate_privileged = np.mean(y_binary[privileged_mask])
        else:
            selection_rate_privileged = 0
        #print(f"priviledged selection rate: {selection_rate_privileged}, unpriviledged selection rate: {selection_rate_unprivileged}")
        if selection_rate_privileged == 0:
            if selection_rate_unprivileged == 0:
                disp_impact = 0
            else:
                disp_impact = float("inf")
        else:
            disp_impact = selection_rate_unprivileged / selection_rate_privileged
        stat_parity = selection_rate_unprivileged - selection_rate_privileged

        output_string = f"Feature: {feature_name}, Event: {event} | stat. parity: {stat_parity}, disp. impact: {disp_impact}, amount: {amount} out of {total_amount}"
        output.append(output_string)
    return output


def sklearn_to_custom_tree(sklearn_tree, feature_names=None, class_names=None, feature_indices=None):
    """Converts an sklearn decision tree to a custom DecisionTreeClassifier"""
    tree_ = sklearn_tree.tree_

    def build_node(node_id, depth):
        """
        Recursively builds a custom Node from an sklearn tree node.
        """
        # Extract information from the sklearn tree
        feature_index = sklearn_tree.tree_.feature[node_id]
        threshold = sklearn_tree.tree_.threshold[node_id]
        num_samples = sklearn_tree.tree_.n_node_samples[node_id]
        value = sklearn_tree.tree_.value[node_id]
        left_child = sklearn_tree.tree_.children_left[node_id]
        right_child = sklearn_tree.tree_.children_right[node_id]
        
        # Check if it's a leaf node
        if left_child == -1 and right_child == -1:
            # Get the output class for leaf
            index = np.argmax(value) if value.ndim > 1 else value[0]
            output = sklearn_tree.classes_[index]
            #print(output)
            return Node(
                node_id=node_id,
                num_samples=num_samples,
                output=output,
                depth=depth
            )
        
        # Build the current node
        node = Node(
            node_id=node_id,
            feature_index=feature_index,
            threshold=threshold,
            num_samples=num_samples,
            depth=depth
        )
        
        # Recursively build left and right children
        node.left = build_node(left_child, depth + 1)
        node.right = build_node(right_child, depth + 1)
        
        return node

    # Initialize the custom decision tree
    custom_tree = DecisionTreeClassifier(
        feature_names=feature_names,
        class_names=class_names,
        feature_indices=feature_indices
    )
    
    # Build the root node
    custom_tree.root = build_node(0,0)
    
    return custom_tree


def copy_decision_tree(tree):
    if tree is None:
        return None
    
    # Create a new DecisionTreeClassifier object
    new_tree = DecisionTreeClassifier(
        id_counter=tree.id_counter,
        feature_names=copy.deepcopy(tree.feature_names),
        feature_indices=copy.deepcopy(tree.feature_indices),
        class_names=copy.deepcopy(tree.class_names)
    )
    
    def copy_node(node):
        if node is None:
            return None
        
        return Node(
            node_id=node.node_id,
            feature_index=node.feature_index,
            threshold=node.threshold,
            num_samples=node.num_samples,
            removed_features=copy.deepcopy(node.removed_features),
            left=copy_node(node.left),
            right=copy_node(node.right),
            output=node.output,
            depth=node.depth
        )

    # Copy the root node
    new_tree.root = copy_node(tree.root)
    
    return new_tree