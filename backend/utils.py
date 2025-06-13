import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import sys
import argparse

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from trace_generator import *
from data_processing import *
from decision_tree import *
from plotting import *

def generate_data(num_cases, model_name, prefix_length):
    process_model = build_process_model(model_name)
    folder_name = model_name
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    X, y, class_names, feature_names, feature_indices = generate_processed_data(process_model, categorical_attributes=categorical_attributes, numerical_attributes=numerical_attributes, num_cases=num_cases, prefix_length=prefix_length, folder_name=folder_name)
    return X, y, class_names, feature_names, feature_indices

# define neural network architecture
def build_nn(input_dim, output_dim, hidden_units=[512, 256, 128, 64], learning_rate=1e-3):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# train and save neural network
def train_nn(X_train, y_train, folder_name=None, hidden_units=[512, 256, 128, 64], model_name="nn.keras", epochs=10, batch_size=32, learning_rate=1e-3):
    input_dim = X_train.shape[1]  # Number of input features (attributes + events)
    output_dim = y_train.shape[1]  # Number of possible events (classes)
    print("training neural network:")
    print(f"input dimension: {input_dim}")
    print(f"output dimension: {output_dim}")
    model = build_nn(input_dim, output_dim, hidden_units=hidden_units, learning_rate=learning_rate)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    print("--------------------------------------------------------------------------------------------------")
    if folder_name:
        save_nn(model, folder_name, model_name)
    return model

def train_sklearn_dt(X_train, y_train, ccp_alpha=0.001, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    print("training decision tree:")
    dt = SklearnDecisionTreeClassifier(ccp_alpha=ccp_alpha, max_depth=max_depth, 
                                       min_samples_split=min_samples_split, 
                                       min_samples_leaf=min_samples_leaf)
    dt.fit(X_train, y_train)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def train_dt(X_train, y_train, ccp_alpha=0.001, max_depth=None, min_samples_split=2, min_samples_leaf=1, folder_name=None, model_name=None, feature_names=None, feature_indices=None, class_names=None):
    dt = train_sklearn_dt(X_train, y_train, ccp_alpha=ccp_alpha, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    dt = sklearn_to_custom_tree(dt, feature_names=feature_names, class_names=class_names, feature_indices=feature_indices)
    num_nodes = dt.count_nodes()
    dt.id_counter = num_nodes
    if model_name and folder_name:
        save_dt(dt, folder_name, model_name)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def train_custom_dt(X_train, y_train, folder_name=None, model_name=None, feature_names=None, feature_indices=None, class_names=None):
    print("training decision tree:")
    dt = DecisionTreeClassifier(class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    dt.fit(X_train, y_train)
    if model_name and folder_name:
        save_dt(dt, folder_name, model_name)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def save_nn(model, folder_name, file_name):
    #print(f"saving {file_name}...")
    full_path = os.path.join('models', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    model.save(file_path)

def load_nn(folder_name, file_name):
    #print(f"loading {file_name}...")
    file_name = os.path.join('models', folder_name, file_name)
    model = load_model(file_name)
    return model
    
def save_dt(dt, folder_name, file_name):
    #print(f"saving {file_name}...")
    full_path = os.path.join('models', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    save_tree_to_json(dt, file_path)

def load_dt(folder_name, file_name):
    #print(f"loading {file_name}...")
    file_path = os.path.join('models', folder_name, file_name)
    return load_tree_from_json(file_path)

def evaluate_nn(model, X_test, y_test):
    print("testing nn:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'accuracy: {test_accuracy:.3f}, loss: {test_loss:.3f}')
    print("--------------------------------------------------------------------------------------------------")
    return test_accuracy


def calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds):
    networks = {
        "base": nn_base,
        "enriched": nn_enriched,
        "modified": nn_modified,
    }
    disp_imp_results = {}
    stat_par_results = {}
    y = {}
    X_adjusted = remove_attribute_features(X, feature_indices, base_attributes)

    for name, nn in networks.items():
        if name == "base":
            y[name] = nn.predict(X_adjusted)
        else:
            y[name] = nn.predict(X)

    for decision in critical_decisions:
        if isinstance(decision.previous, list):
            previous_indices = [feature_names.index("-1. Event = " + prev) for prev in decision.previous]
        else:
            previous_indices = [feature_names.index("-1. Event = " + decision.previous)]
        for attribute in decision.attributes:
            feature_indices_to_use = feature_indices.get(attribute, [])
            for feature_index in feature_indices_to_use:
                for event in decision.possible_events:
                    outer_key = (feature_names[feature_index], event)
                    disp_imp_results[outer_key] = {}
                    stat_par_results[outer_key] = {}
                    for name, nn in networks.items():
                        if event not in class_names:
                            raise ValueError(f"Event '{event}' not found in class_names.")
                        if isinstance(class_names, list):
                            event_index = class_names.index(event)
                        elif isinstance(class_names, np.ndarray):
                            event_index = np.where(class_names == event)[0][0]
                        
                        # Adjust filter_mask to check for any of the previous features being 1
                        filter_mask = np.any(X[:, previous_indices] == 1, axis=1)
                        
                        X_filtered = X[filter_mask]
                        y_filtered = y[name][filter_mask]
                        protected_attribute = X_filtered[:, feature_index]

                        stat_par, disp_imp = get_fairness_metrics(y_filtered, protected_attribute, event_index, feature_names[feature_index], numerical_thresholds)

                        disp_imp_results[outer_key][name] = disp_imp
                        stat_par_results[outer_key][name] = stat_par
                        print(f"Statistical Parity for {feature_names[feature_index]}, {event}, {name}: {stat_par}")
 
    return stat_par_results, disp_imp_results

def remove_attribute_features(X, feature_indices, base_attributes):
    remove_indices = [idx for attr, indices in feature_indices.items() if attr not in base_attributes for idx in indices]
    return np.delete(X, remove_indices, axis=1)

def get_fairness_metrics(y, protected_attribute, event_index, feature_name, numerical_thresholds):
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    y_binary = np.zeros_like(y)
    y_binary[y == event_index] = 1
    total_amount = np.sum(y_binary == 1)
    amount = np.sum((y_binary == 1) & (protected_attribute == 1))
    if total_amount < 1:
        return 0, 1

    if feature_name in numerical_thresholds:
        # Use threshold to separate groups for numerical attributes
        threshold = numerical_thresholds[feature_name]
        unprivileged_mask = protected_attribute <= threshold
        privileged_mask = protected_attribute > threshold
    else:
        # Default handling for binary attributes
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

    if selection_rate_privileged == 0:
        disp_impact = float("inf") if selection_rate_unprivileged != 0 else 0
    else:
        disp_impact = selection_rate_unprivileged / selection_rate_privileged

    stat_parity = abs(selection_rate_unprivileged - selection_rate_privileged)
    #print(f"stat. parity: {stat_parity}, amount: {amount} out of {total_amount}, privileged: {np.sum(protected_attribute == 1)}, unprivileged: {np.sum(protected_attribute == 1)}")
    #print(f"selection_rate_unprivileged: {selection_rate_unprivileged}, selection_rate_privileged: {selection_rate_privileged}")
    return stat_parity, disp_impact


def evaluate_dt(dt, X_test, y_test):
    print("testing dt:")
    y_argmax = np.argmax(y_test, axis=1)
    accuracy = dt.score(X_test, y_argmax)
    print(f"Accuracy: {accuracy:.3f}")
    print("")
    dt.visualize()
    print("")
    print("--------------------------------------------------------------------------------------------------")
    return accuracy

def distill_nn(nn, X):
    print("distilling nn:")
    softmax_predictions = nn.predict(X)
    y = np.argmax(softmax_predictions, axis=1)
    print("--------------------------------------------------------------------------------------------------")
    return y

def calculate_comparable_fairness_single(nn_model, X, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds):
    stat_par_results = {}
    disp_imp_results = {}
    
    y = nn_model.predict(X)
    
    for decision in critical_decisions:
        if isinstance(decision.previous, list):
            previous_indices = [feature_names.index("-1. Event = " + prev) for prev in decision.previous]
        else:
            previous_indices = [feature_names.index("-1. Event = " + decision.previous)]
        
        for attribute in decision.attributes:
            feature_indices_to_use = feature_indices.get(attribute, [])
            for feature_index in feature_indices_to_use:
                for event in decision.possible_events:
                    key = (feature_names[feature_index], event)
                    
                    if event not in class_names:
                        raise ValueError(f"Event '{event}' not found in class_names.")
                    
                    event_index = class_names.index(event) if isinstance(class_names, list) else np.where(class_names == event)[0][0]
                    filter_mask = np.any(X[:, previous_indices] == 1, axis=1)
                    X_filtered = X[filter_mask]
                    y_filtered = y[filter_mask]
                    protected_attribute = X_filtered[:, feature_index]
                    
                    stat_par, disp_imp = get_fairness_metrics(y_filtered, protected_attribute, event_index, feature_names[feature_index], numerical_thresholds)
                    
                    stat_par_results[key] = stat_par
                    disp_imp_results[key] = disp_imp 
                    
                    print(f"Statistical Parity for {feature_names[feature_index]}, {event}, model: {stat_par}")
    
    return stat_par_results, disp_imp_results


def finetune_all(nn, X_train, y_modified, y_distilled_tree, y_distilled, X_test, y_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds):
    best_accuracy = 0
    dp_threshold = 0.2
    best_mode = None
    best_nn = None
    max_dp = None
    
    modes = ['changed_complete', 'simple']
    
    for mode in modes:
        nn_finetuned = clone_model(nn)
        nn_finetuned.set_weights(nn.get_weights())
        nn_finetuned = finetune_nn(nn_finetuned, X_train, y_modified, y_distilled=y_distilled, y_distilled_tree=y_distilled_tree, X_test=X_test, y_test=y_test, mode=mode)
        accuracy_score = evaluate_nn(nn_finetuned, X_test, y_test)
        
        stat_par_results, _ = calculate_comparable_fairness_single(nn_finetuned, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds)
        max_parity = max(stat_par_results.values()) 
        
        print(f"Mode: {mode}, Accuracy: {accuracy_score}, Max Demographic Parity: {max_parity}")
        
        if max_parity < dp_threshold and accuracy_score > best_accuracy:
            best_mode = mode
            best_nn = nn_finetuned
            best_accuracy = accuracy_score
            max_dp = max_parity
        elif best_nn is None:
            best_mode = mode
            best_nn = nn_finetuned
            best_accuracy = accuracy_score
            max_dp = max_parity
    
    print(f"Best mode: {best_mode}, Accuracy: {best_accuracy}, Max Demographic Parity: {max_dp}")
    return best_nn, best_mode



def finetune_nn(nn, X_train, y_modified, y_distilled_tree=None, y_distilled=None, X_test=None, y_test=None, epochs=5, batch_size=32, learning_rate=1e-3, mode="changed_complete"):
    # if mode is simple, just train with y_modified
    print(f"Finetuning with mode: {mode}")
    if mode == "simple":
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        if X_test is None:
            nn.fit(X_train, y_modified, epochs=epochs, batch_size=batch_size)
        else:
            nn.fit(X_train, y_modified, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # if mode is changed complete, use the samples that changed value
    elif mode == "changed_complete":
        changed_rows = np.any(y_distilled_tree != y_modified, axis=1)
        changed_indices = np.where(changed_rows)[0]
        y_changed_complete = y_distilled.copy()
        y_changed_complete[changed_indices] = y_modified[changed_indices]
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Changed {len(changed_indices)} out of {len(X_train)} samples")
        if X_test is None:
            nn.fit(X_train, y_changed_complete, epochs=epochs, batch_size=batch_size)
        else:
            nn.fit(X_train, y_changed_complete, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    else:
        raise(f"mode {mode} doesn't exist!")

    return nn


def find_missing_ids(dt_distilled, dt_modified):
    modified_node_ids = dt_distilled.collect_node_ids()
    distilled_node_ids = dt_modified.collect_node_ids()
    missing_ids = [item for item in distilled_node_ids if item not in modified_node_ids]
    return missing_ids


def print_samples(n, X, y, class_names, feature_names, numerical_attributes):
    if len(feature_names) != X.shape[1]:
        raise ValueError("The length of feature_names must match the number of columns in X.")
    
    for i, sample in enumerate(X[:n]):
        active_indices = np.where(sample != 0)[0]
        active_features = []
        for idx in active_indices:
            feature_name = feature_names[idx]
            if feature_name in numerical_attributes:
                active_features.append(f"{feature_name}={sample[idx]:.2f}")
            else:
                active_features.append(feature_name)
        target_activity = class_names[np.argmax(y[i])]
        print(f"Sample {i}: Features: {', '.join(active_features)}, Target: {target_activity}")
        
def print_latex_table():
    def format_mean_std(column_name):
        mean = df[column_name].mean()
        std = df[column_name].std()
        return remove_leading_zero(mean, std)

    # Compute average DP values across all related columns
    def compute_dp_avg(pattern):
        dp_cols = [col for col in df.columns if "dp" in col]
        dp_model_cols = [col for col in dp_cols if pattern in col]
        if len(dp_model_cols) > 1:
            dp_avg = df[dp_model_cols].mean(axis=0)
        else:
            dp_avg = df[dp_model_cols[0]]
        mean_val = dp_avg.mean()
        std_val = dp_avg.std()
        return remove_leading_zero(mean_val, std_val)
    
    def remove_leading_zero(mean, std):
        mean = f"{mean:.3f}"[1:]
        std = f"{std:.3f}"[1:]
        return f"{mean} $\\pm$ {std}"

    # List of folder names corresponding to table rows
    folder_names = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi_2012"]

    latex_table = ""
    for folder in folder_names:
        df = load_data(folder, "results_df.pkl")
        latex_table += f"        {folder} & "
        latex_table += f" {format_mean_std('base_accuracy')} & "
        latex_table += f" {format_mean_std('modified_accuracy')} & "
        latex_table += f" {format_mean_std('enriched_accuracy')} & "
        latex_table += f" {compute_dp_avg('base')} & "
        latex_table += f" {compute_dp_avg('modified')} & "
        latex_table += f" {compute_dp_avg('enriched')} \\\\\n"

    print(latex_table)

def print_latex_table_ablation_all():
    ablations = ["ablation_bias", "ablation_attributes", "ablation_decisions"]
    for ablation in ablations:
        df = load_data(ablation, "results_df.pkl")
        print_latex_table_ablation(df)

def print_latex_comparison():
    def format_mean_std(column_name):
        mean = df[column_name].mean()
        std = df[column_name].std()
        return remove_leading_zero(mean, std)
    
    def remove_leading_zero(mean, std):
        mean = f"{mean:.3f}"[1:]
        std = f"{std:.3f}"[1:]
        return f"{mean} $\\pm$ {std}"

    # List of folder names corresponding to table rows
    folder_names = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi_2012"]

    latex_table = ""
    for folder in folder_names:
        df = load_data(folder, "results_df.pkl")
        latex_table += f"        {folder} & "
        latex_table += f" {format_mean_std('modified_accuracy')} & "
        latex_table += f" {format_mean_std('modified_tree_accuracy')} \\\\\n"

    print(latex_table)

def print_latex_table_ablation(df):
    # Define the relevant columns
    accuracy_cols = ['base_accuracy', 'modified_accuracy', 'enriched_accuracy']
    dp_cols = ['dp_base', 'dp_modified', 'dp_enriched']
    tree_cols = ['num_nodes', 'removed_nodes', 'depth']
    all_cols = ["param"] + accuracy_cols + dp_cols + tree_cols
    
    # Compute mean and std grouped by 'param'
    df = df[all_cols]
    grouped = df.groupby('param')
    mean_std = grouped.agg(['mean', 'std'])
    
    # Formatting helper
    def format_value(mean, std, decimal_places=2):
        if decimal_places == 2:
            mean = f"{mean:.2f}"[1:]
            std = f"{std:.2f}"[1:]
            return f"{mean} $\\pm$ {std}"
        else:
            return f"{mean:.{decimal_places}f} $\\pm$ {std:.{decimal_places}f}"
    
    # Start LaTeX table
    latex_str = ""
  
    for param, row in mean_std.iterrows():
        latex_str += f"        {param} &  "
        latex_str += " &  ".join([format_value(row[(col, 'mean')], row[(col, 'std')]) for col in accuracy_cols + dp_cols])
        latex_str += " \\\\ \n"
    
    latex_str += "\n"

    for param, row in mean_std.iterrows():
        latex_str += f"        {param} &  "
        latex_str += " &  ".join([format_value(row[(col, 'mean')], row[(col, 'std')], decimal_places=1) for col in tree_cols])
        latex_str += " \\\\ \n"

    print(latex_str)

def run_demo():
    print_latex_comparison()
    #mine_bpm("hospital_billing.xes", "hb")
    #ablation_experiment("bias", "ablation_bias", np.arange(0.6, 0.8, 0.1), ["time_delta", "age"], ["time_delta", "age"], num_cases=1000, folds=3)

def k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, parity_keys, folds=5, prefix_length=3, modify_mode="retrain", finetuning_mode=None, folder_name=None):
    results = []
    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, categorical_attributes)
    feature_names = create_feature_names(class_names, attribute_pools, numerical_attributes, prefix_length)
    feature_indices = create_feature_indices(class_names, attribute_pools, numerical_attributes, prefix_length)
    
    folds = k_fold_cross_validation(df, categorical_attributes, numerical_attributes, critical_decisions, prefix_length=3, k=folds)
    
    for i, (X_train, y_train, X_test, y_test, numerical_thresholds) in tqdm(enumerate(folds), desc="evaluating model:"):
        X_train_base = remove_attribute_features(X_train, feature_indices, base_attributes)
        X_test_base = remove_attribute_features(X_test, feature_indices, base_attributes)
        nn_base = train_nn(X_train_base, y_train)
        base_accuracy = evaluate_nn(nn_base, X_test_base, y_test)

        nn_enriched = train_nn(X_train, y_train)
        y_distilled = nn_enriched.predict(X_train)
        y_encoded = np.argmax(y_distilled, axis=1)
        
        dt_distilled = train_dt(X_train, y_encoded, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
        enriched_accuracy = evaluate_nn(nn_enriched, X_test, y_test)
        
        modified_tree_accuracy = evaluate_dt(dt_distilled, X_test, y_test)
        num_nodes = dt_distilled.count_nodes()
        nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
        removed_nodes = 0
        depth = get_max_depth(dt_distilled.root)
        
        y_distilled_tree = dt_distilled.predict(X_train)
        y_distilled_tree = to_categorical(y_distilled_tree, num_classes=len(dt_distilled.class_names))
        
        nn_modified = clone_model(nn_enriched)
        nn_modified.set_weights(nn_enriched.get_weights())
        
        if nodes_to_remove:
            if modify_mode == "retrain":
                y_encoded = np.argmax(y_train, axis=1)
                iterations = len(categorical_attributes + numerical_attributes) - len(base_attributes)
                for _ in range(iterations):
                    nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
                    print(f"Nodes to remove: {nodes_to_remove}")
                    if not nodes_to_remove:
                        break
                    removed_nodes += len(nodes_to_remove)
                    for node_id in nodes_to_remove:
                        dt_distilled.delete_node(X_train, y_encoded, node_id)
            else:
                print(f"Nodes to remove: {nodes_to_remove}")
                removed_nodes += len(nodes_to_remove)
                for node_id in nodes_to_remove:
                    dt_distilled.delete_branch(node_id)
            
            modified_tree_accuracy = evaluate_dt(dt_distilled, X_test, y_test)
            y_modified = dt_distilled.predict(X_train)
            y_modified = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))
            
            if finetuning_mode is not None:
                nn_modified = finetune_nn(nn_modified, X_train, y_modified, y_distilled=y_distilled, y_distilled_tree=y_distilled_tree, X_test=X_test, y_test=y_test, mode=finetuning_mode)
                best_mode = finetuning_mode
            else:
                nn_modified, best_mode = finetune_all(nn_modified, X_train, y_modified, y_distilled_tree, y_distilled, X_test, y_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds)
            modified_accuracy = evaluate_nn(nn_modified, X_test, y_test)
        else:
            modified_accuracy = enriched_accuracy
            best_mode = None
        
        stat_par_result, _ = calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds)
        
        row = {
            "fold_id": i,
            "base_accuracy": base_accuracy,
            "modified_accuracy": modified_accuracy,
            "enriched_accuracy": enriched_accuracy,
            "modified_tree_accuracy": modified_tree_accuracy,
            "num_nodes": num_nodes,
            "removed_nodes": removed_nodes,
            "depth": depth,
            "best_mode": best_mode
        }
        
        for outer_key, outer_value in stat_par_result.items():
            for inner_key, inner_value in outer_value.items():
                if outer_key in parity_keys:
                    row[f"dp_{outer_key}_{inner_key}"] = inner_value
        
        print(row)
        results.append(row)
        
    results_df = pd.DataFrame(results)
    if folder_name:
        save_data(results_df, folder_name, "results_df.pkl")
        run_results(folder_name)
    return results_df

def ablation_experiment(experiment_type, folder_name, num_range, numerical_attributes, base_attributes, bias=0.7, num_cases=10000, folds=5):
    all_results = []
    for param in num_range:
        if param == 0:
            param = 1
        print(f"Analyzing {experiment_type}: {param}")
        process_model = build_process_model(f"ablation_{experiment_type}")
        critical_decisions = []

        if experiment_type == "decisions":
            categorical_attributes = ["a_0"]
            parity_keys = [("a_0 = A", "A_0")]
            process_model.add_categorical_attribute("a_0", [("A", 0.5), ("B", 0.5)])
            process_model.add_activity("A_0")
            process_model.add_activity("B_0")
            process_model.add_activity("C_0")
            process_model.add_activity("D_0")
            
            conditions_top = {("a_0", "A"): bias, ("a_0", "B"): 1 - bias}
            conditions_bottom = {("a_0", "A"): 1 - bias, ("a_0", "B"): bias}
            process_model.add_transition("start", "A_0", conditions=conditions_top)
            process_model.add_transition("start", "B_0", conditions=conditions_bottom)
            process_model.add_transition("A_0", "C_0", conditions=conditions_top)
            process_model.add_transition("A_0", "D_0", conditions=conditions_bottom)
            process_model.add_transition("B_0", "C_0", conditions=conditions_top)
            process_model.add_transition("B_0", "D_0", conditions=conditions_bottom)

            critical_decisions.append(Decision(attributes=["a_0"], possible_events=["A_0", "B_0"], to_remove=True, previous="start"))
            
            for n in range(1, param):
                attr, prev_c, prev_d = f"a_{n}", f"C_{n-1}", f"D_{n-1}"
                conditions_top = {(attr, "A"): bias, (attr, "B"): 1 - bias}
                conditions_bottom = {(attr, "A"): 1 - bias, (attr, "B"): bias}
                process_model.add_categorical_attribute(attr, [("A", 0.5), ("B", 0.5)])
                categorical_attributes.append(attr)
                parity_keys.append((f"{attr} = A", f"A_{n}"))
                critical_decisions.append(Decision(attributes=[attr], possible_events=[f"A_{n}", f"B_{n}"], to_remove=True, previous=[prev_c, prev_d]))
                
                for act in ["A", "B", "C", "D"]:
                    process_model.add_activity(f"{act}_{n}")
                
                for prev in [prev_c, prev_d]:
                    process_model.add_transition(prev, f"A_{n}", conditions=conditions_top)
                    process_model.add_transition(prev, f"B_{n}", conditions=conditions_bottom)

                for act in ["A", "B"]:
                    process_model.add_transition(f"{act}_{n}", f"C_{n}", conditions=conditions_top)
                    process_model.add_transition(f"{act}_{n}", f"D_{n}", conditions=conditions_bottom)

            process_model.add_transition(f"C_{param-1}", "end", conditions={})
            process_model.add_transition(f"D_{param-1}", "end", conditions={})
        
        elif experiment_type == "attributes":
            categorical_attributes = []
            parity_keys = []
            for n in range(param):
                attr = f"a_{n}"
                process_model.add_categorical_attribute(attr, [("A", 0.5), ("B", 0.5)])
                categorical_attributes.append(attr)
                parity_keys.append((f"{attr} = A", "collect history"))
                critical_decisions.append(Decision(attributes=[attr], possible_events=["collect history", "refuse screening"], to_remove=True, previous="asses eligibility"))
                
                process_model.add_transition("asses eligibility", "collect history", conditions={(attr, "A"): bias, (attr, "B"): 1 - bias})
                process_model.add_transition("asses eligibility", "refuse screening", conditions={(attr, "A"): 1 - bias, (attr, "B"): bias})
                if n % 2 == 1:
                    bias_2 = 1 - bias
                else:
                    bias_2 = bias
                process_model.add_transition("collect history", "prostate screening", conditions={(attr, "A"): bias_2, (attr, "B"): 1 - bias_2})
                process_model.add_transition("collect history", "mammary screening", conditions={(attr, "A"): 1 - bias_2, (attr, "B"): bias_2})
        
        elif experiment_type == "bias":
            bias = param
            categorical_attributes = ["gender"]
            parity_keys = [("gender = male", "collect history")]
            critical_decisions.append(Decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True, previous="asses eligibility"))
            process_model.add_transition("asses eligibility", "collect history", conditions={("gender", "male"): bias, ("gender", "female"): 1 - bias})
            process_model.add_transition("asses eligibility", "refuse screening", conditions={("gender", "male"): 1 - bias, ("gender", "female"): bias})
        
        trace_generator = TraceGenerator(process_model=process_model)
        df = cases_to_dataframe(trace_generator.generate_traces(num_cases=num_cases))
        results_df = k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, parity_keys, folds=folds, prefix_length=3, modify_mode="cut", folder_name=None)
        if experiment_type == "decisions":
            results_df["param"] = param * 2
        else:
            results_df["param"] = param
        all_results.append(results_df)

    final_results_df = pd.concat(all_results, ignore_index=True)
    dp_base_cols = [col for col in final_results_df.columns if col.startswith("dp_") and col.endswith("_base")]
    dp_modified_cols = [col for col in final_results_df.columns if col.startswith("dp_") and col.endswith("_modified")]
    dp_enriched_cols = [col for col in final_results_df.columns if col.startswith("dp_") and col.endswith("_enriched")]
    if dp_base_cols:
        final_results_df["dp_base"] = final_results_df[dp_base_cols].mean(axis=1)
    if dp_modified_cols:
        final_results_df["dp_modified"] = final_results_df[dp_modified_cols].mean(axis=1)
    if dp_enriched_cols:
        final_results_df["dp_enriched"] = final_results_df[dp_enriched_cols].mean(axis=1)
    final_results_df = final_results_df.drop(columns=dp_base_cols + dp_modified_cols + dp_enriched_cols)
    save_data(final_results_df, folder_name, "results_df.pkl")
    run_results(folder_name)



def run_evaluation(folder_name, folds=5, prefix_length=3):
    if folder_name is None:
        folder_names = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi_2012"]
    else:
        folder_names = [folder_name]

    for folder_name in folder_names:
        df = load_data(folder_name, "df.pkl")
        categorical_attributes, numerical_attributes = get_attributes(folder_name)
        categorical_attributes_base, numerical_attributes_base = get_base_attributes(folder_name)
        base_attributes = categorical_attributes_base + numerical_attributes_base
        critical_decisions = get_critical_decisions(folder_name)
        parity_keys = get_parity_key(folder_name)
        k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, parity_keys, folds=folds, prefix_length=prefix_length, folder_name=folder_name)

def run_ablation(folder_name):
    if folder_name is None:
        folder_names = ["ablation_bias", "ablation_attributes", "ablation_decisions"]
    else:
        folder_names = [folder_name]

    for folder_name in folder_names:
        if folder_name == "ablation_bias":
            ablation_experiment("bias", "ablation_bias", np.arange(0.5, 1.05, 0.05), ["time_delta", "age"], ["time_delta", "age"])
        elif folder_name == "ablation_attributes":
            ablation_experiment("attributes", "ablation_attributes", np.arange(0, 11, 2), ["time_delta", "age"], ["time_delta", "age"])
        elif folder_name == "ablation_decisions":
            ablation_experiment("decisions", "ablation_decisions", np.arange(0, 11, 2), ["time_delta"], ["time_delta"])

def run_load(folder_name):
    if folder_name is None:
        folder_names = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi_2012"]
    else:
        folder_names = [folder_name]

    for folder_name in folder_names:
        if "cs" in folder_name:
            process_model = build_process_model("cs")
            trace_generator = TraceGenerator(process_model=process_model)
            cases = trace_generator.generate_traces(num_cases=10000)
            df = cases_to_dataframe(cases)
            print(df.head(20))
            save_data(df, "cs", "df.pkl")
        elif "hb" in folder_name:
            df = load_xes_to_df("hospital_billing.xes", num_cases=20000)
            rules = get_rules(folder_name)
            df = enrich_df(df, rules, folder_name)
            print(df.head(20))
            save_data(df, folder_name, "df.pkl")
        elif "bpi" in folder_name:
            df = load_xes_to_df("bpi_2012.xes")
            df = df[df['activity'].str.startswith('A_')]
            rules = get_rules(folder_name)
            df = enrich_df(df, rules, folder_name)
            print(df.head(20))
            save_data(df, folder_name, "df.pkl")

def run_results(folder_name):
    if folder_name is None:
        folder_names = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi_2012", "ablation_bias", "ablation_attributes", "ablation_decisions"]
    else:
        folder_names = [folder_name]

    for folder_name in folder_names:
        print(f"\nResults for {folder_name}:")
        df = load_data(folder_name, "results_df.pkl")
        with pd.option_context('display.max_columns', None):
            print(df)
        if "param" in df.columns:
            #process_and_plot_ablation(df, folder_name)
            numeric_columns = df.drop(columns=["fold_id", "param"], errors="ignore").select_dtypes(include=[np.number]).columns
            for param_value, group in df.groupby("param"):
                print(f"\nStatistics for param = {param_value}:")
                for col in numeric_columns:
                    mean_val = group[col].mean()
                    std_val = group[col].std()
                    print(f"{col}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")
        else:
            #plot_metrics(df, folder_name)
            numeric_columns = df.drop(columns=["fold_id"], errors="ignore").select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"{col}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")
        print("--------------------------------------------------------------------------------------------------")

def run_plot():
    plot_all()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, default=None, help='Name of the folder to use')
    parser.add_argument('--mode', choices=['a', 'd', 'e', 'l', 'p', 'r'], default=None, help='Name of the mode to use')
    parser.add_argument('--prefix_length', type=int, default=3, help='Value for n-gram (default: 3)')
    parser.add_argument('--folds', type=int, default=5, help='Value for folds (default: 5)')
    parser.add_argument('--num_cases', type=int, default=1000, help='Number of cases to process (default: 1000)')

    args = parser.parse_args()
    experiments = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi_2012"]
    ablations = ["ablation_bias", "ablation_attributes", "ablation_decisions"]

    #print pwd
    print(f"Current working directory: {os.getcwd()}")

    

if __name__ == "__main__":
    main()
    print("Done and dusted!")