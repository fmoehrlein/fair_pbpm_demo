print("Starting importing necessary modules...")
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import time
from datetime import datetime, timedelta
import threading

from utils import *
from cleanup import start_cleanup_thread

app = Flask(__name__)
CORS(app)

@app.route("/api/hello_world", methods=["POST"])
def hello_world():
    data = request.json
    name = data.get("name", "World")
    return jsonify({"message": f"Hello, {name}!"})

@app.route("/api/load_xes", methods=["POST"])
def load_xes():
    file = request.files["file"]
    folder_name = request.form["folder_name"]
    save_path = os.path.join("data", folder_name)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "event_log.xes")
    file.save(file_path)
    df = load_xes_to_df("event_log.xes", folder_name)
    columns = df.columns.tolist()
    columns.remove("case_id")
    columns.remove("activity")
    columns.remove("time:timestamp")

    stats = {
        "num_cases": df["case_id"].nunique(),
        "num_events": len(df),
        "events_per_case": df.groupby("case_id").size().mean(),
        "attributes": columns,
    }
    return jsonify(stats)


@app.route("/api/process_and_train", methods=["POST"])
def process_and_train():
    data = request.json
    folder_name = data.get("folder_name")
    df = load_data(folder_name, "event_log_df.pkl")

    # --- Step 1: Process Data ---
    cat_attributes = data.get("cat_attributes", [])
    num_attributes = data.get("num_attributes", [])
    test_size = data.get("test_split", 0.3)
    prefix_length = data.get("prefix_length", 3)
    shuffle = data.get("shuffle", False)

    X_train, y_train, X_test, y_test = train_test_split_encoding(
        df,
        categorical_attributes=cat_attributes,
        numerical_attributes=num_attributes,
        test_size=test_size,
        prefix_length=prefix_length,
        shuffle=shuffle,
    )

    save_data(X_train, folder_name, "X_train.pkl")
    save_data(y_train, folder_name, "y_train.pkl")
    save_data(X_test, folder_name, "X_test.pkl")
    save_data(y_test, folder_name, "y_test.pkl")

    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, cat_attributes)
    feature_names = create_feature_names(
        class_names, attribute_pools, num_attributes, prefix_length
    )
    feature_indices = create_feature_indices(
        class_names, attribute_pools, num_attributes, prefix_length
    )

    save_data(class_names, folder_name, "class_names.pkl")
    save_data(feature_names, folder_name, "feature_names.pkl")
    save_data(feature_indices, folder_name, "feature_indices.pkl")

    # --- Step 2: Train Model ---
    hidden_units = data.get("hidden_units", [512, 256, 128, 64])
    epochs = data.get("epochs", 5)
    learning_rate = data.get("learning_rate", 0.001)
    batch_size = data.get("batch_size", 32)

    model = train_nn(
        X_train,
        y_train,
        folder_name=folder_name,
        model_name="nn.keras",
        hidden_units=hidden_units,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    y_pred = model.predict(X_test)
    nn_evaluation = calculate_metrics(y_test, y_pred)
    save_json(nn_evaluation, folder_name, "nn_evaluation.json")

    return jsonify(
        {
            "status": "data processed and model trained",
            "nn_evaluation": nn_evaluation,
            "params": data,
        }
    )


@app.route("/api/distill_tree", methods=["POST"])
def distill_tree():
    data = request.json
    folder_name = data.get("folder_name")
    ccp_alpha = data.get("ccp_alpha", 0.001)
    max_depth = data.get("max_depth", None)
    min_samples_split = data.get("min_samples_split", 2)
    min_samples_leaf = data.get("min_samples_leaf", 1)
    model_to_use = data.get("model_to_use", "original")

    if model_to_use == "original":
        nn = load_nn(folder_name, "nn.keras")
        nn_evaluation = load_json(folder_name, "nn_evaluation.json")
    else:
        nn = load_nn(folder_name, "nn_modified.keras")
        nn_evaluation = load_json(folder_name, "nn_modified_evaluation.json")
    X_train = load_data(folder_name, "X_train.pkl")
    X_test = load_data(folder_name, "X_test.pkl")
    y_test = load_data(folder_name, "y_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")

    y_distilled = nn.predict(X_train)
    y_encoded = np.argmax(y_distilled, axis=1)
    dt_distilled = train_dt(
        X_train,
        y_encoded,
        class_names=class_names,
        feature_names=feature_names,
        feature_indices=feature_indices,
        ccp_alpha=ccp_alpha,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    y_distilled_tree = dt_distilled.predict(X_train)
    y_distilled_tree = to_categorical(
        y_distilled_tree, num_classes=len(dt_distilled.class_names)
    )

    tree_json = save_dt(dt_distilled, folder_name, "tree.json")
    save_data(y_distilled, folder_name, "y_distilled.pkl")
    save_data(y_distilled_tree, folder_name, "y_distilled_tree.pkl")

    y_pred = dt_distilled.predict(X_test)
    dt_evaluation = calculate_metrics(y_test, y_pred)
    save_json(dt_evaluation, folder_name, "dt_evaluation.json")

    return jsonify(
        {
            "status": "tree distilled",
            "dt_evaluation": dt_evaluation,
            "nn_evaluation": nn_evaluation,
            "tree": tree_json,
            "params": data,
        }
    )


@app.route("/api/load_tree", methods=["POST"])
def load_tree():
    data = request.json
    folder_name = data.get("folder_name")
    dt_path = os.path.join("models", folder_name, "tree.json")
    try:
        with open(dt_path, "r") as f:
            tree = json.load(f)
        dt_evaluation = load_json(folder_name, "dt_evaluation.json")
        return jsonify(
            {
                "status": "tree loaded",
                "dt_evaluation": dt_evaluation,
                "tree": tree,
                "params": data,
            }
        )
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": f"File not found: {dt_path}",
                    "status": "failed",
                    "params": data,
                }
            ),
            404,
        )
    except json.JSONDecodeError:
        return (
            jsonify(
                {
                    "error": f"Invalid JSON in file: {dt_path}",
                    "status": "failed",
                    "params": data,
                }
            ),
            400,
        )


@app.route("/api/modify_node", methods=["POST"])
def modify_node():
    data = request.json
    folder_name = data.get("folder_name")
    node_id = data.get("node_id")
    threshold = data.get("threshold", None)
    feature_index = data.get("feature_index", None)

    tree = load_dt(folder_name, "tree.json")
    tree.modify_node(node_id, threshold=threshold, feature_index=feature_index)
    tree_json = save_dt(tree, folder_name, "tree.json")

    return jsonify({"status": "node modified", "params": data, "tree": tree_json})


@app.route("/api/modify_retrain", methods=["POST"])
def modify_retrain():
    data = request.json
    folder_name = data.get("folder_name")
    node_id = data.get("node_id")

    X_train = load_data(folder_name, "X_train.pkl")
    y_train = load_data(folder_name, "y_train.pkl")
    y_encoded = np.argmax(y_train, axis=1)
    X_test = load_data(folder_name, "X_test.pkl")
    y_test = load_data(folder_name, "y_test.pkl")
    nn_evaluation = load_json(folder_name, "nn_evaluation.json")

    tree = load_dt(folder_name, "tree.json")
    tree.delete_node(X_train, y_encoded, node_id)
    y_pred = tree.predict(X_test)
    dt_evaluation = calculate_metrics(y_test, y_pred)

    save_json(dt_evaluation, folder_name, "dt_evaluation.json")
    tree_json = save_dt(tree, folder_name, "tree.json")

    return jsonify(
        {
            "status": "subtree retrained",
            "nn_evaluation": nn_evaluation,
            "dt_evaluation": dt_evaluation,
            "tree": tree_json,
            "params": data,
        }
    )


@app.route("/api/modify_cut", methods=["POST"])
def modify_cut():
    data = request.json
    folder_name = data.get("folder_name")
    node_id = data.get("node_id")
    direction = data.get("direction", "auto")

    X_test = load_data(folder_name, "X_test.pkl")
    y_test = load_data(folder_name, "y_test.pkl")
    nn_evaluation = load_json(folder_name, "nn_evaluation.json")

    tree = load_dt(folder_name, "tree.json")
    tree.delete_branch(node_id, direction)
    y_pred = tree.predict(X_test)
    dt_evaluation = calculate_metrics(y_test, y_pred)

    save_json(dt_evaluation, folder_name, "dt_evaluation.json")
    tree_json = save_dt(tree, folder_name, "tree.json")

    return jsonify(
        {
            "status": "branch cut",
            "nn_evaluation": nn_evaluation,
            "dt_evaluation": dt_evaluation,
            "tree": tree_json,
            "params": data,
        }
    )


@app.route("/api/finetune", methods=["POST"])
def finetune():
    data = request.json
    folder_name = data.get("folder_name")
    finetuning_mode = data.get("finetuning_mode", "changed_complete")
    epochs = data.get("epochs", 3)
    learning_rate = data.get("learning_rate", 0.001)
    batch_size = data.get("batch_size", 32)

    X_train = load_data(folder_name, "X_train.pkl")
    X_test = load_data(folder_name, "X_test.pkl")
    y_test = load_data(folder_name, "y_test.pkl")
    y_distilled = load_data(folder_name, "y_distilled.pkl")
    y_distilled_tree = load_data(folder_name, "y_distilled_tree.pkl")
    nn = load_nn(folder_name, "nn.keras")
    dt_distilled = load_dt(folder_name, "tree.json")
    nn_evaluation = load_json(folder_name, "nn_evaluation.json")
    dt_evaluation = load_json(folder_name, "dt_evaluation.json")

    y_modified = dt_distilled.predict(X_train)
    y_modified = to_categorical(
        y_modified, num_classes=len(dt_distilled.class_names)
    )
    nn_modified = finetune_nn(
        nn,
        X_train,
        y_modified,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        y_distilled=y_distilled,
        y_distilled_tree=y_distilled_tree,
        X_test=X_test,
        y_test=y_test,
        mode=finetuning_mode,
    )
    save_nn(nn_modified, folder_name, "nn_modified.keras")
    y_pred = nn_modified.predict(X_test)
    nn_modified_evaluation = calculate_metrics(y_test, y_pred)
    save_json(nn_modified_evaluation, folder_name, "nn_modified_evaluation.json")
    return jsonify(
        {
            "status": "model fine-tuned",
            "nn_evaluation": nn_evaluation,
            "dt_evaluation": dt_evaluation,
            "nn_modified_evaluation": nn_modified_evaluation,
            "params": data,
        }
    )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Starting cleanup thread...")
    #start_cleanup_thread()
    print("Starting backend...")
    app.run(debug=True)
