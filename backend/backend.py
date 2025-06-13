from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd

from utils import *

app = Flask(__name__)
CORS(app)


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello World"})


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


@app.route("/api/process_data", methods=["POST"])
def process_data():
    data = request.json
    folder_name = data.get("folder_name")
    df = load_data(folder_name, "event_log_df.pkl")

    cat_attributes = data.get("cat_attributes", [])
    num_attributes = data.get("num_attributes", [])
    test_size = data.get("test_split", 0.3)  # default 0.3
    prefix_length = data.get("prefix_length", 3)  # if you want to allow override
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

    return jsonify({"status": "processed", "params": data})


@app.route("/api/train_model", methods=["POST"])
def train_model():
    data = request.json
    folder_name = data.get("folder_name")
    X_train = load_data(folder_name, "X_train.pkl")
    y_train = load_data(folder_name, "y_train.pkl")
    X_test = load_data(folder_name, "X_test.pkl")
    y_test = load_data(folder_name, "y_test.pkl")

    hidden_units = data.get("hidden_units", [512, 256, 128, 64])  # expect list
    epochs = data.get("epochs", 10)
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
    accuracy = evaluate_nn(model, X_test, y_test)

    return jsonify(
        {
            "status": "model trained",
            "params": data,
            "accuracy": accuracy,
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

    nn = load_nn(folder_name, "nn.keras")
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
        folder_name=folder_name,
        model_name="tree.json",
        class_names=class_names,
        feature_names=feature_names,
        feature_indices=feature_indices,
        ccp_alpha=ccp_alpha,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    y_distilled_tree = dt_distilled.predict(X_train)
    y_distilled_tree = to_categorical(y_distilled_tree, num_classes=len(dt_distilled.class_names))

    accuracy = evaluate_dt(dt_distilled, X_test, y_test)
    save_data(y_distilled, folder_name, "y_distilled.pkl")
    save_data(y_distilled_tree, folder_name, "y_distilled_tree.pkl")

    return jsonify({"accuracy": accuracy, "status": "tree distilled", "params": data})


@app.route("/api/load_tree", methods=["POST"])
def load_tree():
    data = request.json
    folder_name = data.get("folder_name")
    dt_path = os.path.join("models", folder_name, "tree.json")
    try:
        with open(dt_path, "r") as f:
            tree = json.load(f)
        return jsonify({
            "tree": tree,
            "status": "tree loaded",
            "params": data
        })
    except FileNotFoundError:
        return jsonify({
            "error": f"File not found: {dt_path}",
            "status": "failed",
            "params": data
        }), 404
    except json.JSONDecodeError:
        return jsonify({
            "error": f"Invalid JSON in file: {dt_path}",
            "status": "failed",
            "params": data
        }), 400


@app.route("/api/modify_node", methods=["POST"])
#TODO
def modify_node():
    data = request.json
    return jsonify({"status": "node modified", "params": data})


@app.route("/api/modify_retrain", methods=["POST"])
def modify_retrain():
    data = request.json
    folder_name = data.get("folder_name")
    node_id = data.get("node_id")

    X_train = load_data(folder_name, "X_train.pkl")
    y_train = load_data(folder_name, "y_train.pkl")
    tree = load_dt(folder_name, "tree.json")
    tree.delete_node(X_train, y_train, tree, node_id)
    save_dt(tree, folder_name, "tree.json")

    return jsonify({"status": "subtree retrained", "params": data})


@app.route("/api/modify_cut", methods=["POST"])
def modify_cut():
    data = request.json
    folder_name = data.get("folder_name")
    node_id = data.get("node_id")
    direction = data.get("direction", None)

    tree = load_dt(folder_name, "tree.json")
    tree.delete_branch(node_id, direction)
    save_dt(tree, folder_name, "tree.json")

    return jsonify({"status": "branch cut", "params": data})


@app.route("/api/finetune", methods=["POST"])
def finetune():
    data = request.json
    folder_name = data.get("folder_name")
    finetuning_mode = data.get("finetuning_mode", "changed_complete")

    X_train = load_data(folder_name, "X_train.pkl")
    X_test = load_data(folder_name, "X_test.pkl")
    y_test = load_data(folder_name, "y_test.pkl")
    y_distilled = load_data(folder_name, "y_distilled.pkl")
    y_distilled_tree = load_data(folder_name, "y_distilled_tree.pkl")

    nn = load_nn(folder_name, "nn.keras")
    dt_distilled = load_dt(folder_name, "tree.json")
    y_modified = dt_distilled.predict(X_train)
    y_modified = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))

    nn_modified = finetune_nn(nn, X_train, y_modified, y_distilled=y_distilled, y_distilled_tree=y_distilled_tree, X_test=X_test, y_test=y_test, mode=finetuning_mode)
    accuracy = evaluate_nn(nn_modified, X_test, y_test)
    return jsonify({"accuracy": accuracy, "status": "model fine-tuned", "params": data})


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Starting backend...")
    app.run(debug=True)
