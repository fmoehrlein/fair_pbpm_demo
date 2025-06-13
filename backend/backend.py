from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd

from utils import *

app = Flask(__name__)
CORS(app)

# In-memory storage for event logs
event_logs = {}

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello World"})

@app.route("/api/load_xes", methods=["POST"])
def load_xes():
    file = request.files['file']
    folder_name = request.form['folder_name']
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
    return jsonify({
        "status": "processed",
        "params": data
    })

@app.route("/api/train_model", methods=["POST"])
def train_model():
    data = request.json
    return jsonify({
        "status": "model trained",
        "params": data
    })

@app.route("/api/distill_tree", methods=["POST"])
def distill_tree():
    data = request.json
    return jsonify({
        "status": "tree distilled",
        "params": data
    })

@app.route("/api/modify_node", methods=["POST"])
def modify_node():
    data = request.json
    return jsonify({
        "status": "node modified",
        "params": data
    })

@app.route("/api/modify_retrain", methods=["POST"])
def modify_retrain():
    data = request.json
    return jsonify({
        "status": "model retrained after node modification",
        "params": data
    })

@app.route("/api/modify_cut", methods=["POST"])
def modify_cut():
    data = request.json
    return jsonify({
        "status": "cut operation performed",
        "params": data
    })

@app.route("/api/finetune", methods=["POST"])
def finetune():
    data = request.json
    return jsonify({
        "status": "model fine-tuned",
        "params": data
    })

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Starting backend...")
    app.run(debug=True)
