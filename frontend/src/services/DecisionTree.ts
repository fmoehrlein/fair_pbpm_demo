export default interface DecisionTree {
    root: DecisionNode;
    id_counter: number;
    feature_names: string[];
    class_names: string[];
    feature_indices: {[feature_name: string]: number[]};
}

export interface DecisionNode {
    node_id: number;
    feature_index: number | null;
    threshold: number | null;
    num_samples: number;
    removed_features: number[];
    left: DecisionNode | null;
    right: DecisionNode | null;
    output: number | null;
    depth: number;
}