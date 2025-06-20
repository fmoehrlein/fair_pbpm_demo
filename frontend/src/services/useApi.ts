import DecisionTree from "./DecisionTree";
import {NotificationInstance} from "antd/es/notification/interface";
import {notification} from "antd";


export interface TrainParams {
    prefix_length: number;
    cat_attributes: string[];
    num_attributes: string[];
    sensitive_attributes: string[];
    test_split: number;
    epochs: number;
    learning_rate: number;
    hidden_units: number[];
}

export interface FineTuneParams {
    mode: 'changed_complete';
    learning_rate: number;
    epoch: number;
    batch_size: number;
}

export interface DistillParams {
    min_samples_split: number;
    max_depth: number;
    ccp_alpha: number;
    model_to_use: 'original' | 'latest';
}

export interface RetrainNodeParams {
    node_id: number;
}

export interface CutNodeParams {
    node_id: number;
    direction: 'auto' | 'left' | 'right';
}

export interface ModifyResult {
    nn_evaluation: Metrics;
    dt_evaluation: Metrics;
}

export interface Metrics {
    accuracy: number;
    f1_score: number;
    precision: number;
    recall: number;
}

export interface TrainResult {
    nn_evaluation: Metrics;
}

export interface DistillResult {
    nn_evaluation: Metrics;
    dt_evaluation: Metrics;
}

export interface FineTuneResult {
    nn_evaluation: Metrics;
    dt_evaluation: Metrics;
    nn_modified_evaluation: Metrics;
}

export interface UploadResult {
    attributes: string[];
    events_per_case: number;
    num_cases: number;
    num_events: number;
}

const useApi = (baseUrl: string, notifications: NotificationInstance) => {
    const notifyBadStatus = (status: number, message: string) => {
        console.log("should open a new thingy")
        notifications.error({
            message: `Status ${status}`,
            description: message,
        });
    }

    const fetchTree = async (sessionId: string): Promise<DecisionTree | undefined> => {
        try {
            const response = await fetch(`${baseUrl}/api/load_tree`, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    folder_name: sessionId
                })
            });

            const content = await response.json();
            if(!response.ok) {
                notifyBadStatus(response.status, content.error);
                return undefined;
            }

            return content.tree;
        } catch (e) {
            console.error(e);
            if (e instanceof Error) {
                notifications.error({
                    message: e.message,
                });
            }
            return undefined;
        }
    }

    const uploadXes = async (sessionId: string, xesFile: File | undefined): Promise<UploadResult | undefined> => {
        const formData = new FormData();
        if(xesFile) formData.append("file", xesFile);
        formData.append("folder_name", sessionId);

        try {
            const response = await fetch(`${baseUrl}/api/load_xes`, {
                method: 'POST',
                body: formData
            });
            const content = await response.json();
            if(!response.ok) {
                notifyBadStatus(response.status, content.error);
                return undefined;
            }
            return content;
        } catch (e) {
            console.error(e);
            if (e instanceof Error) {
                notifications.error({
                    message: e.message,
                });
            }
            return undefined;
        }
    }

    const trainModel = async (sessionId: string, params: TrainParams): Promise<TrainResult | undefined> => {
        const res = await genericAction(sessionId, "process_and_train", params);
        if(res) return res as TrainResult;
        return undefined;
    }

    const distillTree = async (sessionId: string, params: DistillParams): Promise<DistillResult | undefined> => {
        const res = await genericAction(sessionId, "distill_tree", params);
        if(res) return res as DistillResult;
        return undefined;
    }

    const cutNode = async (sessionId: string, params: CutNodeParams): Promise<ModifyResult | undefined> => {
        const res = await genericAction(sessionId, "modify", params);
        if(res) return res as ModifyResult;
        return undefined;
    }

    const retrainNode = async (sessionId: string, params: RetrainNodeParams): Promise<ModifyResult | undefined> => {
        const res = await genericAction(sessionId, "modify", params);
        if(res) return res as ModifyResult;
        return undefined;
    }

    const fineTune = async (sessionId: string, params: FineTuneParams): Promise<FineTuneResult | undefined> => {
        const res = await genericAction(sessionId, "finetune", params);
        if(res) return res as FineTuneResult;
        return undefined;
    }

    const genericAction = async (sessionId: string, action: string, params: object): Promise<{[p: string]: any} | undefined> => {
        try {
            const response = await fetch(`${baseUrl}/api/${action}`, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    folder_name: sessionId,
                    ...params,
                })
            });
            const content = await response.json();
            if(!response.ok) {
                notifyBadStatus(response.status, content.error);
                return undefined;
            }
            return content;
        } catch (e) {
            if (e instanceof Error) {
                notifications.error({
                    message: e.name,
                    description: e.message,
                });
            }
            console.error(e);
            return undefined;
        }
    }

    return {
        fetchTree, uploadXes, distillTree, trainModel, cutNode, retrainNode, fineTune,
    }
}


export default useApi;
