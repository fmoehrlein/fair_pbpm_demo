import DecisionTree from "./DecisionTree";


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
}

export interface RetrainNodeParams {
    node_id: number;
}

export interface CutNodeParams {
    node_id: number;
    direction: 'auto' | 'left' | 'right';
}

export interface ModifyResult {}

export interface TrainResult {}

export interface DistillResult {}

export interface FineTuneResult {}

export interface UploadResult {
    attributes: string[];
    events_per_case: number;
    num_cases: number;
    num_events: number;
}

export default class Api {
    private readonly baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    public fetchTree = async (sessionId: string): Promise<DecisionTree | undefined> => {
        try {
            const response = await fetch(`${this.baseUrl}/api/load_tree`, {
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

            if (content.status !== "tree loaded") {
                return undefined;
            }

            return content.tree;
        } catch (e) {
            console.error(e);
            return undefined;
        }
    }

    public uploadXes = async (sessionId: string, xesFile: File): Promise<UploadResult> => {
        const formData = new FormData();
        formData.append("file", xesFile);
        formData.append("folder_name", sessionId);

        const res = await fetch(`${this.baseUrl}/api/load_xes`, {
            method: 'POST',
            body: formData
        });
        return await res.json();
    }

    public trainModel = async (sessionId: string, params: TrainParams): Promise<TrainResult | undefined> => {
        const res = await this.genericAction(sessionId, "process_and_train", params);
        if(res) return res["nn_evaluation"];
        return undefined;
    }

    public distillTree = async (sessionId: string, params: DistillParams): Promise<DistillResult | undefined> => {
        const res = await this.genericAction(sessionId, "distill_tree", params);
        if(res) return res["dt_evaluation"];
        return undefined;
    }

    public cutNode = async (sessionId: string, params: CutNodeParams): Promise<ModifyResult | undefined> => {
        return this.genericAction(sessionId, "modify", params);
    }

    public retrainNode = async (sessionId: string, params: RetrainNodeParams): Promise<ModifyResult | undefined> => {
        return this.genericAction(sessionId, "modify", params);
    }

    public fineTune = async (sessionId: string, params: FineTuneParams): Promise<FineTuneResult | undefined> => {
        return this.genericAction(sessionId, "finetune", params);
    }

    private genericAction = async (sessionId: string, action: string, params: object): Promise<{[p: string]: any} | undefined> => {
        try {
            const response = await fetch(`${this.baseUrl}/api/${action}`, {
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
            const result = await response.json();
            console.log(result)
            return result;
        } catch (e) {
            console.error(e);
            return undefined;
        }
    }
}

