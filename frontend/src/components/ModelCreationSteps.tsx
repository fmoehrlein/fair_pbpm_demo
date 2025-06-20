import React, {ReactElement} from 'react';
import {Button, Divider, Modal, notification, Space, StepProps, Steps, Table, Tag} from "antd";
import XesUpload from "./XesUpload";
import useApi, {
    DistillParams,
    DistillResult,
    FineTuneParams,
    FineTuneResult,
    TrainParams,
    TrainResult,
    UploadResult
} from "../services/useApi";
import TrainParamsInput from "./TrainParamsInput";
import DistillParamsInput from "./DistillParamsInput";
import DecisionTree, {DecisionNode} from "../services/DecisionTree";
import DecisionTreeDisplay from "./DecisionTreeDisplay";
import FineTuneControls from "./FineTuneControls";
import {LoadingOutlined} from "@ant-design/icons";
import MetricsTable from "./MetricsTable";


interface Props {
    sessionId: string;
}


const Context = React.createContext({ name: 'AntDesignNotifications' });

const ModelCreationSteps = ({sessionId}: Props) => {
    const [notifications, contextHolder] = notification.useNotification();
    const contextValue = React.useMemo(() => ({ name: 'Ant Design' }), []);

    const api = useApi("https://apps.pm.iisys.de", notifications);

    const [current, setCurrent] = React.useState(0);
    const [working, setWorking] = React.useState(false);

    const [resultModalOpen, setResultModalOpen] = React.useState(false);
    const [fineTuneResults, setFineTuneResults] = React.useState<FineTuneResult | undefined>(undefined);

    const [xesFile, setXesFile] = React.useState<File | undefined>(undefined);
    const [trainParams, setTrainParams] = React.useState<TrainParams>({
        prefix_length: 3,
        cat_attributes: ['gender'],
        num_attributes: ['age'],
        sensitive_attributes: ['gender', 'age'],
        test_split: 0.3,
        epochs: 5,
        learning_rate: 0.001,
        hidden_units: [512, 256, 128, 64, 32]
    });
    const [distillParams, setDistillParams] = React.useState<DistillParams>({
        min_samples_split: 2,
        max_depth: 100,
        ccp_alpha: 0.001,
        model_to_use: 'original'
    });
    const [fineDistillParams, setFineDistillParams] = React.useState<DistillParams>({
        min_samples_split: 2,
        max_depth: 100,
        ccp_alpha: 0.001,
        model_to_use: 'latest'
    });

    const [trainResult, setTrainResult] = React.useState<TrainResult | undefined>(undefined);
    const [distillResult, setDistillResult] = React.useState<DistillResult | undefined>(undefined);
    const [uploadResult, setUploadResult] = React.useState<UploadResult | undefined>(undefined);

    const [selectedNode, setSelectedNode] = React.useState<DecisionNode | undefined>(undefined);

    const [tree, setTree] = React.useState<DecisionTree | undefined>(undefined);

    const TREE_VIEW = 6;
    const DISTILL_RESULT_VIEW = 5;

    React.useEffect(() => {
        const load = async () => {
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
            setCurrent(TREE_VIEW);
        }

        load();
    }, []);

    const next = () => {
        setCurrent(current + 1);
    }

    const cutNode = async (node: DecisionNode, keep: "left" | "right" | "auto") => {
        if (!sessionId) {
            throw new Error("Should not happen.")
        }

        setWorking(true);
        try {
            await api.cutNode(sessionId, {node_id: node.node_id, direction: keep});
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
        } finally {
            setSelectedNode(undefined);
            setWorking(false);
        }
    }

    const retrainNode = async (node: DecisionNode) => {
        if (!sessionId) {
            throw new Error("Should not happen.")
        }

        setWorking(true);
        try {
            await api.retrainNode(sessionId, {node_id: node.node_id});
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
        } finally {
            setSelectedNode(undefined);
            setWorking(false);
        }
    }

    const applyAlterations = async (ftParams: FineTuneParams) => {
        if (!sessionId) {
            throw new Error("Should not happen.")
        }

        setWorking(true);
        try {
            const fineTuneResults = await api.fineTune(sessionId, ftParams);
            if (!fineTuneResults) {
                throw new Error("Should not happen.");
            }
            setFineTuneResults(fineTuneResults);
            setResultModalOpen(true);
        } finally {
            setWorking(false);
        }
    }

    const keepNewModel = async ()=> {
        setWorking(true);
        try {
            await distill(fineDistillParams);
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
            setResultModalOpen(false);
        } finally {
            setWorking(false);
        }
    }

    const revertFineTune = async () => {
        setWorking(true);
        try {
            await distill(distillParams);
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
            setResultModalOpen(false);
        } finally {
            setWorking(false);
        }
    }

    const steps: (StepProps & {content: ReactElement})[] = [
        {
            title: "Upload",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <Divider>Upload an event log in .xes format...</Divider>
                    <XesUpload onFileSelected={setXesFile}/>
                    <div style={{textAlign: "center"}}>
                        <Button
                            type={"primary"}
                            disabled={!xesFile}
                            onClick={() => upload()}
                            loading={working}
                        >
                            Upload
                        </Button>
                    </div>
                    <Divider>... or use our example from the paper</Divider>
                    <div style={{textAlign: "center"}}>
                        <Button
                            type={"primary"}
                            onClick={() => upload()}
                            loading={working}
                        >
                            Use Cancer Screening Example
                        </Button>
                    </div>
                </div>
            ),
        }, {
            title: "Event Log Statistics",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <Table
                        columns={[
                            {
                                title: "Attributes",
                                dataIndex: "attributes",
                                render: (value: string[]) => value?.map(v => <Tag>{v}</Tag>)
                            },
                            {title: "avg. Events / Case", dataIndex: "events_per_case"},
                            {title: "Cases", dataIndex: "num_cases"},
                            {title: "Events", dataIndex: "num_events"},
                        ]}
                        dataSource={uploadResult ? [uploadResult] : undefined}
                        pagination={false}
                    />
                    <Button onClick={next} type={"primary"}>Next</Button>
                </div>
            ),
            disabled: uploadResult === undefined
        }, {
            title: "Train",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <TrainParamsInput value={trainParams} onChange={setTrainParams}/>
                    <Button type={"primary"} onClick={() => train()} loading={working}>Train</Button>
                </div>
            )
        }, {
            title: "Training Results",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <MetricsTable data={trainResult? [{...trainResult.nn_evaluation, model: "Neural Network"}]: undefined} />
                    <Button onClick={next} type={"primary"}>Next</Button>
                </div>
            ),
            disabled: trainResult === undefined
        }, {
            title: "Distill",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <DistillParamsInput value={distillParams} onChange={setDistillParams}/>
                    <Button type={"primary"} onClick={() => distill(distillParams)} loading={working}>
                        Distill
                    </Button>
                </div>
            )
        }, {
            title: "Distillation Results",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <MetricsTable
                        data={
                            distillResult ? [
                                {...distillResult.nn_evaluation, model: "Neural Network"},
                                {...distillResult.dt_evaluation, model: "Decision Tree"}
                            ]: undefined
                        }
                    />
                    <Button onClick={next} type={"primary"}>Next</Button>
                </div>
            ),
            disabled: distillResult === undefined
        }, {
            title: "Alterations",
            content: (
                <div style={{width: "100%", height: "100%"}}>
                    {
                        tree ? (
                                <>
                                    <Modal
                                        open={resultModalOpen}
                                        title={"Fine tuning results"}
                                        onCancel={() => setResultModalOpen(false)}
                                        onOk={() => setResultModalOpen(false)}
                                        footer={(<Space direction={"horizontal"}>
                                            Distill new decision tree?
                                            <Button onClick={revertFineTune} danger type={"default"} loading={working}>
                                                Revert
                                            </Button>
                                            <Button onClick={keepNewModel} type={"primary"} loading={working}>
                                                Confirm
                                            </Button>
                                        </Space>)}
                                    >
                                        <MetricsTable
                                            data={
                                                fineTuneResults ? [
                                                    {...fineTuneResults.nn_evaluation, model: "Original Neural Network"},
                                                    {...fineTuneResults.nn_modified_evaluation, model: "New Neural Network"},
                                                    {...fineTuneResults.dt_evaluation, model: "Original Decision Tree"},
                                                ] : undefined
                                            }
                                        />
                                        <h1>Distillation Parameters</h1>
                                        <DistillParamsInput value={fineDistillParams} onChange={setFineDistillParams} />
                                    </Modal>
                                    <DecisionTreeDisplay
                                        data={tree}
                                        onNodeClicked={(n) => {
                                            if (n === selectedNode) {
                                                setSelectedNode(undefined);
                                            } else if (n.right === null && n.left === null) {
                                                setSelectedNode(undefined);
                                            } else {
                                                setSelectedNode(n);
                                            }
                                        }}
                                        onBackgroundClicked={() => {
                                            setSelectedNode(undefined);
                                        }}
                                        selectedNode={selectedNode}
                                        onNodeCut={cutNode}
                                        onNodeRetrain={retrainNode}
                                    />
                                    <FineTuneControls
                                        working={working}
                                        selectedNode={selectedNode}
                                        onCutNode={cutNode}
                                        onRetrainNode={retrainNode}
                                        onFineTune={applyAlterations}
                                    />
                                </>
                            )
                            :
                            <LoadingOutlined/>
                    }
                </div>
            ),
            disabled: tree === undefined
        }
    ];

    const upload = async () => {
        setWorking(true);
        try {
            const result = await api.uploadXes(sessionId, xesFile);
            if(result !== undefined) {
                setUploadResult(result);
                next();
            }
        } finally {
            setWorking(false);
        }
    }

    const train = async () => {
        setWorking(true);
        try {
            const result = await api.trainModel(sessionId, trainParams);
            if(result !== undefined) {
                setTrainResult(result);
                next();
            }
        } finally {
            setWorking(false);
        }
    }

    const distill = async (params: DistillParams) => {
        setWorking(true);
        try {
            const distillResult = await api.distillTree(sessionId, params);
            if(distillResult !== undefined) {
                setDistillResult(distillResult);
                setCurrent(DISTILL_RESULT_VIEW);
                const newTree = await api.fetchTree(sessionId);
                setTree(newTree);
            }
        } finally {
            setWorking(false);
        }
    }

    return (
        <Context.Provider value={contextValue}>
            <div style={{width: "100%", height: "100%"}}>
                {contextHolder}
                <Steps
                    items={steps.map(s => ({key: s.title, title: s.title}))}
                    current={current}
                    onChange={(requestedCurrent) => {
                        setCurrent(requestedCurrent);
                    }}
                />
                <div style={{width: "100%", height: "calc(100% - 32px)"}}>
                    {steps[current].content}
                </div>
            </div>
        </Context.Provider>
    );
}

export default ModelCreationSteps;