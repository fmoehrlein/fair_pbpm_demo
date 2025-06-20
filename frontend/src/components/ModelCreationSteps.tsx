import React, {ReactElement} from 'react';
import {Button, Modal, StepProps, Steps, Table, Tag} from "antd";
import XesUpload from "./XesUpload";
import Api, {
    DistillParams,
    DistillResult,
    FineTuneParams,
    Metrics,
    TrainParams,
    TrainResult,
    UploadResult
} from "../services/api";
import TrainParamsInput from "./TrainParamsInput";
import DistillParamsInput from "./DistillParamsInput";
import DecisionTree, {DecisionNode} from "../services/DecisionTree";
import DecisionTreeDisplay from "./DecisionTreeDisplay";
import AlterationControls from "./AlterationControls";
import {LoadingOutlined} from "@ant-design/icons";
import MetricsTable from "./MetricsTable";


interface Props {
    sessionId: string;
}

interface AlterationResults {
    originalNNMetrics: Metrics;
    modifiedNNMetrics: Metrics;
    originalTreeMetrics: Metrics;
    modifiedTreeMetrics: Metrics;
}

const api = new Api("http://localhost:5000");

const ModelCreationSteps = ({sessionId}: Props) => {
    const [current, setCurrent] = React.useState(0);
    const [working, setWorking] = React.useState(false);

    const [resultModalOpen, setResultModalOpen] = React.useState(false);
    const [alterationResults, setAlterationResults] = React.useState<AlterationResults | undefined>(undefined);

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
        model_to_use: 'latest'
    });

    const [trainResult, setTrainResult] = React.useState<TrainResult | undefined>(undefined);
    const [distillResult, setDistillResult] = React.useState<DistillResult | undefined>(undefined);
    const [uploadResult, setUploadResult] = React.useState<UploadResult | undefined>(undefined);

    const [selectedNode, setSelectedNode] = React.useState<DecisionNode | undefined>(undefined);

    const [tree, setTree] = React.useState<DecisionTree | undefined>(undefined);

    React.useEffect(() => {
        const load = async () => {
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
            setCurrent(6);
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

    const applyAlterations = async (ftParams: FineTuneParams, dParams: DistillParams) => {
        if (!sessionId) {
            throw new Error("Should not happen.")
        }

        setWorking(true);
        try {
            const fineTuneResults = await api.fineTune(sessionId, ftParams);
            const distillResults = await api.distillTree(sessionId, dParams);
            if (!fineTuneResults || !distillResults) {
                throw new Error("Should not happen.");
            }
            setAlterationResults({
                originalNNMetrics: fineTuneResults.nn_evaluation,
                modifiedNNMetrics: fineTuneResults.nn_modified_evaluation,
                originalTreeMetrics: fineTuneResults.dt_evaluation,
                modifiedTreeMetrics: distillResults.dt_evaluation
            })
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
            setResultModalOpen(true);

        } finally {
            setWorking(false);
        }
    }

    const distillationDataSource = [];
    if (trainResult) distillationDataSource.push({model: "Neural Network", ...trainResult});
    if (distillResult) distillationDataSource.push({model: "Decision Tree", ...distillResult});

    const steps: (StepProps & {content: ReactElement})[] = [
        {
            title: "Upload",
            content: (
                <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                    <XesUpload onFileSelected={setXesFile}/>
                    <div>
                        <Button
                            type={"primary"}
                            disabled={!xesFile}
                            onClick={() => upload()}
                            loading={working}
                        >
                            Upload
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
                                render: (value: string[]) => value.map(v => <Tag>{v}</Tag>)
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
                <>
                    <TrainParamsInput value={trainParams} onChange={setTrainParams}/>
                    <Button type={"primary"} onClick={() => train()} loading={working}>Train</Button>
                </>
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
                    <Button type={"primary"} onClick={() => distill()} loading={working}>
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
                                        onCancel={() => setResultModalOpen(false)}
                                        onOk={() => setResultModalOpen(false)}
                                    >
                                        {
                                            <MetricsTable
                                                data={
                                                    alterationResults ? [
                                                        {...alterationResults.originalNNMetrics, model: "Original Neural Network"},
                                                        {...alterationResults.modifiedNNMetrics, model: "New Neural Network"},
                                                        {...alterationResults.originalTreeMetrics, model: "Original Decision Tree"},
                                                        {...alterationResults.modifiedTreeMetrics, model: "Modified Decision Tree"},
                                                    ] : undefined
                                                }
                                            />
                                        }
                                    </Modal>
                                    <DecisionTreeDisplay
                                        data={tree}
                                        onNodeClicked={(n) => {
                                            n === selectedNode ? setSelectedNode(undefined) : setSelectedNode(n);
                                            console.log("node clicked")
                                        }}
                                        onBackgroundClicked={() => {
                                            setSelectedNode(undefined);
                                            console.log("background clicked")
                                        }}
                                        selectedNode={selectedNode}
                                        onNodeCut={cutNode}
                                        onNodeRetrain={retrainNode}
                                    />
                                    <AlterationControls
                                        working={working}
                                        selectedNode={selectedNode}
                                        onCutNode={cutNode}
                                        onRetrainNode={retrainNode}
                                        onApplyAlterations={applyAlterations}
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
        if (!xesFile) throw new Error("Cant start without xes file, should not happen, since button is disabled.");

        setWorking(true);
        try {
            const result = await api.uploadXes(sessionId, xesFile);
            setUploadResult(result);
            next();
        } finally {
            setWorking(false);
        }
    }

    const train = async () => {
        setWorking(true);
        try {
            const result = await api.trainModel(sessionId, trainParams);
            setTrainResult(result);
            next();
        } finally {
            setWorking(false);
        }
    }

    const distill = async () => {
        setWorking(true);
        try {
            const distillResult = await api.distillTree(sessionId, distillParams);
            setDistillResult(distillResult);
            next();
        } finally {
            setWorking(false);
        }
    }

    return (
        <div style={{width: "100%", height: "100%"}}>
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
    );
}

export default ModelCreationSteps;