import React from 'react';
import {Badge, Button, Empty, Steps, Table, Tag} from "antd";
import XesUpload from "./XesUpload";
import Api, {DistillParams, DistillResult, TrainParams, TrainResult, UploadResult} from "../services/api";
import TrainParamsInput from "./TrainParamsInput";
import DistillParamsInput from "./DistillParamsInput";


interface Props {
    sessionId: string;
    onFinished: () => void;
}

const api = new Api("http://localhost:5000");

const ModelCreationSteps = ({onFinished, sessionId}: Props) => {
    const [current, setCurrent] = React.useState(0);
    const [working, setWorking] = React.useState(false);

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
        min_samples_split:2,
        max_depth:100,
        ccp_alpha:0.001
    });

    const [trainResult, setTrainResult] = React.useState<TrainResult | undefined>(undefined);
    const [distillResult, setDistillResult] = React.useState<DistillResult | undefined>(undefined);
    const [uploadResult, setUploadResult] = React.useState<UploadResult | undefined>(undefined);

    const next = () => {
        setCurrent(current + 1);
    }

    const distillationDataSource = [];
    if(trainResult) distillationDataSource.push({model: "Neural Network", ...trainResult});
    if(distillResult) distillationDataSource.push({model: "Decision Tree", ...distillResult});

    const steps = [
        {
            title: "Upload",
            content: (
                <>
                    <XesUpload onFileSelected={setXesFile} />
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
                </>
            )
        }, {
            title: "Event Log Statistics",
            content: (
                <>
                    <Table
                        columns={[
                            {title: "Attributes", dataIndex: "attributes", render: (value: string[]) => value.map(v => <Tag>{v}</Tag>)},
                            {title: "avg. Events / Case", dataIndex: "events_per_case"},
                            {title: "Cases", dataIndex: "num_cases"},
                            {title: "Events", dataIndex: "num_events"},
                        ]}
                        dataSource={uploadResult ? [uploadResult] : undefined}
                        pagination={false}
                    />
                    <Button onClick={next} type={"primary"}>Next</Button>
                </>
            )
        }, {
            title: "Train",
            content: (
                <>
                    <TrainParamsInput value={trainParams} onChange={setTrainParams} />
                    <Button type={"primary"} onClick={() => train()} loading={working}>Train</Button>
                </>
            )
        }, {
            title: "Training Results",
            content: (
                <>
                    <Table
                        columns={[
                            {title: "Accuracy", dataIndex: "accuracy", render: (value: number) => (value * 100).toFixed(2) + "%"},
                            {title: "F1", dataIndex: "f1_score", render: (value: number) => (value * 100).toFixed(2) + "%"},
                            {title: "Precision", dataIndex: "precision", render: (value: number) => (value * 100).toFixed(2) + "%"},
                            {title: "Recall", dataIndex: "recall", render: (value: number) => (value * 100).toFixed(2) + "%"},
                        ]}
                        dataSource={trainResult ? [trainResult]: undefined}
                    />
                    <Button onClick={next} type={"primary"}>Next</Button>
                </>
            )
        }, {
            title: "Distill",
            content: (
                <>
                    <DistillParamsInput value={distillParams} onChange={setDistillParams} />
                    <Button type={"primary"} onClick={() => distill()} loading={working}>
                        Distill
                    </Button>
                </>
            )
        }, {
            title: "Distillation Results",
            content: (
                <>
                    <Table
                        columns={[
                            {title: "Model", dataIndex: "model"},
                            {title: "Accuracy", dataIndex: "accuracy", render: (value: number) => (value * 100).toFixed(2) + "%"},
                            {title: "F1", dataIndex: "f1_score", render: (value: number) => (value * 100).toFixed(2) + "%"},
                            {title: "Precision", dataIndex: "precision", render: (value: number) => (value * 100).toFixed(2) + "%"},
                            {title: "Recall", dataIndex: "recall", render: (value: number) => (value * 100).toFixed(2) + "%"},
                        ]}
                        dataSource={distillationDataSource}
                    />
                    <Button onClick={onFinished} type={"primary"}>Next</Button>
                </>
            )
        }
    ];

    const upload = async () => {
        if(!xesFile) throw new Error("Cant start without xes file, should not happen, since button is disabled.");

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
        <>
            <Steps
                items={steps.map(s => ({key: s.title, title: s.title}))}
                current={current}
            />
            <div style={{maxWidth: 900, marginTop: 25, marginLeft: "auto", marginRight: "auto"}}>
                {steps[current].content}
            </div>
        </>
    );
}

export default ModelCreationSteps;