import React from "react";
import {DecisionNode} from "../services/DecisionTree";
import {Button, Form, InputNumber, Modal, Popconfirm, Select, Space} from "antd";
import {ArrowLeftOutlined, ArrowRightOutlined, BulbOutlined} from "@ant-design/icons";
import {DistillParams, FineTuneParams} from "../services/api";
import FineTuneParamsInput from "./FineTuneParamsInput";
import DistillParamsInput from "./DistillParamsInput";

interface Props {
    selectedNode?: DecisionNode;

    working: boolean;
    onRetrainNode: (node: DecisionNode) => void;
    onCutNode: (node: DecisionNode, mode: "auto" | "left" | "right") => void;
    onFineTune: (params: FineTuneParams) => void;
    onDistill: (params: DistillParams) => void;
}

const Controls = ({selectedNode, onRetrainNode, onCutNode, onFineTune, onDistill, working}: Props) => {
    const [cutModalOpen, setCutModalOpen] = React.useState(false);
    const [fineTuneModalOpen, setFineTuneModalOpen] = React.useState(false);
    const [distillModalOpen, setDistillModalOpen] = React.useState(false);

    const [fineTuneParams, setFineTuneParams] = React.useState<FineTuneParams>({
        mode: 'changed_complete', learning_rate: 0.001, epoch: 5, batch_size: 32
    });
    const [distillParams, setDistillParams] = React.useState<DistillParams>({
        min_samples_split:2,
        max_depth:100,
        ccp_alpha:0.001
    });

    const doCut = (mode: "auto" | "left" | "right") => {
        if(!selectedNode) {
            console.error("No node selected, button should be disabled.");
            return;
        }
        onCutNode(selectedNode, mode);
        setCutModalOpen(false);
    }

    return (
        <div style={{
            position: 'absolute',
            right: 5, top: 5,
            width: 175,
            background: 'white',
            border: 'solid 1px black',
            borderRadius: 3,
            padding: '10px 15px',
            minHeight: 200,
            maxHeight: 'calc(100vh - 250px)',
        }}>
            <div style={{
                fontSize: "1.05em",
                fontWeight: "bold",
                textAlign: "center",
                borderBottom: "solid 1px black"
            }}>
                Controls
            </div>

            <div style={{
                fontSize: "1.05em",
                fontWeight: "bold",
                textAlign: "center",
                borderBottom: "solid 1px black",
                marginBottom: 5,
            }}>
                Selected node
            </div>
            <Space direction={"horizontal"}>
                <div>
                    <Button
                        loading={working}
                        disabled={!selectedNode}
                        type={"primary"}
                        onClick={() => onRetrainNode(selectedNode as DecisionNode)}
                    >
                        Retrain
                    </Button>
                </div>
                <div>
                    <Modal
                        open={cutModalOpen}
                        title={"Which branch should be kept?"}
                        onCancel={() => setCutModalOpen(false)}
                        footer={null}
                    >
                        <Space direction={"horizontal"}>
                            <Button onClick={() => doCut("left")} icon={<ArrowLeftOutlined />}>Left</Button>
                            <Button onClick={() => doCut("auto")} icon={<BulbOutlined />} type={"primary"}>Auto</Button>
                            <Button onClick={() => doCut("right")} icon={<ArrowRightOutlined />}>Right</Button>
                        </Space>
                    </Modal>
                    <Button
                        disabled={!selectedNode}
                        type={"primary"}
                        loading={working}
                        onClick={() => setCutModalOpen(true)}
                    >
                        Remove
                    </Button>
                </div>
            </Space>

            <div>
                <Modal
                    open={fineTuneModalOpen}
                    title={"Fine Tuning Parameters"}
                    onCancel={() => setFineTuneModalOpen(false)}
                    onOk={() => {
                        onFineTune(fineTuneParams);
                        setFineTuneModalOpen(false);
                    }}
                >
                    <FineTuneParamsInput value={fineTuneParams} onChange={setFineTuneParams} />
                </Modal>
                <Button
                    type={"primary"}
                    loading={working}
                    onClick={() => setFineTuneModalOpen(true)}
                >
                    Fine Tune Model
                </Button>
            </div>

            <div>
                <Modal
                    open={distillModalOpen}
                    onCancel={() => setDistillModalOpen(false)}
                    onOk={() => onDistill(distillParams)}
                >
                    <DistillParamsInput value={distillParams} onChange={setDistillParams} />
                </Modal>
                <Button
                    type={"primary"}
                    loading={working}
                    onClick={() => setDistillModalOpen(true)}
                >
                    Distill Tree
                </Button>
            </div>
        </div>
    );
}

export default Controls;