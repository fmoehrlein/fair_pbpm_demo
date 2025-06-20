import React from "react";
import {DecisionNode} from "../services/DecisionTree";
import {Button, Modal, Space} from "antd";
import {ArrowLeftOutlined, ArrowRightOutlined, BulbOutlined} from "@ant-design/icons";
import {DistillParams, FineTuneParams} from "../services/api";
import FineTuneParamsInput from "./FineTuneParamsInput";
import DistillParamsInput from "./DistillParamsInput";

interface Props {
    selectedNode?: DecisionNode;

    working: boolean;
    onRetrainNode: (node: DecisionNode) => void;
    onCutNode: (node: DecisionNode, mode: "auto" | "left" | "right") => void;
    onApplyAlterations: (fineTuneParams: FineTuneParams, distillParams: DistillParams) => void;
}

const Controls = ({selectedNode, onRetrainNode, onCutNode, onApplyAlterations, working}: Props) => {
    const [cutModalOpen, setCutModalOpen] = React.useState(false);
    const [modalOpen, setModalOpen] = React.useState(false);

    const [fineTuneParams, setFineTuneParams] = React.useState<FineTuneParams>({
        mode: 'changed_complete', learning_rate: 0.001, epoch: 5, batch_size: 32
    });
    const [distillParams, setDistillParams] = React.useState<DistillParams>({
        min_samples_split:2,
        max_depth:100,
        ccp_alpha:0.001,
        model_to_use: 'latest'
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
            right: 15, top: 100,
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
                    open={modalOpen}
                    title={"Parameters"}
                    onCancel={() => setModalOpen(false)}
                    onOk={() => {
                        onApplyAlterations(fineTuneParams, distillParams);
                    }}
                >
                    <h1>Fine Tuning Parameters</h1>
                    <FineTuneParamsInput value={fineTuneParams} onChange={setFineTuneParams} />
                    <h1>Distillation Parameters</h1>
                    <DistillParamsInput value={distillParams} onChange={setDistillParams} />
                </Modal>
                <Button
                    type={"primary"}
                    loading={working}
                    onClick={() => setModalOpen(false)}
                >
                    Fine Tune Model
                </Button>
            </div>
        </div>
    );
}

export default Controls;