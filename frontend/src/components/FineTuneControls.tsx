import React from "react";
import {DecisionNode} from "../services/DecisionTree";
import {Button, Modal, Space} from "antd";
import {FineTuneParams} from "../services/useApi";
import FineTuneParamsInput from "./FineTuneParamsInput";

interface Props {
    selectedNode?: DecisionNode;

    working: boolean;
    onRetrainNode: (node: DecisionNode) => void;
    onCutNode: (node: DecisionNode, mode: "auto" | "left" | "right") => void;
    onFineTune: (fineTuneParams: FineTuneParams) => void;
}

const FineTuneControls = ({onFineTune, working}: Props) => {

    const [modalOpen, setModalOpen] = React.useState(false);

    const [fineTuneParams, setFineTuneParams] = React.useState<FineTuneParams>({
        mode: 'changed_complete', learning_rate: 0.001, epoch: 5, batch_size: 32
    });

    return (
        <Space direction={"vertical"} style={{position: "absolute", top: 250, right: 15, transform: "translateX(-50%)", textAlign: "center", border: "solid 1px #888", borderRadius: 5, padding: 15, background: "#FFFA"}}>
            <div style={{fontWeight: "bold", fontSize: "1.1em"}}>Controls</div>
            <div style={{maxWidth: 150}}>Apply decision tree modifications and fine tune neural network</div>
            <Modal
                open={modalOpen}
                title={"Parameters"}
                onCancel={() => setModalOpen(false)}
                onOk={() => {
                    setModalOpen(false);
                    onFineTune(fineTuneParams);
                }}
            >
                <h1>Fine Tuning Parameters</h1>
                <FineTuneParamsInput value={fineTuneParams} onChange={setFineTuneParams} />
            </Modal>
            <Button
                type={"primary"}
                loading={working}
                onClick={() => setModalOpen(true)}
            >
                Fine Tune Model
            </Button>
        </Space>
    );
}

export default FineTuneControls;