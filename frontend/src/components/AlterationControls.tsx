import React from "react";
import {DecisionNode} from "../services/DecisionTree";
import {Button, Modal} from "antd";
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

const AlterationControls = ({onApplyAlterations, working}: Props) => {

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

    return (
        <>
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
        </>
    );
}

export default AlterationControls;