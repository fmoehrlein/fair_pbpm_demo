import {DecisionNode} from "../services/DecisionTree";
import {Button, Modal, Space} from "antd";
import {ArrowLeftOutlined, ArrowRightOutlined, BulbOutlined} from "@ant-design/icons";
import React from "react";


interface Props {
    node: DecisionNode;
    working: boolean;
    onRetrainNode: (node: DecisionNode) => void;
    onCutNode: (node: DecisionNode, mode: "auto" | "left" | "right") => void;
}

const NodeAlterationActions = ({node, working, onCutNode, onRetrainNode}: Props) => {
    const [cutModalOpen, setCutModalOpen] = React.useState(false);
    
    const doCut = (mode: "auto" | "left" | "right") => {
        if(!node) {
            console.error("No node selected, button should be disabled.");
            return;
        }
        onCutNode(node, mode);
        setCutModalOpen(false);
    }

    return (
        <Space direction={"horizontal"} className={"nodrag"}>
            <div>
                <Button
                    loading={working}
                    disabled={!node}
                    type={"default"}
                    onClick={(e) => {
                        e.stopPropagation();
                        onRetrainNode(node as DecisionNode);
                    }}
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
                        <Button onClick={() => doCut("left")} icon={<ArrowLeftOutlined/>}>Left</Button>
                        <Button onClick={() => doCut("auto")} icon={<BulbOutlined/>} type={"primary"}>Auto</Button>
                        <Button onClick={() => doCut("right")} icon={<ArrowRightOutlined/>}>Right</Button>
                    </Space>
                </Modal>
                <Button
                    disabled={!node}
                    type={"default"}
                    loading={working}
                    onClick={(e) => {
                        e.stopPropagation();
                        setCutModalOpen(true);
                    }}
                >
                    Remove
                </Button>
            </div>
        </Space>
    );
}

export default NodeAlterationActions;