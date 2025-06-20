import React, {ReactElement} from 'react';
import {Handle, Position} from "@xyflow/react";

import './DecisionNodeDisplay.css';
import {DecisionNode} from "../../services/DecisionTree";
import AlterationControls from "../AlterationControls";
import NodeAlterationActions from "../NodeAlterationActions";
import {Space} from "antd";


export interface Props  {
    data: {
        label: ReactElement,
        onClick: () => void,
        selected?: boolean,
        node: DecisionNode,
        onCutNode: (node: DecisionNode) => void,
        onRetrainNode: (node: DecisionNode) => void,
    };
}

const DecisionNodeDisplay = (props: Props) => {
    const renderControls = () => {
        if (props.data.node.left === null && props.data.node.right === null) return null;
        if (!props.data.selected) return null;
        return (
            <NodeAlterationActions
                node={props.data.node}
                working={false}
                onRetrainNode={props.data.onRetrainNode}
                onCutNode={props.data.onCutNode}
            />
        );
    }

    return (
        <div
            className={`decision-node ${props.data.selected? 'selected' : ''}`}
            onClick={props.data.onClick}
        >
            <Space direction={"vertical"}>
                <div>
                    {props.data.label}
                </div>

                {renderControls()}
            </Space>

            <Handle type={"source"} position={Position.Bottom} isConnectable={false} />
            <Handle type={"target"} position={Position.Top} isConnectable={false} />
        </div>
    );
}

export default DecisionNodeDisplay;