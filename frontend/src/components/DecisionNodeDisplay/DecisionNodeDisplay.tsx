import React, {ReactElement} from 'react';
import {Handle, Position} from "@xyflow/react";

import './DecisionNodeDisplay.css';

const DecisionNodeDisplay = (props: {data: {label: ReactElement, onClick: () => void, selected?: boolean}}) => {
    return (
        <div
            className={`decision-node ${props.data.selected? 'selected' : ''}`}
            onClick={props.data.onClick}
        >
            <div>
                {props.data.label}
            </div>

            <Handle type={"source"} position={Position.Bottom} isConnectable={false} />
            <Handle type={"target"} position={Position.Top} isConnectable={false} />
        </div>
    );
}

export default DecisionNodeDisplay;