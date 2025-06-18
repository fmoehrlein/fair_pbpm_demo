import React from 'react';
import {BaseEdge, EdgeLabelRenderer, getSimpleBezierPath} from "@xyflow/react";

export interface Props {
    id: string;
    sourceX: number;
    sourceY: number;
    targetX: number;
    targetY: number;
    label?: React.ReactNode;
}

const LabeledEdge = ({id, sourceX, sourceY, targetX, targetY, label}: Props) => {
    const [edgePath, labelX, labelY] = getSimpleBezierPath({sourceX, sourceY, targetX, targetY});

    return (
        <>
            <BaseEdge id={id} path={edgePath}/>
            <EdgeLabelRenderer>
                <div style={{position: 'absolute', transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`}}>
                    {label}
                </div>
            </EdgeLabelRenderer>
        </>
    );
}

export default LabeledEdge;