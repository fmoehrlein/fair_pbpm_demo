import React from 'react';

import {Background, Controls, Edge, MiniMap, Node, ReactFlow, useEdgesState, useNodesState} from "@xyflow/react";
import '@xyflow/react/dist/style.css';

import DecisionTree, {DecisionNode} from "../services/DecisionTree";
import LabeledEdge from "./LabeledEdge";
import DecisionNodeDisplay from "./DecisionNodeDisplay/DecisionNodeDisplay";

interface Props {
    data: DecisionTree;
    selectedNode?: DecisionNode;
    onNodeClicked?: (node: DecisionNode) => void;
    onBackgroundClicked?: () => void;
}

const edgeTypes = {"labeled-edge": LabeledEdge};
const nodeTypes = {"decision-node": DecisionNodeDisplay};

const DecisionTreeDisplay = (props: Props) => {
    const getMaxDepth = (root: DecisionNode): number => {
        if (root.left && root.right) return Math.max(getMaxDepth(root.left), getMaxDepth(root.right));
        if (root.left) return getMaxDepth(root.left);
        if (root.right) return getMaxDepth(root.right);
        return root.depth;
    }

    const renderNodeLabel = (node: DecisionNode) => {
        if(node.output !== null) return `${props.data.class_names[node.output]}`;
        if(node.feature_index !== null && node.threshold !== null) {
            if(props.data.feature_names[node.feature_index].indexOf(" = ") !== -1) {
                return props.data.feature_names[node.feature_index];
            }
            return `${props.data.feature_names[node.feature_index]} > ${node.threshold.toFixed(2)}`;
        }
        console.error("Could not render label for node", node);
        return null;
    }

    const getNodes = (root: DecisionNode, rootPos: {x: number, y: number}, offset: {x: number, y: number}): Node[] => {
        const ret: Node[] = [{
            id: root.node_id.toString(),
            position: rootPos,
            data: {
                label: renderNodeLabel(root),
                onClick: () => props.onNodeClicked?.(root),
                selected: root.node_id === props.selectedNode?.node_id
            },
            type: "decision-node"
        }];
        if(root.left) {
            const width = (getMaxDepth(root.left) - root.depth + 1) * offset.x;
            ret.push(...getNodes(root.left, {
                x: rootPos.x + width,
                y: rootPos.y + offset.y
            }, offset));
        }
        if(root.right) {
            const width = (getMaxDepth(root.right) - root.depth + 1) * offset.x;
            ret.push(...getNodes(root.right, {
                x: rootPos.x - width,
                y: rootPos.y + offset.y
            }, offset));
        }
        return ret;
    }

    const getEdges = (root: DecisionNode): Edge[] => {
        const ret: Edge[] = [];
        if(root.left) {
            ret.push({
                id: `e-${root.node_id}-${root.left.node_id}`,
                source: root.node_id.toString(),
                target: root.left.node_id.toString(),
                label: <div style={{background: "#FDD", border: "#A88 solid 1px",  borderRadius: 3, padding: "0 3px", fontSize: 10, color: "#333"}}>false</div>,
                type: "labeled-edge",
            });
            ret.push(...getEdges(root.left));
        }
        if(root.right) {
            ret.push({
                id: `e-${root.node_id}-${root.right.node_id}`,
                source: root.node_id.toString(),
                target: root.right.node_id.toString(),
                label: <div style={{background: "#EFE", border: "#8A8 solid 1px",borderRadius: 3, padding: "0 3px", fontSize: 10, color: "#333"}}>true</div>,
                type: "labeled-edge",
            });
            ret.push(...getEdges(root.right));
        }
        return ret;
    }

    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

    React.useEffect(() => {
        console.log("building tree")
        setNodes(getNodes(props.data.root, {x: 0, y: 0}, {x: 40, y: 100}));
        setEdges(getEdges(props.data.root));
    }, [props.data, props.selectedNode, props.onNodeClicked]);

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            edgeTypes={edgeTypes}
            nodeTypes={nodeTypes}
            onPaneClick={props.onBackgroundClicked}
        >
            <MiniMap/>
            <Controls/>
            <Background />
        </ReactFlow>
    );
}


export default DecisionTreeDisplay;