import React from 'react';
import DecisionTreeDisplay from "./components/DecisionTreeDisplay";
import DecisionTree, {DecisionNode} from "./services/DecisionTree";
import Controls from "./components/Controls";
import Api, {DistillParams, FineTuneParams} from "./services/api";
import ModelCreationSteps from "./components/ModelCreationSteps";
import {LoadingOutlined} from "@ant-design/icons";


const api = new Api("https://apps.pm.iisys.de");

function App() {
    const [selectedNode, setSelectedNode] = React.useState<DecisionNode | undefined>(undefined);
    const [user, setUser] = React.useState<string | null>(null);
    const [working, setWorking] = React.useState(false);

    const [tree, setTree] = React.useState<DecisionTree | undefined>(undefined);

    const initializeSession = () => {
        let sessionId = localStorage.getItem("session_id");
        if (!localStorage.getItem("session_id")) {
            sessionId = crypto.randomUUID();
            localStorage.setItem("session_id", sessionId);
        }
        setUser(sessionId);
        return sessionId as string;
    }

    React.useEffect(() => {
        const load = async () => {
            const sessionId = initializeSession();
            const newTree = await api.fetchTree(sessionId);
            setTree(newTree);
        }

        load();
    }, []);

    const cutNode = async (node: DecisionNode, keep: "left" | "right" | "auto") => {
        if(!user) {throw new Error("Should not happen.")}

        setWorking(true);
        try {
            await api.cutNode(user, {node_id: node.node_id, direction: keep});
            const newTree = await api.fetchTree(user);
            setTree(newTree);
        } finally {
            setSelectedNode(undefined);
            setWorking(false);
        }
    }

    const retrainNode = async (node: DecisionNode) => {
        if(!user) {throw new Error("Should not happen.")}

        setWorking(true);
        try {
            await api.retrainNode(user, {node_id: node.node_id});
            const newTree = await api.fetchTree(user);
            setTree(newTree);
        } finally {
            setSelectedNode(undefined);
            setWorking(false);
        }
    }

    const fineTune = async (params: FineTuneParams) => {
        if(!user) {throw new Error("Should not happen.")}

        setWorking(true);
        try {
            await api.fineTune(user, params);
        } finally {
            setWorking(false);
        }
    }

    const distill = async (params: DistillParams) => {
        if(!user) {throw new Error("Should not happen.")};

        setWorking(true);
        try {
            await api.distillTree(user, params);
            const newTree = await api.fetchTree(user);
            setTree(newTree);
        } finally {
            setWorking(false);
        }
    }

    const renderContent = () => {
        if(tree) {
            return (
                <>
                    <DecisionTreeDisplay
                        data={tree}
                        onNodeClicked={(n) => n === selectedNode ? setSelectedNode(undefined) : setSelectedNode(n)}
                        onBackgroundClicked={() => setSelectedNode(undefined)}
                        selectedNode={selectedNode}
                    />
                    <Controls
                        working={working}
                        selectedNode={selectedNode}
                        onCutNode={cutNode}
                        onRetrainNode={retrainNode}
                        onFineTune={fineTune}
                        onDistill={distill}
                    />
                </>
            );
        }

        if (user) {
            return (
                <div style={{padding: 25}}>
                    <ModelCreationSteps
                        sessionId={user}
                        onFinished={async () => {
                            const newTree = await api.fetchTree(user);
                            setTree(newTree);
                        }}
                    />
                </div>
            );
        }

        return <LoadingOutlined />;
    }

    return (
        <div style={{width: '100vw', height: '100vh'}}>
            {renderContent()}
        </div>
    );
}

export default App;
