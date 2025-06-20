import React from 'react';
import DecisionTreeDisplay from "./components/DecisionTreeDisplay";
import DecisionTree, {DecisionNode} from "./services/DecisionTree";
import Controls from "./components/Controls";
import Api, {DistillParams, FineTuneParams} from "./services/api";
import ModelCreationSteps from "./components/ModelCreationSteps";
import {LoadingOutlined} from "@ant-design/icons";


const api = new Api("http://localhost:5000");

function App() {
    const [user, setUser] = React.useState<string | null>(null);
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
        initializeSession();
    }, []);

    const renderContent = () => {
        if (user) {
            return (
                <div style={{width: "100%", height: "100%"}}>
                    <ModelCreationSteps
                        sessionId={user}
                    />
                </div>
            );
        }

        return <LoadingOutlined style={{margin: "auto"}} />;
    }

    return (
        <div style={{width: '100vw', height: '100vh', padding: "50px 75px 25px 75px", boxSizing: "border-box"}}>
            {renderContent()}
        </div>
    );
}

export default App;
