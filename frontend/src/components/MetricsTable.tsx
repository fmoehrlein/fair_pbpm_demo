import {Table} from "antd";
import React from "react";


interface Props {
    data?: {model: string, accuracy: number, f1_score: number, precision: number, recall: number}[];
}

const MetricsTable = ({data}: Props) => {
    return (
        <Table
            columns={[
                {
                    title: "Model",
                    dataIndex: "model",
                },
                {
                    title: "Accuracy",
                    dataIndex: "accuracy",
                    render: (value: number) => (value * 100).toFixed(2) + "%"
                },
                {
                    title: "F1",
                    dataIndex: "f1_score",
                    render: (value: number) => (value * 100).toFixed(2) + "%"
                },
                {
                    title: "Precision",
                    dataIndex: "precision",
                    render: (value: number) => (value * 100).toFixed(2) + "%"
                },
                {
                    title: "Recall",
                    dataIndex: "recall",
                    render: (value: number) => (value * 100).toFixed(2) + "%"
                },
            ]}
            dataSource={data}
        />
    );
}

export default MetricsTable;