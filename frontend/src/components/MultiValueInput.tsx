import React from 'react';
import {Button, Input, Space} from "antd";
import {DeleteOutlined, PlusOutlined} from "@ant-design/icons";

interface Props {
    what?: string;
    values: string[];
    onChange: (values: string[]) => void;
}

const MultiValueInput = ({values, onChange, what}: Props) => {
    return (
        <>
            {
                values.map((v, i) => {
                    return (
                        <Space direction={"vertical"} key={i}>
                            <Space.Compact>
                                <Input
                                    value={v}
                                    onChange={(e) => {
                                        values.splice(i, 1, e.target.value);
                                        onChange([...values]);
                                    }}
                                />
                                <Button
                                    icon={<DeleteOutlined />}
                                    danger
                                    type={"default"}
                                    onClick={() => {
                                        values.splice(i, 1);
                                        onChange([...values]);
                                    }}
                                />
                            </Space.Compact>
                        </Space>
                    );
                })
            }
            <Button
                icon={<PlusOutlined />}
                type={"default"}
                onClick={() => {
                    onChange([...values, ""]);
                }}
            >
                New {what ? what: ''}
            </Button>
        </>
    )
}

export default MultiValueInput;