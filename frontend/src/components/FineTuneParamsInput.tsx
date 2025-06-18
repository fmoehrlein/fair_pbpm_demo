import {FineTuneParams} from "../services/api";
import {Form, InputNumber, Select} from "antd";
import React from "react";

const FineTuneParamsInput = ({value, onChange}: {value: FineTuneParams, onChange: (params: FineTuneParams) => void}) => {
    return (
        <Form style={{maxWidth: 500, marginLeft: "auto", marginRight: "auto"}}>
            <Form.Item label={"Epochs"}>
                <InputNumber value={value.epoch} onChange={e => onChange({...value, epoch: e || value.epoch})}/>
            </Form.Item>
            <Form.Item label={"Batch Size"}>
                <InputNumber value={value.batch_size} onChange={e => onChange({...value, batch_size: e || value.batch_size})}/>
            </Form.Item>
            <Form.Item label={"Learning Rate"}>
                <InputNumber value={value.learning_rate} onChange={e => onChange({...value, learning_rate: e || value.learning_rate})}/>
            </Form.Item>
            <Form.Item label={"Mode"}>
                <Select value={value.mode}>
                    <Select.Option value={"changed_complete"}>Changed Complete</Select.Option>
                </Select>
            </Form.Item>
        </Form>
    );
}

export default FineTuneParamsInput;