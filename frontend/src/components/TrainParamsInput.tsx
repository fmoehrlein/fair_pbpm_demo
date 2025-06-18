import {Button, Form, InputNumber} from "antd";
import MultiValueInput from "./MultiValueInput";
import React from "react";
import {TrainParams} from "../services/api";

const TrainParamsInput = ({value, onChange}: {value: TrainParams, onChange: (params: TrainParams) => void}) => {
    return (
        <Form>
            <Form.Item label={'Prefix Length'}>
                <InputNumber
                    value={value.prefix_length}
                    onChange={e => onChange({
                        ...value,
                        prefix_length: e || value.prefix_length
                    })}
                />
            </Form.Item>
            <Form.Item label={'Categorical Attributes'}>
                <MultiValueInput
                    values={value.cat_attributes}
                    onChange={vs => onChange({
                        ...value,
                        cat_attributes: vs
                    })}
                    what={'Categorical attribute'}
                />
            </Form.Item>
            <Form.Item label={'Numerical Attributes'}>
                <MultiValueInput
                    values={value.num_attributes}
                    onChange={vs => onChange({
                        ...value,
                        num_attributes: vs
                    })}
                    what={'Numerical attribute'}
                />
            </Form.Item>
            <Form.Item label={'Sensitive Attributes'}>
                <MultiValueInput
                    values={value.sensitive_attributes}
                    onChange={vs => onChange({
                        ...value,
                        sensitive_attributes: vs
                    })}
                    what={'Sensitive attribute'}
                />
            </Form.Item>
            <Form.Item label={'Test split'}>
                <InputNumber
                    value={value.test_split}
                    onChange={e => onChange({
                        ...value,
                        test_split: e || value.test_split
                    })}
                    min={0}
                    max={1}
                    step={.05}
                />
            </Form.Item>
            <Form.Item label={'Num epochs'}>
                <InputNumber
                    value={value.epochs}
                    onChange={e => onChange({
                        ...value,
                        epochs: e || value.epochs
                    })}
                    min={1}
                    step={1}
                />
            </Form.Item>
            <Form.Item label={'Learning Rate'}>
                <InputNumber
                    value={value.learning_rate}
                    onChange={e => onChange({
                        ...value,
                        learning_rate: e || value.learning_rate
                    })}
                    min={0}
                />
            </Form.Item>
            <Form.Item label={'Hidden layers'}>
                <MultiValueInput
                    values={value.hidden_units.map(v => v.toString())}
                    onChange={vs => onChange({
                        ...value,
                        hidden_units: vs.map(v => Number(v) || 1)
                    })}
                    what={'Hidden layer'}
                />
            </Form.Item>
        </Form>
    );
}

export default TrainParamsInput;