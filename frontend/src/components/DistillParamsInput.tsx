import {DistillParams} from "../services/api";
import {Form, InputNumber} from "antd";
import React from "react";

const DistillParamsInput = ({value, onChange}: {value: DistillParams, onChange: (value: DistillParams) => void}) => {
    return (
        <Form>
            <Form.Item label={'Min Samples Split'}>
                <InputNumber
                    value={value.min_samples_split}
                    onChange={e => onChange({
                        ...value,
                        min_samples_split: e || value.min_samples_split
                    })}
                />
            </Form.Item>
            <Form.Item label={'Max Depth'}>
                <InputNumber
                    value={value.max_depth}
                    onChange={e => onChange({
                        ...value,
                        max_depth: e || value.max_depth
                    })}
                />
            </Form.Item>
            <Form.Item label={'CCP Alpha'}>
                <InputNumber
                    value={value.ccp_alpha}
                    onChange={e => onChange({
                        ...value,
                        ccp_alpha: e || value.ccp_alpha
                    })}
                />
            </Form.Item>
        </Form>
    );
}

export default DistillParamsInput;