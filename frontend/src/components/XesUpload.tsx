import React from 'react';

import {Upload} from 'antd';
import {InboxOutlined} from '@ant-design/icons';


interface Props {
    onFileSelected: (file: File) => void;
}

const XesUpload = ({onFileSelected}: Props) => {
    return (
        <Upload.Dragger
            beforeUpload={(file) => {
                onFileSelected(file);
                return false;
            }}
            style={{marginBottom: 5}}
        >
            <p className="ant-upload-drag-icon">
                <InboxOutlined />
            </p>
            <p className="ant-upload-text">Upload event log</p>
            <p className="ant-upload-hint">
                Drag and drop an event log in XES format here to upload it.
            </p>
        </Upload.Dragger>
    );
}

export default XesUpload;