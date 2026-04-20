"""
文件上传streamlit

1.前端界面打造
2.文件上传按钮
3.文件数据提取
3.实例化inbound_vector，将提取的文件数据存入向量库
4.文件信息反馈，如文件类型，大小
5.返回文件是否存入的日志
"""

import streamlit as st
from charm_vector import Vector_charm
st.title("文件上传")

upload_file = st.file_uploader(
    label="文件上传",
    type=['txt', 'pdf'],
    accept_multiple_files=False
)
if upload_file is not None:
    file_name = upload_file.name
    file_size = upload_file.size / 1024
    file_type = upload_file.type

    serve = Vector_charm()

    st.write(f"文件类型：{file_type}")
    st.write(f"文件名称：{file_name}")
    st.write(f"文件大小：{file_size:.2f}kb")

    text = upload_file.getvalue().decode("utf-8")
    st.text_area(text, height=100)

    res = serve.upload_vector(text)
    st.write(res)



