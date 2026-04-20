"""
向量库存储（使用md5检索是否已经入库）

1.定义 数据转MD5字符串 的函数
2.定义 MD5数据检索 的函数
3.定义 MD5数据的存储，存入MD5文件
4.定义 数据入库的类
        self:传入的数据，声明向量库
    （1）.导入嵌入模型
    （2）.将上传的文本分割成多段数据
    （3）.将数据转化成MD5数据
    （4）.使用MD5检索
        未检索到：（1）.将原来的数据存入向量库
                （2）.存入MD5文件


"""
import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_str_md5(string):
    er_data =string.encode(encoding = 'utf-8')
    md5_obj = hashlib.md5(er_data).hexdigest()
    return md5_obj

def check_md5(MD5_data):
    if not os.path.exists("./md5.txt"):   #调用os模块查看文件，不存在则创造文件，并返回库中没有相同数据的标识
        open("./md5.txt","w",encoding="utf-8").close()
        return True
    with open("./md5.txt","r",encoding="utf-8") as f:
        for line in f:
            if MD5_data == line.strip():
                return False
        return True

def save_md5(MD5_data):
    with open("./md5.txt", "w",encoding="utf-8") as f:
        f.write(MD5_data+"\n")

class Vector_charm():
    def __init__(self):
        os.makedirs("D:/zhuomian/rag项目/pythonProject2/self_rag/chroma_base",exist_ok=True) #没有就创建向量库所在的文件
        #声明向量库
        self.chroma =Chroma(
            collection_name="vector_rag",  #向量库名称
            embedding_function=DashScopeEmbeddings(),   #嵌入模型
            persist_directory="D:/zhuomian/rag项目/pythonProject2/self_rag/chroma_base"  #向量存储路径
        )
        #文本分割器
        self.splite =RecursiveCharacterTextSplitter(
            chunk_size=50,  #分割后的文本段长度
            chunk_overlap=20,  #连续文本段的重叠字符数
            separators=["," , "." , "!" , "?" , "\n" , "\n\n" , "，" , "。" , "！" , ],  #段落划分的符号
            length_function=len  #计算长度的功能函数，长度统计依据
        )
    def upload_vector(self, data):
        md5_data = get_str_md5(data)
        if check_md5(md5_data):
            if len(data) > 300:
                new_list = self.splite.split_text(data)   #返回文本切割后的多个数据的列表
            else :
                new_list = [data]

            self.chroma.add_texts(new_list)
            save_md5(md5_data)

            return "文件正在入库"
        else:
            return "库中已经存在此文件"

