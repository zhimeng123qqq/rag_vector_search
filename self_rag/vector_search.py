"""
向量检索返回

声明向量库
返回检索结果
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

class VectorSearch():
    def __init__(self):
        self.embeddings = DashScopeEmbeddings()
        self.chroma=Chroma(
            collection_name="vector_rag",
            embedding_function=self.embeddings,
            persist_directory="D:/zhuomian/rag项目/pythonProject2/self_rag/chroma_base"
        )

    def get_search(self):
        return self.chroma.as_retriever(search_kwargs={"k":2})


if __name__=="__main__":
    search=VectorSearch().get_search()
    res=search.invoke("我身高145，体重35kg，尺码推荐")
    print(res)