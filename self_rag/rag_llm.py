"""
rag检索问答系统

1.定义 获取历史记录 的函数
2.定义  rag问答系统的类
        （1）.声明向量库
        （2）.导入llm模型
        （3）.提示词模板
        （4）.问答langchain链

"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from vector_search import VectorSearch
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

store={}
def history_get(session_id: str) :
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
class rag_llm():
    def __init__(self):
        self.vector_search = VectorSearch()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "你是一个温柔的美女助理，以参考资料为主，并根据历史消息回答问题，不知道就说不知道。参考资料：{context}"),
                ("system", "历史消息如下"),
                MessagesPlaceholder("history"),
                ("user", "{input}")
            ]
        )
        self.chat_model =ChatTongyi(model="qwen3-max")
        self.chain = self.get_chain()

    def get_chain(self):

        def get_input(value):
            return value["input"]
        def put_next(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["history"] = value["input"]["history"]
            new_value["context"] = value["context"]

            return new_value
        search =self.vector_search.get_search()
        base_chain = (
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(get_input) | search
            }
            |RunnableLambda(put_next)
            | self.prompt | self.chat_model | StrOutputParser()

        )

        up_chain = RunnableWithMessageHistory(
            base_chain,
            history_get,
            input_messages_key="input",
            history_messages_key="history",
        )

        return up_chain


if __name__ == "__main__":
    service = rag_llm()
    session_config = {
        "configurable": {"session_id": "user_002"}
    }
    res = service.chain.invoke({"input":"我身高145，体重35kg，尺码推荐"}, session_config)
    print("最终回答：", res)
