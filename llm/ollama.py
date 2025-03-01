""" Description: This file contains the implementation of the `ollama` class.
This class is responsible for handling the interaction with the OpenAI API for 
language generation.
And it is compatible with all of the OpenAI Compatible endpoints, including Ollama, 
OpenAI, and more.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Iterator, Optional
from openai import OpenAI, AzureOpenAI
from langchain import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from datetime import datetime
from copy import deepcopy

from rich.prompt import Prompt

from .llm_interface import LLMInterface


class LLM(LLMInterface):

    def __init__(
        self,
        base_url: str,
        model: str,
        system: str,
        callback=print,
        organization_id: str = "z",
        project_id: str = "z",
        llm_api_key: str = "z",
        verbose: bool = False,
        use_azure: bool = False,
        azure_endpoint: str = "z",
        azure_api_version: str = "z",
        azure_api_key: str = "z"
    ):
        """
        Initializes an instance of the `ollama` class.

        Parameters:
        - base_url (str): The base URL for the OpenAI API.
        - model (str): The model to be used for language generation.
        - system (str): The system to be used for language generation.
        - callback [DEPRECATED] (function, optional): The callback function to be called after each API call. Defaults to `print`.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to an empty string.
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to an empty string.
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to an empty string.
        - verbose (bool, optional): Whether to enable verbose mode. Defaults to `False`.
        """

        self.base_url = base_url
        self.model = model
        self.system = system
        self.callback = callback
        self.memory = []
        self.verbose = verbose
        if not use_azure:
            self.client = OpenAI(
                base_url=base_url,
                organization=organization_id,
                project=project_id,
                api_key=llm_api_key,
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                api_key=azure_api_key,
            )

        self.__set_system(system)

        if self.verbose:
            self.__printDebugInfo()

        # 记忆
        self.mem_collection_name = "mem_collection"
        self.milvus_client = MilvusClient(uri="./mem_milvus.db")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.embedding_cache = self._load_embedding_cache()
        self.dim = self.embedding_model._client.get_sentence_embedding_dimension()
        self._init_milvus_collection()

    def _load_embedding_cache(self):
        cache_file = Path("embedding_cache.pkl")
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_embedding_cache(self):
        with open("embedding_cache.pkl", "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def _generate_embedding(self, text):
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        vector = self.embedding_model.embed_query(text)
        self.embedding_cache[cache_key] = vector
        return vector

    def _init_milvus_collection(self):
        if not self.milvus_client.has_collection("memory_vectors"):
            self.milvus_client.create_collection(
                collection_name=self.mem_collection_name,
                dimension=self.dim,
                primary_field_name="id",
                vector_field_name="vector",
                auto_id=True,
                enable_dynamic_field=True
            )

    def __set_system(self, system):
        """
        Set the system prompt
        system: str
            the system prompt
        """
        self.system = system
        self.memory.append(
            {
                "role": "system",
                "content": system,
            }
        )

    def __print_memory(self):
        """
        Print the memory
        """
        print("Memory:\n========\n")
        # for message in self.memory:
        print(self.memory)
        print("\n========\n")

    def __printDebugInfo(self):
        print(" -- Base URL: " + self.base_url)
        print(" -- Model: " + self.model)
        print(" -- System: " + self.system)

    def chat_iter(self, prompt: str, image_data: Optional[str]=None, image_list=None) -> Iterator[str]:

        self.memory.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        messages = deepcopy(self.memory)

        # 检索添加记忆
        ## 这里的prompt应该结合上下文，生成检索词
        search_prompt = PromptTemplate(
            template="""基于以下对话历史和用户的最新输入，生成一个完整的检索句子，以便从记忆数据库中检索相关信息。

- 对话历史：{chat_history}
- 用户当前输入：{query}
- 用户名：Misakar
- 你的名字：Ghost

请以自然语言的方式生成一个检索句子，使其能够反映当前查询的语境和意图，注意将检索查询中的用户名、时间替换为真实值，直接输出检索句子的内容""",
            input_variables = ["query", "chat_history"]
        )
        gpt4o_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_API_KEY"),
            openai_api_version="2023-07-01-preview",
            deployment_name="gpt-4o-2024-08-06",
            temperature=0
        )
        query_chain = (
            search_prompt
            | gpt4o_llm.bind(response_format={"type": "text"})
            | StrOutputParser()
        )
        search_query = query_chain.invoke({"query": prompt, "chat_history": messages})
        print("-- search_query: ", search_query)
        query_embedding = self._generate_embedding(search_query)
        mem_results = self.milvus_client.search(
            collection_name=self.mem_collection_name,
            data = [query_embedding],
            output_fields=["source", "source_time", "memory_time", "content", "delete"],
            limit = 50,
        )[0]
        memories = ""
        for result in mem_results:
            mem = result["entity"]
            if mem.get("delete", False):
                continue
            memories += f"-记忆来源:{mem['source']}; -记忆来源时间:{mem['source_time']}; -记忆相关时间:{mem['memory_time']}; -记忆内容:{mem['content']}\n"
            print(mem)
        messages[0]["content"] += f"\n当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if memories:
            messages[0]["content"] += f"\n你有如下长期记忆:\n{memories}"

        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data.decode()}",
                        }
                    }
                ]
            })

        print("image_list==>", len(image_list))
        print("image_list==>", [image["frame_time"] for image in image_list])
        if image_list:
            messages.append({
                "role": "user",
                "content": []
            })
            for idx, image in enumerate(image_list):
                messages[-1]["content"].append({
                    "type": "text",
                    "text": f"下面是第{idx}帧，截取时间是{image['frame_time']}"
                })
                messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image['frame'].decode()}"
                    }
                })

        self.verbose = True
        if self.verbose:
            # self.__print_memory()
            print(" -- Base URL: " + self.base_url)
            print(" -- Model: " + self.model)
            # print(" -- Messages: ", messages)

        chat_completion = []
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=True,
            )
        except Exception as e:
            print("Error calling the chat endpoint: " + str(e))
            self.__printDebugInfo()
            return "Error calling the chat endpoint: " + str(e)

        # a generator to give back an iterator to the response that will store
        # the complete response in memory once the iteration is done
        def _generate_and_store_response():
            complete_response = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                yield chunk.choices[0].delta.content
                complete_response += chunk.choices[0].delta.content

            self.memory.append(
                {
                    "role": "assistant",
                    "content": complete_response,
                }
            )

            def serialize_memory(memory, filename):
                with open(filename, "w") as file:
                    json.dump(memory, file, ensure_ascii=False)

            serialize_memory(self.memory, "mem.json")

            # 异步提取和存储记忆
            print("--- store memory ----")
            self.store_memory()

            return

        return _generate_and_store_response()

    def chat_with_image(self, image_data: str) -> Iterator[str]:
        """
        处理图片并返回回复的迭代器
        
        Parameters:
        - image_data (str): base64编码的图片数据
        
        Returns:
        - Iterator[str]: AI回复的迭代器
        """
        # 构建包含图片的消息
        self.memory.append(
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": "请说出你看到这张图片后的感想"
                    # },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    },
                ]
            }
        )
        
        if self.verbose:
            self.__print_memory()
            print(" -- Base URL: " + self.base_url)
            print(" -- Model: " + self.model)
            print(" -- System: " + self.system)
            print(" -- Image data received")

        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.memory,
                model=self.model,
                stream=True,
            )
            
            def _generate_and_store_response():
                complete_response = ""
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content is None:
                        chunk.choices[0].delta.content = ""
                    yield chunk.choices[0].delta.content
                    complete_response += chunk.choices[0].delta.content

                self.memory.append(
                    {
                        "role": "assistant",
                        "content": complete_response,
                    }
                )

                def serialize_memory(memory, filename):
                    with open(filename, "w") as file:
                        json.dump(memory, file)

                serialize_memory(self.memory, "mem.json")
                return

            return _generate_and_store_response()
            
        except Exception as e:
            print("Error processing image: " + str(e))
            self.__printDebugInfo()
            raise e

    def store_memory(self):
        """
        1. 从最近20轮对话历史中提取记忆
        2. 根据语义对相似记忆去重
        """
        memory = deepcopy(self.memory)
        memory.pop(0)
        latest_chat_history = memory[-20:]
        mem_prompt_tpl = PromptTemplate(
            template="""请从下面的用户与你的对话历史中提取你觉得需要记住的关键信息，要理解对话历史中潜在的意图和属性，并输出 JSON 格式：
- 结合上下文进行推理，识别实体的属性（比如性别、年龄、种类等），例如：
  - 如果用户说“我有只狗叫奶豆”，可以推理出奶豆的物种是“狗”。
  - 如果用户提到“某本书说柴犬不能吃人的食物”，并且之前提到“奶豆”吃狗粮，那可以推理出“奶豆”是柴犬。
  - 如果用户提到“奶豆是柴犬，**她**喜欢吃狗粮”，那么可以推理出奶豆是**母柴犬**。
- 对于可以推理出来的隐含关系，请在 JSON 中标记 `"inferred": true`，否则标记 `"inferred": false`。
- 对于用户告知错误的记忆，需要标记`"delete": true`，否则标记 `"delete": false`。
- 识别记忆相关时间信息（格式为YYYY-MM-DD），对于“最近”这类时间概念可以根据当前时间推测一个时间（比如当前时间是2025-02-10，那么最近可以是2025-02-10），如果没有时间那么时间未知
- 识别记忆的来源（来源于特定用户、书籍、其他人等）。
- content: 对应的原始记忆片段，将原始片段中的人物、时间（包括“最近”等时间概念）、地点具体化成你知道的信息，比如：“奶豆去年是我们的宠物” 替换为 “奶豆2024年是misakar和hitomi的宠物“
- 结果以 JSON 格式返回，每个JSON Item只包含一个记忆片段
---

**示例：**
**用户与你的对话历史：**
"用户：我有只狗叫奶豆，她去年喜欢吃狗粮，今年我姐和我说她喜欢吃鱼干。我上次看到有本书里面提到柴犬不能吃人的食物，所以我都喂她狗粮。另外奶豆不是公狗，是母狗"

**输出示例（优化后的 JSON）：**
```json
{{
    "memories": [
        {{
            "memory_time": "2024-02-08",
            "source": "来自用户Misakar告诉了我",
            "source_time": "2025-02-08",
            "inferred": false,
            "delete": false,
            "content": "奶豆在2024年喜欢吃狗粮"
        }},
        {{
            "memory_time": "2025-02-08",
            "source": "用户Misakar的姐姐anri告诉Misakar，Misakar告诉了我",
            "source_time": "2025-02-08",
            "inferred": false,
            "delete": false,
            "content": "misakar的姐姐anri告诉misakar奶豆在2025年喜欢吃鱼干"
        }},
        {{
            "memory_time": "未知",
            "source": "由 Ghost 通过推理得出，依据是Misakar曾提到柴犬不能吃人的食物，并且Misakar喂奶豆狗粮",
            "source_time": "2025-02-08",
            "inferred": true,
            "delete": false,
            "content": "奶豆属于柴犬"
        }},
        {{
            "memory_time": "未知",
            "source": "由misakar告诉我",
            "source_time": "2025-02-08",
            "inferred": false,
            "delete": true,
            "content": "奶豆是公狗"
        }}
    ]
}}

用户的名字是：Misakar
你的名字是：Ghost
对话时间是：{cur_time}
对话历史中 role="user" 表示是用户说的话，role="assistant" 表示是你说的话
用户与你的对话历史: {chat_history}""",
            input_variables=["cur_time", "chat_history"]
        )

        gpt4o_llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_API_KEY"),
            openai_api_version="2023-07-01-preview",
            deployment_name="gpt-4o-2024-08-06",
            temperature=0
        )

        mem_chain = (
            mem_prompt_tpl
            | gpt4o_llm.bind(response_format={"type": "json_object"})
            | JsonOutputParser()
        )
        memories = mem_chain.invoke({"cur_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "chat_history": latest_chat_history})["memories"]
        print("-- debug: 提取记忆")
        print(memories)

        print("-- debug: 计算记忆内容embedding")
        add_mem_list = []
        del_mem_list = []
        for mem in memories:
            if mem.get("delete", False):
                del_mem_list.append(mem["content"])
                continue
            # combined_text = f"记忆所属时间：{mem['memory_time']}; 记忆来源：{mem['source']}; 记忆来源时间：{mem['source_time']}; 记忆内容：{mem['content']}"
            mem_vector = self._generate_embedding(mem['content'])
            # 相似记忆去重
            existing_memories = self.milvus_client.search(
                collection_name=self.mem_collection_name,
                data=[mem_vector],
                output_fields=["source", "source_time", "memory_time", "content", "inferred"],
                limit=1
            )[0]
            if existing_memories and existing_memories[0]["distance"] > 0.99:
                print(f"-- 记忆被去重：已有记忆:{existing_memories[0]["entity"]}, 新记忆:{mem}")
                continue
            add_mem_list.append({
                "vector": mem_vector,
                "memory_time": mem.get("memory_time", ""),
                "source": mem["source"],
                "source_time": mem["source_time"],
                "content": mem.get("content"),
                "inferred": mem["inferred"],
                "delete": False,
            })

        print("-- debug: 存储记忆")
        if add_mem_list:
            insert_result = self.milvus_client.insert(self.mem_collection_name, add_mem_list)
            print(insert_result)

        print("-- debug: 软删除错误记忆")
        delete_data = []  # 保存更新数据

        for del_mem in del_mem_list:
            # 获取待删除记忆的 embedding
            del_mem_embedding = self._generate_embedding(del_mem)

            # 检索相似记忆
            search_results = self.milvus_client.search(
                collection_name=self.mem_collection_name,
                data=[del_mem_embedding],
                output_fields=["vector", "source", "source_time", "memory_time", "content", "inferred", "delete"],
                limit=20  # 搜索最相似的 20 条记忆
            )

            # 遍历检索结果
            for hits in search_results[0]:
                if hits["entity"].get("delete", False):
                    continue
                memory_id = hits["id"]
                similarity = hits["distance"]  # 得到余弦相似度
                content = hits["entity"]["content"]
                if similarity > 0.97:
                    # 添加待更新的 ID 和更新数据
                    data = hits["entity"]
                    data["id"] = memory_id
                    data["delete"] = True
                    delete_data.append(data)
                    print(f"Marked memory {content}, {similarity} as deleted.")
                else:
                    print(f"Memory {content}, {similarity} is not similar enough to {del_mem}.")

        if delete_data:
            print(f"delete_data: {delete_data}")
            # 批量更新删除标记
            try:
                self.milvus_client.upsert(
                    collection_name=self.mem_collection_name,
                    data=delete_data
                )
                print(f"Successfully updated {len(delete_data)} memories to be deleted.")
            except Exception as e:
                print(f"Error updating memories: {str(e)}")
        else:
            print("No memories to update.")

    def handle_interrupt(self, heard_response: str) -> None:
        if self.memory[-1]["role"] == "assistant":
            self.memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self.memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )
        self.memory.append(
            {
                "role": "system",
                "content": "[Interrupted by user]",
            }
        )


def test():
    llm = LLM(
        base_url="http://localhost:11434/v1",
        model="llama3:latest",
        callback=print,
        system='You are a sarcastic AI chatbot who loves to the jokes "Get out and touch some grass"',
        organization_id="organization_id",
        project_id="project_id",
        llm_api_key="llm_api_key",
        verbose=True,
    )
    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        chat_complet = llm.chat_iter(input(">> "))

        for chunk in chat_complet:
            if chunk:
                print(chunk, end="")


if __name__ == "__main__":
    test()
