"""Description: This file contains the implementation of the `ollama` class.
This class is responsible for handling the interaction with the OpenAI API for language generation.
Compatible with all of the OpenAI Compatible endpoints, including Ollama, OpenAI, and more.
"""

from typing import Iterator
from mem0 import Memory
from openai import OpenAI, AzureOpenAI
from loguru import logger
from .llm_interface import LLMInterface
import json
import os


class LLM(LLMInterface):

    def __init__(
        self,
        user_id: str,
        base_url: str,
        model: str,
        system: str,
        mem0_config: dict,
        organization_id: str = "z",
        project_id: str = "z",
        llm_api_key: str = "z",
        verbose: bool = False,
        use_azure: bool = False,
        azure_endpoint: str = "z",
        azure_api_version: str = "z",
        azure_api_key: str = "z"
    ):
        self.base_url = base_url
        self.model = model
        self.system = system
        self.mem0_config = mem0_config
        self.user_id = user_id

        # 处理neo4j配置
        if 'graph_store' in self.mem0_config and self.mem0_config['graph_store']['provider'] == 'neo4j':
            config = self.mem0_config['graph_store']['config']
            if '${NEO4JPASS}' in config['password']:
                neo4j_pass = os.getenv('NEO4JPASS')
                if not neo4j_pass:
                    raise ValueError("环境变量NEO4JPASS未设置，请设置Neo4j数据库密码")
                config['password'] = neo4j_pass
                logger.info("已从环境变量中读取Neo4j密码")

        self.conversation_memory = []
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

        self.system = system
        self.conversation_memory = [
            {
                "role": "system",
                "content": system,
            }
        ]

        logger.debug("Initializing Memory...")
        # Initialize Memory with the configuration
        self.mem0 = Memory.from_config(self.mem0_config)
        logger.debug("Memory Initialized...")

    def _rewrite_query(self, query: str) -> str:
        """使用大模型改写查询，补充上下文"""
        try:
            system_prompt = "你是一个查询改写助手。你的任务是理解用户的输入，并将其改写成更完整的查询语句，以便更好地检索相关记忆。请确保改写后的查询包含完整的上下文信息。"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请将以下输入改写成更完整的查询语句：\n{query}"}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=False
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            logger.debug(f"Original query: {query}")
            logger.debug(f"Rewritten query: {rewritten_query}")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query
    
    def _get_relevant_memories(self, query: str) -> str:
        """获取相关记忆，基于关系图构建记忆字符串"""
        logger.debug("Searching relevant memories...")
        try:
            # 确保查询不为空
            if not query or not query.strip():
                logger.debug("Empty query provided")
                return ""
            
            # 使用大模型改写查询
            rewritten_query = self._rewrite_query(query)
            logger.debug(f"rewritten_query: {rewritten_query}")

            # 搜索相关记忆，使用改写后的查询
            try:
                relevant_memories_list = self.mem0.search(query=rewritten_query, user_id=self.user_id)
                logger.debug(f"Search result: {relevant_memories_list}")
            except Exception as search_error:
                import traceback
                error_stack = traceback.format_exc()
                logger.error(f"Error during memory search:\n{error_stack}")
                logger.error(f"Error details: {str(search_error)}")
                return ""

            # 验证搜索结果格式
            if not relevant_memories_list:
                logger.debug("Empty search result")
                return ""

            # 处理字符串类型的搜索结果
            if isinstance(relevant_memories_list, str):
                logger.debug(f"String search result: {relevant_memories_list}")
                return relevant_memories_list

            # 处理字典类型的搜索结果
            if isinstance(relevant_memories_list, dict):
                relations = relevant_memories_list.get('relations', [])
                if not isinstance(relations, list):
                    logger.debug("Invalid relations format: not a list")
                    return ""

                memories = []
                for relation in relations:
                    try:
                        if not isinstance(relation, dict):
                            logger.warning(f"Invalid relation format: {relation}")
                            continue

                        source = str(relation.get('source', '')).strip()
                        relationship = str(relation.get('relationship', '')).strip()
                        destination = str(relation.get('destination', '')).strip()
                        target = str(relation.get('target', '')).strip()

                        if source and relationship and (destination or target):
                            # 将关系转换为自然语言描述
                            memory = f"{source} {relationship} {destination if destination else target}"
                            memories.append(memory)
                        else:
                            logger.warning(f"Skipping incomplete relation: {relation}")
                    except Exception as e:
                        logger.error(f"Error processing relation {relation}: {str(e)}")
                        continue

                if not memories:
                    logger.debug("No valid memories found")
                    return ""

                return "\n".join(memories)

            logger.debug(f"Unsupported search result type: {type(relevant_memories_list)}")
            return ""

        except Exception as e:
            logger.error(f"Error while processing relations: {str(e)}")
            return ""

    def _update_system_prompt(self, relevant_memories: str):
        """更新系统提示词，加入相关记忆"""
        if relevant_memories:
            logger.debug("Adding relevant memories to system prompt...")
            self.conversation_memory[0] = {
                "role": "system",
                "content": f"""{self.system}
                
                ## Relevant Memories
                Here are something you recall from the past:
                ===== Some relevant memories =====
                {relevant_memories}
                ===== end of relevant memories =====
                
                """,
            }
        else:
            logger.debug("No relevant memories found...")
            self.conversation_memory[0] = {
                "role": "system",
                "content": f"""{self.system}""",
            }

    def chat_iter(self, prompt: str) -> Iterator[str]:
        logger.debug("All Mem:")
        logger.debug(self.mem0.get_all(user_id=self.user_id))

        # 获取相关记忆
        relevant_memories = self._get_relevant_memories(prompt)
        
        # 更新系统提示词
        self._update_system_prompt(relevant_memories)

        self.conversation_memory.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        this_conversation_mem = [prompt]

        chat_completion = []
        try:
            logger.debug("Calling the chat endpoint with...")
            logger.debug(self.conversation_memory)
            chat_completion = self.client.chat.completions.create(
                messages=self.conversation_memory,
                model=self.model,
                stream=True,
            )
        except Exception as e:
            logger.error("Error calling the chat endpoint: " + str(e))
            logger.error(self.mem0_config)
            return "Error calling the chat endpoint: " + str(e)

        def _generate_and_store_response():
            complete_response = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                yield chunk.choices[0].delta.content
                complete_response += chunk.choices[0].delta.content

            self.conversation_memory.append(
                {
                    "role": "assistant",
                    "content": complete_response,
                }
            )

            # 获取所有现有记忆并合并
            existing_memories = self.mem0.get_all(user_id=self.user_id)
            if existing_memories and isinstance(existing_memories, dict) and 'relations' in existing_memories:
                # 将关系转换为自然语言描述
                memory_strings = []
                for relation in existing_memories['relations']:
                    if isinstance(relation, dict):
                        source = relation.get('source', '').strip()
                        relationship = relation.get('relationship', '').strip()
                        target = relation.get('target', '').strip()
                        if source and relationship and target:
                            memory_string = f"{source} {relationship} {target}"
                            memory_strings.append(memory_string)
                
                # 将现有记忆和新记忆合并
                combined_memories = memory_strings + this_conversation_mem
            else:
                combined_memories = this_conversation_mem
            
            # 存储合并后的记忆
            logger.debug(self.mem0.add(combined_memories, user_id=self.user_id))

            logger.debug(f"Mem0 Added... {this_conversation_mem}")

            logger.debug("All Mem:")
            logger.debug(self.mem0.get_all(user_id=self.user_id))

            def serialize_memory(memory, filename):
                with open(filename, "w") as file:
                    json.dump(memory, file)

            serialize_memory(self.conversation_memory, "mem.json")
            return

        return _generate_and_store_response()

    def chat_with_image(self, image_data: str) -> Iterator[str]:
        """处理图片对话并返回回复的迭代器
        
        Parameters:
        - image_data (str): base64编码的图片数据
        
        Returns:
        - Iterator[str]: AI回复的迭代器
        """
        # 构建包含图片的消息
        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这张图片，并告诉我你的感想"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ]
        }
        
        self.conversation_memory.append(image_message)

        try:
            logger.debug("Calling the chat endpoint with...")
            logger.debug(self.conversation_memory)
            chat_completion = self.client.chat.completions.create(
                messages=self.conversation_memory,
                model=self.model,
                stream=True,
            )
        except Exception as e:
            logger.error("Error calling the chat endpoint: " + str(e))
            logger.error(self.mem0_config)
            return "Error calling the chat endpoint: " + str(e)

        def _generate_and_store_response():
            complete_response = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                yield chunk.choices[0].delta.content
                complete_response += chunk.choices[0].delta.content

            assistant_response = {
                "role": "assistant",
                "content": complete_response,
            }
            
            self.conversation_memory.append(assistant_response)
            this_conversation_mem = [complete_response]
            # 获取所有现有记忆并合并
            existing_memories = self.mem0.get_all(user_id=self.user_id)
            if existing_memories and isinstance(existing_memories, dict) and 'relations' in existing_memories:
                # 将关系转换为自然语言描述
                memory_strings = []
                for relation in existing_memories['relations']:
                    if isinstance(relation, dict):
                        source = relation.get('source', '').strip()
                        relationship = relation.get('relationship', '').strip()
                        target = relation.get('target', '').strip()
                        if source and relationship and target:
                            memory_string = f"{source} {relationship} {target}"
                            memory_strings.append(memory_string)
                
                # 将现有记忆和新记忆合并
                combined_memories = memory_strings + this_conversation_mem
            else:
                combined_memories = this_conversation_mem
            
            # 存储合并后的记忆
            logger.debug(self.mem0.add(combined_memories, user_id=self.user_id))
            logger.debug(f"Mem0 Added... {this_conversation_mem}")

            def serialize_memory(memory, filename):
                with open(filename, "w") as file:
                    json.dump(memory, file)

            serialize_memory(self.conversation_memory, "mem.json")
            return

        return _generate_and_store_response()

    def handle_interrupt(self, heard_response: str) -> None:
        if self.conversation_memory[-1]["role"] == "assistant":
            self.conversation_memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self.conversation_memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )
        self.conversation_memory.append(
            {
                "role": "system",
                "content": "[Interrupted by user]",
            }
        )


def test():

    test_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test",
                "host": "localhost",
                "port": 6333,
                "embedding_model_dims": 768,  # Change this according to your local model's dimensions
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:latest",
                "temperature": 0,
                "max_tokens": 8000,
                "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "mxbai-embed-large:latest",
                # Alternatively, you can use "snowflake-arctic-embed:latest"
                "ollama_base_url": "http://localhost:11434",
            },
        },
    }

    llm = LLM(
        user_id="rina",
        base_url="http://localhost:11434/v1",
        model="llama3:latest",
        mem0_config=test_config,
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
