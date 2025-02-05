import os
from mem0 import Memory

config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4JPASS")
        }
    },
    "version": "v1.1"
}

m = Memory.from_config(config_dict=config)
# m.add("我喜欢路畅", user_id="misakar")
# m.add("我喜欢喂奶豆吃狗粮", user_id="misakar")
all_memories = m.get_all(user_id="misakar")
print("---- all memories ----")
print(all_memories)

print("---- search results ----")
print(m.search("misakar喜欢谁", user_id="misakar"))
