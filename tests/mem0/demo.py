import os
from mem0 import Memory


m = Memory()
m.add("我喜欢吃车厘子🍒", user_id="misakar", metadata={"category": "喜欢吃什么"})
m.add("我喜欢写代码", user_id="misakar", metadata={"category": "喜好做什么"})

all_memories = m.get_all()
print("---- all memories ---")
print(all_memories)

related_memories = m.search(query="misakar喜欢吃什么", user_id="misakar")
print("---- searched result ----")
print([m for m in related_memories if m["score"] > 0.5])
