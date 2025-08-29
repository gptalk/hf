from transformers import pipeline

# 下载模型（HF自动帮你下）
chat_bot = pipeline("text-generation", model="gpt2", device=0)  # device=0用第一块显卡，加速！

# 问问题
question = "What is AI?"
answer = chat_bot(question, max_length=50)
print(answer[0]['generated_text'])