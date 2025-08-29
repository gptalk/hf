from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

dataset = load_dataset("glue", "mrpc")  # 一个句子分类数据集
print(dataset['train'][0])  # 看第一条数据

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理
def preprocess_function(examples):
	return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 训练参数
training_args = TrainingArguments(
	output_dir="./results",
	evaluation_strategy="epoch",
	learning_rate=2e-5,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=8,
	num_train_epochs=3,
	weight_decay=0.01,
	logging_dir="./logs",
	logging_steps=10,
)

# Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=encoded_dataset["train"],
	eval_dataset=encoded_dataset["validation"],
)

# 开始训练
trainer.train()
