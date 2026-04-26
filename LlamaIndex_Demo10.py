from pymilvus import MilvusClient, DataType
import time
from zhipuai import ZhipuAI

# 初始化客户端（复用之前的配置）
milvus_client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
COLLECTION_NAME = "milvus_partition_demo"
zhipu_client = ZhipuAI(api_key="ef8d257de3fe4295acb233582cd0b234.xKBQwQzOHerFaxFI")

# 1. 重建集合（先清空旧数据）
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

schema = milvus_client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("category", DataType.VARCHAR, max_length=100)  # 新增「分类字段」，用于分区
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# 创建 HNSW 索引（适配 1024 维向量的最优参数）
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",  # 文本语义检索必选余弦相似度
    params={"M": 16, "efConstruction": 128}  # 1024维向量推荐参数
)
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)

# 2. 创建分区（按「文档分类」分区：AI基础、Milvus教程、RAG实战）
milvus_client.create_partition(COLLECTION_NAME, "ai_basic")    # AI基础分区
milvus_client.create_partition(COLLECTION_NAME, "milvus_tutorial")  # Milvus教程分区
milvus_client.create_partition(COLLECTION_NAME, "rag_practice")    # RAG实战分区
print("✅ 3个分区创建成功")

# 3. 按分区插入数据（不同分类的文本插入对应分区）
def text_to_embedding(text):
    try:
        res = zhipu_client.embeddings.create(model="embedding-2", input=text)
        return [float(x) for x in res.data[0].embedding]
    except:
        return []

# 待插入数据（带分类标签）
knowledge_data = [
    # AI基础分区数据
    {"text": "向量是大模型语义表示的核心形式，由Embedding模型生成", 
     "category": "ai_basic", "embedding": text_to_embedding("向量是大模型语义表示的核心形式")},
    # Milvus教程分区数据
    {"text": "Milvus的HNSW索引通过调整M参数可平衡检索精度和速度", 
     "category": "milvus_tutorial", "embedding": text_to_embedding("Milvus的HNSW索引调整M参数")},
    # RAG实战分区数据
    {"text": "RAG流程中，Milvus检索结果需和Prompt拼接后传给大模型", 
     "category": "rag_practice", "embedding": text_to_embedding("RAG流程中Milvus检索结果拼接Prompt")},
]

# 按分区插入（核心：指定partition_name）
for data in knowledge_data:
    if data["embedding"]:
        milvus_client.insert(
            COLLECTION_NAME,
            data=[data],
            partition_name=data["category"]  # 插入对应分区
        )
milvus_client.flush(COLLECTION_NAME)
print(f"✅ 按分区插入完成，总数据量：{milvus_client.get_collection_stats(COLLECTION_NAME)['row_count']} 条")

# 4. 按分区检索（只查Milvus教程分区，速度更快）
query_question = "Milvus的HNSW索引怎么调优？"
query_vector = text_to_embedding(query_question)

# 只检索「milvus_tutorial」分区
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    partition_names=["milvus_tutorial"],  # 指定分区
    limit=2,
    output_fields=["text", "category"]
)

# 打印结果
print("\n🎯 仅检索Milvus教程分区的结果：")
if search_results and len(search_results[0])>0:
    res = search_results[0][0]
    print(f"相似度：{1-res['distance']:.4f}")
    print(f"文本：{res['entity']['text']}")
    print(f"所属分区：{res['entity']['category']}")