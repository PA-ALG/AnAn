import os

from llama_index.core.indices.vector_store import VectorIndexRetriever

# from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import PromptTemplate, StorageContext, VectorStoreIndex, \
    SimpleDirectoryReader, Settings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, StepDecomposeQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
import warnings

from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, \
    DashScopeTextEmbeddingType

from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.retrievers.bm25 import BM25Retriever
from stemmer.stemmer import Stemmer

from llm.liteqwen import LiteQwen
from post_retrieval.postprocessor.refine import LLMRefineContentPostProcessor
from post_retrieval.postprocessor.remote_rank import RemoteRankPostprocessor
from pre_retrieval.prompts import DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT
from prompts.default_prompt_selectors import DEFAULT_TEXT_QA_PROMPT_SEL
from query_engine.standard_rag_engine import StandardModularRAGQueryEngine
from response_synthesizers.factory import get_response_synthesizer
from retriever.custom import CustomRetriever
from retriever.remote import AnAnRetriever

warnings.filterwarnings('ignore')

# _ = load_dotenv(find_dotenv())  # 导入环境
# config = dotenv_values(".env")

# embedding/llm 基于阿里云Dashcope平台

API_KEY = ""
os.environ["DASHSCOPE_API_KEY"] = API_KEY

def init_embedding_model():

    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
        api_key=API_KEY
    )
    Settings.embed_model = embed_model

    # TEST EMBED MODEL
    text_to_embedding = ["风急天高猿啸哀"]
    # Call text Embedding
    result_embeddings = embed_model.get_text_embedding_batch(text_to_embedding)
    # requests and embedding result index is correspond to.
    for index, embedding in enumerate(result_embeddings):
        if embedding is None:  # if the correspondence request is embedding failed.
            print("The %s embedding failed." % text_to_embedding[index])
        else:
            print("Dimension of embeddings: %s" % len(embedding))
            print(
                "Input: %s, embedding is: %s"
                % (text_to_embedding[index], embedding[:5])
            )
    return embed_model

def init_llm():
    dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_TURBO, api_key=API_KEY)
    # llm = LiteQwen()
    return dashscope_llm
def set_index():
    # set embeder
    # 加载大模型
    # Settings.llm = Ollama(model="qwen2:1.5b", request_timeout=30.0, temperature=0)
    Settings.llm = init_llm()
    # load data
    documents = SimpleDirectoryReader("./data").load_data()

    # Sliding windows chunking & Extract nodes from documents
    node_parser = SentenceWindowNodeParser.from_defaults(
        # how many sentences on either side to capture
        window_size=3,
        # the metadata key that holds the window of surrounding sentences
        window_metadata_key="window",
        # the metadata key that holds the original sentence
        original_text_metadata_key="original_sentence"
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    # indexing & storing
    persist_dir = "storeQ"
    os.makedirs(persist_dir, exist_ok=True)

    try:
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
        )
        index = load_index_from_storage(storage_context)
    except:
        index = VectorStoreIndex(nodes=nodes, embed_model=init_embedding_model())
        index.storage_context.persist(persist_dir=persist_dir)

    return index, nodes


def set_pre_retrieval():
    stepback_decompose_query_engine = StepDecomposeQueryTransform(llm=init_llm(),
                                                        step_decompose_query_prompt=DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT)
    return stepback_decompose_query_engine


def set_retrieval(embed_model, nodes, index):
    # bm25_retriever = BM25Retriever.from_defaults(
    #     nodes=nodes,
    #     similarity_top_k=5,
    # )
    # vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5, embed_model=embed_model)
    # hybrid_retriever = CustomRetriever(vector_retriever, bm25_retriever, mode="AND", alpha=0.3)
    api_retriever = AnAnRetriever(api_url="https://localhost:5000/api/v1/query")
    return api_retriever


def set_postprocessor():
    llm_refine_context_node_processor = LLMRefineContentPostProcessor() # 改写
    # remote_reranker = RemoteRankPostprocessor() # 对接rank service
    # return [remote_reranker]
    return [llm_refine_context_node_processor]


def set_response_synthesizer():
    response_synthesizer = get_response_synthesizer(
        llm=init_llm(),
        text_qa_template=DEFAULT_TEXT_QA_PROMPT_SEL,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE
    )
    return response_synthesizer

if __name__ == '__main__':
    # 预先的准备：embedding model 、节点索引
    # 注意：节点索引应用于线下验证，如用于线上的索引，我们则无需预先准备上面两者，而直接从构建preretrieval即可
    embed_model = init_embedding_model()
    index, nodes = set_index()

    # RAG流程内各模块
    pre_retrieval = set_pre_retrieval()
    retriever = set_retrieval(embed_model, nodes, index)
    post_retrival = set_postprocessor()
    response_synthesizer = set_response_synthesizer()

    # 模块构建RAG pipline
    standard_modular_rag_query_engine = StandardModularRAGQueryEngine(
        pre_retrival=pre_retrieval,
        retriever=retriever,
        post_retrival=post_retrival,
        response_synthesizer=response_synthesizer
    )

    # 测试
    query = "荨麻疹，急性的，医生开药要怎么开？"
    while query:
        response = standard_modular_rag_query_engine.query(query)
        print("------------------")
        print(f"Question: {str(query)}")
        print("------------------")
        print(f"Response: {str(response)}")
        print("------------------")
        query = input("提问：")

    # if response.metadata['selector_result'].ind == 0:
    #     window = response.source_nodes[0].node.metadata["window"]  # 长度为3的窗口，包含了文本两侧的上下文。
    #     sentence = response.source_nodes[0].node.metadata["original_sentence"]  # 检索到的文本
    #     print(f"Window: {window}")
    #     print("------------------")
    #     print(f"Original Sentence: {sentence}")
    #     print("------------------")

"""
示例1：
Question: How many people are on the deck after ten o'clock?
------------------
Response: After ten o'clock, there were only three or five pairs of men and women on the deck.
------------------
Window: Sun with two empty chairs.  Fortunately, the cigarette incident just now fell into their eyes.  That evening, there was a sea breeze and the boat was a bit bumpy.  After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words.  Fang Hongjian and Miss Bao walked side by side without saying a word.  A big wave shook the hull of the ship, and Miss Bao couldn't stand steadily.  Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily. 
------------------
Original Sentence: After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words. 
------------------
示例2：
Question: Who did Fang Hongjian kiss?
------------------
Response: Fang Hongjian kissed Miss Bao.
------------------
Window: After ten o'clock, there were only three or five pairs of men and women on the deck, all hiding in the dark shadows that could not be illuminated by the lights, whispering sweet words.  Fang Hongjian and Miss Bao walked side by side without saying a word.  A big wave shook the hull of the ship, and Miss Bao couldn't stand steadily.  Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily.  Miss Bao's lips hinted, her body obediently, and this hurried and rough kiss gradually stabilized, growing perfectly close.  Miss Bao deftly pushed away Fang Hongjian's arm, took a deep breath, and said, "I'm suffocating you!  I'm catching a cold and can't breathe in my nose - it's too cheap for you, you haven't begged me to love you yet.
------------------
Original Sentence: Fang Hongjian hooked her waist and leaned against the railing, kissing her greedily. 
------------------
示例3：
Question: what is computer?
------------------
Response: A computer is an electronic device that can be programmed to carry out a sequence of arithmetic or logical operations automatically. It is a versatile and complex tool that has become an integral part of modern life. Computers can perform a wide variety of tasks, such as processing and storing large amounts of information, displaying graphics and text, connecting to the internet, and running applications that help with productivity, creativity, and communication.
At its most basic level, a computer consists of a central processing unit (CPU) that performs most of the calculations, and memory that stores both data and instructions for the CPU. It also includes input devices (like keyboards and mice), output devices (like monitors and printers), and storage devices (like hard drives or solid-state drives) for long-term data storage.
Computers can range in size from large mainframes that fill entire rooms to small and powerful devices like smartphones that can fit in your pocket. They can be general-purpose, like personal computers, or designed for specific tasks, like embedded systems in cars or appliances.
The field of computer science is dedicated to the study of computers, including their design, hardware and software components, applications, and the theoretical limits of computation.
------------------
"""
