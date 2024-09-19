# AnAn
A Modular RAG&amp;Agent app framework.


## 1. 前置
### 1. index
考虑index构建方式较为多样（上传知识库切分/index， 提前构建等等），故暂不考虑纳入当前流程，而是由用户自行决定。

### 2. LLM
建议基于CustomLLM实现，目前已经实现 liteqwen版本，zhipuai版本


## 2. pre_retrieval
基础组件：BaseQueryTransform

可以调用LLaMA-index已实现的，或者自定义。

方法：
```def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:```

## 3. retrieval
基础组件：BaseRetriever

可以调用LLaMA-index已实现的，或者自定义。已实现：

1. CustomRetriever：bm25/vector search混合检索器
2. AnAnRetriever：调用安安检索服务的检索器

方法：
```def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]```

## 4. post_retrieval
基础组件：BaseNodePostprocessor

可以调用LLaMA-index已实现的，或者自定义。已实现：

1. RemoteRankPostprocessor：调用安安重排序模型服务

方法：
```    
def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
```

## 5. response_synthesizer
基础组件：BaseSynthesizer

可以调用LLaMA-index已实现的，或者自定义。

## 6. query engine
基础组件：BaseQueryEngine

可以调用LLaMA-index已实现的，或者自定义。已实现：

1. StandardModularRAGQueryEngine：按照以上模块初始化运行的标准ModularRAG query engine

重点方法：
```    
def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
```


