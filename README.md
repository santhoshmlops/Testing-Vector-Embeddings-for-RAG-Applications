# Testing Vector Embeddings Models for RAG-Applications

#### Objective
The primary goal of this project is to evaluate and compare various vector embedding models in the context of Retrieval-Augmented Generation (RAG) applications. This involves determining which models produce the most accurate and contextually relevant embeddings for use in RAG systems. The evaluation will be based on the cosine similarity metric to quantify the similarity between model outputs for the same queries.

#### Background
Retrieval-Augmented Generation (RAG) is a hybrid approach that enhances the performance of generative models by incorporating relevant information retrieved from large datasets. This involves two main steps: retrieving relevant documents or passages using vector embeddings and then using these retrieved pieces of information to generate responses. The quality of the vector embeddings directly impacts the retrieval accuracy and, consequently, the overall performance of the RAG system. Evaluating different embedding models ensures that the most effective ones are utilized, leading to better retrieval and generation results.

#### Methodology
1. **Model Selection**:
   - Select a diverse set of vector embedding models, including Hugging Face models:
  
     * sentence-transformers/all-MiniLM-L6-v2
     * jinaai/jina-embeddings-v2-base-en
     * maidalun1020/bce-embedding-base_v1
     * BAAI/bge-small-en-v1.5
     * infgrad/stella-base-en-v2
     * thenlper/gte-small
     * Snowflake/snowflake-arctic-embed-m
     * avsolatorio/GIST-Embedding-v0
       

2. **Data Preparation**:
   - Curate a dataset consisting of text queries and relevant documents or passages. This dataset should be representative of the types of information typically handled by the RAG system.

     documents = ["The cat sat on the mat.",
     
        "The dog barked at the mailman.",
     
        "The quick brown fox jumps over the lazy dog.",
     
        "I love playing with my cat.",
     
        "The mailman delivered the package.",
     
        "She sells seashells by the seashore.",
     
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
     
        "A journey of a thousand miles begins with a single step.",
     
        "I am the greatest!",
     
        "All that glitters is not gold."],
     
     ids = ["id1", "id2","id3","id4","id5","id6","id7","id8","id9","id10"]

4. **Embedding Generation**:
   - Implement a Python script to process the text queries and relevant documents through each selected model, generating corresponding vector embeddings.

5. **Cosine Similarity Calculation**:
   - Compute the cosine similarity between the query embeddings and the document embeddings produced by different models. The cosine similarity metric will help determine how well the models retrieve relevant documents based on the similarity of the embeddings.

6. **Performance Evaluation**:
   - Analyze the cosine similarity scores to evaluate which models retrieve the most relevant documents for each query. Higher cosine similarity indicates better retrieval performance.

7. **Result Interpretation**:
   - Interpret the results to identify the models that provide the best retrieval performance. Discuss the scenarios where specific models excel and potential reasons for their performance differences.
  
  ### Query_texts = "What did the dog do?"
  
| Model Name | Ids | Distance Score |  Retrieved Documents | Result |
|----------|:-------------:|----------|:-------------:|:-------------:|
| sentence-transformers/all-MiniLM-L6-v2 | ids: [['id2', 'id3']] | distances: [[0.5399942398071289, 0.5448101758956909]] | documents': [['The dog barked at the mailman.' 'The quick brown fox jumps over the lazy dog.']] | 1 |
| BAAI/bge-small-en-v1.5 | ids: [['id2', 'id3']] | distances: [[0.334303081035614, 0.45860934257507324]] | documents': [['The dog barked at the mailman.','The quick brown fox jumps over the lazy dog.']] | 2 |
| infgrad/stella-base-en-v2 | ids: [['id2', 'id3']] | distances: [[0.34219467639923096, 0.38874876499176025]] | documents': [['The dog barked at the mailman.' 'The quick brown fox jumps over the lazy dog.']] | 3 |
| avsolatorio/GIST-Embedding-v0 |ids: [['id2', 'id3']] | distances: [[0.23246049880981445, 0.26549404859542847]] | documents': [['The dog barked at the mailman.' 'The quick brown fox jumps over the lazy dog.']] | 4 |
| Snowflake/snowflake-arctic-embed-m |ids: [['id2', 'id3']] | distances: [[0.1772594451904297, 0.18935298919677734]] | documents': [['The dog barked at the mailman.' 'The quick brown fox jumps over the lazy dog.']] | 5 |
| thenlper/gte-small  |ids: [['id2', 'id3']] | distances: [[0.12128406763076782, 0.14693737030029297]] | documents': [['The dog barked at the mailman.' 'The quick brown fox jumps over the lazy dog.']] | 6 |
| maidalun1020/bce-embedding-base_v1 | ids: [['id2', 'id7']] | distances: [[0.49716949462890625, 0.6792410612106323]] | documents': [['The dog barked at the mailman.','How much wood would a woodchuck chuck if a woodchuck could chuck wood?']] | 7 |
| jinaai/jina-embeddings-v2-base-en | ids: [['id9', 'id6'] | distances: [[0.528051495552063, 0.5929942727088928]] | documents': [['I am the greatest!', 'She sells seashells by the seashore.']] | 8 | 


#### Expected Outcomes
- A detailed comparison of vector embedding models based on their ability to retrieve relevant documents in RAG applications.
- Insights into the most effective models for enhancing RAG systems.
- Recommendations for selecting vector embedding models to optimize retrieval and generation components in RAG systems.

