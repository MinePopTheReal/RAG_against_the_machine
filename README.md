```text
<|im_start|>system
{sont but}<|im_end|>
<|im_start|>user
{message utilisateur}<|im_end|>
<|im_start|>assistant
{reponse du model}<|im_end|>
<|im_start|>user
{message utilisateur}<|im_end|>
```

## Instructions
    Indexing command:
    ```
    uv run python3 -m src index
    # Optional argument:
        --max_chunk_size (default -> 2000)
        --repository_path (default -> 'data/raw/vllm-0.10.1')
        --chunk_overlap (default -> 0)
        --output_path (default -> 'data/processed')
    ```
    Search_dataset command:
    ```
    uv run python3 -m src search_dataset
    # Mandatory argument:
        --dataset_path (default 'data/datasets/UnansweredQuestions/dataset_docs_public.json')
        --save_directory (default 'data/output/search_results/UnansweredQuestions')

    # Optional argument:
        --k (default -> 5)
    ```
    ```
    uv run python3 -m src answer_dataset
    
    --student_search_results_path "data/output/output.json" 
    --save_directory "data/output/output1.json"
    ```

## Ressources
[Python Fire guide](https://google.github.io/python-fire/guide/)
[Best Matching 25 doc](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/)
[Rag Blog](https://blog.stephane-robert.info/docs/developper/programmation/python/rag-introduction/)
[RAG doc](https://docs.langchain.com/oss/python/deepagents/rag#index-langchain-documentation)
[Langchain doc](https://docs.langchain.com/oss/python/deepagents/rag#load-documents)
[Langchain doc 2](https://docs.langchain.com/oss/python/integrations/splitters/code_splitter)
[Opening files recursively doc](https://www.geeksforgeeks.org/python/how-to-iterate-over-files-in-directory-using-python/)