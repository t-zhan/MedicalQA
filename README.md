# MedicalQA
This is a medical question answering system with RAG on a medical knowledge graph.

## Environment Requirements
CUDA 11.8  
Python 3.12  
PyTorch 2.5.1  
streamlit 1.40.1  
All the packages can be installed by `conda env create -f environment.yaml`.

## Project Structure
MedicalQA  
├──&ensp;data  
│&ensp;&ensp;&ensp;&ensp;├──&ensp;kg_info  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├── fixed_entity_relation  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;entity  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;relation  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;└──&ensp;match_info  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;└──&ensp;raw_data  
│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
├──&ensp;models  
│&ensp;&ensp;&ensp;&ensp;├──&ensp;Meta-Llama-3.1-8B-Instruct  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;├──&ensp;Qwen2.5-14B-Instruct  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
│&ensp;&ensp;&ensp;&ensp;└──&ensp;text2vec-base-chinese  
│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;└──&ensp;...  
├──&ensp;src  
│&ensp;&ensp;&ensp;&ensp;├──&ensp;knowledge_graph  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;\_\_init\_\_.py  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;disease_to_cypher.py  
│&ensp;&ensp;&ensp;&ensp;├──&ensp;match  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;\_\_init\_\_.py  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;index_type.py  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;match_utils.py  
│&ensp;&ensp;&ensp;&ensp;├──&ensp;question_answer  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;\_\_init\_\_.py  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;├──&ensp;generate.py  
│&ensp;&ensp;&ensp;&ensp;│&ensp;&ensp;&ensp;&ensp;└──&ensp;load.py  
│&ensp;&ensp;&ensp;&ensp;└──&ensp;ui  
│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;├──&ensp;\_\_init\_\_.py  
│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;├──&ensp;login.py  
│&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;└──&ensp;user_data_storage.py  
├──&ensp;tmp  
│&ensp;&ensp;&ensp;&ensp;└──&ensp;user_credentials.json  
├──&ensp;main.py  
├──&ensp;logo.jpg  
├──&ensp;environment.yaml  
├──&ensp;LICENSE  
└──&ensp;README.md  

## Preprocess  
`python src/preprocess/convert_rawjson_to_line.py`  
`python src/preprocess/old_new_merge.py`  
`python src/preprocess/other_mecical_merge.py`  

These will generate 4 json files in `data/raw_data`.
## Run
`streamlit run main.py`

## Demo
https://github.com/user-attachments/assets/f785cacf-d4de-47c2-8832-b8334e3af99a

## Acknowledge
[基于RAG与大模型技术的医疗问答系统](https://github.com/honeyandme/RAGQnASystem)
