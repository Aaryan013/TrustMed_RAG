# Outputs from meditrust (4).ipynb

## Cell 1
```text
/kaggle/input/datasets/summersamar/medical-q-and-a-data/Diabetes_Digestive_Kidney_Cleaned.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/all_questions_answers.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/Genetics_Home_Reference.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/others.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/cancer.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/Genetic_and_Rare_Diseases.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/Neurological_Disorders_Stroke.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/seniorHealth.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/Heart_Lung_Blood.csv
/kaggle/input/datasets/summersamar/medical-q-and-a-data/MedQuAD_csv/Disease_Control_Prevention.csv
/kaggle/input/datasets/aaryan801/embeddings-chunks/medical_embeddings.npy
/kaggle/input/datasets/aaryan801/embeddings-chunks/chunked_medical_dataset (1).csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/growth_hormone_receptorQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/MedicalQuestionAnswering.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/Disease_Control_and_PreventionQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/Genetic_and_Rare_DiseasesQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/CancerQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/Neurological_Disorders_and_StrokeQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/Heart_Lung_and_BloodQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/SeniorHealthQA.csv
/kaggle/input/datasets/gvaldenebro/cancer-q-and-a-dataset/OtherQA.csv
```

## Cell 2
```text

Final Shape: (16359, 6)
```
```text
                topic          focus           qtype  \
0  Urological Disease  Kidney Stones     information   
1  Urological Disease  Kidney Stones     information   
2  Urological Disease  Kidney Stones  susceptibility   
3  Urological Disease  Kidney Stones          causes   
4  Urological Disease  Kidney Stones     information   

                                        question  \
0        What is (are) Kidney Stones in Adults ?   
1        What is (are) Kidney Stones in Adults ?   
2  Who is at risk for Kidney Stones in Adults? ?   
3          What causes Kidney Stones in Adults ?   
4        What is (are) Kidney Stones in Adults ?   

                                              answer  \
0  A kidney stone is a solid piece of material th...   
1  The urinary tract is the bodys drainage system...   
2  Anyone can get a kidney stone, but some people...   
3  Kidney stones can form when substances in the ...   
4  Four major types of kidney stones can form:\n ...   

                                                 url  
0  https://www.niddk.nih.gov/health-information/u...  
1  https://www.niddk.nih.gov/health-information/u...  
2  https://www.niddk.nih.gov/health-information/u...  
3  https://www.niddk.nih.gov/health-information/u...  
4  https://www.niddk.nih.gov/health-information/u...  
```
```html
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topic</th>
      <th>focus</th>
      <th>qtype</th>
      <th>question</th>
      <th>answer</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>information</td>
      <td>What is (are) Kidney Stones in Adults ?</td>
      <td>A kidney stone is a solid piece of material th...</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>information</td>
      <td>What is (are) Kidney Stones in Adults ?</td>
      <td>The urinary tract is the bodys drainage system...</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>susceptibility</td>
      <td>Who is at risk for Kidney Stones in Adults? ?</td>
      <td>Anyone can get a kidney stone, but some people...</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>causes</td>
      <td>What causes Kidney Stones in Adults ?</td>
      <td>Kidney stones can form when substances in the ...</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>information</td>
      <td>What is (are) Kidney Stones in Adults ?</td>
      <td>Four major types of kidney stones can form:\n ...</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
    </tr>
  </tbody>
</table>
</div>
```

## Cell 3
```text
Index(['topic', 'focus', 'qtype', 'question', 'answer', 'url'], dtype='object')
```

## Cell 4
```text
                                             context
0  Topic: Urological Disease\nFocus: Kidney Stone...
1  Topic: Urological Disease\nFocus: Kidney Stone...
2  Topic: Urological Disease\nFocus: Kidney Stone...
3  Topic: Urological Disease\nFocus: Kidney Stone...
4  Topic: Urological Disease\nFocus: Kidney Stone...
```
```html
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Topic: Urological Disease\nFocus: Kidney Stone...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Topic: Urological Disease\nFocus: Kidney Stone...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Topic: Urological Disease\nFocus: Kidney Stone...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Topic: Urological Disease\nFocus: Kidney Stone...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Topic: Urological Disease\nFocus: Kidney Stone...</td>
    </tr>
  </tbody>
</table>
</div>
```

## Cell 5
*No output*

## Cell 6
*No output*

## Cell 7
*No output*

## Cell 8
```text
<Figure size 1000x500 with 1 Axes>
```

## Cell 9
```text
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (2.0.2)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.3)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.3)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
```

## Cell 10
```text
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m549.1/549.1 kB[0m [31m11.1 MB/s[0m eta [36m0:00:00[0m00:01[0m
[?25h
```

## Cell 11
```text
Final Chunked Shape: (39326, 7)
```
```text
                topic          focus           qtype  \
0  Urological Disease  Kidney Stones     information   
1  Urological Disease  Kidney Stones     information   
2  Urological Disease  Kidney Stones     information   
3  Urological Disease  Kidney Stones  susceptibility   
4  Urological Disease  Kidney Stones          causes   

                                        question  \
0        What is (are) Kidney Stones in Adults ?   
1        What is (are) Kidney Stones in Adults ?   
2        What is (are) Kidney Stones in Adults ?   
3  Who is at risk for Kidney Stones in Adults? ?   
4          What causes Kidney Stones in Adults ?   

                                                 url  chunk_id  \
0  https://www.niddk.nih.gov/health-information/u...         1   
1  https://www.niddk.nih.gov/health-information/u...         2   
2  https://www.niddk.nih.gov/health-information/u...         0   
3  https://www.niddk.nih.gov/health-information/u...         0   
4  https://www.niddk.nih.gov/health-information/u...         1   

                                             context  
0  Answer:\n    A kidney stone is a solid piece o...  
1  Urolithiasis is the medical term used to descr...  
2  Topic: Urological Disease\n\n    Focus: Kidney...  
3  Topic: Urological Disease\n\n    Focus: Kidney...  
4  Answer:\n    Kidney stones can form when subst...  
```
```html
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topic</th>
      <th>focus</th>
      <th>qtype</th>
      <th>question</th>
      <th>url</th>
      <th>chunk_id</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>information</td>
      <td>What is (are) Kidney Stones in Adults ?</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
      <td>1</td>
      <td>Answer:\n    A kidney stone is a solid piece o...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>information</td>
      <td>What is (are) Kidney Stones in Adults ?</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
      <td>2</td>
      <td>Urolithiasis is the medical term used to descr...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>information</td>
      <td>What is (are) Kidney Stones in Adults ?</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
      <td>0</td>
      <td>Topic: Urological Disease\n\n    Focus: Kidney...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>susceptibility</td>
      <td>Who is at risk for Kidney Stones in Adults? ?</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
      <td>0</td>
      <td>Topic: Urological Disease\n\n    Focus: Kidney...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Urological Disease</td>
      <td>Kidney Stones</td>
      <td>causes</td>
      <td>What causes Kidney Stones in Adults ?</td>
      <td>https://www.niddk.nih.gov/health-information/u...</td>
      <td>1</td>
      <td>Answer:\n    Kidney stones can form when subst...</td>
    </tr>
  </tbody>
</table>
</div>
```

## Cell 12
```text
(39326, 10000)
```

## Cell 13
*No output*

## Cell 14
```text

TOPIC: Diabetes
QUESTION: What is (are) Prevent diabetes problems: Keep your diabetes under control ?
SCORE: 0.7819

RETRIEVED CONTEXT:

Topic: Diabetes

    Focus: Diabetes Management
    
    Question Type: information

    Question:
    What is (are) Prevent diabetes problems: Keep your diabetes under control ?

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) Prevent diabetes problems: Keep your diabetes under control ?
SCORE: 0.7819

RETRIEVED CONTEXT:

Topic: Diabetes

    Focus: Diabetes Management
    
    Question Type: information

    Question:
    What is (are) Prevent diabetes problems: Keep your diabetes under control ?

====================================================================================================

TOPIC: Diabetes
QUESTION: How to diagnose Prevent diabetes problems: Keep your diabetes under control ?
SCORE: 0.7348

RETRIEVED CONTEXT:

Topic: Diabetes

    Focus: Diabetes Management
    
    Question Type: exams and tests

    Question:
    How to diagnose Prevent diabetes problems: Keep your diabetes under control ?

====================================================================================================
```

## Cell 15
*No output*

## Cell 16
```text
modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]
```
```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```
```text
config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]
```
```text
README.md: 0.00B [00:00, ?B/s]
```
```text
sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]
```
```text
config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]
```
```text
model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]
```
```text
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
```
```text
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
```
```text
tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]
```
```text
vocab.txt: 0.00B [00:00, ?B/s]
```
```text
tokenizer.json: 0.00B [00:00, ?B/s]
```
```text
special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]
```
```text
config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]
```

## Cell 17
```text
Batches:   0%|          | 0/615 [00:00<?, ?it/s]
```

## Cell 18
```text
(39326, 384)
```

## Cell 19
*No output*

## Cell 20
*No output*

## Cell 21
*No output*

## Cell 22
*No output*

## Cell 23
*No output*

## Cell 24
*No output*

## Cell 25
```text

TOPIC: Genetic & Rare Diseases
QUESTION: What is (are) Diabetes mellitus type 1 ?
CHUNK ID: 3
SCORE: 0.7143

RETRIEVED CONTEXT:

. Improper control can cause recurrence of high blood sugar, or abnormally low blood sugar (hypoglycemia) during exercise or when eating is delayed. If not treated, the condition can be life-threatening. Over many years, chronic high blood sugar may be associated with a variety of complications that affect many parts of the body.

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) I Can Lower My Risk for Type 2 Diabetes: A Guide for American Indians ?
CHUNK ID: 1
SCORE: 0.6829

RETRIEVED CONTEXT:

Answer:
    Diabetes causes blood glucose levels to be above normal. People with diabetes have problems converting food to energy. After food is eaten, it is broken down into a sugar called glucose. Glucose is then carried by the blood to cells throughout the body. The hormone insulin, made in the pancreas, helps the body change blood glucose into energy. People with diabetes, however, either no longer make enough insulin, or their insulin doesn't work properly, or both.
                
Type 2 diabetes

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) Diagnosis of Diabetes and Prediabetes ?
CHUNK ID: 1
SCORE: 0.6476

RETRIEVED CONTEXT:

Answer:
    Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia.
                
Diabetes is a disorder of metabolismthe way the body uses digested food for energy. The digestive tract breaks down carbohydratessugars and starches found in many foodsinto glucose, a form of sugar that enters the bloodstream. With the help of the hormone insulin, cells throughout the body absorb glucose and use it for energy. Insulin is made in the pancreas, an organ located behind the stomach. As the blood glucose level rises after a meal, the pancreas is triggered to release insulin. Within the pancreas, clusters of cells called islets contain beta cells, which make the insulin and release it into the blood.

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) Diabetes, Heart Disease, and Stroke ?
CHUNK ID: 2
SCORE: 0.6375

RETRIEVED CONTEXT:

Over time, high blood glucose levels damage nerves and blood vessels, leading to complications such as heart disease and stroke, the leading causes of death among people with diabetes. Uncontrolled diabetes can eventually lead to other health problems as well, such as vision loss, kidney failure, and amputations.

====================================================================================================

TOPIC: others
QUESTION: What is (are) Hypoglycemia ?
CHUNK ID: 2
SCORE: 0.6373

RETRIEVED CONTEXT:

Hypoglycemia means low blood glucose, or blood sugar. Your body needs glucose to have enough energy. After you eat, your blood absorbs glucose. If you eat more sugar than your body needs, your muscles, and liver store the extra. When your blood sugar begins to fall, a hormone tells your liver to release glucose.    In most people, this raises blood sugar. If it doesn't, you have hypoglycemia, and your blood sugar can be dangerously low. Signs include        - Hunger    - Shakiness    - Dizziness    - Confusion    - Difficulty speaking    - Feeling anxious or weak       In people with diabetes, hypoglycemia is often a side effect of diabetes medicines. Eating or drinking something with carbohydrates can help. If it happens often, your health care provider may need to change your treatment plan.    You can also have low blood sugar without having diabetes. Causes include certain medicines or diseases, hormone or enzyme deficiencies, and tumors

====================================================================================================
```

## Cell 26
```text
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m398.1/398.1 kB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
[?25h
```

## Cell 27
```text
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m29.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m43.6 MB/s[0m eta [36m0:00:00[0m
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m73.1/73.1 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
[?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
bigframes 2.35.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
google-adk 1.25.1 requires google-cloud-bigquery-storage>=2.0.0, which is not installed.
google-colab 1.0.0 requires jupyter-server==2.14.0, but you have jupyter-server 2.12.5 which is incompatible.
google-colab 1.0.0 requires pandas==2.2.2, but you have pandas 2.3.3 which is incompatible.
google-colab 1.0.0 requires requests==2.32.4, but you have requests 2.34.2 which is incompatible.
gcsfs 2025.3.0 requires fsspec==2025.3.0, but you have fsspec 2026.2.0 which is incompatible.[0m[31m
[0m
```

## Cell 28
*No output*

## Cell 29
```text
Connected Successfully!
```

## Cell 30
```text
True
```

## Cell 31
```text
CollectionsResponse(collections=[CollectionDescription(name='meditrust_rag'), CollectionDescription(name='meditrust_rag_v2')])
```

## Cell 32
```text
CollectionInfo(status=<CollectionStatus.GREEN: 'green'>, optimizer_status=<OptimizersStatusOneOf.OK: 'ok'>, warnings=None, indexed_vectors_count=0, points_count=0, segments_count=2, config=CollectionConfig(params=CollectionParams(vectors=VectorParams(size=384, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None), shard_number=1, sharding_method=None, replication_factor=1, write_consistency_factor=1, read_fan_out_factor=None, read_fan_out_delay_ms=None, on_disk_payload=True, sparse_vectors=None), hnsw_config=HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000, max_indexing_threads=0, on_disk=False, payload_m=None, inline_storage=None), optimizer_config=OptimizersConfig(deleted_threshold=0.2, vacuum_min_vector_number=1000, default_segment_number=0, max_segment_size=None, memmap_threshold=None, indexing_threshold=10000, flush_interval_sec=5, max_optimization_threads=None, prevent_unoptimized=None), wal_config=WalConfig(wal_capacity_mb=32, wal_segments_ahead=0, wal_retain_closed=1), quantization_config=None, strict_mode_config=StrictModeConfigOutput(enabled=True, max_query_limit=None, max_timeout=None, unindexed_filtering_retrieve=False, unindexed_filtering_update=False, search_max_hnsw_ef=None, search_allow_exact=None, search_max_oversampling=None, upsert_max_batchsize=None, search_max_batchsize=None, max_collection_vector_size_bytes=None, read_rate_limit=None, write_rate_limit=None, max_collection_payload_size_bytes=None, max_points_count=None, filter_max_conditions=None, condition_max_size=None, multivector_config=None, sparse_config=None, max_payload_index_count=100, max_resident_memory_percent=None), metadata=None), payload_schema={}, update_queue=UpdateQueueInfo(length=0, deferred_points=None))
```

## Cell 33
```text
Payload count: 39326
```

## Cell 34
*No output*

## Cell 35
```text
status=<CollectionStatus.GREEN: 'green'> optimizer_status=<OptimizersStatusOneOf.OK: 'ok'> warnings=None indexed_vectors_count=33536 points_count=39326 segments_count=2 config=CollectionConfig(params=CollectionParams(vectors=VectorParams(size=384, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None), shard_number=1, sharding_method=None, replication_factor=1, write_consistency_factor=1, read_fan_out_factor=None, read_fan_out_delay_ms=None, on_disk_payload=True, sparse_vectors=None), hnsw_config=HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000, max_indexing_threads=0, on_disk=False, payload_m=None, inline_storage=None), optimizer_config=OptimizersConfig(deleted_threshold=0.2, vacuum_min_vector_number=1000, default_segment_number=0, max_segment_size=None, memmap_threshold=None, indexing_threshold=10000, flush_interval_sec=5, max_optimization_threads=None, prevent_unoptimized=None), wal_config=WalConfig(wal_capacity_mb=32, wal_segments_ahead=0, wal_retain_closed=1), quantization_config=None, strict_mode_config=StrictModeConfigOutput(enabled=True, max_query_limit=None, max_timeout=None, unindexed_filtering_retrieve=False, unindexed_filtering_update=False, search_max_hnsw_ef=None, search_allow_exact=None, search_max_oversampling=None, upsert_max_batchsize=None, search_max_batchsize=None, max_collection_vector_size_bytes=None, read_rate_limit=None, write_rate_limit=None, max_collection_payload_size_bytes=None, max_points_count=None, filter_max_conditions=None, condition_max_size=None, multivector_config=None, sparse_config=None, max_payload_index_count=100, max_resident_memory_percent=None), metadata=None) payload_schema={} update_queue=UpdateQueueInfo(length=0, deferred_points=None)
```

## Cell 36
*No output*

## Cell 37
```text

TOPIC: Senior Health
QUESTION: What are the symptoms of Diabetes ?
CHUNK ID: 2
SCORE: 0.7916

RETRIEVED CONTEXT:

Diabetes is often called a "silent" disease because it can cause serious complications even before you have symptoms. Symptoms can also be so mild that you dont notice them. An estimated 8 million people in the United States have type 2 diabetes and dont know it, according to 2012 estimates by the Centers for Disease Control and Prevention (CDC). Common Signs Some common symptoms of diabetes are: - being very thirsty  - frequent urination  - feeling very hungry or tired  - losing weight without trying  - having sores that heal slowly  - having dry, itchy skin  - loss of feeling or tingling in the feet  - having blurry eyesight. being very thirsty frequent urination feeling very hungry or tired losing weight without trying having sores that heal slowly having dry, itchy skin loss of feeling or tingling in the feet having blurry eyesight. Signs of type 1 diabetes usually develop over a short period of time. The signs for type 2 diabetes develop more gradually

====================================================================================================

TOPIC: Diabetes
QUESTION: What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?
CHUNK ID: 0
SCORE: 0.7636

RETRIEVED CONTEXT:

Topic: Diabetes

    Focus: Diabetes Overview
    
    Question Type: symptoms

    Question:
    What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?

    Answer:
    The signs and symptoms of diabetes are
                
- being very thirsty  - urinating often  - feeling very hungry  - feeling very tired  - losing weight without trying  - sores that heal slowly  - dry, itchy skin  - feelings of pins and needles in your feet  - losing feeling in your feet  - blurry eyesight
                
Some people with diabetes dont have any of these signs or symptoms. The only way to know if you have diabetes is to have your doctor do a blood test.

====================================================================================================

TOPIC: Senior Health
QUESTION: What are the symptoms of Diabetes ?
CHUNK ID: 0
SCORE: 0.7489

RETRIEVED CONTEXT:

Topic: Senior Health

    Focus: Diabetes
    
    Question Type: symptoms

    Question:
    What are the symptoms of Diabetes ?

    Answer:
    Many people with diabetes experience one or more symptoms, including extreme thirst or hunger, a frequent need to urinate and/or fatigue. Some lose weight without trying. Additional signs include sores that heal slowly, dry, itchy skin, loss of feeling or tingling in the feet and blurry eyesight. Some people with diabetes, however, have no symptoms at all.

====================================================================================================

TOPIC: others
QUESTION: What is (are) Diabetes Type 1 ?
CHUNK ID: 2
SCORE: 0.7382

RETRIEVED CONTEXT:

Diabetes means your blood glucose, or blood sugar, levels are too high. With type 1 diabetes, your pancreas does not make insulin. Insulin is a hormone that helps glucose get into your cells to give them energy. Without insulin, too much glucose stays in your blood. Over time, high blood glucose can lead to serious problems with your heart, eyes, kidneys, nerves, and gums and teeth.     Type 1 diabetes happens most often in children and young adults but can appear at any age. Symptoms may include       -  Being very thirsty     -  Urinating often     -  Feeling very hungry or tired     -  Losing weight without trying     -  Having sores that heal slowly     -  Having dry, itchy skin     -  Losing the feeling in your feet or having tingling in your feet     -  Having blurry eyesight        A blood test can show if you have diabetes. If you do, you will need to take insulin for the rest of your life

====================================================================================================

TOPIC: Diabetes
QUESTION: What are the symptoms of Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes ?
CHUNK ID: 1
SCORE: 0.689

RETRIEVED CONTEXT:

Question:
    What are the symptoms of Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes ?

    Answer:
    The signs and symptoms of type 2 diabetes can be so mild that you might not even notice them. Nearly 7 million people in the United States have type 2 diabetes and dont know they have the disease. Many have no signs or symptoms. Some people have symptoms but do not suspect diabetes.
                
Symptoms include
                
- increased thirst  - increased hunger  - fatigue  - increased urination, especially at night  - unexplained weight loss  - blurred vision  - numbness or tingling in the feet or hands  - sores that do not heal
                
Many people do not find out they have the disease until they have diabetes problems, such as blurred vision or heart trouble. If you find out early that you have diabetes, you can get treatment to prevent damage to your body.

====================================================================================================
```

## Cell 38
```text

TOPIC: Genetic & Rare Diseases
QUESTION: What is (are) Diabetes mellitus type 1 ?
CHUNK ID: 3
SCORE: 0.7143

RETRIEVED CONTEXT:

. Improper control can cause recurrence of high blood sugar, or abnormally low blood sugar (hypoglycemia) during exercise or when eating is delayed. If not treated, the condition can be life-threatening. Over many years, chronic high blood sugar may be associated with a variety of complications that affect many parts of the body.

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) I Can Lower My Risk for Type 2 Diabetes: A Guide for American Indians ?
CHUNK ID: 1
SCORE: 0.6829

RETRIEVED CONTEXT:

Answer:
    Diabetes causes blood glucose levels to be above normal. People with diabetes have problems converting food to energy. After food is eaten, it is broken down into a sugar called glucose. Glucose is then carried by the blood to cells throughout the body. The hormone insulin, made in the pancreas, helps the body change blood glucose into energy. People with diabetes, however, either no longer make enough insulin, or their insulin doesn't work properly, or both.
                
Type 2 diabetes

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) Diagnosis of Diabetes and Prediabetes ?
CHUNK ID: 1
SCORE: 0.6476

RETRIEVED CONTEXT:

Answer:
    Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia.
                
Diabetes is a disorder of metabolismthe way the body uses digested food for energy. The digestive tract breaks down carbohydratessugars and starches found in many foodsinto glucose, a form of sugar that enters the bloodstream. With the help of the hormone insulin, cells throughout the body absorb glucose and use it for energy. Insulin is made in the pancreas, an organ located behind the stomach. As the blood glucose level rises after a meal, the pancreas is triggered to release insulin. Within the pancreas, clusters of cells called islets contain beta cells, which make the insulin and release it into the blood.

====================================================================================================

TOPIC: Diabetes
QUESTION: What is (are) Diabetes, Heart Disease, and Stroke ?
CHUNK ID: 2
SCORE: 0.6375

RETRIEVED CONTEXT:

Over time, high blood glucose levels damage nerves and blood vessels, leading to complications such as heart disease and stroke, the leading causes of death among people with diabetes. Uncontrolled diabetes can eventually lead to other health problems as well, such as vision loss, kidney failure, and amputations.

====================================================================================================

TOPIC: others
QUESTION: What is (are) Hypoglycemia ?
CHUNK ID: 2
SCORE: 0.6373

RETRIEVED CONTEXT:

Hypoglycemia means low blood glucose, or blood sugar. Your body needs glucose to have enough energy. After you eat, your blood absorbs glucose. If you eat more sugar than your body needs, your muscles, and liver store the extra. When your blood sugar begins to fall, a hormone tells your liver to release glucose.    In most people, this raises blood sugar. If it doesn't, you have hypoglycemia, and your blood sugar can be dangerously low. Signs include        - Hunger    - Shakiness    - Dizziness    - Confusion    - Difficulty speaking    - Feeling anxious or weak       In people with diabetes, hypoglycemia is often a side effect of diabetes medicines. Eating or drinking something with carbohydrates can help. If it happens often, your health care provider may need to change your treatment plan.    You can also have low blood sugar without having diabetes. Causes include certain medicines or diseases, hormone or enzyme deficiencies, and tumors

====================================================================================================
```

## Cell 39
*No output*

## Cell 40
*No output*

## Cell 41
*No output*

## Cell 42
*No output*

## Cell 43
*No output*

## Cell 44
```text

TOPIC: Diabetes
QUESTION: What are the symptoms of Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes ?
CHUNK ID: 0
SCORE: 20.4421

CONTEXT:

Topic: Diabetes

    Focus: Type 2 Diabetes Risk
    
    Question Type: symptoms

    Question:
    What are the symptoms of Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes ?

====================================================================================================

TOPIC: Genetic & Rare Diseases
QUESTION: What are the symptoms of Brittle diabetes ?
CHUNK ID: 0
SCORE: 17.2257

CONTEXT:

Topic: Genetic & Rare Diseases

    Focus: Brittle diabetes
    
    Question Type: symptoms

    Question:
    What are the symptoms of Brittle diabetes ?

    Answer:
    What are the symptoms of brittle diabetes? The main symptom of brittle diabetes is severe instability of blood glucose levels with frequent and unpredictable episodes of hypoglycemia and/or ketoacidosis that cause a disruption of daily activities. Three clinical presentations have been described: Predominant hyperglycemia with recurrent ketoacidosis, Predominant hypoglycemia, and Mixed hyper- and hypoglycemia. Patients with brittle diabetes have wide swings in their blood sugar levels and often experience differing blood sugar responses to the same dose and type of insulin. Complications such as neuropathy, nephropathy, and retinopathy are common. Most patients are females in their twenties of thirties, though any age or gender can be affected.

====================================================================================================

TOPIC: Diabetes
QUESTION: Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?
CHUNK ID: 0
SCORE: 17.0766

CONTEXT:

Topic: Diabetes

    Focus: Type 2 Diabetes Risk
    
    Question Type: susceptibility

    Question:
    Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?

====================================================================================================

TOPIC: Diabetes
QUESTION: Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?
CHUNK ID: 0
SCORE: 17.0766

CONTEXT:

Topic: Diabetes

    Focus: Type 2 Diabetes Risk
    
    Question Type: susceptibility

    Question:
    Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?

====================================================================================================

TOPIC: Diabetes
QUESTION: Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?
CHUNK ID: 0
SCORE: 17.0766

CONTEXT:

Topic: Diabetes

    Focus: Type 2 Diabetes Risk
    
    Question Type: susceptibility

    Question:
    Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?

====================================================================================================
```

## Cell 45
*No output*

## Cell 46
```text

TOPIC: Senior Health
QUESTION: What are the symptoms of Diabetes ?
CHUNK ID: 2
HYBRID SCORE: 0.6

CONTEXT:

Diabetes is often called a "silent" disease because it can cause serious complications even before you have symptoms. Symptoms can also be so mild that you dont notice them. An estimated 8 million people in the United States have type 2 diabetes and dont know it, according to 2012 estimates by the Centers for Disease Control and Prevention (CDC). Common Signs Some common symptoms of diabetes are: - being very thirsty  - frequent urination  - feeling very hungry or tired  - losing weight without trying  - having sores that heal slowly  - having dry, itchy skin  - loss of feeling or tingling in the feet  - having blurry eyesight. being very thirsty frequent urination feeling very hungry or tired losing weight without trying having sores that heal slowly having dry, itchy skin loss of feeling or tingling in the feet having blurry eyesight. Signs of type 1 diabetes usually develop over a short period of time. The signs for type 2 diabetes develop more gradually

====================================================================================================

TOPIC: Diabetes
QUESTION: Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?
CHUNK ID: 0
HYBRID SCORE: 0.5471

CONTEXT:

Topic: Diabetes

    Focus: Type 2 Diabetes Risk
    
    Question Type: susceptibility

    Question:
    Who is at risk for Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes? ?

====================================================================================================

TOPIC: Diabetes
QUESTION: What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?
CHUNK ID: 0
HYBRID SCORE: 0.4864

CONTEXT:

Topic: Diabetes

    Focus: Diabetes Overview
    
    Question Type: symptoms

    Question:
    What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?

    Answer:
    The signs and symptoms of diabetes are
                
- being very thirsty  - urinating often  - feeling very hungry  - feeling very tired  - losing weight without trying  - sores that heal slowly  - dry, itchy skin  - feelings of pins and needles in your feet  - losing feeling in your feet  - blurry eyesight
                
Some people with diabetes dont have any of these signs or symptoms. The only way to know if you have diabetes is to have your doctor do a blood test.

====================================================================================================

TOPIC: Senior Health
QUESTION: What are the symptoms of Diabetes ?
CHUNK ID: 0
HYBRID SCORE: 0.4267

CONTEXT:

Topic: Senior Health

    Focus: Diabetes
    
    Question Type: symptoms

    Question:
    What are the symptoms of Diabetes ?

    Answer:
    Many people with diabetes experience one or more symptoms, including extreme thirst or hunger, a frequent need to urinate and/or fatigue. Some lose weight without trying. Additional signs include sores that heal slowly, dry, itchy skin, loss of feeling or tingling in the feet and blurry eyesight. Some people with diabetes, however, have no symptoms at all.

====================================================================================================

TOPIC: Diabetes
QUESTION: What are the symptoms of Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes ?
CHUNK ID: 0
HYBRID SCORE: 0.4

CONTEXT:

Topic: Diabetes

    Focus: Type 2 Diabetes Risk
    
    Question Type: symptoms

    Question:
    What are the symptoms of Am I at Risk for Type 2 Diabetes? Taking Steps to Lower Your Risk of Getting Diabetes ?

====================================================================================================
```

## Cell 47
```text
config.json:   0%|          | 0.00/799 [00:00<?, ?B/s]
```
```text
model.safetensors:   0%|          | 0.00/1.11G [00:00<?, ?B/s]
```
```text
Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]
```
```text
XLMRobertaForSequenceClassification LOAD REPORT from: BAAI/bge-reranker-base
Key                             | Status     |  | 
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
```
```text
tokenizer_config.json:   0%|          | 0.00/443 [00:00<?, ?B/s]
```
```text
sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]
```
```text
tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]
```
```text
special_tokens_map.json:   0%|          | 0.00/279 [00:00<?, ?B/s]
```
```text
README.md: 0.00B [00:00, ?B/s]
```

## Cell 48
*No output*

## Cell 49
```text

TOPIC: Senior Health
QUESTION: What are the symptoms of Diabetes ?
RERANK SCORE: 0.9994

CONTEXT:

Topic: Senior Health

    Focus: Diabetes
    
    Question Type: symptoms

    Question:
    What are the symptoms of Diabetes ?

    Answer:
    Many people with diabetes experience one or more symptoms, including extreme thirst or hunger, a frequent need to urinate and/or fatigue. Some lose weight without trying. Additional signs include sores that heal slowly, dry, itchy skin, loss of feeling or tingling in the feet and blurry eyesight. Some people with diabetes, however, have no symptoms at all.

====================================================================================================

TOPIC: Diabetes
QUESTION: What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?
RERANK SCORE: 0.999

CONTEXT:

Topic: Diabetes

    Focus: Diabetes Overview
    
    Question Type: symptoms

    Question:
    What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?

    Answer:
    The signs and symptoms of diabetes are
                
- being very thirsty  - urinating often  - feeling very hungry  - feeling very tired  - losing weight without trying  - sores that heal slowly  - dry, itchy skin  - feelings of pins and needles in your feet  - losing feeling in your feet  - blurry eyesight
                
Some people with diabetes dont have any of these signs or symptoms. The only way to know if you have diabetes is to have your doctor do a blood test.

====================================================================================================

TOPIC: Senior Health
QUESTION: What are the symptoms of Diabetes ?
RERANK SCORE: 0.9947

CONTEXT:

Diabetes is often called a "silent" disease because it can cause serious complications even before you have symptoms. Symptoms can also be so mild that you dont notice them. An estimated 8 million people in the United States have type 2 diabetes and dont know it, according to 2012 estimates by the Centers for Disease Control and Prevention (CDC). Common Signs Some common symptoms of diabetes are: - being very thirsty  - frequent urination  - feeling very hungry or tired  - losing weight without trying  - having sores that heal slowly  - having dry, itchy skin  - loss of feeling or tingling in the feet  - having blurry eyesight. being very thirsty frequent urination feeling very hungry or tired losing weight without trying having sores that heal slowly having dry, itchy skin loss of feeling or tingling in the feet having blurry eyesight. Signs of type 1 diabetes usually develop over a short period of time. The signs for type 2 diabetes develop more gradually

====================================================================================================
```

## Cell 51
*No output*

## Cell 52
*No output*

## Cell 53
```text
config.json: 0.00B [00:00, ?B/s]
```
```text
tokenizer_config.json: 0.00B [00:00, ?B/s]
```
```text
spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]
```
```text
tokenizer.json: 0.00B [00:00, ?B/s]
```
```text
special_tokens_map.json: 0.00B [00:00, ?B/s]
```

## Cell 54
```text
model.safetensors:   0%|          | 0.00/990M [00:00<?, ?B/s]
```
```text
Loading weights:   0%|          | 0/282 [00:00<?, ?it/s]
```
```text
The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
```
```text
generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]
```

## Cell 55
```text
diabetes is a disease in which the body can not produce enough insulin to maintain blood sugar levels.
```

## Cell 56
*No output*

## Cell 57
*No output*

## Cell 58
*No output*

## Cell 59
```text
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
```
```text

QUESTION:

What are symptoms of diabetes?

ANSWER:

Being very thirsty, frequent urination, feeling very hungry or tired, losing weight without trying, having sores that heal slowly, having dry, itchy skin, loss of feeling or tingling in the feet, having blurry eyesight
```

## Cell 60
*No output*

## Cell 61
*No output*

## Cell 62
*No output*

## Cell 63
```text

QUESTION:

What are symptoms of diabetes?

ANSWER:

Being very thirsty, frequent urination, feeling very hungry or tired, losing weight without trying, having sores that heal slowly, having dry, itchy skin, loss of feeling or tingling in the feet, having blurry eyesight

CONFIDENCE:

High

SOURCES:

1. Senior Health → What are the symptoms of Diabetes ?
2. Diabetes → What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?
```

## Cell 64
*No output*

## Cell 66
*No output*

## Cell 67
```text
{'query': 'What are symptoms of diabetes?', 'answer': 'Being very thirsty, frequent urination, feeling very hungry or tired, losing weight without trying, having sores that heal slowly, having dry, itchy skin, loss of feeling or tingling in the feet, having blurry eyesight', 'sources': [{'topic': 'Senior Health', 'focus': 'Diabetes', 'question': 'What are the symptoms of Diabetes ?', 'url': 'https://www.nia.nih.gov/health/diabetes'}, {'topic': 'Diabetes', 'focus': 'Diabetes Overview', 'question': 'What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?', 'url': 'https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes'}], 'confidence': 'High'}
```

## Cell 68
```text
{'query': 'Can diabetes cure brain cancer naturally?', 'answer': 'I could not find reliable medical information in the retrieved documents.', 'confidence': 'Low', 'sources': []}
```

## Cell 69
*No output*

## Cell 70
*No output*

## Cell 71
```text

DEBUG
<class 'list'>
3

DOC 1
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 2
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 3
dict_keys(['score', 'topic', 'question', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

QUESTION:

Who won IPL 2025?

ANSWER:

I could not find reliable medical information in the retrieved documents.

CONFIDENCE:

Low

SOURCES:


RETRIEVAL EXPLANATIONS:

```

## Cell 72
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mNameError[0m                                 Traceback (most recent call last)[0;32m/tmp/ipykernel_57/4242480756.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0;32mfor[0m [0mdoc[0m [0;32min[0m [0mretrieved_docs[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m     [0mprint[0m[0;34m([0m[0mdoc[0m[0;34m.[0m[0mget[0m[0;34m([0m[0;34m"url"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;31mNameError[0m: name 'retrieved_docs' is not defined
```

## Cell 73
```text

DOC 1
QUESTION: What are the symptoms of Diabetes ?
URL: https://www.nia.nih.gov/health/diabetes

DOC 2
QUESTION: What are the symptoms of Your Guide to Diabetes: Type 1 and Type 2 ?
URL: https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes

DOC 3
QUESTION: What are the symptoms of Diabetes ?
URL: https://www.nia.nih.gov/health/diabetes
```

## Cell 74
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mKeyError[0m                                  Traceback (most recent call last)[0;32m/tmp/ipykernel_57/520940651.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mprint[0m[0;34m([0m[0mresult[0m[0;34m[[0m[0;34m"sources"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[0;31mKeyError[0m: 'sources'
```

## Cell 76
*No output*

## Cell 77
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mNameError[0m                                 Traceback (most recent call last)[0;32m/tmp/ipykernel_57/2809926464.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0mretrieved[0m [0;34m=[0m [0mhybrid_search[0m[0;34m([0m[0mquery[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m reranked = rerank_documents(
[0m[1;32m      6[0m     [0mquery[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0mretrieved[0m[0;34m[0m[0;34m[0m[0m
[0;31mNameError[0m: name 'rerank_documents' is not defined
```

## Cell 78
```text
====================================================================================================
OUT-OF-DOMAIN TESTING
====================================================================================================
```
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mKeyError[0m                                  Traceback (most recent call last)[0;32m/tmp/ipykernel_57/2725554680.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtest_out_of_domain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[0;32m/tmp/ipykernel_57/1629722610.py[0m in [0;36mtest_out_of_domain[0;34m()[0m
[1;32m     21[0m     [0;32mfor[0m [0mquery[0m [0;32min[0m [0mqueries[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0;34m[0m[0m
[0;32m---> 23[0;31m         [0mresult[0m [0;34m=[0m [0mtrustworthy_rag[0m[0;34m([0m[0mquery[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0;34m[0m[0m
[1;32m     25[0m         [0mprint[0m[0;34m([0m[0;34m"\nQUERY:"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m/tmp/ipykernel_57/2314316381.py[0m in [0;36mtrustworthy_rag[0;34m(query)[0m
[1;32m      4[0m     [0;31m# Generate RAG Answer[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0;31m# ---------------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m     [0mresult[0m [0;34m=[0m [0mgenerate_rag_answer[0m[0;34m([0m[0mquery[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m     [0mretrieved_docs[0m [0;34m=[0m [0mresult[0m[0;34m[[0m[0;34m"retrieved_docs"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m/tmp/ipykernel_57/3320523244.py[0m in [0;36mgenerate_rag_answer[0;34m(query, retrieval_k, final_k)[0m
[1;32m     25[0m     [0;31m# Step 3 — Build Context[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m     [0;31m# ---------------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m     context = build_context(
[0m[1;32m     28[0m         [0mreranked_docs[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m     )
[0;32m/tmp/ipykernel_57/3836616793.py[0m in [0;36mbuild_context[0;34m(docs)[0m
[1;32m      8[0m             f"""
[1;32m      9[0m             [0mTopic[0m[0;34m:[0m [0;34m{[0m[0mdoc[0m[0;34m[[0m[0;34m'topic'[0m[0;34m][0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m             [0mFocus[0m[0;34m:[0m [0;34m{[0m[0mdoc[0m[0;34m[[0m[0;34m'focus'[0m[0;34m][0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m             [0mQuestion[0m [0mType[0m[0;34m:[0m [0;34m{[0m[0mdoc[0m[0;34m[[0m[0;34m'qtype'[0m[0;34m][0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m
[0;31mKeyError[0m: 'focus'
```

## Cell 79
```text
<function generate_rag_answer at 0x7a4907783ec0>
```

## Cell 80
```text

DEBUG
<class 'list'>
3

DOC 1
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 2
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 3
dict_keys(['score', 'topic', 'question', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])
```
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mKeyError[0m                                  Traceback (most recent call last)[0;32m/tmp/ipykernel_57/2115715764.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtrustworthy_rag[0m[0;34m([0m[0;34m"Who won IPL 2025?"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[0;32m/tmp/ipykernel_57/2314316381.py[0m in [0;36mtrustworthy_rag[0;34m(query)[0m
[1;32m      4[0m     [0;31m# Generate RAG Answer[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0;31m# ---------------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m     [0mresult[0m [0;34m=[0m [0mgenerate_rag_answer[0m[0;34m([0m[0mquery[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m     [0mretrieved_docs[0m [0;34m=[0m [0mresult[0m[0;34m[[0m[0;34m"retrieved_docs"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m/tmp/ipykernel_57/4050675493.py[0m in [0;36mgenerate_rag_answer[0;34m(query, retrieval_k, final_k)[0m
[1;32m     32[0m     [0;31m# Step 3 — Build Context[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m     [0;31m# ---------------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m     context = build_context(
[0m[1;32m     35[0m         [0mreranked_docs[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m     )
[0;32m/tmp/ipykernel_57/3836616793.py[0m in [0;36mbuild_context[0;34m(docs)[0m
[1;32m      8[0m             f"""
[1;32m      9[0m             [0mTopic[0m[0;34m:[0m [0;34m{[0m[0mdoc[0m[0;34m[[0m[0;34m'topic'[0m[0;34m][0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m             [0mFocus[0m[0;34m:[0m [0;34m{[0m[0mdoc[0m[0;34m[[0m[0;34m'focus'[0m[0;34m][0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m             [0mQuestion[0m [0mType[0m[0;34m:[0m [0;34m{[0m[0mdoc[0m[0;34m[[0m[0;34m'qtype'[0m[0;34m][0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m
[0;31mKeyError[0m: 'focus'
```

## Cell 81
```text

DOC 1
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score'])

DOC 2
dict_keys(['score', 'topic', 'question', 'context', 'chunk_id', 'hybrid_score'])

FOUND OLD DOCUMENT
{'score': np.float64(3.168174891622995), 'topic': 'Heart_Lung_Blood', 'question': 'Who is at risk for Hypersensitivity Pneumonitis? ?', 'context': "Answer:\n    People who repeatedly breathe in foreign substances are at risk for hypersensitivity pneumonitis (HP). These substances, which also are known as antigens, include molds, dusts, and chemicals. However, most people who breathe in these substances don't develop HP.\n                \nPeople at increased risk include:\n                \nFarm and dairy cattle workers\n                \nPeople who use hot tubs often\n                \nPeople who are exposed to molds or dusts from humidifiers, heating systems, or wet carpeting\n                \nBird fanciers (people who keep pet birds) and poultry handlers\n                \nFlorists and landscapers, especially those who use liquid chemicals on lawns and gardens\n                \nPeople who work in grain and flour processing and loading\n                \nLumber milling, construction, wood stripping, and paper and wallboard workers\n                \nPeople who make plastics or electronics, and those who paint or work with other chemicals", 'chunk_id': np.int64(1), 'hybrid_score': 0.39999998500700895}

DOC 3
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score'])

DOC 4
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score'])

DOC 5
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score'])
```

## Cell 82
```text
status=<CollectionStatus.GREEN: 'green'> optimizer_status=<OptimizersStatusOneOf.OK: 'ok'> warnings=None indexed_vectors_count=33536 points_count=39326 segments_count=2 config=CollectionConfig(params=CollectionParams(vectors=VectorParams(size=384, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None), shard_number=1, sharding_method=None, replication_factor=1, write_consistency_factor=1, read_fan_out_factor=None, read_fan_out_delay_ms=None, on_disk_payload=True, sparse_vectors=None), hnsw_config=HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000, max_indexing_threads=0, on_disk=False, payload_m=None, inline_storage=None), optimizer_config=OptimizersConfig(deleted_threshold=0.2, vacuum_min_vector_number=1000, default_segment_number=0, max_segment_size=None, memmap_threshold=None, indexing_threshold=10000, flush_interval_sec=5, max_optimization_threads=None, prevent_unoptimized=None), wal_config=WalConfig(wal_capacity_mb=32, wal_segments_ahead=0, wal_retain_closed=1), quantization_config=None, strict_mode_config=StrictModeConfigOutput(enabled=True, max_query_limit=None, max_timeout=None, unindexed_filtering_retrieve=False, unindexed_filtering_update=False, search_max_hnsw_ef=None, search_allow_exact=None, search_max_oversampling=None, upsert_max_batchsize=None, search_max_batchsize=None, max_collection_vector_size_bytes=None, read_rate_limit=None, write_rate_limit=None, max_collection_payload_size_bytes=None, max_points_count=None, filter_max_conditions=None, condition_max_size=None, multivector_config=None, sparse_config=None, max_payload_index_count=100, max_resident_memory_percent=None), metadata=None) payload_schema={} update_queue=UpdateQueueInfo(length=0, deferred_points=None)
```

## Cell 83
*No output*

## Cell 84
```text

DEBUG
<class 'list'>
3

DOC 1
dict_keys(['score', 'topic', 'question', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 2
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 3
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])
```
```text
{'query': 'Does garlic cure cancer?',
 'answer': 'I could not find reliable medical information in the retrieved documents.',
 'confidence': 'Low',
 'sources': [],
 'retrieval_explanations': []}
```

## Cell 85
```text

DEBUG
<class 'list'>
3

DOC 1
dict_keys(['score', 'topic', 'focus', 'qtype', 'question', 'url', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 2
dict_keys(['score', 'topic', 'question', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])

DOC 3
dict_keys(['score', 'topic', 'question', 'context', 'chunk_id', 'hybrid_score', 'rerank_score'])
```
```text
{'query': 'Can vaccines cause autism?',
 'answer': 'I could not find reliable medical information in the retrieved documents.',
 'confidence': 'Low',
 'sources': [],
 'retrieval_explanations': []}
```

## Cell 86
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mNameError[0m                                 Traceback (most recent call last)[0;32m/tmp/ipykernel_57/462417054.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mbm25_documents[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mkeys[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[0;31mNameError[0m: name 'bm25_documents' is not defined
```

## Cell 87
```error
[0;31m---------------------------------------------------------------------------[0m[0;31mNameError[0m                                 Traceback (most recent call last)[0;32m/tmp/ipykernel_57/1058846551.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mprint[0m[0;34m([0m[0mchunks_df[0m[0;34m.[0m[0mcolumns[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[0;31mNameError[0m: name 'chunks_df' is not defined
```

## Cell 88
*No output*

