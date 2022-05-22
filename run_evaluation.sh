#! /bin/bash

set -e

OUTPUT=$1

mkdir -p $OUTPUT

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O $OUTPUT/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O $OUTPUT/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O $OUTPUT/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O $OUTPUT/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O $OUTPUT/NaturalQuestions.jsonl.gz
wget http://participants-area.bioasq.org/MRQA2019/ -O $OUTPUT/BioASQ.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz -O $OUTPUT/TextbookQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz -O $OUTPUT/RelationExtraction.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz -O $OUTPUT/DROP.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz -O $OUTPUT/DuoRC.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz -O $OUTPUT/RACE.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/models/BERT/_MIX_6.tar.gz -O $OUTPUT/_MIX_6.tar.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/models/BERT/_MIX_6_large.tar.gz -O $OUTPUT/_MIX_6_large.tar.gz


python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/SQuAD.jsonl.gz   $OUTPUT/SQuAD_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/NewsQA.jsonl.gz   $OUTPUT/NewsQA_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/TriviaQA.jsonl.gz   $OUTPUT/TriviaQA_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/SearchQA.jsonl.gz   $OUTPUT/SearchQA_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/HotpotQA.jsonl.gz   $OUTPUT/HotpotQA_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/NaturalQuestions.jsonl.gz   $OUTPUT/NaturalQuestions_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/BioASQ.jsonl.gz   $OUTPUT/BioASQ_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/TextbookQA.jsonl.gz   $OUTPUT/TextbookQA_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/RelationExtraction.jsonl.gz   $OUTPUT/RelationExtraction_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/DROP.jsonl.gz   $OUTPUT/DROP_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/DuoRC.jsonl.gz   $OUTPUT/DuoRC_eval.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6.tar.gz $OUTPUT/RACE.jsonl.gz   $OUTPUT/RACE_eval.json --cuda_device 0

python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/SQuAD.jsonl.gz   $OUTPUT/SQuAD_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/NewsQA.jsonl.gz   $OUTPUT/NewsQA_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/TriviaQA.jsonl.gz   $OUTPUT/TriviaQA_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/SearchQA.jsonl.gz   $OUTPUT/SearchQA_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/HotpotQA.jsonl.gz   $OUTPUT/HotpotQA_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/NaturalQuestions.jsonl.gz   $OUTPUT/NaturalQuestions_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/BioASQ.jsonl.gz   $OUTPUT/BioASQ_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/TextbookQA.jsonl.gz   $OUTPUT/TextbookQA_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/RelationExtraction.jsonl.gz   $OUTPUT/RelationExtraction_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/DROP.jsonl.gz   $OUTPUT/DROP_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/DuoRC.jsonl.gz   $OUTPUT/DuoRC_eval_large.json --cuda_device 0
python ./MRQA-Shared-Task-2019/baseline/predict.py $OUTPUT/_MIX_6_large.tar.gz $OUTPUT/RACE.jsonl.gz   $OUTPUT/RACE_eval_large.json --cuda_device 0
