

# CancerBERT NER


Use CancerBERT for named entity recognition.


### Folder Description:
```
BERT-NER
|____ data		            # train data
|____ middle_data	            # middle data (label id map)
|____ output			    # output (final model, predict results)
|____ BERT_NER.py		    # mian code
|____ conlleval.pl		    # eval code
|____ run_ner.sh    		   # run model and eval result

```


### What's in run_ner.sh:
```
python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=cased_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=3.0   \
    --output_dir=./output/result_dir

```


### RESULTS:(On test set)
#### Parameter setting:
* do_lower_case=False 
* num_train_epochs=4.0
* crf=False
  
```

```
### Result description:
Here i just use the default paramaters, but as Google's paper says a 0.2% error is reasonable(reported 92.4%).
Maybe some tricks need to be added to the above model. 



### reference:



