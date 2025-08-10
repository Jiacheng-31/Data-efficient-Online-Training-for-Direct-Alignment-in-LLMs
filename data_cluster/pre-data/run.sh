python download.py

# python looklook.py /fs-computility/llmit_d/shared/zhangchi/wjc/DPO/dataset/ultra-train.jsonl ultra-3000.jsonl -k 3000


python turn_to_json.py UltraFeedback/train.jsonl time1.json

wait

python ultra_turn_tp.py time1.json time2.json

wait

python Ultra_new.py time2.json ultra_train.json

wait