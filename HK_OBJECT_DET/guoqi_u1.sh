mv projects/GuoQi.yml projects/Old/GuoQi.yml -i
mv projects/New/GuoQi.yml projects/GuoQi.yml -i
mv projects/display_GuoQi.yml projects/Old/display_GuoQi.yml -i
mv projects/New/display_GuoQi.yml projects/display_GuoQi.yml -i
python XML_2_COCO.py --project GuoQi  --is_train True
python XML_2_COCO.py --project GuoQi