mv projects/GuoQi.yml projects/Old/GuoQi.yml -i -b
mv projects/New/GuoQi.yml projects/GuoQi.yml -i -b
mv projects/display_GuoQi.yml projects/Old/display_GuoQi.yml -i -b
mv projects/New/display_GuoQi.yml projects/display_GuoQi.yml -i -b
python XML_2_COCO.py --project GuoQi  --is_train True
python XML_2_COCO.py --project GuoQi