mv projects/GuoQi.yml projects/New/GuoQi.yml -i -b
mv projects/Old/GuoQi.yml projects/GuoQi.yml -i -b
mv projects/display_GuoQi.yml projects/New/display_GuoQi.yml -i -b
mv projects/Old/display_GuoQi.yml projects/display_GuoQi.yml -i -b
mv weights/efficientdet-d0_guoqi.pth weights/New/efficientdet-d0_guoqi.pth -i -b
mv weights/Old/efficientdet-d0_guoqi.pth weights/efficientdet-d0_guoqi.pth -i -b
python XML_2_COCO.py --project GuoQi  --is_train True
python XML_2_COCO.py --project GuoQi