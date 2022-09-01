mv projects/GuoQi.yml projects/New/GuoQi.yml -i
mv projects/Old/GuoQi.yml projects/GuoQi.yml -i
mv projects/display_GuoQi.yml projects/New/display_GuoQi.yml -i
mv projects/Old/display_GuoQi.yml projects/display_GuoQi.yml -i
mv weights/efficientdet-d0_guoqi.pth weights/New/efficientdet-d0_guoqi.pth -i
mv weights/Old/efficientdet-d0_guoqi.pth weights/efficientdet-d0_guoqi.pth -i
python XML_2_COCO.py --project GuoQi  --is_train True
python XML_2_COCO.py --project GuoQi