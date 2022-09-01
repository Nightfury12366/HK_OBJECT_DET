mv projects/DiBiao.yml projects/New/DiBiao.yml -i
mv projects/Old/DiBiao.yml projects/DiBiao.yml -i
mv projects/display_DiBiao.yml projects/New/display_DiBiao.yml -i
mv projects/Old/display_DiBiao.yml projects/display_DiBiao.yml -i
mv weights/efficientdet-d0_guoqi.pth weights/New/efficientdet-d0_guoqi.pth -i
mv weights/Old/efficientdet-d0_guoqi.pth weights/efficientdet-d0_guoqi.pth -i
python XML_2_COCO.py --project DiBiao  --is_train True
python XML_2_COCO.py --project DiBiao