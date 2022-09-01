mv projects/DiBiao.yml projects/New/DiBiao.yml -i -b
mv projects/Old/DiBiao.yml projects/DiBiao.yml -i -b
mv projects/display_DiBiao.yml projects/New/display_DiBiao.yml -i -b
mv projects/Old/display_DiBiao.yml projects/display_DiBiao.yml -i -b
mv weights/efficientdet-d0_dibiao.pth weights/New/efficientdet-d0_dibiao.pth -i -b
mv weights/Old/efficientdet-d0_dibiao.pth weights/efficientdet-d0_dibiao.pth -i -b
python XML_2_COCO.py --project DiBiao  --is_train True
python XML_2_COCO.py --project DiBiao