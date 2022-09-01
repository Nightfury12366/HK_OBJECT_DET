mv projects/DiBiao.yml projects/Old/DiBiao.yml -i
mv projects/New/DiBiao.yml projects/DiBiao.yml -i
mv projects/display_DiBiao.yml projects/Old/display_DiBiao.yml -i
mv projects/New/display_DiBiao.yml projects/display_DiBiao.yml -i
python XML_2_COCO.py --project DiBiao  --is_train True
python XML_2_COCO.py --project DiBiao