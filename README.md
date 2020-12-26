# Dataset Maker
![Build Status](https://github.com/zactodd/dataset_maker/workflows/build/badge.svg)

## Labled data conversion:
| Problem | Method | Import | Export |
| :---: | :---: | :---: | :---: |
| **Object Detection** | VGG VIA | :heavy_check_mark: |  :heavy_check_mark: |
|  | YOLO | :heavy_check_mark: | :heavy_check_mark: |
|  | COCO*[1]  | :heavy_check_mark: | :heavy_check_mark: |
|  | Pascal VOC | :heavy_check_mark: | :heavy_check_mark: |
|  | Tensorflow Object Detection CSV | :heavy_check_mark: | :heavy_check_mark: |
|  | OIDv4 | :heavy_check_mark: | :heavy_check_mark: |
|  | IBM Cloud | :heavy_check_mark: | :heavy_check_mark: |
|  | VoTT CSV | :heavy_check_mark: | :heavy_check_mark: |
|  | CreateML | :heavy_check_mark: | :heavy_check_mark: |
| **Instance Segemenation** | VGG VIA | :heavy_check_mark: | :heavy_check_mark: |
| | COCO*[1]  | :heavy_check_mark: | :heavy_check_mark: |

[1] Does not work on crowd data.

Conversion script can be aceess using the following
```bash
python datasetmaker/scripts/localisation_format_conversion.py image_dir annotation_dir download_dir in_format out_format
```

For example:
```bash
python datasetmaker/scripts/localisation_format_conversion.py images annotations new_annoations COCO YOLO
```
