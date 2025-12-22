'''pole_class.py'''

COCO_CLASSES_LIST = ['greenpole', 'redpole']

yolo_cls_to_ssd = [1, 2]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 2:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
