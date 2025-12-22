'''pole_class.py'''

COCO_CLASSES_LIST = ['redball', 'greenball', 'yellowball']

yolo_cls_to_ssd = [1, 2, 3]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == 3:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
