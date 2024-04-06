import shutil
from tqdm import tqdm
from utils.general import np, pd, Path, xyxy2xywh
import json

# Load local dataset
dir = Path(r"D:\Pycharm\dataset\retail_product_checkout_dataset")  # dataset root dir
labels_path = dir / "labels"


# Convert labels
categories_names = 'id', 'name', 'supercategory'
images_names = 'file_name', 'width', 'height', 'id'
annotations_names = 'area', 'bbox', 'category_id', 'id', 'image_id', 'iscrowd', 'point_xy', 'segmentation'


def generate_multiple_labels():
    if labels_path.exists():
        shutil.rmtree(labels_path)
    labels_path.mkdir(parents=True, exist_ok=True)  # create labels dir
    for d in 'instances_val2019.json', 'instances_train2019.json', 'instances_test2019.json':
        with open(dir / d) as json_data:
            data = json.load(json_data)
        df_images = pd.DataFrame(data['images'], columns=images_names)
        df_annotations = pd.DataFrame(data['annotations'], columns=annotations_names)
        # df_categories = pd.DataFrame(data['categories'], columns=categories_names)
        # 生成训练集、测试集、验证集txt 每行存储图片路径
        with open((dir / d).with_suffix('.txt').__str__().replace('instances_', '').replace('2019', ''), 'w') as f:
            for index, row in df_images.iterrows():
                f.writelines(f'./images/{row["file_name"]}\n')
        for index, image in tqdm(df_images.iterrows(), desc=f'Converting {dir / d}'):
            cls = 0  # 单类别模式
            with open((dir / 'labels' / image['file_name']).with_suffix('.txt'), 'a') as f:
                w_image, h_image = image['width'], image['height']
                image_id = image['id']
                annotations = df_annotations[df_annotations['image_id'] == image_id]
                if annotations is None or len(annotations) == 0:
                    raise Exception('No annotations found for image {}'.format(image_id))
                for _, annotation in annotations.iterrows():
                    r = annotation['bbox']
                    w, h = r[2], r[3]  # bbox width, height
                    xywh = xyxy2xywh(np.array([[r[0] / w_image, r[1] / h_image, (r[0] + w) / w_image, (r[1] + h) / h_image]]))[0]  # instance
                    f.write(f"{cls} {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n")  # write label
                    # xywh = xyxy2xywh(np.array([[r[0], r[1], (r[0] + w), (r[1] + h)]]))[0]  # instance
                    # f.write(f"{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")  # write label


def generate_single_labels():
    """
    需要预先生成多标签
    """
    d = 'instances_test2019.json'
    with open(dir / d) as json_data:
        data = json.load(json_data)
    df_images = pd.DataFrame(data['images'], columns=images_names)
    total_size = df_images.shape[0]
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    df_train_images = df_images[:train_size]
    df_val_images = df_images[train_size:train_size + val_size]
    df_test_images = df_images[train_size + val_size:]
    # df_categories = pd.DataFrame(data['categories'], columns=categories_names)
    # 生成训练集、测试集、验证集txt 每行存储图片路径
    for name, _df_images in [('train', df_train_images), ('val', df_val_images), ('test', df_test_images)]:
        with open((dir / f'single_{name}.txt'), 'w') as f:
            for index, row in _df_images.iterrows():
                f.writelines(f'./images/{row["file_name"]}\n')


if __name__ == '__main__':
    # generate_multiple_labels()
    generate_single_labels()
