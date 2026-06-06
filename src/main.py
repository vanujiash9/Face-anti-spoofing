from data.face_dataset import FaceDatasetClass
from torch.utils.data import DataLoader

train_dataset = FaceDatasetClass(
    image_dir="data/train/images/",
    label_file="data/train/train_labels.txt"
)

test_dataset = FaceDatasetClass(
    image_dir="data/test/images/",
    label_file="data/test/test_labels.txt"
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
