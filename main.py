import torch
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.masks = sorted(os.listdir(os.path.join(root, "masks")))

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert mask to numpy array (0s and 1s)
        mask = np.array(mask)
        mask[mask != 0] = 1

        # Apply transformations
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # Convert mask to tensor
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        # Define the target dictionary
        target = {}
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["masks"] = mask
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Replace the last layer with a new layer for your dataset
num_classes = 3  # Number of classes in your custom dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Replace the mask predictor with a new mask predictor for your dataset
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                          hidden_layer, num_classes)

# Load your custom dataset
dataset = CustomDataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Train the model on your dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()

# Test the model on some images from your dataset
model.eval()

with torch.no_grad():
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        for i, output in enumerate(outputs):
            # Visualize the predicted masks for each image
            masks = output['masks'].cpu().numpy()
            for j in range(masks.shape[0]):
                mask = masks[j, 0]
                plt.imshow(mask)
                plt.show()
