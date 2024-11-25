import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# 1. Load Data from .npz Files
train_data = np.load('train.npz')
valid_data = np.load('valid.npz')

train_images = train_data['images'].reshape(-1, 64, 64, 3)
train_masks = train_data['semantic_masks'].reshape(-1, 64, 64)
valid_images = valid_data['images'].reshape(-1, 64, 64, 3)
valid_masks = valid_data['semantic_masks'].reshape(-1, 64, 64)

print("Train Images Shape:", train_images.shape)
print("Train Masks Shape:", train_masks.shape)
print("Valid Images Shape:", valid_images.shape)
print("Valid Masks Shape:", valid_masks.shape)

# Visualize a sample
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(train_images[0].reshape(64, 64, 3))
plt.title("Sample Image")
plt.subplot(1, 2, 2)
plt.imshow(train_masks[0].reshape(64, 64), cmap='gray')
plt.title("Sample Mask")
plt.show()

# 2. Define Custom Dataset for PyTorch
class MNISTSegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.int64)
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask)

# 3. Create DataLoader
train_dataset = MNISTSegmentationDataset(train_images, train_masks)
valid_dataset = MNISTSegmentationDataset(valid_images, valid_masks)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# 4. Define UNet Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=11)  # 11 classes for 0-9 + background
model.to(device)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropy for multiclass segmentation
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Training Loop with Checkpoint Saving and Progress Display
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Progress bar for training
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        
        for images, masks in train_progress:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Update progress bar description with current loss
            train_progress.set_postfix({"Loss": loss.item()})

        # Average train loss
        avg_train_loss = train_loss / len(train_loader)

        # Validation step with progress bar
        model.eval()
        val_loss = 0
        val_progress = tqdm(valid_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, masks in val_progress:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Update progress bar description with current validation loss
                val_progress.set_postfix({"Val Loss": loss.item()})

        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'unet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

# 6. Train the Model
train_and_validate(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50)

# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import segmentation_models_pytorch as smp
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import OneCycleLR
# import os
# import sys
# from torchvision import transforms

# import ssl
# from tqdm import tqdm 

# ssl._create_default_https_context = ssl._create_unverified_context

# # Combined transform for both image and mask
# class JointTransform:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.RandomRotation(10),  # Rotation within 10 degrees
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter for RGB images
#             transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
#             transforms.ToTensor()
#         ])

#     def __call__(self, img, mask):
#         # Apply the same transformation to both image and mask
#         img = self.transform(img)
#         mask = self.transform(mask)
#         return img, mask

# # Dataset class for U-Net with joint transformation
# class MNISTDD_Dataset(Dataset):
#     def __init__(self, images_path, masks_path, joint_transform=None):
#         self.images_path = images_path
#         self.masks_path = masks_path
#         self.joint_transform = joint_transform
#         self.images = sorted(os.listdir(images_path))
#         self.masks = sorted(os.listdir(masks_path))

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.images_path, self.images[idx])).convert("RGB")
#         mask = Image.open(os.path.join(self.masks_path, self.masks[idx]))
        
#         # if self.transform:
#         #     img = self.transform(img)
#         # if self.mask_transform:
#         #     mask = self.mask_transform(mask)


#         if self.joint_transform:
#             img, mask = self.joint_transform(img, mask)
#         else:
#             img = transforms.ToTensor()(img)
#             mask = transforms.ToTensor()(mask)


#         return img, mask.long()

# # Data transformations with augmentation
# joint_transform = JointTransform()

# # image_transform = transforms.Compose([
# #     # transforms.RandomHorizontalFlip(),  # Horizontal flip
# #     transforms.RandomRotation(10),  # Rotation within 10 degrees
# #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter for RGB images
# #     transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
# #     transforms.ToTensor()
# # ])

# # Convert mask to binary for simplified debugging
# # def binary_mask_transform(mask):
# #     mask = mask > 0  # Set all non-background pixels to 1
# #     return mask.float()

# # mask_transform = transforms.Compose([
# #     # transforms.RandomHorizontalFlip(),
# #     # transforms.RandomRotation(10),
# #     transforms.ToTensor(),
# #     # transforms.Lambda(binary_mask_transform)
# #     transforms.Lambda(lambda x: x.long())  # Scale mask values to 0-10
# # ])

# # Subset the dataset for debugging
# dataset = MNISTDD_Dataset("unet/train/images", "unet/train/masks", joint_transform)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# # dataset = MNISTDD_Dataset("unet/train/images", "unet/train/masks", image_transform, mask_transform)
# # dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# # Define U-Net
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.enc1 = DoubleConv(n_channels, 64)
#         self.enc2 = DoubleConv(64, 128)
#         self.enc3 = DoubleConv(128, 256)
#         self.enc4 = DoubleConv(256, 512)

#         self.pool = nn.MaxPool2d(2)
        
#         self.bottleneck = DoubleConv(512, 1024)

#         self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec4 = DoubleConv(1024, 512)
#         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = DoubleConv(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = DoubleConv(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = DoubleConv(128, 64)

#         self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))

#         bottleneck = self.bottleneck(self.pool(enc4))

#         dec4 = self.dec4(torch.cat((self.up4(bottleneck), enc4), dim=1))
#         dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), dim=1))
#         dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
#         dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))

#         return self.out_conv(dec1)
# # Define U-Net model with resnet50 encoder and add dropout to the decoder
# class UNetWithDropout(smp.Unet):
#     def __init__(self, encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=11, dropout_rate=0.3):
#         super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
#         self.dropout = torch.nn.Dropout2d(p=dropout_rate)
        
#     def forward(self, x):
#         features = self.encoder(x)
#         decoder_output = self.decoder(*features)
        
#         # Apply dropout in the decoder
#         decoder_output = self.dropout(decoder_output)
        
#         masks = self.segmentation_head(decoder_output)
#         return masks
    
# # Define model
# # model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=11)
# # # model = model.to(device)
# # criterion = nn.CrossEntropyLoss()  # Suitable for multi-class segmentation

# model = UNetWithDropout(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=11)
# weights = torch.tensor([0.1] + [1.0] * 9 + [0.5])  # Lower weight for background (10)

# criterion = nn.CrossEntropyLoss(weight=weights)  # Suitable for multi-class segmentation

# # criterion = torch.nn.BCEWithLogitsLoss()  # Ignore background in loss
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # Hyperparameter tuning - try different learning rates and batch sizes if needed
# # Here we use lr=1e-3, which is adjustable based on observed convergence

# # Increase the number of epochs

# scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=50)
# num_epochs = 50

# # Train U-Net with checkpoint saving and debugging
# # try:
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     # early_stop = False

#     # Using tqdm for progress bar in batch loop
#     with tqdm(train_dataloader, unit="batch") as tepoch:
#         tepoch.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
#         for images, masks in tepoch:
    
#     # for batch_idx, (images, masks) in enumerate(dataloader):
#             optimizer.zero_grad()
#             outputs = model(images)

#             # Debug output stats
#             # print("Outputs min:", outputs.min().item(), "Outputs max:", outputs.max().item())
#             if masks.dim() == 4:
#                 masks = masks.squeeze(1)

#             # Ensure masks are in correct shape and data type
#             loss = criterion(outputs, masks.long())
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             total_loss += loss.item()
#             # print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
#             tepoch.set_postfix(loss=loss.item())
#             # if loss.item() < 0.01:  # Early stop if loss is low, indicating successful overfitting
#             #     print("Model has successfully overfitted on the small batch.")
#             #     early_stop = True
#             #     break
    

#     # if early_stop:
#     #     break

#     avg_t_loss = total_loss / len(train_dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Training Loss: {avg_t_loss:.4f}")

#     # Debugging: Check unique values in model output after each epoch
#     with torch.no_grad():
#         test_outputs = model(images[:1])  # Take a sample image from the batch
#         test_seg_mask = test_outputs.argmax(dim=1)
#         print("Unique values in test_seg_mask after argmax:", torch.unique(test_seg_mask))

#     # # Save the full model after each epoch
#     # torch.save(model, f"unet_model_epoch_{epoch+1}.pth")
#     # print(f"Checkpoint saved for epoch {epoch+1}")

# # except KeyboardInterrupt:
# #     # Save the model state when manually interrupted
# #     torch.save(model, "unet_model_interrupted.pth")
# #     print("Training interrupted. Model saved as unet_model_interrupted.pth")
# #     sys.exit(0)

# # # Final model save
# # torch.save(model, "unet_model_final.pth")
# # print("Training completed. Final model saved as unet_model_final.pth")

# # Gradient check after a backward pass
# # print("\nGradient check for conv1 layer:")
# # for param in model.encoder.conv1.parameters():
# #     if param.grad is not None:
# #         print(param.grad.mean().item())  # Ensure gradients are non-zero
# #     else:
# #         print("No gradient found.")

# # Evaluate the predictions on a small validation set to see if model can generalize
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for images, masks in val_dataloader:
#             outputs = model(images)
#             if masks.dim() == 4:
#                 masks = masks.squeeze(1)
#             loss = criterion(outputs, masks.long())
#             val_loss += loss.item()
#     avg_val_loss = val_loss / len(val_dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] completed, Validation Loss: {avg_val_loss:.4f}")

#     # Save model checkpoint after each epoch
#     checkpoint_path = f"unet_model_epoch_{epoch+1}.pth"
#     torch.save(model.state_dict(), checkpoint_path)
#     print(f"Checkpoint saved for epoch {epoch+1} at {checkpoint_path}")


# # with torch.no_grad():
# #     for i in range(5):
# #         img, mask = small_dataset[i]
# #         pred = model(img.unsqueeze(0))  # Add batch dimension
# #         pred_class = (pred > 0).float()  # Convert logits to binary
# #         print(f"Sample {i} - Predicted unique classes:", torch.unique(pred_class))
# #         print(f"Sample {i} - True mask unique values:", torch.unique(mask))