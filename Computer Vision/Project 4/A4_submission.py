import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_segmentation_mask(segmentation_mask):
    """
    Visualizes the segmentation mask with specific colors for each digit and background.
    
    :param segmentation_mask: np.ndarray, 2D array where each pixel value corresponds to a digit (0-9) or background (10)
    """
    # Define the color map for each digit and background
    color_map = {
        0: "red",
        1: "green",
        2: "blue",
        3: "magenta",
        4: "cyan",
        5: "yellow",
        6: "purple",
        7: "forestgreen",
        8: "orange",
        9: "maroon",
        10: "black"  # Background
    }

    # Create a color map for the visualization
    cmap = mcolors.ListedColormap([color_map[i] for i in range(11)])
    bounds = np.arange(-0.5, 11, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the segmentation mask with color mapping
    # plt.figure(figsize=(6, 6))
    # plt.imshow(segmentation_mask, cmap=cmap, norm=norm)
    # plt.colorbar(ticks=np.arange(11), label='Digit')
    # plt.title("Semantic Segmentation Mask Visualization")
    # plt.axis("off")
    # plt.show()

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
    
def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.zeros((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.zeros((N, 4096), dtype=np.int32)

    # add your code here to fill in pred_class and pred_bboxes
    # model_detection = YOLO("YOLO/cp/yolov5_checkpoint_epoch_15.pt")  # Load trained YOLOv5 model
    # model_detection = YOLO("YOLO/cp/yolov5_checkpoint_epoch_4.pt")  # Load trained YOLOv5 model
    model_detection = YOLO("yolov5_checkpoint_epoch_23s.pt")  # Load trained YOLOv5 model
    
    # model_segmentation = UNetWithDropout(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=11)
    model_segmentation = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=11) 
    state_dict = torch.load("unet_epoch_5.pth", map_location=torch.device('cpu'))

    # model_segmentation = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=11)
    # # model_segmentation.load_state_dict("unet_model_epoch_10.pth")
    # state_dict = torch.load("unet_model_epoch_15.pth")
    model_segmentation.load_state_dict(state_dict, strict = False)
    model_segmentation.eval()

    # Transformation for images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    for i in range(N):
        img = images[i].reshape(64, 64, 3).astype(np.uint8)
        img_tensor = transform(img).unsqueeze(0)

        # Step 1: Detection with YOLO
        with torch.no_grad():
            det_results = model_detection(img_tensor)
            if len(det_results[0].boxes) >= 2:  # Ensure at least two detections
                boxes = det_results[0].boxes.xyxy.cpu().numpy()
                classes = det_results[0].boxes.cls.cpu().numpy().astype(int)

                # Filter for top two detections by confidence if needed
                sorted_indices = np.argsort(classes)
                classes = classes[sorted_indices[:2]]
                boxes = boxes[sorted_indices[:2]]
                pred_class[i] = sorted(classes)
                pred_bboxes[i] = boxes

        # Step 2: Segmentation with UNet
        with torch.no_grad():
            seg_results = model_segmentation(img_tensor)

            # Debugging output
            # print("Segmentation model output shape:", seg_results.shape)
            # print("Unique values in seg_results before argmax:", torch.unique(seg_results))
            seg_mask = seg_results.argmax(dim=1).cpu().numpy().reshape(64, 64)
            # seg_mask = (torch.sigmoid(seg_results) > 0.5).int().cpu().numpy().reshape(4096)
            # pred_seg[i] = seg_mask
            
            # Get the predicted class for each pixel (0-9 for digits, 10 for background)
            # seg_mask = seg_results.argmax(dim=1).cpu().numpy().reshape(64, 64)  # Reshape to 64x64 for visualization
            
            # Debugging: Check unique values in seg_mask
            # print("Unique values in seg_mask after argmax:", np.unique(seg_mask))

            # Store segmentation result in flattened format
            pred_seg[i] = seg_mask.flatten()

            # Visualize the segmentation mask for each image
            visualize_segmentation_mask(seg_mask)

    return pred_class, pred_bboxes, pred_seg
