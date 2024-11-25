from ultralytics import YOLO
# import torch
import sys

# Initialize model, choosing YOLOv5s or any variant
model = YOLO("yolov5su.pt")  # Pre-trained YOLOv5s weights

# train from checkpoint
model = YOLO("yolov5_checkpoint_epoch_20.pt")

# Define training parameters
epochs = 50
checkpoint_interval = 1  # Save after each epoch

try:
    for epoch in range(1, epochs + 1):
        # Train the model for one epoch
        model.train(data="mnistdd.yaml", epochs=1, imgsz=64, batch=16, augment=True)

        # Save checkpoint at each specified interval
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f"yolov5_checkpoint_epoch_{epoch}.pt"
            model.save(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

except KeyboardInterrupt:
    # Save the current state on manual interruption
    model.save("yolov5_interrupted_checkpoint.pt")
    print("Training interrupted. Model saved as yolov5_interrupted_checkpoint.pt")
    sys.exit(0)

# Save the final model after all epochs
model.save("yolov5_final_model.pt")
print("Training completed. Final model saved as yolov5_final_model.pt")


# from ultralytics import YOLO
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import OneCycleLR
# from tqdm import tqdm
# import torch
# import sys

# # Initialize model, choosing YOLOv5s or any variant
# model = YOLO("yolov5mu.pt")  # Pre-trained YOLOv5s weights

# # model.train(data="mnistdd.yaml", epochs=1, imgsz=64, batch=16, augment=True)
# model.overrides['optimizer'] = 'AdamW'  # Set optimizer to AdamW
# # model.overrides['lr0'] = 1e-4  # Start with a lower initial learning rate
# model.overrides['box'] = 0.05  # Lower initial box loss
# model.overrides['cls'] = 0.6  # Class loss scaling
# # model.overrides['iou'] = 0.7  # Increase IOU threshold for better localization
# # model.overrides['obj'] = 0.5  # Object loss scaling
# # model.overrides['momentum'] = 0.937  # Momentum optimization

# # Define training parameters
# epochs = 50
# checkpoint_interval = 1  # Save after each epoch

# # Define optimizer manually
# # optimizer = AdamW(model.model.parameters(), lr=1e-3)
# # scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs)

# try:
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch [{epoch}/{epochs}]")
        
#         # Training progress bar for each epoch
#         with tqdm(total=3308, unit="batch") as pbar:  # Assuming 3308 batches for display
#             # for batch_idx, _ in enumerate(model.train(data="mnistdd.yaml", epochs=1, imgsz=64, batch=16, augment=True, optimizer=optimizer)):
#                 # Update progress bar
#                 # pbar.update(1)

#             # model.train(data="mnistdd.yaml", epochs=1, imgsz=64, batch=16, augment=True, warmup_epochs=3)
#             model.train(
#                 data="mnistdd.yaml",         # Dataset config file
#                 epochs=1,                  # Increase epochs for better training
#                 batch=16,                    # Batch size (adjust as per available memory)
#                 imgsz=128,                   # Higher resolution can improve accuracy
#                 lr0=0.001,                   # Starting learning rate
#                 # lrf=0.01,                    # Final learning rate as a fraction of initial
#                 momentum=0.937,              # Momentum
#                 weight_decay=0.0005,         # Weight decay for regularization
#                 iou=0.6,                     # Adjust IoU threshold for NMS
#                 augment=True,                # Use augmentation
#                 val=True                     # Validate on the validation set after each epoch
#             )
#             pbar.update(3308)
        
#         # Step the scheduler
#         # scheduler.step()
#         # Train the model for one epoch
#         # model.train(data="mnistdd.yaml", epochs=1, imgsz=64, batch=16, augment=True)
#         # if scheduler is None:
#         #     optimizer = model.optimizer
#         #     scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs)

#         # Save checkpoint at each specified interval
#         if epoch % checkpoint_interval == 0:
#             checkpoint_path = f"yolov5_checkpoint_epoch_{epoch}.pt"
#             model.save(checkpoint_path)
#             print(f"Checkpoint saved at {checkpoint_path}")

# except KeyboardInterrupt:
#     # Save the current state on manual interruption
#     model.save("yolov5_interrupted_checkpoint.pt")
#     print("Training interrupted. Model saved as yolov5_interrupted_checkpoint.pt")
#     sys.exit(0)

# # Save the final model after all epochs
# model.save("yolov5_final.pt")
# print("Training completed. Final model saved as yolov5_final.pt")
