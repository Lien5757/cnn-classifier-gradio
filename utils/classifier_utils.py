import os
import cv2
import copy
import timm
import math
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_train_val_data(train_dir=r'datasets\train', val_dir=r'datasets\val', batch_size=32, data_balance=True, num_workers=0, device='cuda'):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),                       # Resize to 256x256
        transforms.CenterCrop((224, 224)),                   # Randomly crop to 224x224
        transforms.RandomHorizontalFlip(p=0.5),              # Random horizontal flip with 50% probability
        transforms.RandomVerticalFlip(p=0.5),                # Random vertical flip with 50% probability
        transforms.RandomRotation(degrees=30),               # Random rotation within ±30 degrees
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        transforms.ToTensor(),                               # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                      # Normalize with ImageNet mean
            std=[0.229, 0.224, 0.225]                        # Normalize with ImageNet std
        )
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),                       # Resize to 256x256
        transforms.CenterCrop((224, 224)),                  # Central crop to 224x224
        transforms.ToTensor(),                              # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize with ImageNet mean
                            std=[0.229, 0.224, 0.225])     # Normalize with ImageNet std
    ])

    ## Datasets
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_val)

    ## Calculate class weights for WeightedRandomSampler
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    weights = torch.tensor([1.0 / math.log(1.02 + count) for count in class_counts], device=device)

    if data_balance:
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=False)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, val_dataloader, weights


def load_test_data(test_dir=r'datasets\test', batch_size=32):
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),   
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))  # Normalize for RGB channels
        ])
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataset, test_dataloader

def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)): # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(std).view(3, 1, 1).to(image.device)
        return image * std + mean


def show_images_from_batch(X, num_images=5, scale=5):
    X = X.cpu()
    images = []
    for i in range(min(num_images, X.size(0))):
        img = X[i]
        img = denormalize(img).permute(1, 2, 0).numpy()  # Convert to numpy array in (H, W, C)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        images.append(img)

    # Stack images horizontally
    stacked_images = np.hstack(images)
    cv2.imshow('Stacked Training Images', stacked_images)
    cv2.waitKey(10)

# training function:　Chage to train_one_epoch
def train_one_epoch(epoch_num, dataloader, model, loss_fn, optimizer, device, period=100):  ## train_one_epoch
    size = len(dataloader.dataset)
    total_loss = 0
    total_correct = 0

    for batch, (X, y) in enumerate(dataloader): # X:image y:label
        X, y = X.to(device), y.to(device)

        ## show Images
        # if epoch_num == 1 and batch < 5: # Display images only in the first 5 batches of the first epoch
        #     show_images_from_batch(X, num_images=5)

        # Forward
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() # 加权累加损失
        total_correct += (pred.argmax(1) == y).sum().item()  # 累加正确预测数

        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % period == 0: # 100 show一次
            loss, current = loss.item(), batch * len(X)
            print(f"Batch: {batch+1}  ==>  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    acc = total_correct / size
    total_loss = total_loss / len(dataloader) # len(dataloader):4
    return model, total_loss, acc

## Building testing function: calc loss
def validation(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)  # 數據集樣本數
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            total_correct += (pred.argmax(1) == y).sum().item()

    acc = total_correct / size
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def plot_loss_acc(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_dir):
    x = [i + 1 for i in range(len(train_loss_list))]
    fig = plt.figure("Training Process: Acc and Loss", figsize=(10, 5))
    fig.clf()

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss_list, c='r', label="Train Loss")
    plt.plot(x, val_loss_list, c='b', label="Validation Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc_list, c='r', label="Train Accuracy")
    plt.plot(x, val_acc_list, c='b', label="Validation Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.ylim((0, 1))
    
    plt.tight_layout()
    file_path = f"{save_dir}/training_process.png"
    plt.savefig(file_path)
    plt.close(fig)
    return file_path

def train(train_loader, val_loader, model, save_dir, num_epoch = 100, learning_rate = 0.001, opt = 'sgd', early_stop = 10):
    ## Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = model.to(device)

    ## Choose or build a loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()   #nn.MSELoss()
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  ## adam()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    ## training n epochs
    train_loss_list = list()
    train_acc_list = list()
    test_loss_list = list()
    test_acc_list = list()
    best_acc = 0
    no_improve = 0
    for t in range(num_epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        ## train & validation
        model, train_loss, train_acc = train_one_epoch(t+1, train_loader, model, loss_fn, optimizer, device)
        test_loss, test_acc = validation(val_loader, model, loss_fn, device)
        print(f"Train/Val Result: \n Accuracy: {(100*train_acc):>0.1f}/{(100*test_acc):>0.1f} %, Avg loss: {train_loss:>8f}/{test_loss:>8f} \n")
        if test_acc > best_acc:
            best_acc = test_acc
            ## save static model
            torch.save( model.state_dict(), f"{save_dir}/best_model.pth" )
            print("Saving best model: ", best_acc, " accuracy")
            no_improve = 0
        else:
            no_improve = no_improve +1
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        plot_loss_acc(train_loss_list, test_loss_list, train_acc_list, test_acc_list, save_dir)
        if no_improve > early_stop:
            torch.save(model.state_dict(), f"{save_dir}/final_model.pth")
            print("Early Stop and save final_model.pth: ", t, " epoch")
    print("Done!")

    ## saving model
    torch.save(model.state_dict(), f"{save_dir}/final_model.pth")
    print("Saved PyTorch Model State to final_model.pth")
    return

def predict(model, dataloader, device):
    size = len(dataloader.dataset)
    print('Testing Datasize:', size)

    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    return y_true, y_pred

def display_confusion_matrix(y_true, y_pred, class_names, is_show=False, is_save=True, save_dir=''):
    # Figure setup
    if len(class_names) <= 4:
        fig_size = (6, 5)
        fontsize = 10
        label_size = 10
    else:
        fig_size = (10, 8)
        fontsize = 9
        label_size = 8

    # Draw Non-normalized cm
    cm_raw = confusion_matrix(y_true, y_pred)
    cmd_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=class_names)

    plt.figure(figsize=fig_size)
    cmd_raw.plot(cmap="Blues", ax=plt.gca(), values_format='d')
    cmd_raw.ax_.set(title="Confusion Matrix (Raw)", xlabel="Predicted", ylabel="True")
    
    for text in cmd_raw.ax_.texts:
        text.set_fontsize(fontsize)
    cmd_raw.ax_.tick_params(axis='x', labelsize=label_size)
    cmd_raw.ax_.tick_params(axis='y', labelsize=label_size)

    if is_save:
        raw_save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(raw_save_path, bbox_inches='tight')
    if is_show:
        plt.show()

    # Draw normalized cm
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    cmd_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)

    plt.figure(figsize=fig_size)
    cmd_norm.plot(cmap="Blues", ax=plt.gca())
    cmd_norm.ax_.set(title="Confusion Matrix (Normalized)", xlabel="Predicted", ylabel="True")
    
    for text in cmd_norm.ax_.texts:
        value = float(text.get_text())
        if value < 0.01:
            text.set_text("0")
        else:
            text.set_text(f"{value:.2f}")
        text.set_fontsize(fontsize)
    cmd_norm.ax_.tick_params(axis='x', labelsize=label_size)
    cmd_norm.ax_.tick_params(axis='y', labelsize=label_size)

    if is_save:
        norm_save_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plt.savefig(norm_save_path, bbox_inches='tight')
    if is_show:
        plt.show()

    return norm_save_path

def predicting(device, model, save_dir, test_dir, class_names=['cats', 'dogs']):
    _, test_loader = load_test_data(test_dir, batch_size = 64)
    model_path = os.path.join(save_dir,'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true, y_pred = predict(model, test_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    display_confusion_matrix(y_true, y_pred, class_names, save_dir=save_dir)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

def predicting_with_heatmap(device, model, save_dir, test_dir, class_names=['cats', 'dogs'], target_layer=10):
    # Load test data
    _, test_loader = load_test_data(test_dir, batch_size = 64)
    
    # Load the best model
    model_path = os.path.join(save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize lists for metrics
    y_true, y_pred = [], []
    
    # Create directories for saving results
    heatmap_dir = os.path.join(save_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Process each batch in the test loader
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        # Append results for metrics calculation
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

        # Generate and save heatmaps for each image
        for i in range(images.size(0)):
            # Get the single image and its label
            image = images[i].unsqueeze(0)  # Add batch dimension
            label = labels[i].item()
            pred = predictions[i].item()

            # Preprocess image for heatmap generation
            heatmap = grad_cam(model, target_layer=target_layer, input_image=image)

            # Reverse normalization
            denormalized_image = denormalize(images[i])
            original_image = denormalized_image.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
            original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Prepare heatmap
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            heatmap_on_image = show_heatmap_on_image(heatmap, original_image)

            # Add text: True Label and Prediction
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color_true = (0,0,0)
            if pred != label:
                color_pred = (0,0,255)
                print(f'batch_{batch_idx}_image_{i}.jpg predict wrong!')
            else:
                color_pred = (0,0,0)

            true_label_name = class_names[label]
            pred_label_name = class_names[pred]

            original_image = cv2.putText(
                original_image.copy(), 
                f"label: {true_label_name}", 
                (10, 30), 
                font, 
                font_scale, 
                color_true, 
                1, 
                cv2.LINE_AA
            )
            heatmap_on_image = cv2.putText(
                heatmap_on_image.copy(), 
                f"pred: {pred_label_name}", 
                (10, 30), 
                font, 
                font_scale, 
                color_pred, 
                1, 
                cv2.LINE_AA
            )
            stacked_image = np.hstack((original_image, heatmap_on_image))

            # Save the image
            save_path = os.path.join(heatmap_dir, f'batch_{batch_idx}_image_{i}.jpg')
            cv2.imwrite(save_path, stacked_image)
            # cv2.imshow('Predict & heatmap:', stacked_image)
            # cv2.waitKey(0)
    
    # Calculate and display metrics
    display_confusion_matrix(y_true, y_pred, class_names, save_dir=save_dir)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("Confusion Matrix:")
    print(cm)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

def cross_validate_model(
    model, 
    dataset, 
    k=5, 
    num_epochs=10, 
    batch_size=64, 
    lr=0.001,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
    save_dir='',
    early_stop=20
):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()

    fold_accuracies = []
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k}')
        
        # Create data loaders for this fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Reinitialize a new instance of the model for each fold
        fold_model = copy.deepcopy(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-3)

        best_acc = 0
        no_improve = 0

        for epoch in range(num_epochs):
            fold_model.train()
            fold_model, train_loss, train_acc = train_one_epoch(epoch+1, train_loader, fold_model, criterion, optimizer, device)
            val_loss, val_acc = validation(val_loader, fold_model, criterion, device)

            print(f"Train/Val Result: \n Accuracy: {(100*train_acc):>0.1f}/{(100*val_acc):>0.1f} %, Avg loss: {train_loss:>8f}/{val_loss:>8f} \n")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save( model.state_dict(), f"{save_dir}/best_model.pth" )
                print("Saving best model: ", best_acc, " accuracy")
                no_improve = 0
            else:
                if no_improve >= early_stop:
                    print("Early stopping triggered.")
                    break
            
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            # Plot loss and accuracy after each epoch (optional)
            plot_loss_acc(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_dir)

        # Append the final validation accuracy of each fold
        fold_accuracies.append(val_acc)

    # Print average accuracy across all folds
    avg_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy across {k} folds: {avg_accuracy:.4f}')

    # Saving the final model
    torch.save(fold_model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print("Saved PyTorch Model State to final_model.pth")

    return fold_model

def get_model_target_layer(save_dir, device, num_classes=2 ,model_name="AlexNet"):
    if model_name == "ResNet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        model_path = os.path.join(save_dir, 'best_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model = model.to(device)
        model.eval()

        target_layer = model.layer4[-1]
    
    elif model_name == "EfficientNet-b0":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)  # <-- key change here
        model_path = os.path.join(save_dir, 'best_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        target_layer = model.blocks[-1]

    elif model_name == "DenseNet121":
        model = torchvision.models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        model_path = os.path.join(save_dir, 'best_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        target_layer = model.features[-1]

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, target_layer

def grad_cam(model, input_image, target_layer, class_index=None):
    """
    Generate Grad-CAM heatmap for a specific class prediction.

    Args:
        model: Trained model (e.g., AlexNet).
        target_layer: Layer to compute Grad-CAM (e.g., 'conv5').
        input_image: Input image tensor with shape [1, C, H, W].
        class_index: Index of the target class (default: predicted class).

    Returns:
        heatmap: Grad-CAM heatmap as a numpy array.
    """
    model.eval()

    # Hook to capture the feature maps and gradients
    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Register hooks on the target layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Perform forward pass
    input_image = input_image.to(next(model.parameters()).device)
    output = model(input_image)

    # If class_index is None, use the predicted class
    if class_index is None:
        class_index = torch.argmax(output, dim=1).item()

    # Compute gradients of the target class w.r.t. feature maps
    model.zero_grad()
    class_score = output[:, class_index]
    class_score.backward()

    # Average the gradients across the spatial dimensions (global average pooling)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the feature maps using the pooled gradients
    feature_maps = feature_maps[0].cpu().detach().numpy()
    pooled_gradients = pooled_gradients.cpu().detach().numpy()
    for i in range(feature_maps.shape[0]):
        feature_maps[i, :, :] *= pooled_gradients[i]

    # Compute the heatmap by averaging the weighted feature maps
    heatmap = np.mean(feature_maps, axis=0)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]

    # Cleanup hooks
    forward_handle.remove()
    backward_handle.remove()

    return heatmap

def show_heatmap_on_image(heatmap, original_image, save_path='', filename='', alpha=0.6, colormap=cv2.COLORMAP_JET):
    # Convert heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Resize heatmap to match original image size
    original_image = np.array(original_image)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Convert heatmap to RGB if needed
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Overlay the heatmap on the original image
    overlayed_image = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

    # Display the image
    # cv2.imwrite(os.path.join(save_path, filename), overlayed_image)
    # cv2.imshow('Heatmap', overlayed_image)
    # cv2.waitKey(0)

    return overlayed_image