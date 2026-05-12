import os
import cv2
import time
import numpy as np
import timm
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import logging
from logging import Handler
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gradio as gr

from utils.classifier_utils import load_train_val_data, train_one_epoch, validation, denormalize, load_test_data, plot_loss_acc, grad_cam, show_heatmap_on_image, display_confusion_matrix, get_model_target_layer

class GradioLogHandler(Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)

    def get_logs(self):
        return "\n".join(self.logs)
    
def setup_logging(save_dir, ui_handler=None):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Avoiding repulicate handler
    if logger.handlers:
        return logger

    log_file = os.path.join(save_dir, f"train.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Gradio handler
    if ui_handler:
        ui_handler.setFormatter(formatter)
        logger.addHandler(ui_handler)

    return logger

def show_images(dataloader, num_images=5):
    images = []

    # Extract one patch in the dataloader
    batch_idx, (data, _) = next(enumerate(dataloader))

    # Make sure the data run in cpu
    data = data.cpu() # NumPy cannot operate on GPU tensors

    for i in range(min(num_images, data.size(0))):
        img = data[i]
        img = denormalize(img).permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
        images.append(img)

    # Horizontally stack the images
    stacked_images = np.hstack(images)

    return stacked_images

def train(data_name, model, save_dir, opt, num_epoch, batch_size, early_stop, logger, ui_log_handler, config):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")    

    # Datasets
    data_dir = os.path.join('datasets', data_name)
    train_dir = fr'{data_dir}\train'
    val_dir = fr'{data_dir}\val'

    # Dataloader
    train_loader, val_loader, weights = load_train_val_data(train_dir=train_dir, val_dir=val_dir, batch_size=batch_size, device=device)

    # Save training config
    training_config = {
        'training': {
            'batch_size': batch_size,
            'num_epoch': num_epoch,
            'early_stop': early_stop,
            'device': str(device),
        },
        'data': {
            'data_name': data_name,
            'save_path': save_dir,
            'num_train_samples': len(train_loader.dataset),
            'num_val_samples': len(val_loader.dataset),
        }
    }
    config = {**config, **training_config} # config |= training_config (python>3.9)

    config_file = os.path.join(save_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {config_file}")

    # Load model
    model = model.to(device)

    # Loss function & optimizer
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Training n epochs
    train_loss_list = list()
    train_acc_list = list()
    val_loss_list = list()
    val_acc_list = list()
    best_acc = 0
    no_improve = 0

    start_time = time.perf_counter()
    for t in range(num_epoch):
        logger.info(f"Epoch {t+1}/{num_epoch} " + "-"*50)

        # Training & validation
        start_epoch_time = time.perf_counter()
        image_np = show_images(train_loader)

        model, train_loss, train_acc = train_one_epoch(t+1, train_loader, model, loss_fn, opt, device)
        val_loss, val_acc = validation(val_loader, model, loss_fn, device)
        logger.info(f"Train - Loss: {train_loss:.6f}, Acc: {100*train_acc:.2f}%")
        logger.info(f"Val   - Loss: {val_loss:.6f}, Acc: {100*val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save( model.state_dict(), f"{save_dir}/best_model.pth" )
            logger.info(f"✓ New best model saved! Accuracy: {100*best_acc:.2f}%")
            no_improve = 0
        else:
            no_improve = no_improve +1
            logger.info(f"No improvement for {no_improve} epoch(s)")

            if no_improve >= early_stop:
                torch.save(model.state_dict(), f"{save_dir}/final_model.pth")
                logger.info(f"Early stopping at epoch {t+1}")
                break
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        fig = plot_loss_acc(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_dir)

        yield fig, image_np, ui_log_handler.get_logs()
        end_epoch_time = time.perf_counter()
        total_epoch_time = round(end_epoch_time - start_epoch_time)
        logger.info(f"Epoch time: {total_epoch_time}s\n")

    end_time = time.perf_counter()
    total_time = round(end_time - start_time)
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info(f"Training complete!")
    logger.info(f"Best_val_acc: {100*best_acc:.2f}%")
    logger.info(f"Final_train_acc: {100*train_acc_list[-1]}")
    logger.info(f"Final_val_acc: {val_acc_list[-1]}")
    logger.info(f'Total_epoch: {len(train_loss_list)}')
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
    yield fig, image_np, ui_log_handler.get_logs()

    # Save model
    torch.save(model.state_dict(), f"{save_dir}/final_model.pth")

def set_exp_idx(data_name, model_name, project_name):
    idx = 1
    while True:
        project_name_idx = f"{project_name}_exp{idx}"
        save_dir = os.path.join('experiments', data_name, model_name, project_name_idx)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            return save_dir
        idx += 1

def gradio_train(data_name, model_name, opt_name, lr, weight_decay, num_epoch, batch_size, early_stop):
    num_classes = len(os.listdir(os.path.join('datasets', data_name, 'train')))
    model = load_model(model_name=model_name, num_classes=num_classes)

    project_name = f"{opt_name}_{lr}_{weight_decay}_{batch_size}_{num_epoch}_{early_stop}"
    save_dir = set_exp_idx(data_name, model_name, project_name)

    # Set log
    ui_log_handler = GradioLogHandler()
    logger = setup_logging(save_dir, ui_handler=ui_log_handler)

    logger.info(f"Starting training: {data_name}, classes:{num_classes}")
    logger.info(f"Model:{model_name}, loss:CrossEntropyLoss, class_weighted:True")
    logger.info(f"Optimizer:{opt_name}, lr:{lr}, weight_decay:{weight_decay}")
    logger.info(f"Epochs:{num_epoch}, batch_size:{batch_size}, early_stop:{early_stop}")
    
    # Set Training Config
    config = {
        'model': model_name,
        'optimizer': {
            'name': opt_name,
            'lr': lr,
            'weight_decay': weight_decay
        },
        'loss': 'CrossEntropyLoss',
        'class_weighted': True
    }

    opt = load_opt(model=model, opt_name=opt_name, lr=lr, weight_decay=weight_decay)
    yield from train(data_name, model, save_dir, opt, num_epoch, batch_size, early_stop, logger, ui_log_handler, config)

def test_all(data_name, model_name, checkpoint_dir, isShowHeatmap=True):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # Datasets
    data_dir = os.path.join('datasets', data_name)
    train_dir = fr'{data_dir}\train'
    test_dir = fr'{data_dir}\test'

    # Classes
    class_names = os.listdir(train_dir) # ['cats', 'dogs']
    if len(class_names) > 2:
        class_names = [name for name in class_names] # name.split('_')[1][:2] # name.split('_')[0][:2]
    # print('class label:', class_names)

    # Load test data
    _, test_loader = load_test_data(test_dir, batch_size=8)
    
    # Load the best model
    model_dir = os.path.join('experiments', data_name, model_name, checkpoint_dir)
    model, target_layer = get_model_target_layer(save_dir=model_dir, device=device, num_classes=len(class_names),model_name=model_name)
    
    # Save dir
    pred_save_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(pred_save_dir, exist_ok=True)
    
    # Initialize lists for metrics
    y_true, y_pred = [], []
    
    correct_heatmaps_path = []
    wrong_heatmaps_path = []

    # Process each batch in the test loader
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        # Append results for metrics calculation
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

        if isShowHeatmap:
            batch_start = batch_idx * test_loader.batch_size

            # Generate and save heatmaps for each image
            for i in range(images.size(0)):
                # Get filename
                dataset_index = batch_start + i
                filepath, _ = test_loader.dataset.imgs[dataset_index]
                filename = os.path.basename(filepath)

                # Get the single image and its label
                image = images[i].unsqueeze(0)  # Add batch dimension
                label = labels[i].item()
                pred = predictions[i].item()

                # Preprocess image for heatmap generation
                heatmap = grad_cam(model, input_image=image, target_layer=target_layer)

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
                color_true = (0,255,0)

                # Get label's name
                true_label_name = class_names[label]
                pred_label_name = class_names[pred]

                if pred != label:
                    # save path
                    wrong_heatmap_dir = os.path.join(pred_save_dir, 'heatmaps', 'wrong', f'{true_label_name}_{pred_label_name}')
                    os.makedirs(wrong_heatmap_dir, exist_ok=True)

                    color_pred = (0,0,255)
                    save_path = os.path.join(wrong_heatmap_dir, filename)
                    wrong_heatmaps_path.append(save_path)
                else:
                    # save path
                    correct_heatmap_dir = os.path.join(pred_save_dir, 'heatmaps', 'correct', f'{pred_label_name}')
                    os.makedirs(correct_heatmap_dir, exist_ok=True)

                    color_pred = (0,255,0)
                    save_path = os.path.join(correct_heatmap_dir, filename)
                    correct_heatmaps_path.append(save_path)

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
                cv2.imwrite(save_path, stacked_image)

    # Calculate and display metrics
    cm_path = display_confusion_matrix(y_true, y_pred, class_names, save_dir=pred_save_dir)
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Compute overkill rate
    ok_class_idx = len(class_names) - 1
    overkill_fn = cm[ok_class_idx, :ok_class_idx].sum()      # good predicted as defect
    total_good = cm[ok_class_idx].sum()            # all actual good samples
    overkill_rate = overkill_fn / total_good if total_good > 0 else 0

    # Save to .txt
    save_metric_path = os.path.join(pred_save_dir, "evaluation_results.txt")
    with open(save_metric_path, "w") as f:
        f.write(f"{checkpoint_dir}\n")
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Overkill Rate: {overkill_rate:.4f}\n")

    # Form the result msg
    result_text = format_evaluation_result(cm, accuracy, precision, recall, f1, overkill_rate, pred_save_dir)

    return cm_path, result_text

def format_evaluation_result(cm, accuracy, precision, recall, f1, overkill_rate, save_dir):
    result = []
    
    result.append("Evaluation Results")
    result.append("=" * 30)
    
    result.append("Confusion Matrix:")
    result.append(str(cm))
    result.append("")
    
    result.append(f"Accuracy      : {accuracy:.4f}")
    result.append(f"Precision     : {precision:.4f}")
    result.append(f"Recall        : {recall:.4f}")
    result.append(f"F1 Score      : {f1:.4f}")
    result.append(f"Overkill Rate : {overkill_rate:.4f}")
    
    result.append("")
    result.append(f"Evaluation results saved to: {save_dir}")
    
    return "\n".join(result)

# Get model
def get_model_folders_dynamic(data_name, model_name):
    path = os.path.join('experiments', data_name, model_name)
    if not os.path.exists(path):
        return []
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

def update_model_folders(data_name, model_name):
    return gr.update(choices=get_model_folders_dynamic(data_name, model_name))

# Get data dir
def get_folders(path):
    try:
        folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        return folders
    except Exception as e:
        return [f"Error: {str(e)}"]

def update_folder_choices():
    return get_folders(data_root)


def load_opt(model, opt_name, lr, weight_decay):
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # weight_decay=1e-4

    elif opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported model: {opt_name}")
    
    return opt

def load_model(model_name, num_classes=2):
    if model_name == "ResNet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "EfficientNet-b0":
        model = timm.create_model("efficientnet_b0", pretrained=True)

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
 
    elif model_name == "DenseNet121":
        model = torchvision.models.densenet121(pretrained=True)

        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

if __name__ == "__main__":
    data_root = r'datasets'
    with gr.Blocks() as demo:            
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        data_dir = gr.Dropdown(label="Select Data Folder", choices=get_folders(data_root), interactive=True)
                        update_btn = gr.Button("Refresh dataset list!")
                        update_btn.click(
                            fn=update_folder_choices,
                            inputs=[],
                            outputs=[data_dir]
                        )
                    model_name = gr.Radio(choices=["ResNet18", "EfficientNet-b0", "DenseNet121"], label="Model", value="ResNet18")
                    opt = gr.Radio(choices=["sgd", "adam"], label="Optimizer", value="sgd")
                    lr = gr.Number(label="Learning Rate", value=0.001, interactive=True)
                    weight_decay = gr.Number(label="L2:weight decay", value=0.0001, interactive=True)
                    num_epoch = gr.Number(label="Epochs", value=200, interactive=True)
                    batch_size = gr.Number(label="Batch Size", value=64, interactive=True)
                    early_stop = gr.Number(label="Early stop", value=50, interactive=True)
                    
                with gr.Column():
                    training_image = gr.Image(label="Training Process", interactive=False, type="filepath")
                    data_samples_output = gr.Image(label="Batch Samples", type="numpy")
                    log_box = gr.Textbox(label="Training Log", lines=10, interactive=False)

            # Execution Button
            train_button = gr.Button("Start Training")
            train_button.click(
                gradio_train,
                inputs=[data_dir, model_name, opt, lr, weight_decay, num_epoch, batch_size, early_stop],
                outputs=[training_image, data_samples_output, log_box]
            )

        with gr.Tab("Test"):
            with gr.Row():
                with gr.Column():
                    data_dir = gr.Dropdown(label="Select Datasets", choices=get_folders(data_root), interactive=True)
                    model_name = gr.Radio(choices=["ResNet18", "EfficientNet-b0", "DenseNet121"], label="Model", value="ResNet18")
                    checkpoint_dir = gr.Dropdown(label="Model Directory")
                    isSaveHeatmap = gr.Checkbox(label="Generate heatmaps?")

                with gr.Column():
                    cm_output = gr.Image(label="Confusion Matrix", type="filepath")
                    end_msg = gr.Textbox(label="Result Message")

            with gr.Row():
                    predict_button = gr.Button("Start Prediction")
                    predict_button.click(
                        test_all,
                        inputs=[data_dir, model_name, checkpoint_dir, isSaveHeatmap],
                        outputs=[cm_output, end_msg]
                    )
        model_name.change(
            update_model_folders,
            inputs=[data_dir, model_name],
            outputs=checkpoint_dir
            )

    demo.launch()