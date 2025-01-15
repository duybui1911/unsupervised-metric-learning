import os
import random
import argparse
import torch

from glob import glob
import torch.utils.data as data
from tqdm import tqdm
from pytorch_metric_learning import losses, miners, distances
from transformers import get_cosine_schedule_with_warmup

import configs as cfg
import vision_transformer as vits
from data.dataset import ShoesDataset
from data.augmentation.augmentations import DataAugmentation

def train_epoch(model, train_loader, optimizer, scheduler, criterion, miner, device):
    def forward_step(images, label):
        optimizer.zero_grad()
        outputs = []
        labels = []
        for image in images:
            image = image.to(device)
            augment_embed = model(image)
            outputs.append(augment_embed)
            labels.append(label)
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        hard_pairs = miner(outputs, labels)
        loss = criterion(outputs, labels, hard_pairs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.item()

    model.train()
    running_loss = 0.0
    for idx, batch_data in tqdm(enumerate(train_loader)):
        images = batch_data[:-1]
        label = batch_data[-1]
        label = label.to(device)
        loss = forward_step(images, label)
        running_loss += loss
    epoch_loss = running_loss / (idx + 1)

    return model, optimizer, epoch_loss

def eval(model, loader, criterion, miner,device):
    def forward_step(image, label):
        image = image.to(device)
        outputs = model(image)
        hard_pairs = miner(outputs, label)
        loss = criterion(outputs, label, hard_pairs)
        return loss.item()

    model.eval()
    running_loss = 0
    for idx, (image1, image2, image3, label) in tqdm(enumerate(loader)):
        label = label.to(device)
        merge_image12 = [(img, l) for img, l in zip(image1, label)] + [(img, l) for img, l in zip(image2, label)]
        random.shuffle(merge_image12)
        image1 = torch.stack([img for img, _ in merge_image12[:len(merge_image12)//2]])
        label1 = torch.stack([l for _, l in merge_image12[:len(merge_image12)//2]])
        image2 = torch.stack([img for img, _ in merge_image12[len(merge_image12)//2:]])
        label2 = torch.stack([l for _, l in merge_image12[len(merge_image12)//2:]])

        loss1 = forward_step(image1, label1)
        loss2 = forward_step(image2, label2)
        loss3 = forward_step(image3, label)
        
        running_loss += (loss1 + loss2 + loss3) / 3

    return running_loss/len(loader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser('Unsupervised Metric Learning')
    parser.add_argument('--input_data', default='', type=str)
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument('--output_dir', default='save_weights')

    args = parser.parse_args()

    data_folder = args.input_data
    best_path = f'{args.output_dir}/dino_{cfg.MODEL_NAME}_{cfg.PATCH_SIZE}_best.pth'
    last_path = f'{args.output_dir}/dino_{cfg.MODEL_NAME}_{cfg.PATCH_SIZE}_last.pth'

    pre_train = True if os.path.exists(args.pretrained_weights) else False
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(args.pretrained_weights):
        pre_train = True
    else:
        pre_train = False

    # Feature Extractor
    train_processor = DataAugmentation(
        cfg.GLOBAL_CROP_RATIO,
        cfg.LOCAL_CROP_RATIO,
        cfg.NUM_LOCAL,
        cfg.GLOBAL_IMG_SIZE,
        cfg.LOCAL_IMG_SIZE
    )

    model = vits.__dict__[cfg.MODEL_NAME](patch_size=cfg.PATCH_SIZE, num_classes=0)
    if pre_train:
        model.load_state_dict(torch.load(best_path))
    else:
        if cfg.PATCH_SIZE == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif cfg.PATCH_SIZE == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)

    print('Model parameters: ', count_parameters(model))

    # Load data
    class_folder = glob(data_folder + "/*/*")
    class_idx_list = [i for i in range(len(class_folder))]

    miner = miners.BatchHardMiner()
    criterion = losses.ContrastiveLoss(
        pos_margin=1., 
        neg_margin=0., 
        distance=distances.CosineSimilarity()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = model.to(device)

    initial_lr = cfg.LEARNING_RATE
    warmup_epochs = cfg.WARMUP_EPOCH
    total_epochs = cfg.EPOCHS
    if cfg.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    elif cfg.OPTIMIZER == 'adamw':
        betas=(cfg.ADAMW_BETA1, cfg.ADAMW_BETA2)
        optimizer = torch.optim.AdamW(model.parameters(), betas=betas, lr=initial_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    best_loss = float('inf')
    for epoch in range(total_epochs):
        random.shuffle(class_folder)
        running_loss = 0.0
        final_class_idx_list = []
        final_class_folder = []
        for idx, folder_name in zip(class_idx_list, class_folder):
            if "products.csv" in folder_name:
                continue
            folder_file = glob(folder_name + "/*")
            final_class_folder.extend(folder_file)
            final_class_idx_list.extend([idx] * len(folder_file))

        train_ds = ShoesDataset(final_class_folder, final_class_idx_list, transform=train_processor)
        train_loader = data.DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,  num_workers=4)
        if epoch == 0:
            print("Num paths = ", len(final_class_idx_list))
            print("Num labels = ", len(final_class_folder))
            
            print("Train batch: ", len(final_class_idx_list)/cfg.BATCH_SIZE)
            print("Start LR: ", optimizer.param_groups[0]['lr'] )
            total_steps = len(train_loader) * total_epochs
            warmup_steps = len(train_loader) * warmup_epochs

            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=warmup_steps, 
                                                        num_training_steps=total_steps)

        model.train()
        model, optimizer, running_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, miner, device)
        
        print(f"Epoch [{epoch+1}/{total_epochs}], "
            f"Train Loss: {running_loss:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), best_path)
        
        torch.save(model.state_dict(), last_path)


if __name__ == "__main__":
    main()