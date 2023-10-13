import torch
import torchvision
from dataset import BubbleDataset
from dataset import SlugDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.multiprocessing
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFont, ImageDraw, ImageOps
import textwrap
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
torch.multiprocessing.set_sharing_strategy('file_system')
import datetime
import os

label_dict = {"Invalid":[0., 0., 0.],"Bubbly":[1., 0., 0.],"Bubbly Slug":[1., 1., 0.],"Slug Annular with some Bubbles":[1., 1., 1.],"Slug":[0., 1., 0.],"Slug Annular":[0., 1., 1.],"Annular with Bubbles":[1., 0., 1.],"Annular":[0., 0., 1.]}
class_dict = {"Invalid":[0., 0., 0.],"4":[1., 0., 0.],"6":[1., 1., 0.],"7":[1., 1., 1.],"2":[0., 1., 0.],"3":[0., 1., 1.],"5":[1., 0., 1.],"1":[0., 0., 1.]}

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    y_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        y_labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)

def save_best(state, filename="best_model.pth.tar"):
    print("=> Saving Best Model")
    torch.save(state, filename)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = BubbleDataset(
        image_dir=train_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BubbleDataset(
        image_dir=val_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, loss_fn, writer, device="cuda",epochs = None):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    folder="saved_images/"
    transform = T.ToPILImage()


    with torch.no_grad():
        test_loss = 0
        number_of_images = 0
        ylist = []
        predlist = []
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y = torch.reshape(y,(y.shape[0],y.shape[1]))
            #y = torch.reshape(y,(-1,))
            test_loss += loss_fn(model(x),y)
            preds = torch.round(model(x))
            number_of_images += x.shape[0]
            #!Saving Images
            for i in range(len(x)):
                y_list= y[i].cpu().numpy().tolist()
                pred_list = np.rint(preds[i].cpu().numpy().tolist())
                label_text_y = [k for k, v in label_dict.items() if v == y_list][0]
                label_text_pred = [k for k, v in label_dict.items() if v == pred_list.tolist()][0]
                #! Saving Images where Labels do not match
                if label_text_pred != label_text_y:
                    img = transform(x[i])
                    img = ImageOps.expand(img, border=50, fill=(255,255,255))
                    draw = ImageDraw.Draw(img)
                    text_image = "User defined this frame as: " + str(label_text_y) + ", whereas the model predicted " + str(label_text_pred) + " i.e. " + str(np.round(preds[i].cpu().numpy().tolist(),2))
                    font = ImageFont.truetype("arial.ttf", 15)
                    #draw.text((0,0),text_image,(0,0,0),font=font)
                    margin = 0
                    offset = 0
                    for line in textwrap.wrap(text_image, width=120):
                        draw.text((margin, offset), line, font=font, fill="#000000")
                        offset += font.getsize(line)[1]
                    img.save( f"Mismatches/{label_text_y}_image_{idx}_{i}.png")
                    #! Saving Images where Labels do not match
                img = transform(x[i])
                img = ImageOps.expand(img, border=50, fill=(255,255,255))
                draw = ImageDraw.Draw(img)
                text_image = "User defined this frame as: " + str(label_text_y) + ", whereas the model predicted " + str(label_text_pred) + " i.e. " + str(np.round(preds[i].cpu().numpy().tolist(),2))
                font = ImageFont.truetype("arial.ttf", 15)
                margin = 0
                offset = 0
                for line in textwrap.wrap(text_image, width=120):
                    draw.text((margin, offset), line, font=font, fill="#000000")
                    offset += font.getsize(line)[1]
                img.save( f"{folder}/{label_text_y}_{idx}_{i}.png")
                #!Saving Images
            #!Saving Confusion Matrix
            for i in range(len(y)):
                ylist.append([k for k, v in label_dict.items() if v in y[i:i+1].tolist()][0])
                predlist.append([k for k, v in label_dict.items() if v in preds[i:i+1].tolist()][0])
    array = confusion_matrix(ylist, predlist, labels=['Bubbly', 'Bubbly Slug', 'Slug Annular with some Bubbles', 'Slug', 'Slug Annular','Annular with Bubbles','Annular'],normalize='true')
    df_cm = pd.DataFrame(array, index = ['Bubbly', 'Bubbly Slug', 'Slug Annular with some Bubbles', 'Slug', 'Slug Annular','Annular with Bubbles','Annular'],
    columns = ['Bubbly', 'Bubbly Slug', 'Slug Annular with some Bubbles', 'Slug', 'Slug Annular','Annular with Bubbles','Annular'])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Confusion Matrix (True Labels Against Predictions)')
    sn.heatmap(df_cm, annot=True)
    wrap_labels(ax, 10)
    if epochs is not None:
        fig.savefig(f"{os.getcwd()}/confusion_matrix_epoch/epoch_{epochs}.png")
    #!Saving Confusion Matrix

    check_accuracy.test_loss = test_loss.item()/number_of_images
    print(f"The Average Test Loss is {check_accuracy.test_loss * 100} %")
    model.train()
    return test_loss/len(loader)

def save_predictions_as_imgs(
    epoch,loader, model, folder="saved_images/", device="cuda"
):
    transform = T.ToPILImage()
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = torch.reshape(y,(y.shape[0],y.shape[1]))
        with torch.no_grad():
            preds = model(x)
        #! Save Images
        for i in range(len(x)):
            y_list= y[i].numpy().tolist()
            pred_list = np.rint(preds[i].cpu().numpy().tolist())
            label_text_y = [k for k, v in label_dict.items() if v == y_list][0]
            label_text_pred = [k for k, v in label_dict.items() if v == pred_list.tolist()][0]
            
            if label_text_pred != label_text_y:
                img = transform(x[i])
                img = ImageOps.expand(img, border=50, fill=(255,255,255))
                draw = ImageDraw.Draw(img)
                text_image = "User defined this frame as: " + str(label_text_y) + ", whereas the model predicted " + str(label_text_pred) + " i.e. " + str(np.round(preds[i].cpu().numpy().tolist(),2))
                font = ImageFont.truetype("arial.ttf", 15)
                #draw.text((0,0),text_image,(0,0,0),font=font)
                margin = 0
                offset = 0
                for line in textwrap.wrap(text_image, width=120):
                    draw.text((margin, offset), line, font=font, fill="#000000")
                    offset += font.getsize(line)[1]
                img.save( f"Mismatches/{label_text_y}_image_{idx}_{i}.png")
            img = transform(x[i])
            img = ImageOps.expand(img, border=50, fill=(255,255,255))
            draw = ImageDraw.Draw(img)
            text_image = "User defined this frame as: " + str(label_text_y) + ", whereas the model predicted " + str(label_text_pred) + " i.e. " + str(np.round(preds[i].cpu().numpy().tolist(),2))
            font = ImageFont.truetype("arial.ttf", 15)
            margin = 0
            offset = 0
            for line in textwrap.wrap(text_image, width=120):
                draw.text((margin, offset), line, font=font, fill="#000000")
                offset += font.getsize(line)[1]
            img.save( f"{folder}/{label_text_y}_{idx}_{i}.png")


    model.train()