# AI-vs-Real-art-Differentiation

This is a project that our group designed to enhance the classification model's performance on differentiating AI generated images and real images. 

## Environments

```
pip install timm
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

For the Segment Anything Model's checkpoints, please go to [SAM's official repository](https://github.com/facebookresearch/segment-anything)

## Dataset We Used and Generated

```
1. train-dataset --- This is the dataset used for training, including images that are labeled AI and Real-art
2. test-dataset --- This is the dataset used for testing
3. cropped_test_dataset --- This dataset composed of images after segmentation from test-dataset, each image will have their single folder. In the folder, it includes all the segmented pieces and orginal image
```

## Codes That are Used for Differentiation

```
train.py	--- This is the train code used for training ResNet model using the train-dataset
train_rexnet.py	--- This is the train code used for training ReXNet model using the train-dataset
predict_2_labels.py --- Code used for testing models' performance on the test-dataset
predict_sam_2_labels.py --- Code used for testing model's performance on the cropped_test_dataset
```

## Other Codes

```
sam.py	--- Code used for segmenting images, it will output .png file for the images
convert_png_jpg.py	--- Convert .png files to .jpg files if needed
predict.py	--- If your model is trained with multiple labels more labels, like 5 we used, please run this code
predict_sam.py	--- Code for testing performance on cropped images for the model with 5 labels
```

## Checkpoints

```
Under chkpt folder, you can find following checkpoints
new_resnet_152_trained_state_dict.pth	--- ResNet trained with 2 labels, AI and Real-art
new_rexnet_150_trained_state_dict.pth	--- ReXNet trained with 2 labels, AI and Real-art
resnet_152_trained_state_dict.pth	--- ResNet trained with 5 labels, Stable Diffusion, Latent Diffusion, Midjourney, DALLE, Real-arts
rexnet_150_trained_state_dict.pth	--- ReXNet trained with 5 labels, Stable Diffusion, Latent Diffusion, Midjourney, DALLE, Real-arts
```

