# Faster_RCNN_Tensorflow
This repo is based on the official tensorflow model at https://github.com/tensorflow/models/tree/master/object_detection.

## Getting Started

### Preparing Inputs
The model uses data in the tfrecord format. So first we need to convert the training and validation images along with their bounding box info into tfrecords.

The dataset directory must contain both the images as well as the annotations in the Pascal VOC format. 

The Directory structure should be the following:
  
    Data_Dir
      Annotations
      data
        JPEGImages
      train.txt
      val.txt
    
* " train.txt " and " val.txt " contain the filenames(eg. Img001, without filepath or extension) of images that compose the training and validation sets. The filenames are on new lines and care should be taken that the last line of the file is not a new line.

* The " data " folder contains all the images(both training and validation)

* The " Annotations " folder contains all the annotation files which have been generated in Pascal VOC format. The files are of the form $Img_name.xml. Where $Img_name is the name of the image that the annotation details corresspond to.

Run the following commands in the models folder to convert the dataset into training and validation tfrecords

```
python object_detection/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=$ABSOLUTE_PATH_TO_DATA_DIR --set=train \
    --output_path=custom_train.record
python object_detection/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=$ABSOLUTE_PATH_TO_DATA_DIR --set=val \
    --output_path=custom_val.record
```

### Retraining

* Download a suitable exported model for retraining extract this to your desired path. For this repo, I have used COCO-pretrained Faster R-CNN with Resnet-101 model(http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)
* In the object_detection/faster_rcnn_resnet101_CUSTOM.config file enter config details for the following

   fine_tune_checkpoint - $ABSOLUTE_PATH_TO_EXTRACTED_MODEL_FOR_RETRAINING
   
   input_path - $ABSOLUTE_PATH_TO_TRAINING_OR_VALIDATION_TFRECORDS
  
   label_map_path - $ABSOLUTE_PATH_TO_TRAINING_OR_VALIDATION_LABEL_MAP
   
 * Run the training job - 
 ```
 # From the tensorflow/models/ directory
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
 ```
${PATH_TO_YOUR_PIPELINE_CONFIG} - points to the pipeline config (In this case object_detection/faster_rcnn_resnet101_CUSTOM.config)
${PATH_TO_TRAIN_DIR} - points to the directory in which training checkpoints and events will be written to

### Evaluation
```
# From the tensorflow/models/ directory
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
```
${PATH_TO_TRAIN_DIR}  - Points to the directory in which training checkpoints were saved (same as the training job)

${PATH_TO_YOUR_PIPELINE_CONFIG} - Points to the pipeline config

${PATH_TO_EVAL_DIR}  - Points to the directory in which evaluation events will be saved

### Running Tensorboard 
```
tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}
```
${PATH_TO_MODEL_DIRECTORY} - Path to the directory that contains the train and eval directories

### Exporting the graph

Once evaluation phase yields sufficient accuracy, we can export the model to a Tensorflow graph proto to be used for inference(prediction)

```
# From tensorflow/models
python object_detection/export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH}|model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory faster_rcnn_inference_graph.pb
```

