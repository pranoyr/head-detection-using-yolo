# Head Detection Using YOLO Algorithm
The objective is to train a YOLO algorithm to detect multiple heads from a frame.

## Getting Started
### Prerequisites
1. TensorFlow
2. Keras

### Download the pre-trained weights for the backend
Download full_yolo_backend.h5 from https://drive.google.com/file/d/1Q9WhhRlqQbA4jgBkCDrynvgquRXZA_f8/view?usp=sharing
Put it in the root directory.
### Dataset
 Download the dataset and put it in the root directory.
 
 Images      - https://drive.google.com/open?id=1zn-AGmsBqVheFPnDTXWBpeo3XRH1Ho15
 
 Annotations - https://drive.google.com/open?id=1LiTDMWk0KglGueJCaxgneEA_ltvEbUDV
 
### Training
Run train.py

The weights for the front-end will be saved in the file name "model.h5".

I have uploaded the front-end weights.
https://drive.google.com/file/d/1wg4q9cc6q04oRr_Xaf9GVhhbTSH4-ena/view?usp=sharing


### Testing
Give the path to your image in predict.py

Run predict.py

## License
This project is licensed under the MIT License 

