**Steps to run:
**
1. clone project into local and run: pip install torch bing_image_downloader
2. fill in your own parameters and run the following in the project folder using bash: python image_classification_model_builder.py --dir 'your_directory_here' --model 'model_name_here' --labels 'label1,label2,label3'
3. Now a folder structure containing a training folder and a training script has been generated, run that training script to create a model checkpoint file
4. Run inference on the model checkpoint you have generated with the example code in inference_example.py
