# Sportsperson-Image-Classification
![classification_image](https://user-images.githubusercontent.com/83235872/154677412-4d3fe23c-6391-474d-9cd6-a9979d58396a.jpg)
This is a classification project in machine learning. We classify sportsperson from different sports. The 5 sporstperson which are used here are-

1.Maria Sharapova

2.Serena Williams

3.Virat Kohli

4.Roger Federer

5.Lionel Messi

I have used dataset of around 70 images for each of the sportsperson which is provided in the dataset folder. Using opencv haarcascades feature, I have tried to detect 2 eyes and a face for each of the image. For ease, I created a function which goes through all images and if the image detects two eyes i.e. if the face is clearly visible, it will select that image and will crop the image which is then saved to the local folder. After this, I have manually examined the folder for any of the unwanted images.
After the data cleaning process is done, I then trained the model using SVM where I achieved 0.77 precision score. I also used gridsearchCV to find the best model among svm, random forest and logistic regression, out of which SVM gives the best score. Using some images from the dataset as test images, I achieved around 0.75 precision score.

Technologies used in this project,

1.Python

2.Numpy and OpenCV for data cleaning

3.Matplotlib & Seaborn for data visualization

4.Sklearn for model building

5.Jupyter notebook, visual studio code and pycharm as IDE

6.Python flask for http server

7.HTML/CSS/Javascript for UI
