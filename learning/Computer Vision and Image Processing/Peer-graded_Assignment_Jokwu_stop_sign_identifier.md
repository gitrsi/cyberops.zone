> :bulb: Notes on "Peer-graded Assignment: Build a Regression Model in Keras"


# Project Overview

In this course, you learned how to train and use custom classifiers. For the project in this module, you will develop a new custom classifier using one of the classification methods you learnt and then deploy it as a web app using Code Engine. There are various advantages to deploying your classifier. First, you can share your classifier with anyone in the world. They simply need to enter the URL of your model into any web browser. Second, you can showcase your web app in your portfolio and your potential future employers can interact with your project.

# Project Scenario

You have been employed as a Junior Data Scientist by Jokwu, a self-driving car start-up in Capetown, South Africa. Jokwu has created the hardware and parts of the car, and they are beginning to create sensors; the next step is to have a working model that identifies traffic signs. The project and product team have decided to start with stop signs - is it a stop sign or not?

As a first step, you have been given a dataset and tasked with training a model that identifies the stop signs in an image. This will be integrated into a motion detector as a next step.

# Project Tasks

Your job is to load the training images, create features, and train the model. You will then deploy the model to Code Engine so your manager can upload an image of a stop sign and your image classifier will classify the image and tell them to what accuracy it predicts it is correct. You will utilize CV Studio to develop this model and then deploy it as a web app by completing the tasks below.

Once you are done, please take screenshots of your results to upload them for a peer evaluation. This final project will be worth 25% of your final grade.

## Task 1: Gather and Upload Your Data

__[Stop sign images](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/dataset/stop.zip)__

__[not stop sign images](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/dataset/not_stop.zip)__

## Task 2: Train Your Classifier

Create a New training run, enter a Name, choose Jupyter notebook as a “Training tool”, and select Convolutional Neural Networks (CNN) with PyTorch

## Task 3: Deploy Your Model

Deploy "Test-1-click Deploy your Model to Cloud (Code Engine)"

## Task 4: Test Your Classifier

1.  Open the Jupyter notebook from the Train model phase

2.  Add a cell to the end of the model training notebook

    imageNames = ['YongeStreet.jpg']
    for imageName in imageNames:
        image = Image.open(imageName)
        transform = composed = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        x = transform(image)
        z=model(x.unsqueeze_(0))
        _,yhat=torch.max(z.data, 1)
        # print(yhat)
        prediction = "Not Stop"
        if yhat == 1:
            prediction ="Stop"
        imshow_(transform(image),imageName+": Prediction = "+prediction)

3.  Download the test images

__[Test images](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/dataset/test_set_stop_not_stop.zip)__

4.  Upload the images to the Skill Network Labs

5.  Test classifier

6.  Take screenshots of take screenshots of each image and its prediction


## Task 5: Submit Your Assignment and Evaluate Your Peers

Upload the screenshots of your results from the 1st image for a stop sign 

Upload the screenshots of your results from the 2nd image for a traffic stop sign

Upload the screenshots of your results from the 3rd image for a traffic stop sign

Upload the screenshots of your results from the 4th image for a traffic stop sign

Upload the screenshots of your results from the 1st image for a "not -stop sign" 

Upload the screenshots of your results from the 2nd image for a "not -stop sign" 

Upload the screenshots of your results from the 3rd image for a "not -stop sign" 

Upload the screenshots of your results from the final image for a "not -stop sign" 

find an image of a stop sign online and classify it correctly

find an image of a street without a stop sign  online and classify it correctly



Find an image of a street without a stop sign or with a stop sign that will be incorrectly classified by your classifier  
    Your answer needs to be a little bit longer. Write a few sentences to complete your assignment.


Find more data and re-train your classifier to classify the misclassified sample,  show your new classifier works on the miss classified image 
    Your answer needs to be a little bit longer. Write a few sentences to complete your assignment.
























