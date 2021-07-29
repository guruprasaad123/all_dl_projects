# Task

You‌ ‌have‌ ‌to‌ ‌submit‌ ‌a‌ ‌write‌ ‌up‌ ‌(with‌ ‌proper‌ ‌images)‌ ‌about‌ ‌how‌ ‌recurrent‌ ‌neural‌ ‌networks‌ ‌such‌ ‌as‌ ‌LSTM ,‌ ‌GRU‌ ‌based‌ ‌models‌ ‌can‌ ‌be‌ ‌utilized‌ ‌in‌ ‌Autonomous‌ ‌Vehicles.‌ ‌ ‌

### Solution

- we can train a object detection model on traffic signs datasets like GTSRB datasets or any other datasets on a server/gpu
  - to detect and recognize traffic signs 
  - to detect and recognize the road lines
  - to detect and recognize any other car/bike/vehicles

- after the training the model with reasonable accuracy then we can have that trained object detection model which can detect the road line , crossing lines , traffic signals like stop signals , traffic lights ( red , light , green )
- then that image as input would be passed as input for LSTM model which does the further processing
- we can train the LSTM model on the server/gpu
  - to draw the lines between the while crossed lines
  - incase of red signal detected that means to stop
  - incase of green signal detected that means to start
  - incase of car/bike/any vehicle detected that means to slow down a bit 
  - and based on various signal how should the model output
- later that trained model LSTM model can be used to draw line on the road between the white crossed line determining the clear path and also output the command like slow , stop , steer straight , steer right , steer left , make u turn relating the action that autonomous vehicle should do based on the situation/input making it dynamic
- so when we input a video file , then that video file can be separated into a bunch of images which can be feed into the trained object detection model + LSTM Model which would output the lines and the command related to what vehicle should do
- And we can use rasp bi as a medium to run our experiments

