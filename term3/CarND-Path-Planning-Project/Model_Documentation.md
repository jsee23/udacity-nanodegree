# Model Documentation

# Overview
I didn't use a State Machine like proposed in the class videos. Instead, it's based on the skeleton code presented by Aaron in the walkthrough video.

We use a small spline library to generate way points.

## Planning
The implementation is quite simple: I always try to reach the maximal allowed speed. If the distance to the car in front (and in the same lane) is lower than a minimum value, the car speed will be decreased to the speed of that car. In this situation, we check if we have enough space to switch the lane and if the cars on the other lanes driving faster.

If yes, we will switch the lanes. The important part was that I had to flatten/strech the spline depending on the speed. Additionally, we directly switch to the speed which is possible on the new lane.

If we can switch the lane safely is defined by two fixed values: there should be no car 30 meters ahead and 8 meters behind us.

## Possible improvements
The observation of the 'safe space' on the other lanes, could be improved by caluculating those values speed dependent. This will improve the lane changing in case of low speed and improve the safety in case of higher speed.