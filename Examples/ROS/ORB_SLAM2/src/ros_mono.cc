/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include"../../../include/System.h"

using namespace std;

class ImageGrabber
{
private: 
    ros::NodeHandle nh2;

    ros::Publisher slamTf; 

public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){

        slamTf = nh2.advertise<geometry_msgs::TransformStamped>("/slam/tf",10);
    }

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM2::System* mpSLAM;
    cv::Mat M1l,M2l,M1r,M2r,Twc;
    float q[4];
    geometry_msgs::PoseStamped msg;
    geometry_msgs::TransformStamped tf1;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 Mono path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe("/camera/rgb/image_raw", 1, &ImageGrabber::GrabImage,&igb);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());

    if(mpSLAM->GetFramePose(Twc, q)){

        tf1.header.stamp = msg->header.stamp;
        tf1.header.frame_id = "world" ;
        tf1.child_frame_id = "slam" ;
        tf1.transform.translation.x = Twc.at<float>(2);//Twc.at<float>(0);
        tf1.transform.translation.y = -Twc.at<float>(0);//Twc.at<float>(1);
        tf1.transform.translation.z = -Twc.at<float>(1);//Twc.at<float>(2);
        tf1.transform.rotation.x = q[2];//q[0];
        tf1.transform.rotation.y = -q[0];//q[1];
        tf1.transform.rotation.z = -q[1];//q[2];
        tf1.transform.rotation.w = q[3];
        slamTf.publish(tf1);
    }
}


