#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>
#include <Eigen/SVD>  
#include <Eigen/Dense>   
#include <chrono> 
#include <vector>
using namespace std;
using namespace cv;
using namespace Eigen;    
using namespace Eigen::internal;    
using namespace Eigen::Architecture;  
using namespace chrono; 

struct Normal
{
    Normal(int a, int b, Vector3f v)
    {
        this->row = a, this->col = b, this->normal = v;
    }
    int row;
    int col;
    Vector3f normal;
} ;

int main(void){
    auto start1 = system_clock::now();
    char buff1[100];
    double cx = 330.1758117675781;
    double cy = 236.1585693359375;
    double fx = 615.7573852539062;
    double fy = 615.7640380859375;
    float data[7] ={0,0,0,0,0,0,1};
    Eigen::Quaterniond q( data[6], data[3], data[4], data[5] );
    Eigen::Isometry3d T(q);
    T.pretranslate( Eigen::Vector3d( data[0], data[1], data[2] ));
    vector<Normal> m_normal;
    sprintf(buff1,"/media/tx2/Drive/wuzida/ORB_SLAM2/depth_try/depth.png");
    Mat depth=imread(buff1,CV_LOAD_IMAGE_UNCHANGED);
    depth.convertTo(depth,CV_32F,0.001);

    MatrixXf A(9,3); 
    for(size_t y=0;y<depth.rows-5;y=y+3){
        const float* d_row_ptr = depth.ptr<float>(y);
        for(size_t x=0;x<depth.cols-5;x = x+3){    
            if ( *d_row_ptr++!=0 ){ 
                float c_depth = depth.ptr<float>( y )[ x];
                Eigen::Vector3d point; 
                point[2] = depth.ptr<float>( y )[x] ; // z in wordl
                point[0] = (x-cx)*point[2]/fx;   // x in world
                point[1] = (y-cy)*point[2]/fy;   //y in world
                Eigen::Vector3d o_pointWorld =T*point;  // origin point location

                int cout = 0;
                for(size_t i=1;i<4;i++){
                    for(size_t j=1;j<4;j++){
                        point[2] = depth.ptr<float>( y+i )[x+j]; // z in wordl
                        point[0] = (x+j-cx)*point[2]/fx;   // x in world
                        point[1] = (y+i-cy)*point[2]/fy;   //y in world
                        Eigen::Vector3d pointWorld =T*point;
                        A(cout,0) = pointWorld[0] - o_pointWorld[0];                          // use Cartesian coordinate
                        A(cout,1) = pointWorld[1] - o_pointWorld[1];                         
                        A(cout++,2) = pointWorld[2] - o_pointWorld[2];                         
                  }    
                }
                JacobiSVD<Eigen::MatrixXf> svd(A, ComputeThinU | ComputeThinV );  
                MatrixXf V = svd.matrixV();
                Eigen::Vector3f c_normal ={V(0,2),V(1,2),V(2,2)};
                Normal temp(y,x, c_normal);
                m_normal.push_back(temp);
   // std::cout<<"V :\n"<<c_normal<<std::endl; 
             } 
        }
    }

    auto end1 = system_clock::now();
    auto duration = duration_cast<microseconds>(end1-start1);
    cout << "time consumption " << double(duration.count())*microseconds::period::num /microseconds::period::den << " s" <<endl;

    return 0;  
//compile conmmand: g++ depth_try.cpp -o test `pkg-config --cflags --libs eigen3 opencv`    
}
