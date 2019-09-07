// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>             // for cout
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include "../cv-helpers.hpp"    // Helper functions for conversions between RealSense and OpenCV

int main(int argc, char * argv[]) try
{
  using namespace cv;
  using namespace std;
  using namespace rs2;

  const size_t inWidth      = 1280;
  const size_t inHeight     = 720;
  const float WHRatio       = inWidth / (float)inHeight;

  // Start the camera
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
  pipeline pipe;
  auto config = pipe.start(cfg);

  auto profile = config.get_stream(RS2_STREAM_COLOR)
                 .as<video_stream_profile>();

  rs2::align align_to(RS2_STREAM_COLOR);

  Size cropSize;
  if (profile.width() / (float)profile.height() > WHRatio){
    cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                    profile.height());
  }
  else{
    cropSize = Size(profile.width(),
                  static_cast<int>(profile.width() / WHRatio));
  }

  Rect crop(Point((profile.width() - cropSize.width) / 2,
            (profile.height() - cropSize.height) / 2),
            cropSize);

  //const auto window_name = "Display Image";
  //namedWindow(window_name, WINDOW_AUTOSIZE);
  namedWindow("win_cv_src", CV_WINDOW_AUTOSIZE);
  namedWindow("win_cv_bin", CV_WINDOW_AUTOSIZE);
  namedWindow("result", CV_WINDOW_AUTOSIZE);
  // Skips some frames to allow for auto-exposure stabilization
  for (int i = 0; i < 10; i++) pipe.wait_for_frames();

  while (waitKey(1) < 0)
  {

    // Wait for the next set of frames
    auto data = pipe.wait_for_frames();
    // Make sure the frames are spatially aligned
    data = align_to.process(data);

    auto color_frame = data.get_color_frame();
    auto depth_frame = data.get_depth_frame();

    // If we only received new depth frame,
    // but the color did not update, continue
    static int last_frame_number = 0;
    if (color_frame.get_frame_number() == last_frame_number) continue;
    last_frame_number = color_frame.get_frame_number();

    // Convert RealSense frame to OpenCV matrix:
    auto color_mat = frame_to_mat(color_frame);
    auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

    imshow("win_cv_src",color_mat);

    //　2値化
    Mat hsv_mat,dst,show;
    cvtColor(color_mat,hsv_mat,CV_BGR2HSV);
    inRange(hsv_mat, Scalar(0,200,80), Scalar(10,255,255), dst);
    //morphologyEx(dst, dst, MORPH_OPEN,
    //getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1),2);
    morphologyEx(dst, dst, MORPH_CLOSE,
    getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1),8);

    //ラベリング
    Mat img_lab;
    Mat stats, center;
    int nLabels = connectedComponentsWithStats(dst,img_lab,stats,center);//OpenCVラベリング関数
    //面積最大のラベルを抽出
    std::vector<int>area;
    for(int i=1;i<nLabels;i++)area.push_back(stats.at<int>(i,CC_STAT_AREA));

    std::vector<int>::iterator it = std::max_element(area.begin(),area.end());
    size_t index = std::distance(area.begin(),it) + 1;
    compare(img_lab,index,dst,CMP_EQ);//ラベルindexを抜き出し

    //輪郭内の面積計算
    int epsilon = 1;
    int area_output=0,area_i,area_max=0;
    // 輪郭の検出
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    // 2値画像，輪郭（出力），階層構造（出力），輪郭抽出モード，輪郭の近似手法
    findContours(dst, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    Mat contour_max;
    Point2f vertices[4];

    double w,h;
    double xc,yc,zc;

    //焦点距離
    double fx = 1392.68;
    double fy = 1390.59;

    for(int i=0; i<contours.size(); i++){
      Mat contour = Mat(contours[i]);
      std::vector<Point> approx;
      approxPolyDP(contour,approx,epsilon,true);
      area_output = contourArea(approx);+area_output;
      area_i = contourArea(approx);

      if(area_i>area_max){
        area_max=area_i;
        contour_max=contour;
        RotatedRect rRect =  minAreaRect(contour_max);
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
          line(color_mat, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
        w=sqrt(pow(vertices[1].x-vertices[0].x,2.0)+pow(vertices[1].y-vertices[0].y,2.0));
        h=sqrt(pow(vertices[2].x-vertices[1].x,2.0)+pow(vertices[2].y-vertices[1].y,2.0));
      }
      cout<< "エリア数" << contours.size() <<endl;
    }

    if(area_output>1000){
      Point cog(center.at<double>(index,0),center.at<double>(index,1));
      circle(color_mat,cog,5,Scalar(255,0,0),-1);
      // Query the distance from the camera to the object in the center of the image
      float dist_to_center = depth_frame.get_distance(cog.x,cog.y);

      // Print the distance
      xc = (((double)cog.x-inWidth/2)/fx)*dist_to_center;
      yc = (((double)cog.y-inHeight/2)/fy)*dist_to_center;
      zc = dist_to_center;

      if(zc > 0){
        printf("%f %f %f\n",xc,yc,zc);
      }else{
        printf("It is too close!\n");
      }

    }else{
      printf("Red area is small\n");
    }

    imshow("win_cv_bin",dst);
    imshow("result",color_mat);
  }

  return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
