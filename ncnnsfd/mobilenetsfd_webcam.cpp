
// Copyright (C) 2018 nviso


#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "net.h"

struct Object{
    cv::Rect rec;
    float prob;
};

static int detect_webcam(int cam_id, ncnn::Net& nvisosfd, float show_threshold)
{
    cv::VideoCapture cap(cam_id);
    cv::Mat raw_img;

    cap >> raw_img;
    if(raw_img.empty()) return 0;

    int img_h = raw_img.size().height;
    int img_w = raw_img.size().width;

    int input_size = 320;

    while(true) 
    {
        cap >> raw_img;
        if(raw_img.empty()) break;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(raw_img.data, ncnn::Mat::PIXEL_BGR, raw_img.cols, raw_img.rows, input_size, input_size);

        ncnn::Mat out;
    
        ncnn::Extractor ex = nvisosfd.create_extractor();
        ex.set_light_mode(true);

        //ex.set_num_threads(1);
        ex.input("data", in);
        ex.extract("detection_out",out);

        std::vector<Object> objects;
        for (int iw=0;iw<out.h;iw++)
        {
            Object object;
            const float *values = out.row(iw);
            //object.class_id = values[0];
            object.prob = values[1];
            object.rec.x = values[2] * img_w;
            object.rec.y = values[3] * img_h;
            object.rec.width = values[4] * img_w - object.rec.x;
            object.rec.height = values[5] * img_h - object.rec.y;
            objects.push_back(object);
        }

        for(int i = 0;i<objects.size();++i)
        {
            Object object = objects.at(i);
            if(object.prob > show_threshold)
            {
                cv::rectangle(raw_img, object.rec, cv::Scalar(0, 255, 0), 2);
                std::ostringstream pro_str;
                pro_str<<object.prob;
                std::string label = pro_str.str();
                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
                cv::rectangle(raw_img, cv::Rect(cv::Point(object.rec.x, object.rec.y- label_size.height-3),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(raw_img, label, cv::Point(object.rec.x, object.rec.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
        cv::imshow("result",raw_img);
        cv::waitKey(30);
    }
    return 0;
}

int main(int argc, char** argv)
{
    char *p;

    const int cam_id = strtol(argv[1], &p, 10);

    ncnn::Net nvisosfd;
    
    const char* parampath = argv[2];
    nvisosfd.load_param(parampath);

    const char* binpath = argv[3];
    nvisosfd.load_model(binpath);

    detect_webcam(cam_id, nvisosfd, 0.7);

    return 0;
}
