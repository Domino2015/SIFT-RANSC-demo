#include <iostream>
#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/xfeatures2d.hpp>

using namespace cv;  //包含cv命名空间
using namespace std;

vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint);

int main() {
    //Create SIFT class pointer
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //读入图片
    Mat img_1 = imread("../3.png");
    Mat img_2 = imread("../4.png");
    imshow("img_1", img_1);
    imshow("img_2", img_2);

    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
    f2d->compute(img_2, keypoints_2, descriptors_2);

    //Matching descriptor vector using BFMatcher
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    cout << "SIFT后一共：" << matches.size() << " 对匹配：" << endl;

    //绘制匹配出的关键点
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    imshow("RANSC_before", img_matches);

    matches = ransac(matches, keypoints_1, keypoints_2);

    Mat img_matches1;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches1);
    imshow("RANSC_after", img_matches1);

    //等待任意按键按下
    waitKey(0);
}

//RANSAC算法实现
vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint) {
    //定义保存匹配点对坐标
    vector<Point2f> srcPoints(matches.size()), dstPoints(matches.size());
    //保存从关键点中提取到的匹配点对的坐标
    for (int i = 0; i < matches.size(); i++) {
        srcPoints[i] = queryKeyPoint[matches[i].queryIdx].pt;
        dstPoints[i] = trainKeyPoint[matches[i].trainIdx].pt;
    }
    //保存计算的单应性矩阵
    Mat homography;
    //保存点对是否保留的标志
    vector<unsigned char> inliersMask(srcPoints.size());
    //匹配点对进行RANSAC过滤
    homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 5, inliersMask);
    //RANSAC过滤后的点对匹配信息
    vector<DMatch> matches_ransac;
    //手动的保留RANSAC过滤后的匹配点对
    for (int i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            matches_ransac.push_back(matches[i]);
            //cout<<"第"<<i<<"对匹配："<<endl;
            //cout<<"queryIdx:"<<matches[i].queryIdx<<"\ttrainIdx:"<<matches[i].trainIdx<<endl;
            //cout<<"imgIdx:"<<matches[i].imgIdx<<"\tdistance:"<<matches[i].distance<<endl;
        }
    }
    cout << "经RANSAC消除误匹配后一共：" << matches_ransac.size() << " 对匹配：" << endl;
    //返回RANSAC过滤后的点对匹配信息
    return matches_ransac;
}