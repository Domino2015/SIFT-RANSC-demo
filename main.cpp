#include <iostream>
#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/xfeatures2d.hpp>

using namespace cv;  //包含cv命名空间
using namespace std;

vector<DMatch> ransac(vector<DMatch> matches, vector<KeyPoint> queryKeyPoint, vector<KeyPoint> trainKeyPoint);
Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri);
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
void transform(Mat &SrcImage,Mat &dst);

int main() {
    //Create SIFT class pointer
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //读入图片
    Mat input_1 = imread("../11.jpg");
    Mat input_2 = imread("../12.jpg");
    Mat img_1,img_2;
    transform(input_1,img_1);
    transform(input_2,img_2);

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

    //获得匹配特征点，并提取最优配对
    sort(matches.begin(),matches.end()); //特征点排序
    //获取排在前N个的最优匹配特征点
    vector<Point2f> imagePoints1,imagePoints2;
    for(int i=0;i<10;i++)
    {
        imagePoints1.push_back(keypoints_1[matches[i].queryIdx].pt);
        imagePoints2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //获取图像1到图像2的投影映射矩阵，尺寸为3*3
    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);
    Mat adjustMat=(Mat_<double>(3,3)<<1.0,0,img_1.cols,0,1.0,0,0,0,1.0);
    Mat adjustHomo=adjustMat*homo;

    //获取最强配对点在原始图像和矩阵变换后图像上的对应位置，用于图像拼接点的定位
    Point2f originalLinkPoint,targetLinkPoint,basedImagePoint;
    originalLinkPoint=keypoints_1[matches[0].queryIdx].pt;
    targetLinkPoint=getTransformPoint(originalLinkPoint,adjustHomo);
    basedImagePoint=keypoints_2[matches[0].trainIdx].pt;

    //图像配准
    Mat imageTransform1;
    warpPerspective(img_1,imageTransform1,adjustMat*homo,Size(img_2.cols+img_1.cols+10,img_2.rows));

    //在最强匹配点的位置处衔接，最强匹配点左侧是图1，右侧是图2，这样直接替换图像衔接不好，光线有突变
    Mat ROIMat=img_2(Rect(Point(basedImagePoint.x,0),Point(img_2.cols,img_2.rows)));
    ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,0,img_2.cols-basedImagePoint.x+1,img_2.rows)));

    imshow("Complete!",imageTransform1);
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

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri)
{
    Mat originelP,targetP;
    originelP=(Mat_<double>(3,1)<<originalPoint.x,originalPoint.y,1.0);
    targetP=transformMaxtri*originelP;
    float x=targetP.at<double>(0,0)/targetP.at<double>(2,0);
    float y=targetP.at<double>(1,0)/targetP.at<double>(2,0);
    return Point2f(x,y);
}

void transform(Mat &SrcImage,Mat &dst)
{
    float F = 512.89;
    float r = 512.89;
    int row = SrcImage.rows;
    int col = SrcImage.cols;

    cv::Mat X, Y;
    meshgrid(cv::Range(1, col), cv::Range(1, row), X, Y);

    cv::Mat mapx1(row, col, CV_32FC1, cv::Scalar(0));
    cv::Mat mapy1(row, col, CV_32FC1, cv::Scalar(0));

    for( int i =  0; i < SrcImage.rows; i++)
    {
        for( int j = 0; j < SrcImage.cols; j++)
        {
            mapx1.at<float>(i, j) = F*tan(((float) j - col/2)/r);
            mapy1.at<float>(i, j) = (((float) i - row/2)/r)*hypot(((float) j - col/2),F);  //垂直

        }
    }
    //remap and then save the image
    mapx1 = mapx1 + col/2;
    mapy1 = mapy1 + row/2;

    remap(SrcImage, dst, mapx1, mapy1, CV_INTER_LINEAR);

}
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
    std::vector<float> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(float(i));
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(float(j));
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}