#include <pangolin/pangolin.h>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <mutex>

template<typename T>
struct Point3d{
    Point3d(T x, T y, T z):
    _x(x), _y(y), _z(z){}
    T _x, _y, _z;
};

class Visualizer{

public:
    Visualizer(size_t H=480, size_t W=640, size_t im_h=480, size_t im_w=640):
    _H(H), _W(W), _im_h(im_h), _im_w(im_w){
    }

    void Run(){

        pangolin::CreateWindowAndBind("Visuazlier",_W,_H);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(_W,_H,420,420,_W/2,_H/2,0.1,1000),
            pangolin::ModelViewLookAt(-0,0.5,-3, 0,0,0, pangolin::AxisY)
        );

        const int UI_WIDTH = 180;

        pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::View& d_image = pangolin::Display("image")
            .SetBounds(0,1/3.0f,2/3.0f,1.0f,640.0/480)
            .SetLock(pangolin::LockRight, pangolin::LockBottom);
            
        pangolin::CreatePanel("ui")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

        pangolin::Var<bool> show_traj("ui.Show_Traj",false,false);
        pangolin::Var<bool> show_img("ui.Show_Img",true,false);
        pangolin::Var<bool> save_frame("ui.Save_Frame",false,false);
        pangolin::Var<bool> record_video("ui.Record",false,false);

        pangolin::GlTexture imageTexture(_im_w, _im_h, GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

        while( !pangolin::ShouldQuit() )
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            if( pangolin::Pushed(save_frame) )
                d_cam.SaveOnRender("frame");
            
            if( pangolin::Pushed(record_video) )
                pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename,flip=true]//screencap.avi");

            d_cam.Activate(s_cam);

            drawCurrentPoints();

            if(show_traj)
                drawTrajectory();
            
            if(show_img){
                d_image.Activate();
                glColor3f(1.0,1.0,1.0);
                imageTexture.Upload(_cur_image.data, GL_BGR, GL_UNSIGNED_BYTE);
                imageTexture.RenderToViewportFlipY();
            }

            pangolin::FinishFrame();
        }
    }

    void drawCurrentPoints(){
        glPointSize(_pointsize);
        glBegin(GL_POINTS);
        double R = 105 / 255.;
        double G = 227 / 255.;
        double B = 251 / 255.;
        glColor4f(R, G, B, 0.7);

        if(_cur_points.empty())
            return;
            
        std::unique_lock<std::mutex> lock(MutexUpdate);
        for(size_t i = 0; i < _cur_points.size(); ++ i){
            auto& p = _cur_points[i];
            auto& c = _cur_colors[i];
            glColor3f(c._x / 255., c._y / 255., c._z / 255.);
            glVertex3f(-p._x, p._y, p._z);
        }

        glEnd();
    }

    void drawTrajectory(){

        glLineWidth(3);
        glColor4f(0.7f,0.0f,0.0f, 0.5f);
        glBegin(GL_LINES);

        if(_total_points.empty())
            return;
            
        std::unique_lock<std::mutex> lock(MutexUpdate);
        size_t N_frame = _total_points.size();
        size_t N_traj = _total_points[0].size();

        for(size_t traj_id = 0; traj_id < N_traj; traj_id += 10){
            for(size_t f_id = 1; f_id < N_frame; ++f_id){
                auto& p0 = _total_points[f_id - 1][traj_id];
                auto& p1 = _total_points[f_id][traj_id];
                glVertex3f(p0._x, p0._y, p0._z);
                glVertex3f(p1._x, p1._y, p1._z);
            }
        }

        glEnd();
    }

    void updateData(const std::vector<Point3d<float>>& points,
                      const std::vector<Point3d<size_t>>& colors,
                      const cv::Mat& image){

        std::unique_lock<std::mutex> lock(MutexUpdate);

        _cur_points.clear();
        _cur_colors.clear();
        for(auto& p: points)
            _cur_points.emplace_back(Point3d<float>(p._x, p._y, p._z));
        for(auto& c: colors)
            _cur_colors.emplace_back(Point3d<size_t>(c._x, c._y, c._z));

        cv::resize(image, _cur_image, cv::Size(_im_w, _im_h));
        _total_points.push_back(_cur_points);
        _total_colors.push_back(_cur_colors);
    }

private:
    std::vector<std::vector<Point3d<float>>> _total_points;
    std::vector<std::vector<Point3d<size_t>>> _total_colors;
    std::vector<Point3d<float>> _cur_points;
    std::vector<Point3d<size_t>> _cur_colors;
    cv::Mat _cur_image;
    float _pointsize = 5;
    std::mutex MutexUpdate;
    double _vis_maxlen = 300.;
    size_t _H, _W, _im_h, _im_w;
};

void loadData(const std::string path,
              std::vector<std::vector<Point3d<float>>>& points, 
              std::vector<std::vector<Point3d<size_t>>>& colors,
              std::vector<std::string>& images_path){
    std::string indexpath = path + "index.txt";
    std::string respath = path + "results/";
    std::string imgpath = path + "images/";
    std::ifstream file(indexpath);
    if(!file){
        std::cerr << "file path: " + indexpath << std::endl;
    }

    std::string idx;
    while(file >> idx){
        std::vector<Point3d<float>> cur_points;
        std::vector<Point3d<size_t>> cur_colors;
        std::string points_file_path = respath + idx + ".txt";
        std::ifstream points_file(points_file_path);
        if(!points_file){
            std::cerr << "file path: " + points_file_path << std::endl;
        }
        float x, y, z;
        int r, g, b;
        while(points_file >> x >> y >> z >> r >> g >> b){
            cur_points.emplace_back(Point3d<float>(x, y, z));
            cur_colors.emplace_back(Point3d<size_t>(r, g, b));
        }
        points.push_back(cur_points);
        colors.push_back(cur_colors);

        std::string image_path = imgpath + idx + ".png";
        images_path.push_back(image_path);
    }

    std::cout << "Total Frames:         " << points.size() << std::endl;
    std::cout << "Total Points NUM:     " << points.size() * points[0].size() << std::endl;
}

int main( int argc, char** argv )
{

    std::vector<std::vector<Point3d<float>>> vPoints;
    std::vector<std::vector<Point3d<size_t>>> vColors;
    std::vector<std::string> vImages_path;

    const std::string root = argv[1];
    loadData(root, vPoints, vColors, vImages_path);

    Visualizer* visualier = new Visualizer();
    std::thread vis_thread(&Visualizer::Run, visualier);

    int N = vPoints.size();
    for(int i = 0; i < vPoints.size(); ++i){
        usleep(3.33e+4);
        std::cout << i << "/" << vPoints.size() << std::endl;

        cv::Mat image = cv::imread(vImages_path[i], cv::IMREAD_UNCHANGED);
        if(image.channels() == 0){
            std::cerr << "can not open the image: " + vImages_path[i] << std::endl;
        }

        visualier->updateData(vPoints[i], vColors[i], image);
    }   

    vis_thread.join();

    return 0;
}
