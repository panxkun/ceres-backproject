#include <iostream>
#include <ceres/ceres.h>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <fstream>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

constexpr int CNT_X = 32;
constexpr int CNT_Y = 24;
constexpr int reso_w = 640;
constexpr int reso_h = 480;
constexpr double focal_x = 400;
constexpr double focal_y = 530;

constexpr int CNT = CNT_X * CNT_Y;
constexpr int CNT_ADJACENT_CONSTRAINT = 4 * CNT_X * CNT_Y - 3 * (CNT_X + CNT_Y) + 2;
constexpr int CNT_DIFFPLANE_CONSTRAINT = CNT_X * CNT_Y -  (CNT_X + CNT_Y) + 1;
constexpr double dx = 0.297 / (CNT_X - 1);
constexpr double dy = 0.210 / (CNT_Y - 1);
constexpr double dxy2 = (dx * dx + dy * dy);

struct Camera{
    Camera(double fx, double fy, double cx, double cy):
    _fx(fx), _fy(fy), _cx(cx), _cy(cy){
    }
    double _fx, _fy, _cx, _cy;
};

template<typename T>
struct Point3d{
    Point3d(T u, T v)
    :_u(u), _v(v){
    }

    Point3d(T x, T y, T z)
    :_x(x), _y(y), _z(z){
    }

    void setDepth(const T& z,  const Camera& cam){
        _z = z;
        _x = z * (_u - cam._cx) / cam._fx;
        _y = z * (_v - cam._cy) / cam._fy;
    }
    T _u, _v;
    T _x, _y, _z;
};

template<typename T>
struct Plane{
    Plane(const Point3d<T>& p1, const Point3d<T>& p2, const Point3d<T>& p3){
        _A = (p2._y - p1._y) * (p3._z - p1._z) - (p2._z - p1._z) * (p3._y - p1._y);
        _B = (p2._z - p1._z) * (p3._x - p1._x) - (p2._x - p1._x) * (p3._z - p1._z);
        _C = (p2._x - p1._x) * (p3._y - p1._y) - (p2._y - p1._y) * (p3._x - p1._x);
        _D = - (_A * p1._x + _B * p1._y + _C * p1._z);

    }

    T _A, _B, _C, _D;
};

struct AdjacentDistanceError{
    AdjacentDistanceError(Point3d<double> points1, Point3d<double> points2, Camera cam):
    _points1(points1), _points2(points2), _cam(cam){
    }

    template<typename T>
    bool operator() (const T* const depth1, const T* const depth2, T* residual) const  
    {
        // TODO: inverse depth
        residual[0] = T((ceres::sqrt( 
                ceres::pow((depth1[0] * (_points1._u - _cam._cx) / _cam._fx - depth2[0] * (_points2._u - _cam._cx) / _cam._fx), 2) +
                ceres::pow((depth1[0] * (_points1._v - _cam._cy) / _cam._fy - depth2[0] * (_points2._v - _cam._cy) / _cam._fy), 2) +
                ceres::pow((depth1[0] - depth2[0]), 2)) - dx));
        return true;
    }

    static ceres::CostFunction* Create(Point3d<double> points1, Point3d<double> points2, Camera cam){
        return (new ceres::AutoDiffCostFunction<AdjacentDistanceError, 1, 1, 1>(
                new AdjacentDistanceError(points1, points2, cam)));
   }

    Point3d<double> _points1;
    Point3d<double> _points2;
    Camera _cam;
};

void Warp3D(const std::vector<Point3d<double>>& points2d, const Camera& cam, double* depth, bool verbose=true){
    
    ceres::Problem problem;
    for(size_t i = 0; i < CNT; ++i){
        problem.AddParameterBlock(depth + i, 1);
    }

    for(size_t r = 0; r < CNT_Y; ++r){
        for(size_t c = 0; c < CNT_X; ++c){
            const size_t idx1 = r * CNT_X + c;
            if(r < CNT_Y - 1){
                const size_t idx2 = idx1 + CNT_X;
                problem.AddResidualBlock(AdjacentDistanceError::Create(points2d[idx1], points2d[idx2], cam), nullptr, depth + idx1, depth + idx2);
            }
            if(c < CNT_X - 1){
                const size_t idx2 = idx1 + 1;
                problem.AddResidualBlock(AdjacentDistanceError::Create(points2d[idx1], points2d[idx2], cam), nullptr, depth + idx1, depth + idx2);
            }
            if(r < CNT_Y - 1 && c < CNT_X - 1){
                const size_t idx2 = idx1 + CNT_X + 1;
                problem.AddResidualBlock(AdjacentDistanceError::Create(points2d[idx1], points2d[idx2], cam), nullptr, depth + idx1, depth + idx2);
            }
            if(c > 0 && r < CNT_Y - 1){
                const size_t idx2 = idx1 + CNT_X - 1;
                problem.AddResidualBlock(AdjacentDistanceError::Create(points2d[idx1], points2d[idx2], cam), nullptr, depth + idx1, depth + idx2);
            }
        }
    }

    ceres::Solver::Options options;    
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  
    options.minimizer_progress_to_stdout = verbose;  

    ceres::Solver::Summary summary;                
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
}

void importData(const string filename, std::vector<Point3d<double>>& points, std::vector<Point3d<int>>& coords){
    std::ifstream points_file(filename);
    if (!points_file) {
        std::cerr << "cannot to open points file: " + string(filename) << std::endl;
        exit(1);   // call system to stop
    }
    double u, v, x, y;
    while(points_file >> u >> v>> x >> y){
        coords.emplace_back(Point3d<int>(u, v));
        points.emplace_back(Point3d<double>(x, y));
	}
	points_file.close();
}

void exportData(std::vector<Point3d<double>>& points3d, double* depth,
                const std::vector<Point3d<int>>& coords, 
                const cv::Mat& image, 
                const Camera& cam,
                const string save_dir){
    
    std::ofstream result(save_dir);
    for(int i = 0; i < points3d.size(); ++i){
        Point3d<double>& p = points3d[i];
        p.setDepth(depth[i], cam);
        cv::Vec3b bgr=image.at<cv::Vec3b>(coords[i]._v, coords[i]._u);
        result  << p._x << " " << p._y << " " << p._z << " "
                << size_t(bgr[2]) << " " << size_t(bgr[1]) << " " << size_t(bgr[0]) << std::endl;
    }
    result.close();
}

std::vector<string> loadDataList(const string filename){
    std::ifstream file(filename + "index.txt");
    if (!file) {
        std::cerr << "cannot to open DataList: " + string(filename) << std::endl;
        exit(1);   // call system to stop
    }
    std::vector<string> datalist;
    string x;
    while(file >> x){
        datalist.emplace_back(x);
	}
	file.close();
    return datalist;
}

void run_on_sequence(const string& path, const std::vector<string>& filesname, const Camera& cam, const cv::Mat& image){

    double* depth = new double[CNT];
    std::fill(depth, depth + CNT, 1.0);

    for(int i = 0; i < filesname.size(); ++i){
        string filename = path + "points/" + filesname[i] + ".txt";
        printf("%d /%lu\t %s\n", i, filesname.size(), filename.c_str());

        std::vector<Point3d<double>> points;
        std::vector<Point3d<int>> coords;
        

        importData(filename.c_str(), points, coords);
        Warp3D(points, cam, depth, false);

        string save_dir = path + "results/" + filesname[i] + ".txt";
        exportData(points, depth, coords, image, cam, save_dir.c_str());

        
    }
    delete[] depth;
    std::cout << "Sequence finished!" << std::endl;
}

void run_on_frame(const string& path, const string filename, const Camera& cam, const cv::Mat& image){

    double* depth = new double[CNT];
    std::fill(depth, depth + CNT, 1.0);

    std::vector<Point3d<double>> points;
    std::vector<Point3d<int>> coords;
    
    importData(filename, points, coords);
    Warp3D(points, cam, depth);

    string save_dir = path + "result.txt";
    exportData(points, depth, coords, image, cam, save_dir.c_str());

    delete[] depth;
}


int main ( int argc, char** argv )
{
    google::InitGoogleLogging(argv[0]);
    Camera cam(focal_x, focal_y, reso_w/2.0, reso_h/2.0);

    const string datapath = string(argv[1]);

    string imagename = datapath + "starry_night_640_480.png";
    cv::Mat image = cv::imread(imagename, cv::IMREAD_UNCHANGED);

    

    // ========================================
    std::cout << std::endl;
    std::cout << "INFO: " << std::endl;
    std::cout << "--CNT_X:                      " << CNT_X << std::endl; 
    std::cout << "--CNT_Y:                      " << CNT_Y << std::endl; 
    std::cout << "--CNT:                        " << CNT << std::endl; 
    std::cout << "--CNT_ADJACENT_CONSTRAINT:    " << CNT_ADJACENT_CONSTRAINT << std::endl; 
    std::cout << "--CNT_DIFFPLANE_CONSTRAINT:   " << CNT_DIFFPLANE_CONSTRAINT << std::endl << std::endl; 
    // ========================================
    

    if(1){
        std::vector<string> filesname = loadDataList(datapath);
        run_on_sequence(datapath, filesname, cam, image);
    }else{
        run_on_frame(datapath, datapath, cam, image);
    }

    return 0;
}

