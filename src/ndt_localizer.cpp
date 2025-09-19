#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

class NDTLocalizer : public rclcpp::Node {
public:
    NDTLocalizer() : Node("ndt_localizer") {
        // parameters
        declare_parameter<std::string>("map_path", "/home/ubuntu/ros2_ws/src/FAST_LIO/PCD/test.pcd");
        declare_parameter<double>("initial_x", 0.0);
        declare_parameter<double>("initial_y", 0.0);
        declare_parameter<double>("initial_z", 0.0);
        declare_parameter<double>("initial_yaw", 0.0);

        std::string map_path;
        get_parameter("map_path", map_path);

        double init_x, init_y, init_z, init_yaw;
        get_parameter("initial_x", init_x);
        get_parameter("initial_y", init_y);
        get_parameter("initial_z", init_z);
        get_parameter("initial_yaw", init_yaw);

        // Map loading
        pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile(map_path, *map_cloud) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load map: %s", map_path.c_str());
            return;
        }

        // Downsample the map
        pcl::VoxelGrid<pcl::PointXYZI> voxel_map;
        voxel_map.setLeafSize(0.5f, 0.5f, 0.5f);
        voxel_map.setInputCloud(map_cloud);
        voxel_map.filter(*target_map_);

        if (target_map_->empty()) {
            RCLCPP_ERROR(this->get_logger(), "Target map is empty after voxel filtering!");
            return;
        }

        // NDT setup
        ndt_.setTransformationEpsilon(0.01);
        ndt_.setStepSize(0.05);
        ndt_.setResolution(0.5);
        ndt_.setMaximumIterations(30);
        ndt_.setInputTarget(target_map_);

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/pcd_segment_obs", 10,
            std::bind(&NDTLocalizer::pointCloudCallback, this, std::placeholders::_1));
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/livox/imu", 10,
            std::bind(&NDTLocalizer::imuCallback, this, std::placeholders::_1));

        pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom/match", 10);

        // initialize pose_ from parameters (useful if map origin != (0,0,0))
        pose_ = Eigen::Matrix4f::Identity();
        pose_(0,3) = static_cast<float>(init_x);
        pose_(1,3) = static_cast<float>(init_y);
        pose_(2,3) = static_cast<float>(init_z);
        {
            // set initial yaw
            float cy = std::cos(static_cast<float>(init_yaw));
            float sy = std::sin(static_cast<float>(init_yaw));
            pose_.block<3,3>(0,0) = Eigen::Matrix3f::Identity();
            pose_(0,0) = cy; pose_(0,1) = -sy;
            pose_(1,0) = sy; pose_(1,1) = cy;
        }

        // initialize IMU orientation to identity to avoid uninitialized reads
        latest_imu_orientation_ = Eigen::Quaternionf::Identity();

        RCLCPP_INFO(this->get_logger(), "NDTLocalizer initialized. Map: %s", map_path.c_str());
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr target_map_{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt_;

    Eigen::Matrix4f pose_; // Current estimated pose
    Eigen::Quaternionf latest_imu_orientation_;
    std::mutex imu_mutex_;


    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!msg) return;
    
        // --- debug: fields ---
        RCLCPP_DEBUG(this->get_logger(), "pointCloudCallback: w=%u h=%u frame=%s",
                     msg->width, msg->height, msg->header.frame_id.c_str());
        bool has_intensity = false;
        for (const auto &f : msg->fields) {
            if (f.name == "intensity") { has_intensity = true; break; }
        }
        RCLCPP_DEBUG(this->get_logger(), "has_intensity=%d", (int)has_intensity);
    
        // --- convert to PointXYZI safely ---
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_xyzi(new pcl::PointCloud<pcl::PointXYZI>());
    
        if (has_intensity) {
            // direct conversion (will map intensity if present)
            pcl::fromROSMsg(*msg, *input_xyzi);
        } else {
            // convert to PointXYZ first, then copy with intensity=0
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_xyz(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::fromROSMsg(*msg, *tmp_xyz);
            input_xyzi->points.reserve(tmp_xyz->points.size());
            for (const auto &p : tmp_xyz->points) {
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
                pcl::PointXYZI pi;
                pi.x = p.x; pi.y = p.y; pi.z = p.z; pi.intensity = 0.0f;
                input_xyzi->push_back(pi);
            }
            input_xyzi->width = static_cast<uint32_t>(input_xyzi->points.size());
            input_xyzi->height = 1;
            input_xyzi->is_dense = true;
        }
    
        if (input_xyzi->empty()) {
            RCLCPP_WARN(this->get_logger(), "Converted input is empty");
            return;
        }
    
        // --- downsample (PointXYZI) ---
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setLeafSize(0.25f, 0.25f, 0.25f);
        voxel.setInputCloud(input_xyzi);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_input(new pcl::PointCloud<pcl::PointXYZI>());
        voxel.filter(*filtered_input);
        if (filtered_input->empty()) {
            RCLCPP_WARN(this->get_logger(), "Filtered input empty after voxel");
            return;
        }
    
        // --- build guess_pose safely ---
        Eigen::Matrix4f guess_pose = pose_;
        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            Eigen::Matrix3f imu_rot = latest_imu_orientation_.toRotationMatrix();
            Eigen::Vector3f euler_prev = pose_.block<3,3>(0,0).eulerAngles(0,1,2);
            Eigen::Vector3f imu_euler   = imu_rot.eulerAngles(0,1,2);
            Eigen::AngleAxisf roll(imu_euler[0], Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf pitch(imu_euler[1], Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf yaw(euler_prev[2], Eigen::Vector3f::UnitZ());
            guess_pose.block<3,3>(0,0) = (yaw * pitch * roll).toRotationMatrix();
        }
    
        // --- NDT (wrapped in try/catch for safety) ---
        try {
            ndt_.setInputSource(filtered_input);
            pcl::PointCloud<pcl::PointXYZI> aligned;
            ndt_.align(aligned, guess_pose);
            if (!ndt_.hasConverged()) {
                RCLCPP_WARN(this->get_logger(), "NDT did not converge");
                return;
            }
            pose_ = ndt_.getFinalTransformation();
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during NDT: %s", e.what());
            return;
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception during NDT");
            return;
        }
    
        // --- publish odom (same as before) ---
        nav_msgs::msg::Odometry odom;
        odom.header = msg->header;
        odom.header.frame_id = "odom";
        odom.pose.pose.position.x = pose_(0, 3);
        odom.pose.pose.position.y = pose_(1, 3);
        odom.pose.pose.position.z = pose_(2, 3);
        Eigen::Matrix3f rot = pose_.block<3,3>(0,0);
        Eigen::Quaternionf q(rot);
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();
        pub_->publish(odom);
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        if (!msg) return;
        std::lock_guard<std::mutex> lock(imu_mutex_);
        latest_imu_orientation_.x() = msg->orientation.x;
        latest_imu_orientation_.y() = msg->orientation.y;
        latest_imu_orientation_.z() = msg->orientation.z;
        latest_imu_orientation_.w() = msg->orientation.w;
        // Note: you may want to normalize to be safe
        latest_imu_orientation_.normalize();
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NDTLocalizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
