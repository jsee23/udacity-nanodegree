#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <thread>
#include <uWS/uWS.h>
#include <vector>

// parameters
const double TIME_INTERVAL_IN_SEC = 0.02;
const double LANE_WIDTH_IN_M = 4.0;
const double PROXIMITY_FRONTIER_IN_M = 30.0;
const double REFERENCE_V = 22.1; // 22.1 m/s = 49.504 mph
const int WAYPOINT_COUNT = 50;
const double MAX_SPEED_CHANGE = 0.2;

enum CarPosition {
  CarPositionUnknown,
  CarAheadLeft,
  CarAheadCenter,
  CarAheadRight,
  CarBehindLeft,
  CarBehindCenter,
  CarBehindRight
};

struct CarInfo {
  int id;
  unsigned int index;
  double vx;
  double vy;
  double speed;
  double s;
  double d;
  double distance;
  int lane;
  double futureS;
  double futureD;
  double futureDistance;
};

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2) {
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x,
                    const vector<double> &maps_y) {

  double closestLen = 100000; // large number
  int closestWaypoint = 0;

  for (int i = 0; i < maps_x.size(); i++) {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x, y, map_x, map_y);
    if (dist < closestLen) {
      closestLen = dist;
      closestWaypoint = i;
    }
  }

  return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x,
                 const vector<double> &maps_y) {

  int closestWaypoint = ClosestWaypoint(x, y, maps_x, maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2((map_y - y), (map_x - x));

  double angle = fabs(theta - heading);
  angle = min(2 * pi() - angle, angle);

  if (angle > pi() / 4) {
    closestWaypoint++;
    if (closestWaypoint == maps_x.size()) {
      closestWaypoint = 0;
    }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta,
                         const vector<double> &maps_x,
                         const vector<double> &maps_y) {
  int next_wp = NextWaypoint(x, y, theta, maps_x, maps_y);

  int prev_wp;
  prev_wp = next_wp - 1;
  if (next_wp == 0) {
    prev_wp = maps_x.size() - 1;
  }

  double n_x = maps_x[next_wp] - maps_x[prev_wp];
  double n_y = maps_y[next_wp] - maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y);
  double proj_x = proj_norm * n_x;
  double proj_y = proj_norm * n_y;

  double frenet_d = distance(x_x, x_y, proj_x, proj_y);

  // see if d value is positive or negative by comparing it to a center point

  double center_x = 1000 - maps_x[prev_wp];
  double center_y = 2000 - maps_y[prev_wp];
  double centerToPos = distance(center_x, center_y, x_x, x_y);
  double centerToRef = distance(center_x, center_y, proj_x, proj_y);

  if (centerToPos <= centerToRef) {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for (int i = 0; i < prev_wp; i++) {
    frenet_s += distance(maps_x[i], maps_y[i], maps_x[i + 1], maps_y[i + 1]);
  }

  frenet_s += distance(0, 0, proj_x, proj_y);

  return {frenet_s, frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s,
                     const vector<double> &maps_x,
                     const vector<double> &maps_y) {
  int prev_wp = -1;

  while (s > maps_s[prev_wp + 1] && (prev_wp < (int)(maps_s.size() - 1))) {
    prev_wp++;
  }

  int wp2 = (prev_wp + 1) % maps_x.size();

  double heading =
      atan2((maps_y[wp2] - maps_y[prev_wp]), (maps_x[wp2] - maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s - maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp] + seg_s * cos(heading);
  double seg_y = maps_y[prev_wp] + seg_s * sin(heading);

  double perp_heading = heading - pi() / 2;

  double x = seg_x + d * cos(perp_heading);
  double y = seg_y + d * sin(perp_heading);

  return {x, y};
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  int lane = 1;
  double targetSpeed = 0.0, desiredSpeed = REFERENCE_V;
  bool laneChangeOngoing = false;

  h.onMessage([&map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
               &map_waypoints_dx, &map_waypoints_dy,
               &lane, &targetSpeed, &desiredSpeed, &laneChangeOngoing](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                      uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    // auto sdata = string(data).substr(0, length);
    // cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of
          // the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          int previousSize = previous_path_x.size();
          double carFutureS = previousSize > 0 ? end_path_s : car_s;
          map<CarPosition, CarInfo> nearestVehicles;

          // sensor fusion for the other cars
          for (unsigned int i = 0; i < sensor_fusion.size(); i++) {
            const auto &vehicle = sensor_fusion.at(i);
            const double vehicleX = vehicle[3];
            const double vehicleY = vehicle[4];

            const double vehicleSpeed =
                sqrt(vehicleX * vehicleX + vehicleY * vehicleY);
            const double vehicleS = double(vehicle[5]);
            const double vehicleD = double(vehicle[6]);
            const int vehicleLane = int(vehicleD / LANE_WIDTH_IN_M);

            // ignore cars of the other side of the street
            if (vehicleD < 0) {
              continue;
            }

            bool vehicleIsAhead = vehicleS > car_s;
            double vehicleFutureS = vehicleS + double(previousSize) * TIME_INTERVAL_IN_SEC * vehicleSpeed;

            CarPosition vehiclePosition = CarPositionUnknown;
            if (vehicleLane == lane) {
                vehiclePosition = vehicleIsAhead ? CarAheadCenter : CarBehindCenter;
            } else if (vehicleLane == lane - 1) {
              vehiclePosition = vehicleIsAhead ? CarAheadLeft : CarBehindLeft;
            } else if (vehicleLane == lane + 1) {
              vehiclePosition = vehicleIsAhead ? CarAheadRight : CarBehindRight;
            } else {
              // ignore these lanes
              continue;
            }

            double vehicleDistance = vehicleS - car_s;
            double vehicleFutureDistance = vehicleFutureS - carFutureS;

            if (nearestVehicles.find(vehiclePosition) == nearestVehicles.end() || vehicleDistance < nearestVehicles[vehiclePosition].distance) {
              nearestVehicles[vehiclePosition] = CarInfo{
                .id = vehicle[0],
                .index = i,
                .vx = vehicleX,
                .vy = vehicleY,
                .speed = vehicleSpeed,
                .s = vehicleS,
                .d = vehicleD,
                .distance = vehicleDistance,
                .lane = vehicleLane,
                .futureS = vehicleFutureS,
                .futureD = vehicleD,
                .futureDistance = vehicleFutureDistance
              };
            }
          }

          /* PATH PLANNING */
          const bool isRightLaneChangePossible =
            ((nearestVehicles.find(CarAheadRight) == nearestVehicles.end()) || nearestVehicles[CarAheadRight].distance > 30.0) &&
            ((nearestVehicles.find(CarBehindRight) == nearestVehicles.end()) || nearestVehicles[CarBehindRight].distance < -8.0) &&
            (lane < 2);
          const bool isLeftLaneChangePossible =
            ((nearestVehicles.find(CarAheadLeft) == nearestVehicles.end()) || nearestVehicles[CarAheadLeft].distance > 30.0) &&
            ((nearestVehicles.find(CarBehindLeft) == nearestVehicles.end()) || nearestVehicles[CarBehindLeft].distance < -8.0) &&
            (lane > 0);
          const bool isRightLaneChangeFeasible =
            nearestVehicles.find(CarAheadRight) == nearestVehicles.end() || nearestVehicles[CarAheadRight].speed > targetSpeed;
          const bool isLeftLaneChangeFeasible =
            nearestVehicles.find(CarAheadLeft) == nearestVehicles.end() || nearestVehicles[CarAheadLeft].speed > targetSpeed;

          const bool laneChangeNeeded =
              (nearestVehicles.find(CarAheadCenter) != nearestVehicles.end()) &&
              (nearestVehicles[CarAheadCenter].distance < PROXIMITY_FRONTIER_IN_M);
          if (laneChangeOngoing) {
            const int currentLane = int(car_d / LANE_WIDTH_IN_M);
            if (currentLane == lane) {
              laneChangeOngoing = false;
            }
            targetSpeed = min(desiredSpeed, targetSpeed + MAX_SPEED_CHANGE);
          } else if (laneChangeNeeded) {

            bool laneChangeDoable = false;
            if (isLeftLaneChangePossible && isLeftLaneChangeFeasible) {
              laneChangeDoable = true;
              lane -= 1;
              desiredSpeed = nearestVehicles.find(CarAheadLeft) == nearestVehicles.end() || nearestVehicles[CarAheadLeft].distance > PROXIMITY_FRONTIER_IN_M ?
                REFERENCE_V : nearestVehicles[CarAheadLeft].speed;
              if (nearestVehicles.find(CarBehindLeft) != nearestVehicles.end()) {
                std::cout << "distance => " <<  nearestVehicles[CarBehindLeft].distance << std::endl; 
              }
            } else if (isRightLaneChangePossible && isRightLaneChangeFeasible) {
              laneChangeDoable = true;
              lane += 1;
              desiredSpeed = nearestVehicles.find(CarAheadRight) == nearestVehicles.end() || nearestVehicles[CarAheadRight].distance > PROXIMITY_FRONTIER_IN_M ?
                REFERENCE_V : nearestVehicles[CarAheadRight].speed;
              if (nearestVehicles.find(CarBehindRight) != nearestVehicles.end()) {
                std::cout << "distance => " <<  nearestVehicles[CarBehindRight].distance << std::endl; 
              }
            }

            if (laneChangeDoable) {
              std::cout << "desired lane = " << lane << std::endl;
              laneChangeOngoing = true;
            } else {
              std::cout << "slowing down => " << nearestVehicles[CarAheadCenter].speed << std::endl;
              desiredSpeed = nearestVehicles[CarAheadCenter].speed;
              targetSpeed = max(targetSpeed - MAX_SPEED_CHANGE, nearestVehicles[CarAheadCenter].speed);
            }
          } else {
            desiredSpeed = REFERENCE_V;
            targetSpeed = min(targetSpeed + MAX_SPEED_CHANGE, REFERENCE_V);
          }

          /* PATH CALCULATION */
          std::vector<double> pts_x;
          std::vector<double> pts_y;

          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);
          double ref_x_prev;
          double ref_y_prev;

          if (previousSize < 2) {
            ref_x_prev = car_x - cos(ref_yaw);
            ref_y_prev = car_y - sin(ref_yaw);
          } else {
            ref_x = previous_path_x[previousSize - 1];
            ref_y = previous_path_y[previousSize - 1];

            ref_x_prev = previous_path_x[previousSize - 2];
            ref_y_prev = previous_path_y[previousSize - 2];
            ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);
          }

          pts_x.push_back(ref_x_prev);
          pts_x.push_back(ref_x);

          pts_y.push_back(ref_y_prev);
          pts_y.push_back(ref_y);

          double speedDependentFactor = max(2.0 * targetSpeed / REFERENCE_V, 1.2);
          vector<double> next_wp0 =
              getXY(car_s + speedDependentFactor*30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 =
              getXY(car_s + speedDependentFactor*30 + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 =
              getXY(car_s + speedDependentFactor*30 + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

          pts_x.push_back(next_wp0[0]);
          pts_x.push_back(next_wp1[0]);
          pts_x.push_back(next_wp2[0]);

          pts_y.push_back(next_wp0[1]);
          pts_y.push_back(next_wp1[1]);
          pts_y.push_back(next_wp2[1]);

          // transforming the X-Y coordinates describing the spline to the
          // vehicle's local coordinate system
          for (unsigned int i = 0; i < pts_x.size(); i++) {

            // the map coordinates of the ith waypoint relative to the car (or
            // reference point)
            double shift_x = pts_x[i] - ref_x;
            double shift_y = pts_y[i] - ref_y;

            // transforming that point to local car coordinates
            pts_x[i] = shift_x * cos(-ref_yaw) - shift_y * sin(-ref_yaw);
            pts_y[i] = shift_x * sin(-ref_yaw) + shift_y * cos(-ref_yaw);
          }

          // start with all of the previous path points from last time
          for (unsigned int i = 0; i < previousSize; i++) {

            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          tk::spline s;
          s.set_points(pts_x, pts_y);

          // calculate how to break up spline points so that we travel at our
          // desired reference velocity...
          double target_x = 30.0;
          double target_y = s(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);

          // ...as presented in Aaron's 'visualisation'
          const double N = target_dist / (TIME_INTERVAL_IN_SEC * targetSpeed);
          const double x_add_on_unit = target_x / N;
          double x_point = 0.0;

          // fill up the rest of our path planner after filling it previuos
          // points.
          for (unsigned int i = 1; i <= WAYPOINT_COUNT - previousSize; i++) {

            x_point += x_add_on_unit;
            double y_point = s(x_point);

            // transforming back from local to global coordinate system
            double x_global = x_point * cos(ref_yaw) - y_point * sin(ref_yaw);
            double y_global = x_point * sin(ref_yaw) + y_point * cos(ref_yaw);

            // we need global coordinates: shifting the point relative to the
            // global map's origo
            x_global += ref_x;
            y_global += ref_y;

            next_x_vals.push_back(x_global);
            next_y_vals.push_back(y_global);
          }

          // TODO: define a path made up of (x,y) points that the car will visit
          // sequentially every .02 seconds
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\"," + msgJson.dump() + "]";

          // this_thread::sleep_for(chrono::milliseconds(1000));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
