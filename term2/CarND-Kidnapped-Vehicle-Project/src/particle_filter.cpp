/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    if (is_initialized)
        return;

    is_initialized = true;
    num_particles = 20;

    default_random_engine engine;
    double std_x = std[0], std_y = std[1], std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i=0; i < num_particles; i++) {
        const Particle p = {
            .id = i,
            .x = dist_x(engine),
            .y = dist_y(engine),
            .theta = dist_theta(engine),
            .weight = 1
        };
        particles.push_back(p);
    }
    weights.resize(num_particles);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine engine;

    // TODO
    for (int i=0; i < num_particles; i++) {
        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        } else {
            particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += (velocity / yaw_rate) * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
        }
        particles[i].theta += yaw_rate * delta_t;

        normal_distribution<double> dist_x(0, std_pos[0]);
        normal_distribution<double> dist_y(0, std_pos[1]);
        normal_distribution<double> dist_theta(0, std_pos[2]);

        particles[i].x += dist_x(engine);
        particles[i].y += dist_y(engine);
        particles[i].theta += dist_theta(engine);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    double previousMinDistribution, nextDistribution;

    for (LandmarkObs &observation : observations) {
        previousMinDistribution = std::numeric_limits<double>::max();
        int id = 0;

        for (unsigned int i=0; i < predicted.size(); i++) {
            nextDistribution = dist(observation.x, observation.y, predicted[i].x, predicted[i].y);
            if (previousMinDistribution >= nextDistribution) {
                previousMinDistribution = nextDistribution;
                id = i;
            }
        }
        observation.id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    for (unsigned int i=0; i < particles.size(); i++) {
        const Particle& particle = particles[i];
        std::vector<LandmarkObs> predictions;
        for (const Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
            if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) < sensor_range) {
                LandmarkObs convertedLandmark = {
                    .id = landmark.id_i,
                    .x = landmark.x_f,
                    .y = landmark.y_f
                };
                predictions.push_back(convertedLandmark);
            }
        }

        // convert observations in map coordinates
        std::vector<LandmarkObs> observationsInMapCoordinates;
        for (const LandmarkObs& observation : observations) {
            const LandmarkObs observationInMapCoordinates = {
                .id = observation.id,
                .x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta),
                .y = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta)
            };
            observationsInMapCoordinates.push_back(observationInMapCoordinates);
        }

        dataAssociation(predictions, observationsInMapCoordinates);
        double probability = 1.0;
        double normalizer = (1 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]));

        for (const LandmarkObs& observation : observationsInMapCoordinates) {
            const LandmarkObs prediction = predictions[observation.id];

            double expWeight = exp(-1.0 * ((pow(observation.x - prediction.x, 2) / (2 * pow(std_landmark[0], 2)) +
                                           (pow(observation.y - prediction.y, 2) / (2 * pow(std_landmark[1], 2))))));
            probability = probability * expWeight * normalizer;
        }

        particles[i].weight = probability;
        weights[i] = probability;
    }
}

void ParticleFilter::resample() {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::discrete_distribution<int> indexes(weights.begin(), weights.end());

    std::vector<Particle> resampledParticles;
    for (unsigned int i=0; i < particles.size(); i++) {
        resampledParticles.push_back(particles[indexes(generator)]);
    }

    particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
