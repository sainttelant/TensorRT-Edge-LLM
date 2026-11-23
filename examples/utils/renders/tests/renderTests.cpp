/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "renders/Render.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

TEST(RenderTrajectoryTest, IntegratesAccelerationAndStopsWithoutReversing)
{
    auto points = Alpaway::integrateTrajectory({{2.0F, 0}, {2.0F, 0}}, {1.0, 0.5});
    ASSERT_EQ(points.size(), 3);
    EXPECT_NEAR(points.back().x, 2.0, 1.0e-9);
    EXPECT_NEAR(points.back().y, 0.0, 1.0e-9);
    points = Alpaway::integrateTrajectory({{-4.0F, 0}, {-4.0F, 0}}, {1.0, 1.0});
    EXPECT_NEAR(points[1].x, 0.125, 1.0e-9);
    EXPECT_NEAR(points.back().x, 0.125, 1.0e-9);
}

TEST(RenderTrajectoryTest, IntegratesSignedCurvatureAsAnArc)
{
    auto const left = Alpaway::integrateTrajectory({{0, 0.5F}}, {2.0, 1.0});
    auto const right = Alpaway::integrateTrajectory({{0, -0.5F}}, {2.0, 1.0});
    EXPECT_NEAR(left.back().x, 2 * std::sin(1.0), 1.0e-9);
    EXPECT_NEAR(left.back().y, 2 * (1 - std::cos(1.0)), 1.0e-9);
    EXPECT_NEAR(right.back().x, left.back().x, 1.0e-9);
    EXPECT_NEAR(right.back().y, -left.back().y, 1.0e-9);
}

TEST(RenderTrajectoryTest, RejectsInvalidAssumptionsAndNonfiniteActions)
{
    EXPECT_THROW(Alpaway::integrateTrajectory({{0, 0}}, {-1.0, 0.1}), std::invalid_argument);
    EXPECT_THROW(Alpaway::integrateTrajectory({{0, 0}}, {1.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(Alpaway::integrateTrajectory({}, {1.0, 0.1}), std::invalid_argument);
    EXPECT_THROW(Alpaway::integrateTrajectory({{0, std::numeric_limits<float>::quiet_NaN()}}, {1.0, 0.1}),
        std::invalid_argument);
}

TEST(RenderDashboardTest, RequiresAssumptionsForBevAndPreservesInput)
{
    Alpaway::Render render;
    cv::Mat const frame(90, 160, CV_8UC3, cv::Scalar(40, 50, 60));
    auto const original = frame.clone();
    std::vector<std::pair<float, float>> const actions(64, {0.5F, 0.03F});
    auto const noPath = render.dashboard(frame, "Proceed cautiously.<i1495>", actions);
    auto const withPath
        = render.dashboard(frame, "Proceed cautiously.<i1495>", actions, Alpaway::TrajectoryPreview{2.0, 0.1});
    EXPECT_EQ(withPath.type(), CV_8UC3);
    EXPECT_EQ(withPath.size(), cv::Size(1600, 900));
    EXPECT_EQ(cv::norm(frame, original, cv::NORM_INF), 0);
    cv::Rect const bev(995, 495, 565, 300);
    cv::Mat mask;
    cv::inRange(noPath(bev), cv::Scalar(0, 200, 0), cv::Scalar(80, 255, 80), mask);
    EXPECT_EQ(cv::countNonZero(mask), 0);
    cv::inRange(withPath(bev), cv::Scalar(0, 200, 0), cv::Scalar(80, 255, 80), mask);
    EXPECT_GT(cv::countNonZero(mask), 0);
    auto const clean = render.dashboard(frame, "Proceed cautiously.", actions);
    EXPECT_EQ(cv::norm(clean, noPath, cv::NORM_INF), 0);
    EXPECT_NO_THROW(render.dashboard(frame, std::string(5000, 'W'), {}));
    EXPECT_NO_THROW(render.dashboard(frame, "<i1495><|im_end|>", {}));
    EXPECT_THROW(render.dashboard(cv::Mat{}, "test", {}), std::invalid_argument);
}

//! A calibrated future path is projected onto the matching input image.
TEST(RenderDashboardTest, ProjectsFuturePathOnlyWithValidCalibration)
{
    Alpaway::Render render;
    cv::Mat const frame(90, 160, CV_8UC3, cv::Scalar(40, 50, 60));
    auto const original = frame.clone();
    std::vector<std::pair<float, float>> const actions(20, {0.0F, 0.0F});
    Alpaway::CameraCalibration calibration{{160, 90}, {50, 0, 80, 0, 50, 40, 0, 0, 1},
        {0, -1, 0, 0, 0, -1, 1, 0, 0}, {0, 1.5, 0}, {0, 0, 0, 0}, false, 0};
    Alpaway::TrajectoryPreview const preview{10.0, 0.1};
    auto const without = render.dashboard(frame, "", actions, preview);
    auto const with = render.dashboard(frame, "", actions, preview, calibration);
    cv::Rect const expectedPath(475, 360, 12, 45);
    cv::Mat green;
    cv::inRange(with(expectedPath), cv::Scalar(0, 200, 0), cv::Scalar(80, 255, 80), green);
    EXPECT_GT(cv::countNonZero(green), 0);
    cv::inRange(without(expectedPath), cv::Scalar(0, 200, 0), cv::Scalar(80, 255, 80), green);
    EXPECT_EQ(cv::countNonZero(green), 0);
    EXPECT_EQ(cv::norm(frame, original, cv::NORM_INF), 0);
    auto const leftTurn = render.dashboard(
        frame, "", std::vector<std::pair<float, float>>(6, {0.0F, 0.1F}), preview, calibration);
    cv::inRange(leftTurn(cv::Rect(410, 395, 35, 40)), cv::Scalar(0, 200, 0), cv::Scalar(80, 255, 80), green);
    EXPECT_GT(cv::countNonZero(green), 0);
    calibration.fisheye = true;
    auto const fisheye = render.dashboard(frame, "", actions, preview, calibration);
    cv::Rect const cameraImage(20, 65, 920, 518);
    EXPECT_GT(cv::norm(with(cameraImage), fisheye(cameraImage), cv::NORM_INF), 0);
    calibration.imageSize.width = 100;
    EXPECT_THROW(render.dashboard(frame, "", actions, preview, calibration), std::invalid_argument);
}
