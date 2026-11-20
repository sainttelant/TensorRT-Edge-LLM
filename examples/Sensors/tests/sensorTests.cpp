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
#include "Cameras_help.hpp"
#include "vehicle_control.hpp"

#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <gtest/gtest.h>

TEST(FrameBufferTest, CopiesFramesAndRejectsStaleOrSkewedSets)
{
    cameras::FrameBuffer source;
    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    auto const now = std::chrono::steady_clock::now() - std::chrono::milliseconds(20);
    for (size_t i = 0; i < 4; ++i)
    {
        source.publish(i, image, now);
    }
    image.setTo(cv::Scalar(0, 0, 0));
    auto frames = source.snapshot(std::chrono::seconds(1), std::chrono::milliseconds(10));
    ASSERT_EQ(frames.size(), 4);
    EXPECT_EQ(frames[0].image.at<cv::Vec3b>(0, 0)[0], 1);
    EXPECT_TRUE(source.snapshot(std::chrono::milliseconds(0), std::chrono::seconds(1)).empty());
    source.publish(0, image, now + std::chrono::milliseconds(20));
    EXPECT_TRUE(source.snapshot(std::chrono::seconds(1), std::chrono::milliseconds(10)).empty());
    EXPECT_THROW(source.publish(0, image, now), std::invalid_argument);
    EXPECT_THROW(source.publish(4, image, now), std::invalid_argument);
}

TEST(TrajectoryHistoryTest, ValidatesAndRoundTrips)
{
    vehicle::TrajectoryHistory history;
    std::vector<vehicle::Point> points(16, {1.0F, 2.0F, 3.0F});
    EXPECT_THROW(history.updateTrajectory({}, 1), std::invalid_argument);
    EXPECT_THROW(history.updateTrajectory(points, 0), std::invalid_argument);
    points[2][0] = std::numeric_limits<float>::infinity();
    EXPECT_THROW(history.updateTrajectory(points, 1), std::invalid_argument);
    points[2][0] = 4.0F;
    history.updateTrajectory(points, 12345);
    char path[] = "/tmp/alpaway-trajectory-test-XXXXXX";
    int const fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    history.saveJson(path);
    vehicle::TrajectoryHistory loaded;
    loaded.loadJson(path);
    EXPECT_EQ(loaded.timestampMs(), 12345);
    EXPECT_EQ(loaded.points(), points);
    EXPECT_EQ(std::remove(path), 0);
}
