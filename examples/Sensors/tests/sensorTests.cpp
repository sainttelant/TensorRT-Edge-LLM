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
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

namespace
{
struct TemporaryCameraFiles
{
    std::string directory;
    std::string video;
    std::string config;

    ~TemporaryCameraFiles()
    {
        std::remove(video.c_str());
        std::remove(config.c_str());
        rmdir(directory.c_str());
    }
};

nlohmann::json cameraConfig(std::string const& type, std::string const& source)
{
    nlohmann::json entries = nlohmann::json::array();
    for (auto const* role : {"front_wide", "front_tele", "cross_left", "cross_right"})
    {
        entries.push_back({{"role", role}, {"type", type}, {"source", source}});
    }
    return {{"stream_cameras", entries}};
}
} // namespace

//! Capture FPS accounts for frames between snapshots and rejects invalid intervals.
TEST(FrameRateTest, EstimatesCapturedFramesBetweenSnapshots)
{
    cameras::Frame first;
    first.sequence = 5;
    first.received = std::chrono::steady_clock::now();
    cameras::Frame last;
    last.sequence = 35;
    last.received = first.received + std::chrono::milliseconds(1500);
    auto const fps = cameras::estimateCaptureFps(first, last);
    ASSERT_TRUE(fps.has_value());
    EXPECT_DOUBLE_EQ(*fps, 20.0);
    last.sequence = first.sequence;
    EXPECT_FALSE(cameras::estimateCaptureFps(first, last).has_value());
    last.sequence = first.sequence + 1;
    last.received = first.received;
    EXPECT_FALSE(cameras::estimateCaptureFps(first, last).has_value());
}

//! Offline sources pause together during inference and resume before end of stream.
TEST(CameraRigTest, PausesAndResumesOfflineVideo)
{
    char directory[] = "/tmp/alpaway-camera-test-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    TemporaryCameraFiles files{
        directory, std::string(directory) + "/video.avi", std::string(directory) + "/config.json"};
    cv::VideoWriter writer(files.video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, cv::Size(32, 32));
    ASSERT_TRUE(writer.isOpened());
    for (int i = 0; i < 20; ++i)
    {
        writer.write(cv::Mat(32, 32, CV_8UC3, cv::Scalar(i, i, i)));
    }
    writer.release();
    {
        std::ofstream config(files.config);
        ASSERT_TRUE(config.is_open());
        config << cameraConfig("file", files.video);
    }
    cameras::CameraRig rig(files.config);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::vector<cameras::Frame> frames;
    while (frames.empty() && std::chrono::steady_clock::now() < deadline)
    {
        frames = rig.snapshot(std::chrono::seconds(1), std::chrono::seconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(frames.size(), size_t{4});
    auto const firstSequence = frames[0].sequence;
    auto const updateDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (frames[0].sequence == firstSequence && std::chrono::steady_clock::now() < updateDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        frames = rig.snapshot(std::chrono::seconds(1), std::chrono::seconds(1));
        ASSERT_EQ(frames.size(), size_t{4});
    }
    ASSERT_GT(frames[0].sequence, firstSequence);
    rig.setOfflinePlaybackPaused(true);
    auto const pausedFrames = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
    ASSERT_EQ(pausedFrames.size(), size_t{4});
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    auto const stillPaused = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
    ASSERT_EQ(stillPaused.size(), size_t{4});
    for (size_t i = 0; i < pausedFrames.size(); ++i)
    {
        EXPECT_EQ(stillPaused[i].sequence, pausedFrames[i].sequence);
    }
    rig.setOfflinePlaybackPaused(false);
    auto const resumeDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    auto resumed = stillPaused;
    while (resumed[0].sequence == stillPaused[0].sequence && std::chrono::steady_clock::now() < resumeDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        resumed = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
        ASSERT_EQ(resumed.size(), size_t{4});
    }
    EXPECT_GT(resumed[0].sequence, stillPaused[0].sequence);
    bool ended = false;
    while (!ended && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        try
        {
            rig.snapshot(std::chrono::seconds(1), std::chrono::seconds(1));
        }
        catch (std::runtime_error const&)
        {
            ended = true;
        }
    }
    EXPECT_TRUE(ended);
}

//! Pausing a mixed rig stops offline playback while the live source continues.
TEST(CameraRigTest, PausesOnlyOfflineSources)
{
    std::string const pipeline
        = "videotestsrc is-live=true ! video/x-raw,width=32,height=32,framerate=10/1 ! videoconvert ! appsink";
    cv::VideoCapture probe(pipeline, cv::CAP_GSTREAMER);
    if (!probe.isOpened())
    {
        GTEST_SKIP() << "GStreamer videotestsrc is unavailable";
    }
    probe.release();

    char directory[] = "/tmp/alpaway-mixed-camera-test-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    TemporaryCameraFiles files{
        directory, std::string(directory) + "/video.avi", std::string(directory) + "/config.json"};
    cv::VideoWriter writer(files.video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, cv::Size(32, 32));
    ASSERT_TRUE(writer.isOpened());
    for (int i = 0; i < 50; ++i)
    {
        writer.write(cv::Mat(32, 32, CV_8UC3, cv::Scalar(i, i, i)));
    }
    writer.release();
    {
        auto config = cameraConfig("file", files.video);
        config["stream_cameras"][1]["type"] = "gstreamer";
        config["stream_cameras"][1]["source"] = pipeline;
        std::ofstream output(files.config);
        output << config;
    }
    cameras::CameraRig rig(files.config);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    std::vector<cameras::Frame> frames;
    while (frames.empty() && std::chrono::steady_clock::now() < deadline)
    {
        frames = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(frames.size(), size_t{4});
    rig.setOfflinePlaybackPaused(true);
    auto const paused = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
    ASSERT_EQ(paused.size(), size_t{4});
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    auto const during = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
    ASSERT_EQ(during.size(), size_t{4});
    EXPECT_EQ(during[0].sequence, paused[0].sequence);
    EXPECT_GT(during[1].sequence, paused[1].sequence);
    EXPECT_EQ(during[2].sequence, paused[2].sequence);
    EXPECT_EQ(during[3].sequence, paused[3].sequence);
    rig.setOfflinePlaybackPaused(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto const resumed = rig.snapshot(std::chrono::seconds(2), std::chrono::seconds(2));
    ASSERT_EQ(resumed.size(), size_t{4});
    EXPECT_GT(resumed[0].sequence, during[0].sequence);
    EXPECT_GT(resumed[2].sequence, during[2].sequence);
    EXPECT_GT(resumed[3].sequence, during[3].sequence);
}

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
    ASSERT_EQ(frames.size(), size_t{4});
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
