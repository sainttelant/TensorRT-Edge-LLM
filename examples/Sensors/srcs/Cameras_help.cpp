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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

namespace cameras
{
namespace
{
std::array<std::string, 4> const kROLES{"front_wide", "front_tele", "cross_left", "cross_right"};

std::vector<Frame> validateFrames(
    std::vector<Frame> frames, std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew)
{
    auto const now = std::chrono::steady_clock::now();
    for (auto const& frame : frames)
    {
        if (frame.image.empty() || frame.received > now || now - frame.received > maxAge)
        {
            return {};
        }
    }
    auto const times = std::minmax_element(frames.begin(), frames.end(),
        [](Frame const& a, Frame const& b) { return a.received < b.received; });
    if (times.second->received - times.first->received > maxSkew)
    {
        return {};
    }
    return frames;
}

struct CameraState
{
    cv::VideoCapture capture;
    std::mutex mutex;
    Frame frame;
    std::string error;
    std::thread worker;
};
} // namespace

class FrameBuffer::Impl
{
public:
    mutable std::mutex mutex;
    std::vector<Frame> frames = std::vector<Frame>(kROLES.size());
};

FrameBuffer::FrameBuffer()
    : mImpl(std::make_unique<Impl>())
{
}

FrameBuffer::~FrameBuffer() = default;

void FrameBuffer::publish(size_t cameraIndex, cv::Mat const& bgr, std::chrono::steady_clock::time_point received)
{
    if (cameraIndex >= kROLES.size() || bgr.empty() || bgr.type() != CV_8UC3)
    {
        throw std::invalid_argument("FrameBuffer requires a valid camera index and an 8-bit BGR image");
    }
    auto image = bgr.clone();
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    auto& frame = mImpl->frames[cameraIndex];
    if (frame.sequence != 0 && received <= frame.received)
    {
        throw std::invalid_argument("Camera timestamps must be strictly increasing");
    }
    frame.image = std::move(image);
    frame.received = received;
    ++frame.sequence;
}

std::vector<Frame> FrameBuffer::snapshot(
    std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew) const
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return validateFrames(mImpl->frames, maxAge, maxSkew);
}

class CameraRig::Impl
{
public:
    std::atomic<bool> stop{false};
    std::vector<std::unique_ptr<CameraState>> cameras;

    ~Impl()
    {
        stop = true;
        for (auto& camera : cameras)
        {
            if (camera->worker.joinable())
            {
                camera->worker.join();
            }
        }
    }
};

CameraRig::CameraRig(std::string const& configPath)
    : mImpl(std::make_unique<Impl>())
{
    std::ifstream input(configPath);
    if (!input)
    {
        throw std::runtime_error("Cannot open camera config: " + configPath);
    }
    auto const config = nlohmann::json::parse(input);
    auto const& entries = config.at("stream_cameras");
    if (!entries.is_array() || entries.size() != kROLES.size())
    {
        throw std::runtime_error("stream_cameras must contain four cameras in model order");
    }
    for (size_t i = 0; i < entries.size(); ++i)
    {
        auto const& entry = entries.at(i);
        if (entry.at("role").get<std::string>() != kROLES[i])
        {
            throw std::runtime_error("Incorrect camera order; expected " + kROLES[i]);
        }
        auto camera = std::make_unique<CameraState>();
        auto const type = entry.at("type").get<std::string>();
        bool opened = false;
        if (type == "usb")
        {
            int const device = entry.at("device").get<int>();
            if (device < 0)
            {
                throw std::runtime_error("Set a non-negative USB device for " + kROLES[i]);
            }
            opened = camera->capture.open(device, cv::CAP_V4L2);
        }
        else if (type == "rtsp" || type == "gstreamer" || type == "file")
        {
            auto const source = entry.at("source").get<std::string>();
            if (source.empty())
            {
                throw std::runtime_error("Set a source for camera " + kROLES[i]);
            }
            int const backend = type == "gstreamer" ? cv::CAP_GSTREAMER : (type == "rtsp" ? cv::CAP_FFMPEG : cv::CAP_ANY);
            if (type == "rtsp")
            {
                constexpr int kTIMEOUT_MS = 5000;
                opened = camera->capture.open(source, backend,
                    {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, kTIMEOUT_MS, cv::CAP_PROP_READ_TIMEOUT_MSEC, kTIMEOUT_MS});
            }
            else
            {
                opened = camera->capture.open(source, backend);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported camera type: " + type);
        }
        if (!opened)
        {
            throw std::runtime_error("Cannot open camera " + kROLES[i]);
        }
        double const fps = camera->capture.get(cv::CAP_PROP_FPS);
        if (type == "file" && (fps <= 0 || !std::isfinite(fps)))
        {
            throw std::runtime_error("Video file has invalid FPS: " + kROLES[i]);
        }
        mImpl->cameras.push_back(std::move(camera));
    }
    for (size_t i = 0; i < mImpl->cameras.size(); ++i)
    {
        auto* camera = mImpl->cameras[i].get();
        bool const isFile = entries.at(i).at("type") == "file";
        double const fps = camera->capture.get(cv::CAP_PROP_FPS);
        camera->worker = std::thread([this, camera, isFile, fps]() {
            try
            {
                auto next = std::chrono::steady_clock::now();
                while (!mImpl->stop)
                {
                    if (isFile)
                    {
                        std::this_thread::sleep_until(next);
                        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(1.0 / fps));
                    }
                    cv::Mat image;
                    if (!camera->capture.read(image) || image.empty())
                    {
                        throw std::runtime_error("Camera stream ended or read failed");
                    }
                    std::lock_guard<std::mutex> lock(camera->mutex);
                    camera->frame.image = std::move(image);
                    camera->frame.received = std::chrono::steady_clock::now();
                    ++camera->frame.sequence;
                }
            }
            catch (std::exception const& error)
            {
                std::lock_guard<std::mutex> lock(camera->mutex);
                camera->error = error.what();
            }
        });
    }
}

CameraRig::~CameraRig() = default;

std::vector<Frame> CameraRig::snapshot(std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew) const
{
    std::vector<Frame> frames;
    for (size_t i = 0; i < mImpl->cameras.size(); ++i)
    {
        auto& camera = *mImpl->cameras[i];
        std::lock_guard<std::mutex> lock(camera.mutex);
        if (!camera.error.empty())
        {
            throw std::runtime_error(kROLES[i] + ": " + camera.error);
        }
        frames.push_back(camera.frame);
    }
    return validateFrames(std::move(frames), maxAge, maxSkew);
}
} // namespace cameras
