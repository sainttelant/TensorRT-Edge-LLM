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
#include "cameraSources.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <nlohmann/json.hpp>

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
} // namespace

std::optional<double> estimateCaptureFps(Frame const& first, Frame const& last)
{
    if (first.sequence == 0 || last.sequence <= first.sequence || last.received <= first.received)
    {
        return std::nullopt;
    }
    double const elapsed = std::chrono::duration<double>(last.received - first.received).count();
    return static_cast<double>(last.sequence - first.sequence) / elapsed;
}

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
    std::vector<std::unique_ptr<Camera>> cameras;

    ~Impl()
    {
        for (auto const& camera : cameras)
        {
            camera->stop();
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
        mImpl->cameras.push_back(makeCamera(entry, kROLES[i]));
    }
    for (auto const& camera : mImpl->cameras)
    {
        camera->start();
    }
}

CameraRig::~CameraRig() = default;

void CameraRig::setOfflinePlaybackPaused(bool paused)
{
    for (auto const& camera : mImpl->cameras)
    {
        if (!camera->isOffline())
        {
            throw std::invalid_argument("Offline playback pause requires all cameras to use file sources");
        }
    }
    for (auto const& camera : mImpl->cameras)
    {
        camera->setPaused(paused);
    }
    if (paused)
    {
        for (auto const& camera : mImpl->cameras)
        {
            camera->waitUntilPaused();
        }
    }
}

std::vector<Frame> CameraRig::snapshot(std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew) const
{
    std::vector<Frame> frames;
    for (size_t i = 0; i < mImpl->cameras.size(); ++i)
    {
        frames.push_back(mImpl->cameras[i]->latestFrame(kROLES[i]));
    }
    return validateFrames(std::move(frames), maxAge, maxSkew);
}
} // namespace cameras
