/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cameraSources.hpp"

#include <cmath>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace cameras
{
namespace
{
std::string getSource(nlohmann::json const& entry, std::string const& role)
{
    auto const source = entry.at("source").get<std::string>();
    if (source.empty())
    {
        throw std::runtime_error("Set a source for camera " + role);
    }
    return source;
}
} // namespace

V4l2Camera::V4l2Camera(nlohmann::json const& entry, std::string const& role)
{
    int const device = entry.at("device").get<int>();
    if (device < 0)
    {
        throw std::runtime_error("Set a non-negative USB device for " + role);
    }
    if (!mCapture.open(device, cv::CAP_V4L2))
    {
        throw std::runtime_error("Cannot open camera " + role);
    }
}

RtspCamera::RtspCamera(nlohmann::json const& entry, std::string const& role)
{
    constexpr int kTIMEOUT_MS = 5000;
    if (!mCapture.open(getSource(entry, role), cv::CAP_FFMPEG,
            {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, kTIMEOUT_MS, cv::CAP_PROP_READ_TIMEOUT_MSEC, kTIMEOUT_MS}))
    {
        throw std::runtime_error("Cannot open camera " + role);
    }
}

GStreamerCamera::GStreamerCamera(nlohmann::json const& entry, std::string const& role)
{
    if (!mCapture.open(getSource(entry, role), cv::CAP_GSTREAMER))
    {
        throw std::runtime_error("Cannot open camera " + role);
    }
}

OfflineCamera::OfflineCamera(nlohmann::json const& entry, std::string const& role)
{
    if (!mCapture.open(getSource(entry, role), cv::CAP_ANY))
    {
        throw std::runtime_error("Cannot open camera " + role);
    }
    double const fps = mCapture.get(cv::CAP_PROP_FPS);
    
    if (fps <= 0 || !std::isfinite(fps))
    {
        throw std::runtime_error("Video file has invalid FPS: " + role);
    }
    mFrameInterval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / fps));
    mOffline = true;
}

std::unique_ptr<Camera> makeCamera(nlohmann::json const& entry, std::string const& role)
{
    auto const type = entry.at("type").get<std::string>();
    if (type == "usb")
    {
        return std::make_unique<V4l2Camera>(entry, role);
    }
    if (type == "rtsp")
    {
        return std::make_unique<RtspCamera>(entry, role);
    }
    if (type == "gstreamer")
    {
        return std::make_unique<GStreamerCamera>(entry, role);
    }
    if (type == "file")
    {
        return std::make_unique<OfflineCamera>(entry, role);
    }
    throw std::runtime_error("Unsupported camera type: " + type);
}
} // namespace cameras
