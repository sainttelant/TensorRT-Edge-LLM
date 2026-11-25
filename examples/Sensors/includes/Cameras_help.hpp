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
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace cameras
{

//! A frame's timestamp is host receipt time, not the camera exposure time.
struct Frame
{
    cv::Mat image;
    std::chrono::steady_clock::time_point received;
    uint64_t sequence{};
};

//! Estimate host-received frames per second between two snapshots of the same source.
std::optional<double> estimateCaptureFps(Frame const& first, Frame const& last);

//! Supply four frames in front-wide, front-tele, cross-left, cross-right order.
class FrameSource
{
public:
    virtual ~FrameSource() = default;
    //! Empty means frames are not yet fresh enough or their host receipt times are too far apart.
    virtual std::vector<Frame> snapshot(
        std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew) const = 0;
};

//! Thread-safe frame ingress for a camera SDK callback or other producer.
class FrameBuffer : public FrameSource
{
public:
    FrameBuffer();
    ~FrameBuffer() override;
    FrameBuffer(FrameBuffer const&) = delete;
    FrameBuffer& operator=(FrameBuffer const&) = delete;

    //! Copy one BGR frame into a camera slot. received must use the host steady clock.
    void publish(size_t cameraIndex, cv::Mat const& bgr, std::chrono::steady_clock::time_point received);
    std::vector<Frame> snapshot(
        std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew) const override;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

//! Continuously captures the latest frame from each configured camera.
class CameraRig : public FrameSource
{
public:
    explicit CameraRig(std::string const& configPath);
    ~CameraRig() override;
    CameraRig(CameraRig const&) = delete;
    CameraRig& operator=(CameraRig const&) = delete;

    //! Returns frames in model camera order, rejecting stale or skewed frames.
    std::vector<Frame> snapshot(
        std::chrono::milliseconds maxAge, std::chrono::milliseconds maxSkew) const override;

    //! Return whether the rig contains an offline video source.
    bool hasOfflineSources() const;

    //! Pause or resume offline video sources without interrupting live cameras.
    void setOfflinePlaybackPaused(bool paused);

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cameras
