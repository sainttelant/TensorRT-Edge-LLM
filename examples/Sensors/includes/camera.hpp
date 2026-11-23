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
#pragma once

#include "Cameras_help.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <opencv2/videoio.hpp>

namespace cameras
{

//! Owns one capture thread and publishes its latest frame.
//! Subclasses configure the capture before start; the worker uses only base-class state.
class Camera
{
public:
    virtual ~Camera();
    void start();
    void stop();
    Frame latestFrame(std::string const& role);
    bool isOffline() const;
    void setPaused(bool paused);
    void waitUntilPaused();

protected:
    cv::VideoCapture mCapture;
    std::chrono::steady_clock::duration mFrameInterval{};
    bool mOffline{false};

private:
    std::atomic<bool> mStop{false};
    std::mutex mMutex;
    std::condition_variable mCondition;
    bool mPaused{false};
    bool mReading{false};
    bool mResetSchedule{false};
    Frame mFrame;
    std::string mError;
    std::thread mWorker;
};
} // namespace cameras
