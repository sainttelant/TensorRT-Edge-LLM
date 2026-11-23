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
#include "camera.hpp"

#include <stdexcept>

namespace cameras
{
Camera::~Camera()
{
    stop();
    if (mWorker.joinable())
    {
        mWorker.join();
    }
}

void Camera::stop()
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mStop = true;
    }
    mCondition.notify_all();
}

bool Camera::isOffline() const
{
    return mOffline;
}

void Camera::setPaused(bool paused)
{
    if (!mOffline)
    {
        throw std::logic_error("Cannot pause a live camera");
    }
    std::lock_guard<std::mutex> lock(mMutex);
    mPaused = paused;
    if (!paused)
    {
        mResetSchedule = true;
    }
    mCondition.notify_all();
}

void Camera::waitUntilPaused()
{
    std::unique_lock<std::mutex> lock(mMutex);
    mCondition.wait(lock, [this]() { return !mReading || mStop; });
}

void Camera::start()
{
    mWorker = std::thread([this]() {
        try
        {
            auto next = std::chrono::steady_clock::now();
            while (!mStop)
            {
                if (mOffline)
                {
                    std::unique_lock<std::mutex> lock(mMutex);
                    mCondition.wait(lock, [this]() { return !mPaused || mStop; });
                    if (mStop)
                    {
                        break;
                    }
                    if (mResetSchedule)
                    {
                        next = std::chrono::steady_clock::now();
                        mResetSchedule = false;
                    }
                    if (mCondition.wait_until(lock, next, [this]() { return mPaused || mStop; }))
                    {
                        continue;
                    }
                    mReading = true;
                    next += mFrameInterval;
                }
                cv::Mat image;
                if (!mCapture.read(image) || image.empty())
                {
                    throw std::runtime_error("Camera stream ended or read failed");
                }
                std::lock_guard<std::mutex> lock(mMutex);
                mFrame.image = std::move(image);
                mFrame.received = std::chrono::steady_clock::now();
                ++mFrame.sequence;
                if (mOffline)
                {
                    mReading = false;
                    mCondition.notify_all();
                }
            }
        }
        catch (std::exception const& error)
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mError = error.what();
            mReading = false;
            mCondition.notify_all();
        }
    });
}

Frame Camera::latestFrame(std::string const& role)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mError.empty())
    {
        throw std::runtime_error(role + ": " + mError);
    }
    return mFrame;
}
} // namespace cameras
