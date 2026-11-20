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

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Alpaway
{

enum Color
{
    RED,
    GREEN,
    BLUE,
    YELLOW,
    CYAN,
    MAGENTA,
};

enum FontFace
{
    FONT_HERSHEY_SIMPLEX,
    FONT_HERSHEY_COMPLEX
};

//! Explicit visualization assumptions, not measured vehicle state.
struct TrajectoryPreview
{
    double initialSpeedMps;
    double timeStepSeconds;
};

//! Integrate (acceleration in m/s^2, curvature in 1/m) into ego-frame (forward, left) meters.
//! Each action is held for one timestep. Speed is clamped at zero; reversing is not modeled.
std::vector<cv::Point2d> integrateTrajectory(
    std::vector<std::pair<float, float>> const& actions, TrajectoryPreview const& preview);

class Render
{
public:
    Render() = default;
    ~Render() = default;
    void drawTinyTriangle(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Point p3, Color color);
    void drawTriangle(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Point p3, Color color);
    void drawCircle(cv::Mat& img, cv::Point center, int radius, Color color);
    void drawLine(cv::Mat& img, cv::Point p1, cv::Point p2, Color color);
    void drawPoint(cv::Mat& img, cv::Point p, Color color);
    void drawPoints(cv::Mat& img, std::vector<cv::Point> points, Color color);
    void drawText(
        cv::Mat& img, cv::Point p, std::string text, Color color, int fontFace, double fontScale, int thickness);

    //! Compose a BGR camera image, readable model text and action/BEV panels without opening a window.
    //! Discrete/control tokens are omitted only from display text; the caller retains raw output.
    cv::Mat dashboard(cv::Mat const& frame, std::string const& explanation,
        std::vector<std::pair<float, float>> const& actions,
        std::optional<TrajectoryPreview> const& preview = std::nullopt);

private:
    cv::Scalar getColor(Color color);
};

} // namespace Alpaway
