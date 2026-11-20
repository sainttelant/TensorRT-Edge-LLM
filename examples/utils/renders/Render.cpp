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
#include "Render.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace Alpaway
{
namespace
{
void validateActions(std::vector<std::pair<float, float>> const& actions)
{
    for (auto const& [accel, curvature] : actions)
    {
        if (!std::isfinite(accel) || !std::isfinite(curvature))
        {
            throw std::invalid_argument("Trajectory actions must be finite");
        }
    }
}

void validatePreview(TrajectoryPreview const& preview)
{
    if (!std::isfinite(preview.initialSpeedMps) || preview.initialSpeedMps < 0
        || !std::isfinite(preview.timeStepSeconds) || preview.timeStepSeconds <= 0)
    {
        throw std::invalid_argument("Trajectory preview requires speed >= 0 and dt > 0, both finite");
    }
}

std::string number(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

std::vector<std::string> wrapText(std::string const& text, int width)
{
    std::vector<std::string> lines;
    std::string line;
    auto fits = [width](std::string const& value) {
        int baseline{};
        return cv::getTextSize(value, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline).width <= width;
    };
    for (char c : text)
    {
        if (c == '\n')
        {
            lines.push_back(line);
            line.clear();
            continue;
        }
        line += c;
        if (!fits(line))
        {
            auto const space = line.find_last_of(' ');
            if (space != std::string::npos && space > 0)
            {
                lines.push_back(line.substr(0, space));
                line.erase(0, space + 1);
            }
            else
            {
                char const last = line.back();
                line.pop_back();
                lines.push_back(line);
                line.assign(1, last);
            }
        }
    }
    if (!line.empty())
    {
        lines.push_back(line);
    }
    return lines;
}

void drawActionPlot(
    Render& render, cv::Mat& canvas, cv::Rect area, std::vector<std::pair<float, float>> const& actions, bool curvature)
{
    render.drawText(canvas, {area.x, area.y + 24}, curvature ? "CURVATURE [1/m]" : "ACCELERATION [m/s^2]", CYAN,
        FONT_HERSHEY_SIMPLEX, 0.65, 2);
    if (actions.empty())
    {
        render.drawText(
            canvas, {area.x, area.y + 65}, "Waiting for model output", YELLOW, FONT_HERSHEY_SIMPLEX, 0.6, 1);
        return;
    }
    double low = 0;
    double high = 0;
    for (auto const& action : actions)
    {
        double const value = curvature ? action.second : action.first;
        low = std::min(low, value);
        high = std::max(high, value);
    }
    double const span = std::max(high - low, 1.0e-6);
    auto yPixel = [&](double value) { return area.y + 60 + cvRound((high - value) / span * (area.height - 90)); };
    render.drawText(
        canvas, {area.x, area.y + 48}, number(low) + " .. " + number(high), YELLOW, FONT_HERSHEY_SIMPLEX, 0.55, 1);
    cv::line(canvas, {area.x, yPixel(0)}, {area.x + area.width, yPixel(0)}, cv::Scalar(90, 90, 90), 1);
    std::vector<cv::Point> pixels;
    for (size_t i = 0; i < actions.size(); ++i)
    {
        int const x = area.x + cvRound(static_cast<double>(i) / std::max(size_t{1}, actions.size() - 1) * area.width);
        pixels.emplace_back(x, yPixel(curvature ? actions[i].second : actions[i].first));
        if (i > 0)
        {
            render.drawLine(canvas, pixels[i - 1], pixels[i], GREEN);
        }
    }
    render.drawPoints(canvas, pixels, YELLOW);
    render.drawText(canvas, {area.x, area.y + area.height}, "Prediction step (not XY coordinates)", CYAN,
        FONT_HERSHEY_SIMPLEX, 0.5, 1);
}
} // namespace

std::vector<cv::Point2d> integrateTrajectory(
    std::vector<std::pair<float, float>> const& actions, TrajectoryPreview const& preview)
{
    validatePreview(preview);
    validateActions(actions);
    if (actions.empty())
    {
        throw std::invalid_argument("Cannot integrate an empty action sequence");
    }
    std::vector<cv::Point2d> points{{0, 0}};
    double speed = preview.initialSpeedMps;
    double heading = 0;
    for (auto const& [accel, curvature] : actions)
    {
        double const nextSpeed = std::max(0.0, speed + accel * preview.timeStepSeconds);
        double const movingTime = accel < 0 ? std::min(preview.timeStepSeconds, speed / -static_cast<double>(accel))
                                            : preview.timeStepSeconds;
        double const distance = (speed + nextSpeed) * 0.5 * movingTime;
        double const halfTurn = distance * curvature * 0.5;
        double const chord = distance * (std::abs(halfTurn) < 1.0e-8 ? 1.0 : std::sin(halfTurn) / halfTurn);
        cv::Point2d const next
            = points.back() + cv::Point2d(chord * std::cos(heading + halfTurn), chord * std::sin(heading + halfTurn));
        heading = std::remainder(heading + 2 * halfTurn, 2 * std::acos(-1.0));
        if (!std::isfinite(next.x) || !std::isfinite(next.y) || !std::isfinite(heading) || !std::isfinite(nextSpeed))
        {
            throw std::invalid_argument("Trajectory integration overflowed");
        }
        points.push_back(next);
        speed = nextSpeed;
    }
    return points;
}

cv::Scalar Render::getColor(Color color)
{
    switch (color)
    {
    case RED: return {0, 0, 255};
    case GREEN: return {0, 255, 0};
    case BLUE: return {255, 0, 0};
    case YELLOW: return {0, 255, 255};
    case CYAN: return {255, 255, 0};
    case MAGENTA: return {255, 0, 255};
    }
    throw std::invalid_argument("Unknown render color");
}

void Render::drawTinyTriangle(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Point p3, Color color)
{
    std::array<cv::Point, 3> const points{p1, p2, p3};
    cv::fillConvexPoly(img, points.data(), static_cast<int>(points.size()), getColor(color), cv::LINE_AA);
}

void Render::drawTriangle(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Point p3, Color color)
{
    drawLine(img, p1, p2, color);
    drawLine(img, p2, p3, color);
    drawLine(img, p3, p1, color);
}

void Render::drawCircle(cv::Mat& img, cv::Point center, int radius, Color color)
{
    if (radius < 0)
    {
        throw std::invalid_argument("Circle radius must not be negative");
    }
    cv::circle(img, center, radius, getColor(color), 2, cv::LINE_AA);
}

void Render::drawLine(cv::Mat& img, cv::Point p1, cv::Point p2, Color color)
{
    cv::line(img, p1, p2, getColor(color), 2, cv::LINE_AA);
}

void Render::drawPoint(cv::Mat& img, cv::Point p, Color color)
{
    cv::circle(img, p, 3, getColor(color), cv::FILLED, cv::LINE_AA);
}

void Render::drawPoints(cv::Mat& img, std::vector<cv::Point> points, Color color)
{
    for (auto const& point : points)
    {
        drawPoint(img, point, color);
    }
}

void Render::drawText(
    cv::Mat& img, cv::Point p, std::string text, Color color, int fontFace, double fontScale, int thickness)
{
    if (!std::isfinite(fontScale) || fontScale <= 0 || thickness <= 0
        || (fontFace != FONT_HERSHEY_SIMPLEX && fontFace != FONT_HERSHEY_COMPLEX))
    {
        throw std::invalid_argument("Invalid render font settings");
    }
    int const font = fontFace == FONT_HERSHEY_SIMPLEX ? cv::FONT_HERSHEY_SIMPLEX : cv::FONT_HERSHEY_COMPLEX;
    cv::putText(img, text, p, font, fontScale, cv::Scalar(0, 0, 0), thickness + 3, cv::LINE_AA);
    cv::putText(img, text, p, font, fontScale, getColor(color), thickness, cv::LINE_AA);
}

cv::Mat Render::dashboard(cv::Mat const& frame, std::string const& explanation,
    std::vector<std::pair<float, float>> const& actions, std::optional<TrajectoryPreview> const& preview)
{
    if (frame.empty() || frame.type() != CV_8UC3)
    {
        throw std::invalid_argument("Dashboard requires a nonempty 8-bit BGR image");
    }
    validateActions(actions);
    if (preview)
    {
        validatePreview(*preview);
    }
    cv::Mat canvas(900, 1600, CV_8UC3, cv::Scalar(22, 24, 30));
    cv::rectangle(canvas, {970, 15, 615, 870}, cv::Scalar(37, 40, 49), cv::FILLED);
    drawText(canvas, {20, 42}, "ALPAMAYO | FRONT WIDE", CYAN, FONT_HERSHEY_SIMPLEX, 1.0, 2);
    double const scale = std::min(920.0 / frame.cols, 518.0 / frame.rows);
    cv::Mat resized;
    cv::resize(frame, resized, {}, scale, scale);
    resized.copyTo(
        canvas(cv::Rect(20 + (920 - resized.cols) / 2, 65 + (518 - resized.rows) / 2, resized.cols, resized.rows)));
    drawText(canvas, {20, 615}, "Matched input frame | Model output is NOT a control command", YELLOW,
        FONT_HERSHEY_SIMPLEX, 0.65, 2);
    drawActionPlot(*this, canvas, {20, 640, 430, 225}, actions, false);
    drawActionPlot(*this, canvas, {500, 640, 430, 225}, actions, true);
    drawText(canvas, {990, 48}, "MODEL EXPLANATION", YELLOW, FONT_HERSHEY_SIMPLEX, 0.9, 2);
    drawText(canvas, {990, 76}, "Unverified model text; not a safety assessment", CYAN, FONT_HERSHEY_SIMPLEX, 0.55, 1);
    static std::regex const kTOKENS(R"(<i[0-9]+>|<\|[^>]*\|>)");
    std::string readable = std::regex_replace(explanation, kTOKENS, "");
    if (readable.find_first_not_of(" \n\r\t") == std::string::npos)
    {
        readable = "No readable explanation. Raw output is empty or contains only model tokens.";
    }
    auto lines = wrapText(readable, 565);
    constexpr size_t kMAX_LINES = 9;
    if (lines.size() > kMAX_LINES)
    {
        lines.resize(kMAX_LINES);
        lines.back() = "... [display truncated; see raw JSON]";
    }
    for (size_t i = 0; i < lines.size(); ++i)
    {
        drawText(canvas, {990, 115 + static_cast<int>(i) * 30}, lines[i], YELLOW, FONT_HERSHEY_SIMPLEX, 0.8, 2);
    }
    cv::Rect const plot(995, 495, 565, 300);
    drawText(canvas, {990, 425}, "PREDICTED PATH | SCHEMATIC BEV", CYAN, FONT_HERSHEY_SIMPLEX, 0.7, 2);
    if (!preview || actions.empty())
    {
        drawText(canvas, {990, 480}, actions.empty() ? "Waiting for model actions" : "XY path unavailable:", YELLOW,
            FONT_HERSHEY_SIMPLEX, 0.7, 2);
        if (!preview)
        {
            drawText(canvas, {990, 520}, "Supply --preview-speed-mps and", YELLOW, FONT_HERSHEY_SIMPLEX, 0.65, 2);
            drawText(canvas, {990, 555}, "--preview-dt to integrate actions.", YELLOW, FONT_HERSHEY_SIMPLEX, 0.65, 2);
        }
        return canvas;
    }
    auto const points = integrateTrajectory(actions, *preview);
    double minX = 0, maxX = 1, minY = -0.5, maxY = 0.5;
    for (auto const& point : points)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }
    double const metersToPixels = std::min((plot.width - 50) / (maxY - minY), (plot.height - 40) / (maxX - minX));
    auto pixel = [&](cv::Point2d const& point) {
        return cv::Point(plot.x + plot.width / 2 - cvRound((point.y - (minY + maxY) / 2) * metersToPixels),
            plot.y + plot.height / 2 - cvRound((point.x - (minX + maxX) / 2) * metersToPixels));
    };
    cv::rectangle(canvas, plot, cv::Scalar(70, 75, 85), 1);
    std::vector<cv::Point> pixels;
    for (auto const& point : points)
    {
        pixels.push_back(pixel(point));
    }
    for (size_t i = 1; i < pixels.size(); ++i)
    {
        drawLine(canvas, pixels[i - 1], pixels[i], GREEN);
        drawCircle(canvas, pixels[i], 3, CYAN);
        if (i % 8 == 0 || i + 1 == pixels.size())
        {
            cv::Point2d const direction = pixels[i] - pixels[i - 1];
            double const length = cv::norm(direction);
            if (length > 0)
            {
                cv::Point2d const forward = direction * (10.0 / length);
                cv::Point2d const side(-forward.y * 0.5, forward.x * 0.5);
                cv::Point2d const center(pixels[i]);
                drawTinyTriangle(
                    canvas, center + forward, center - forward * 0.6 + side, center - forward * 0.6 - side, YELLOW);
            }
        }
    }
    auto const origin = pixel({0, 0});
    drawTriangle(canvas, origin + cv::Point(0, -12), origin + cv::Point(-9, 9), origin + cv::Point(9, 9), MAGENTA);
    drawText(canvas, {990, 465},
        "v0=" + number(preview->initialSpeedMps) + " m/s, dt=" + number(preview->timeStepSeconds) + " s (assumed)",
        YELLOW, FONT_HERSHEY_SIMPLEX, 0.6, 1);
    drawText(canvas, {990, 825}, "Forward: up | Left: left | Span: " + number(maxX - minX) + " m", CYAN,
        FONT_HERSHEY_SIMPLEX, 0.55, 1);
    drawText(canvas, {990, 860}, "No camera projection. Not a validated driving path.", YELLOW, FONT_HERSHEY_SIMPLEX,
        0.55, 1);
    return canvas;
}
} // namespace Alpaway