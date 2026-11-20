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
#include "vehicle_control.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <nlohmann/json.hpp>
#include <unistd.h>

namespace vehicle
{
void TrajectoryHistory::updateTrajectory(std::vector<Point> points, int64_t timestampMs)
{
    constexpr size_t kHISTORY_POINTS = 16;
    if (points.size() != kHISTORY_POINTS || timestampMs <= 0)
    {
        throw std::invalid_argument("Trajectory requires 16 points and a positive timestamp_ms");
    }
    for (auto const& point : points)
    {
        for (auto value : point)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("Trajectory coordinates must be finite");
            }
        }
    }
    mPoints = std::move(points);
    mTimestampMs = timestampMs;
}

void TrajectoryHistory::loadJson(std::string const& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("Cannot open trajectory file: " + path);
    }
    auto const json = nlohmann::json::parse(input);
    updateTrajectory(json.at("trajectory").get<std::vector<Point>>(), json.at("timestamp_ms").get<int64_t>());
}

void TrajectoryHistory::saveJson(std::string const& path) const
{
    if (mPoints.empty())
    {
        throw std::runtime_error("No trajectory to save");
    }
    auto temp = path + ".tmp.XXXXXX";
    std::vector<char> name(temp.begin(), temp.end());
    name.push_back('\0');
    int const fd = mkstemp(name.data());
    if (fd < 0)
    {
        throw std::runtime_error("Cannot create temporary trajectory file for " + path);
    }
    FILE* file = fdopen(fd, "w");
    if (!file)
    {
        close(fd);
        std::remove(name.data());
        throw std::runtime_error("Cannot open temporary trajectory file for " + path);
    }
    auto const data = nlohmann::json{{"timestamp_ms", mTimestampMs}, {"trajectory", mPoints}}.dump() + "\n";
    bool const written = std::fwrite(data.data(), 1, data.size(), file) == data.size();
    bool const closed = std::fclose(file) == 0;
    if (!written || !closed || std::rename(name.data(), path.c_str()) != 0)
    {
        std::remove(name.data());
        throw std::runtime_error("Cannot write trajectory file: " + path);
    }
}

std::vector<Point> const& TrajectoryHistory::points() const
{
    return mPoints;
}

int64_t TrajectoryHistory::timestampMs() const
{
    return mTimestampMs;
}
} // namespace vehicle
