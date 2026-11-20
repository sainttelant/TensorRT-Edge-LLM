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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vehicle
{
using Point = std::array<float, 3>;

//! Stores a complete measured history; this class does not control a vehicle.
class TrajectoryHistory
{
public:
    //! Replace history with 16 time-ordered ego-frame positions and a Unix timestamp in milliseconds.
    void updateTrajectory(std::vector<Point> points, int64_t timestampMs);
    //! Read or atomically write {"timestamp_ms": ..., "trajectory": [[x,y,z], ...]}.
    void loadJson(std::string const& path);
    void saveJson(std::string const& path) const;
    std::vector<Point> const& points() const;
    int64_t timestampMs() const;

private:
    std::vector<Point> mPoints;
    int64_t mTimestampMs{};
};
} // namespace vehicle
