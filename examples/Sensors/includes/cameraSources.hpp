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

#include "camera.hpp"

#include <nlohmann/json_fwd.hpp>

namespace cameras
{
class V4l2Camera final : public Camera
{
public:
    V4l2Camera(nlohmann::json const& entry, std::string const& role);
};

class RtspCamera final : public Camera
{
public:
    RtspCamera(nlohmann::json const& entry, std::string const& role);
};

class GStreamerCamera final : public Camera
{
public:
    GStreamerCamera(nlohmann::json const& entry, std::string const& role);
};

class OfflineCamera final : public Camera
{
public:
    OfflineCamera(nlohmann::json const& entry, std::string const& role);
};

std::unique_ptr<Camera> makeCamera(nlohmann::json const& entry, std::string const& role);
} // namespace cameras
