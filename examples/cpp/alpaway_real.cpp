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
#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "renders/Render.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "vehicle_control.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace rt = trt_edgellm::rt;

namespace
{
struct Args
{
    std::string cameraConfig;
    std::string trajectoryFile;
    std::string engineDir;
    std::string multimodalDir;
    std::string checkpointDir;
    std::string calibrationFile;
    int maxWindows{0};
    bool show{false};
    bool pauseOfflineDuringInference{false};
    std::optional<Alpaway::TrajectoryPreview> preview;
};

void usage(char const* name)
{
    std::cerr << "Usage: " << name
              << " --camera-config PATH --trajectory-file PATH --engine-dir PATH --multimodal-engine-dir PATH"
                 " [--checkpoint-dir PATH] [--max-windows N] [--show]"
                 " [--preview-speed-mps VALUE --preview-dt SECONDS] [--preview-calibration PATH]"
                 " [--pause-offline-during-inference]\n";
}

double parseNumber(char const* input)
{
    std::string const value(input);
    size_t parsed = 0;
    double const number = std::stod(value, &parsed);
    if (parsed != value.size() || !std::isfinite(number))
    {
        throw std::invalid_argument("Preview values must be finite numbers");
    }
    return number;
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    std::optional<double> previewSpeed;
    std::optional<double> previewDt;
    option const options[] = {{"camera-config", required_argument, nullptr, 'c'},
        {"trajectory-file", required_argument, nullptr, 't'}, {"engine-dir", required_argument, nullptr, 'e'},
        {"multimodal-engine-dir", required_argument, nullptr, 'm'}, {"checkpoint-dir", required_argument, nullptr, 'p'},
        {"max-windows", required_argument, nullptr, 'n'}, {"show", no_argument, nullptr, 's'},
        {"preview-speed-mps", required_argument, nullptr, 'v'}, {"preview-dt", required_argument, nullptr, 'd'},
        {"preview-calibration", required_argument, nullptr, 'C'},
        {"pause-offline-during-inference", no_argument, nullptr, 'P'},
        {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'c': args.cameraConfig = optarg; break;
        case 't': args.trajectoryFile = optarg; break;
        case 'e': args.engineDir = optarg; break;
        case 'm': args.multimodalDir = optarg; break;
        case 'p': args.checkpointDir = optarg; break;
        case 'n':
        {
            std::string value(optarg);
            size_t parsed = 0;
            args.maxWindows = std::stoi(value, &parsed);
            if (parsed != value.size() || args.maxWindows <= 0)
            {
                throw std::invalid_argument("max-windows must be a positive integer");
            }
            break;
        }
        case 's': args.show = true; break;
        case 'P': args.pauseOfflineDuringInference = true; break;
        case 'v': previewSpeed = parseNumber(optarg); break;
        case 'd': previewDt = parseNumber(optarg); break;
        case 'C':
            args.calibrationFile = optarg;
            if (args.calibrationFile.empty())
            {
                throw std::invalid_argument("--preview-calibration requires a nonempty path");
            }
            break;
        case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
        default: throw std::invalid_argument("Invalid command line option");
        }
    }
    if (optind != argc || args.cameraConfig.empty() || args.trajectoryFile.empty() || args.engineDir.empty()
        || args.multimodalDir.empty())
    {
        throw std::invalid_argument("Missing required arguments");
    }
    if (previewSpeed.has_value() != previewDt.has_value() || (previewSpeed && (*previewSpeed < 0 || *previewDt <= 0)))
    {
        throw std::invalid_argument("Supply both --preview-speed-mps >= 0 and --preview-dt > 0");
    }
    if (previewSpeed)
    {
        args.preview = Alpaway::TrajectoryPreview{*previewSpeed, *previewDt};
    }
    if (!args.calibrationFile.empty() && (!args.show || !args.preview))
    {
        throw std::invalid_argument("--preview-calibration requires --show, --preview-speed-mps and --preview-dt");
    }
    return args;
}

Alpaway::CameraCalibration loadCalibration(std::string const& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("Cannot open camera calibration: " + path);
    }
    auto const json = nlohmann::json::parse(input);
    auto checkArray = [](nlohmann::json const& value, size_t size, std::string const& name) {
        if (!value.is_array() || value.size() != size)
        {
            throw std::invalid_argument(name + " has an invalid shape");
        }
    };
    checkArray(json.at("image_size"), 2, "image_size");
    checkArray(json.at("ego_to_camera_translation_m"), 3, "ego_to_camera_translation_m");
    for (auto const* name : {"intrinsic", "ego_to_camera_rotation"})
    {
        auto const& value = json.at(name);
        checkArray(value, 3, name);
        for (auto const& row : value)
        {
            checkArray(row, 3, name);
        }
    }
    auto const imageSize = json.at("image_size").get<std::array<int, 2>>();
    auto const intrinsic = json.at("intrinsic").get<std::array<std::array<double, 3>, 3>>();
    auto const rotation = json.at("ego_to_camera_rotation").get<std::array<std::array<double, 3>, 3>>();
    auto const translation = json.at("ego_to_camera_translation_m").get<std::array<double, 3>>();
    auto const model = json.at("distortion_model").get<std::string>();
    if (model != "pinhole" && model != "fisheye")
    {
        throw std::invalid_argument("distortion_model must be 'pinhole' or 'fisheye'");
    }
    auto matrix = [](std::array<std::array<double, 3>, 3> const& values) {
        return cv::Matx33d(values[0][0], values[0][1], values[0][2], values[1][0], values[1][1], values[1][2],
            values[2][0], values[2][1], values[2][2]);
    };
    return {{imageSize[0], imageSize[1]}, matrix(intrinsic), matrix(rotation),
        {translation[0], translation[1], translation[2]}, json.at("distortion").get<std::vector<double>>(),
        model == "fisheye", json.at("ground_z_m").get<double>()};
}

rt::imageUtils::ImageData makeImage(cv::Mat const& bgr)
{
    if (bgr.type() != CV_8UC3)
    {
        throw std::runtime_error("Camera must deliver 8-bit three-channel BGR frames");
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rt::Tensor tensor({1, rgb.rows, rgb.cols, 3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kUINT8, "cameraFrame");
    std::memcpy(tensor.dataPointer<unsigned char>(), rgb.data, rgb.total() * rgb.elemSize());
    return rt::imageUtils::ImageData(std::move(tensor));
}

rt::LLMGenerationRequest makeRequest(
    std::deque<std::vector<cameras::Frame>> const& history, vehicle::TrajectoryHistory const& trajectory)
{
    rt::LLMGenerationRequest request{};
    request.temperature = 1.0F;
    request.topP = 1.0F;
    request.topK = 50;
    request.maxGenerateLength = 128;
    rt::LLMGenerationRequest::Request item;
    rt::Message system;
    system.role = "system";
    system.contents.push_back({"text", "You are a driving assistant that generates safe and accurate actions."});
    rt::Message user;
    user.role = "user";
    for (auto const& frames : history)
    {
        for (auto const& frame : frames)
        {
            user.contents.push_back({"image", ""});
            item.imageBuffers.push_back(makeImage(frame.image));
        }
    }
    user.contents.push_back({"trajectory", ""});
    user.contents.push_back(
        {"text", "output the chain-of-thought reasoning of the driving process, then output the future trajectory."});
    item.pastTrajectory.emplace();
    for (auto const& point : trajectory.points())
    {
        item.pastTrajectory->emplace_back(point[0], point[1], point[2]);
    }
    item.messages.push_back(std::move(system));
    item.messages.push_back(std::move(user));
    request.requests.push_back(std::move(item));
    return request;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto const args = parseArgs(argc, argv);
        std::optional<Alpaway::CameraCalibration> calibration;
        if (!args.calibrationFile.empty())
        {
            calibration = loadCalibration(args.calibrationFile);
        }
        auto plugin = trt_edgellm::loadEdgellmPluginLib();
        if (!plugin)
        {
            throw std::runtime_error("Cannot load the TensorRT Edge LLM plugin");
        }
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        struct StreamGuard
        {
            cudaStream_t stream;
            ~StreamGuard()
            {
                cudaStreamDestroy(stream);
            }
        } streamGuard{stream};
        rt::LLMInferenceRuntime runtime(args.engineDir, args.multimodalDir,
            std::unordered_map<std::string, std::string>{}, stream, rt::ContextCacheConfig{}, args.checkpointDir);
        runtime.setActionNoiseSeed(5);
        if (!runtime.captureDecodingCUDAGraph(stream))
        {
            std::cerr << "Warning: CUDA graph capture failed; using normal execution\n";
        }
        cameras::CameraRig rig(args.cameraConfig);
        cameras::FrameSource& frameSource = rig;
        std::deque<std::vector<cameras::Frame>> history;
        auto nextSample = std::chrono::steady_clock::now();
        auto lastValid = nextSample;
        constexpr auto kINTERVAL = std::chrono::milliseconds(500);
        constexpr auto kMAX_AGE = std::chrono::milliseconds(250);
        constexpr auto kMAX_SKEW = std::chrono::milliseconds(100);
        constexpr auto kTIMEOUT = std::chrono::seconds(5);
        int completed = 0;
        Alpaway::Render renderer;
        cv::Mat display;
        bool haveResult = false;
        std::string const windowName = "Alpamayo | input, explanation and trajectory";

        if (args.show)
        {
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::resizeWindow(windowName, 1920, 1080);
        }
        while (args.maxWindows == 0 || completed < args.maxWindows)
        {
            if (args.show && !display.empty() && cv::waitKey(1) == 27)
            {
                break;
            }
            auto const now = std::chrono::steady_clock::now();
            if (now < nextSample)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            auto frames = frameSource.snapshot(kMAX_AGE, kMAX_SKEW);
            if (frames.empty())
            {
                if (now - lastValid > kTIMEOUT)
                {
                    throw std::runtime_error("Timed out waiting for fresh camera frames within the skew limit");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (frames.size() != 4)
            {
                throw std::runtime_error("Frame source must return exactly four cameras");
            }
            if (!history.empty())
            {
                bool duplicate = false;
                for (size_t i = 0; i < frames.size(); ++i)
                {
                    duplicate |= frames[i].sequence == history.back()[i].sequence;
                }
                if (duplicate)
                {
                    if (now - lastValid > kTIMEOUT)
                    {
                        throw std::runtime_error("Timed out waiting for new camera frames");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                if (now - nextSample > std::chrono::milliseconds(150))
                {
                    history.clear();
                }
            }
            lastValid = now;
            nextSample = now + kINTERVAL;
            history.push_back(std::move(frames));
            if (args.show && !haveResult)
            {
                display = renderer.dashboard(history.back().front().image, "Collecting four camera timestamps...", {},
                    args.preview, calibration);
                cv::imshow(windowName, display);
            }
            if (history.size() != 4)
            {
                continue;
            }
            if (args.pauseOfflineDuringInference)
            {
                rig.setOfflinePlaybackPaused(true);
            }
            vehicle::TrajectoryHistory trajectory;
            trajectory.loadJson(args.trajectoryFile);
            auto const wallNow = std::chrono::system_clock::now();
            auto const latestFrame = history.back().front().received;
            auto const frameTime = wallNow - (std::chrono::steady_clock::now() - latestFrame);
            auto const frameMs
                = std::chrono::duration_cast<std::chrono::milliseconds>(frameTime.time_since_epoch()).count();
            if (std::llabs(frameMs - trajectory.timestampMs()) > 1000)
            {
                throw std::runtime_error("Trajectory timestamp differs from the latest camera frame by over 1 second");
            }
            auto request = makeRequest(history, trajectory);
            rt::LLMGenerationResponse response;
            if (args.show)
            {
                display = renderer.dashboard(
                    history.back().front().image, "Inference running. Waiting for model output...", {}, args.preview,
                    calibration);
                cv::imshow(windowName, display);
                if (cv::waitKey(1) == 27)
                {
                    break;
                }
            }
            auto const inferenceStart = std::chrono::steady_clock::now();
            bool const inferenceSucceeded = runtime.handleRequest(request, response, stream);
            auto const inferenceEnd = std::chrono::steady_clock::now();
            constexpr std::array<char const*, 4> kCAMERA_ROLES{"front_wide", "front_tele", "cross_left", "cross_right"};
            std::ostringstream performance;
            performance << std::fixed << std::setprecision(2) << "inference_ms="
                        << std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count()
                        << " camera_capture_fps(input_window)";
            for (size_t i = 0; i < kCAMERA_ROLES.size(); ++i)
            {
                performance << " " << kCAMERA_ROLES[i] << "=";
                auto const fps = cameras::estimateCaptureFps(history.front()[i], history.back()[i]);
                if (fps)
                {
                    performance << *fps;
                }
                else
                {
                    performance << "unavailable";
                }
            }
            std::cerr << performance.str() << '\n';
            if (!inferenceSucceeded || response.outputTexts.size() != 1
                || response.outputTrajectories.size() != 1 || response.outputTrajectories[0].empty())
            {
                throw std::runtime_error("Alpamayo inference failed or returned no action trajectory");
            }
            nlohmann::json result{{"frame_timestamp_ms", frameMs},
                {"trajectory_timestamp_ms", trajectory.timestampMs()}, {"output_text", response.outputTexts[0]},
                {"output_trajectory", nlohmann::json::array()}};
            for (auto const& point : response.outputTrajectories[0])
            {
                if (!std::isfinite(point.first) || !std::isfinite(point.second))
                {
                    throw std::runtime_error("Action trajectory contains non-finite values");
                }
                result["output_trajectory"].push_back({point.first, point.second});
            }
            std::cout << result.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << std::endl;
            ++completed;
            if (args.show)
            {
                display = renderer.dashboard(history.back().front().image, response.outputTexts[0],
                    response.outputTrajectories[0], args.preview, calibration);
                haveResult = true;
                bool const finalWindow = args.maxWindows > 0 && completed >= args.maxWindows;
                if (finalWindow)
                {
                    renderer.drawText(display, {35, 92}, "Result ready - press any key to exit", Alpaway::YELLOW,
                        Alpaway::FONT_HERSHEY_SIMPLEX, 0.75, 2);
                }
                else if (args.pauseOfflineDuringInference)
                {
                    renderer.drawText(display, {35, 92}, "Press any key for next result; ESC to exit", Alpaway::YELLOW,
                        Alpaway::FONT_HERSHEY_SIMPLEX, 0.75, 2);
                }
                cv::imshow(windowName, display);
                if (cv::waitKey(finalWindow || args.pauseOfflineDuringInference ? 0 : 1) == 27)
                {
                    break;
                }
            }
            if (args.pauseOfflineDuringInference)
            {
                rig.setOfflinePlaybackPaused(false);
            }
            history.clear();
            nextSample = std::chrono::steady_clock::now();
            lastValid = nextSample;
        }
        if (args.show)
        {
            cv::destroyAllWindows();
        }
        return EXIT_SUCCESS;
    }
    catch (std::exception const& error)
    {
        std::cerr << "alpaway_real: " << error.what() << '\n';
        usage(argv[0]);
        return EXIT_FAILURE;
    }
}
