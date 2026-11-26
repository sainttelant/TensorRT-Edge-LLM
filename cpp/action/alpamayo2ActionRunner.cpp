/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "action/alpamayo2ActionRunner.h"

#include "common/checkMacros.h"
#include "common/executionPhase.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/trtInterop.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace trt_edgellm::rt
{

namespace
{
struct DrainOnFailure
{
    cudaStream_t stream;
    int uncaught{std::uncaught_exceptions()};
    ~DrainOnFailure()
    {
        if (std::uncaught_exceptions() > uncaught)
        {
            cudaError_t const status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess)
            {
                LOG_ERROR("Alpamayo2 action stream failed to drain: %s", cudaGetErrorString(status));
            }
        }
    }
};
} // namespace

Alpamayo2ActionRunner::Alpamayo2ActionRunner(std::string const& engineDir, cudaStream_t stream)
{
    ELLM_CHECK(stream != nullptr, "Alpamayo2 requires an explicit CUDA stream");
    DrainOnFailure drain{stream};
    std::ifstream input(engineDir + "/config.json");
    ELLM_CHECK(input.is_open(), "Cannot open Alpamayo2 action config");
    auto const config = nlohmann::json::parse(input);
    ELLM_CHECK(config.at("graph") == "host/action_graph.py" && config.at("attention") == "tree_plugin",
        "Alpamayo2 requires the tree action graph");
    ELLM_CHECK(config.at("kv_cache") == "fp16" && config.at("gemm") == "fp16",
        "Alpamayo2 action runner currently supports only FP16 KV and GEMM");
    mLayers = config.at("num_hidden_layers").get<int32_t>();
    mHeads = config.at("num_key_value_heads").get<int32_t>();
    mHeadDim = config.at("head_dim").get<int32_t>();
    mTokens = config.at("tokens").get<int32_t>();
    mSteps = config.at("steps").get<int32_t>();
    mCandidates = config.at("candidates").get<int32_t>();
    int32_t perSequence = config.at("candidates_per_sequence").get<int32_t>();
    mCapacity = config.at("capacity").get<int32_t>();
    mTheta = config.at("rope_theta").get<float>();
    auto const section = config.at("mrope_section").get<std::vector<int32_t>>();
    ELLM_CHECK(mLayers > 0 && mLayers <= 256 && mHeads > 0 && mHeads <= 256 && mHeadDim > 0 && mHeadDim <= 512
            && mHeadDim % 2 == 0 && mTokens > 0 && mSteps > 0 && mSteps <= 64 && mCandidates > 0
            && mCandidates <= 128 && perSequence > 0 && perSequence <= mCandidates && mCapacity > 0
            && mCapacity <= 64 * kTOKENS_PER_PAGE && std::isfinite(mTheta) && mTheta > 0
            && section.size() == 3 && section[1] > 0 && section[2] > 0,
        "Invalid Alpamayo2 action configuration");
    while (mCandidates % perSequence != 0)
    {
        --perSequence;
    }
    mSectionH = section[1];
    mSectionW = section[2];
    mSequences = mCandidates / perSequence;
    ELLM_CHECK(static_cast<int64_t>(perSequence) * mTokens <= 128
            && static_cast<int64_t>(mCandidates) * mTokens <= INT32_MAX,
        "Alpamayo2 action token dimensions exceed the tree attention limits");
    mBlock = perSequence * mTokens;
    mPrivatePages = (mBlock + kTOKENS_PER_PAGE - 2) / kTOKENS_PER_PAGE + 2;
    mTablePages = computeMaxPagesPerSeq(mCapacity) + mPrivatePages;
    mPoolPages = computeMaxPagesPerSeq(mCapacity) + mSequences * mPrivatePages;

    mRuntime.reset(nvinfer1::createInferRuntime(TrtLogger::instance()));
    ELLM_CHECK(mRuntime != nullptr, "Cannot create Alpamayo2 TensorRT runtime");
    mEngine = deserializeCudaEngineFromFile(*mRuntime, engineDir + "/action.engine");
    ELLM_CHECK(mEngine != nullptr, "Cannot deserialize Alpamayo2 action engine");
    auto checkTensor = [&](std::string const& name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type,
                           std::vector<int64_t> const& expected) {
        ELLM_CHECK(mEngine->getTensorIOMode(name.c_str()) == mode && mEngine->getTensorDataType(name.c_str()) == type,
            "Alpamayo2 tensor type or mode mismatch: " + name);
        auto const dims = mEngine->getTensorShape(name.c_str());
        ELLM_CHECK(dims.nbDims == static_cast<int32_t>(expected.size()), "Alpamayo2 tensor rank mismatch: " + name);
        for (int32_t i = 0; i < dims.nbDims; ++i)
        {
            ELLM_CHECK(expected[static_cast<size_t>(i)] == -1 || dims.d[i] == expected[static_cast<size_t>(i)],
                "Alpamayo2 tensor shape mismatch: " + name);
        }
    };
    using Mode = nvinfer1::TensorIOMode;
    using Type = nvinfer1::DataType;
    int64_t const rows = static_cast<int64_t>(mCandidates) * mTokens;
    checkTensor("noise", Mode::kINPUT, Type::kFLOAT, {rows, 2});
    checkTensor("rope", Mode::kINPUT, Type::kFLOAT, {mTokens, mHeadDim});
    checkTensor("past_lengths", Mode::kINPUT, Type::kINT32, {mSequences});
    checkTensor("attention_sequence_lengths", Mode::kINPUT, Type::kINT32, {mSequences});
    checkTensor("attention_pos_id", Mode::kINPUT, Type::kINT32, {rows});
    checkTensor("actions", Mode::kOUTPUT, Type::kFLOAT, {rows, 2});
    checkTensor("kv_page_table", Mode::kINPUT, Type::kINT32, {mSequences, 2, -1});
    checkTensor("execution_phase_marker", Mode::kINPUT, Type::kINT32, {-1});
    checkTensor("context_sequence_count_carrier", Mode::kINPUT, Type::kINT32, {-1});
    mContext.reset(mEngine->createExecutionContext());
    ELLM_CHECK(mContext != nullptr, "Cannot create Alpamayo2 action execution context");
    ELLM_CHECK(mContext->setOptimizationProfileAsync(0, stream), "Cannot select Alpamayo2 action profile");
    setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);

    auto shape = [&](char const* name, nvinfer1::Dims const& dims) {
        ELLM_CHECK(mContext->setInputShape(name, dims), std::string("Invalid Alpamayo2 input shape: ") + name);
    };
    shape("kv_page_table", {3, {mSequences, 2, mTablePages}});
    shape("execution_phase_marker", {1, {static_cast<int32_t>(ExecutionPhase::kSpecTargetVerify)}});
    shape("context_sequence_count_carrier", {1, {0}});
    for (int32_t layer = 0; layer < mLayers; ++layer)
    {
        auto const name = "pool_" + std::to_string(layer);
        checkTensor(name, Mode::kINPUT, Type::kHALF, {2, -1, kTOKENS_PER_PAGE, mHeads, mHeadDim});
        for (int32_t step = 0; step < mSteps; ++step)
        {
            auto const outputName = "present_" + std::to_string(step) + "_" + std::to_string(layer);
            checkTensor(outputName, Mode::kOUTPUT, Type::kHALF, {2, -1, kTOKENS_PER_PAGE, mHeads, mHeadDim});
        }
        shape(name.c_str(), {5, {2, mPoolPages, kTOKENS_PER_PAGE, mHeads, mHeadDim}});
        mPools.emplace_back(Coords{2, mPoolPages, kTOKENS_PER_PAGE, mHeads, mHeadDim}, DeviceType::kGPU,
            DataType::kHALF, "Alpamayo2ActionRunner::pool");
    }
    for (int32_t layer = 0; layer < mLayers; ++layer)
    {
        for (int32_t step = 0; step < mSteps; ++step)
        {
            auto const outputName = "present_" + std::to_string(step) + "_" + std::to_string(layer);
            auto const dims = mContext->getTensorShape(outputName.c_str());
            ELLM_CHECK(dims.nbDims == 5 && dims.d[0] == 2 && dims.d[1] == mPoolPages
                    && dims.d[2] == kTOKENS_PER_PAGE && dims.d[3] == mHeads && dims.d[4] == mHeadDim,
                "Alpamayo2 action output pool shape mismatch: " + outputName);
        }
    }
    mPageTable = std::make_unique<KVPageTable>(mSequences, mTablePages, mPoolPages);
    mNoiseHost = Tensor({rows, 2}, DeviceType::kCPU, DataType::kFLOAT);
    mNoiseDevice = Tensor({rows, 2}, DeviceType::kGPU, DataType::kFLOAT);
    mActionsHost = Tensor({rows, 2}, DeviceType::kCPU, DataType::kFLOAT);
    mActionsDevice = Tensor({rows, 2}, DeviceType::kGPU, DataType::kFLOAT);
    mLengthsHost = Tensor({mSequences}, DeviceType::kCPU, DataType::kINT32);
    mLengthsDevice = Tensor({mSequences}, DeviceType::kGPU, DataType::kINT32);
    mAttentionLengthsHost = Tensor({mSequences}, DeviceType::kCPU, DataType::kINT32);
    mAttentionLengthsDevice = Tensor({mSequences}, DeviceType::kGPU, DataType::kINT32);
    mPositionsHost = Tensor({rows}, DeviceType::kCPU, DataType::kINT32);
    mPositionsDevice = Tensor({rows}, DeviceType::kGPU, DataType::kINT32);
    mRopePositionsHost = Tensor({3, mTokens}, DeviceType::kCPU, DataType::kINT64);
    mRopePositionsDevice = Tensor({3, mTokens}, DeviceType::kGPU, DataType::kINT64);
    mRopeDevice = Tensor({mTokens, mHeadDim}, DeviceType::kGPU, DataType::kFLOAT);
    mPhaseDevice = Tensor({static_cast<int32_t>(ExecutionPhase::kSpecTargetVerify)}, DeviceType::kGPU, DataType::kINT32);
    mContextCountDevice = Tensor({1}, DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemsetAsync(mPhaseDevice.rawPointer(), 0, mPhaseDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<std::vector<FutureTrajectoryPoint>> Alpamayo2ActionRunner::sampleTrajectory(cudaStream_t stream,
    KVCacheManager const& cache, ActionKvBatchView const& prefix, std::vector<float> const& noise)
{
    ELLM_CHECK(stream != nullptr, "Alpamayo2 requires an explicit CUDA stream");
    DrainOnFailure drain{stream};
    ELLM_CHECK(prefix.batchSize == 1 && prefix.mropeDeltas.size() == 1
            && prefix.kvLengthsHost.getDeviceType() == DeviceType::kCPU
            && prefix.kvLengthsHost.getDataType() == DataType::kINT32 && prefix.kvLengthsHost.getShape().volume() >= 1,
        "Alpamayo2 requires one retained LLM prefix");
    ELLM_CHECK(cache.numLayers() == mLayers && !cache.hasReducedKVCache()
            && cache.getConfig().kvCacheType == DataType::kHALF,
        "Alpamayo2 requires a full FP16 LLM KV cache with matching layers");
    ELLM_CHECK(noise.size() == static_cast<size_t>(mCandidates) * mTokens * 2,
        "Alpamayo2 noise must have shape [candidates, tokens, 2]");
    ELLM_CHECK(std::all_of(noise.begin(), noise.end(), [](float value) { return std::isfinite(value); }),
        "Alpamayo2 noise must contain finite values");
    int32_t const length = prefix.kvLengthsHost.dataPointer<int32_t>()[0];
    ELLM_CHECK(length > 0 && length <= mCapacity, "Alpamayo2 prefix length exceeds the action graph capacity");
    ELLM_CHECK(prefix.mropeDeltas[0] <= std::numeric_limits<int64_t>::max() - length - mTokens,
        "Alpamayo2 RoPE position exceeds the int64 range");
    int32_t const fullPages = length / kTOKENS_PER_PAGE;
    int32_t const remainder = length % kTOKENS_PER_PAGE;
    ELLM_CHECK(fullPages + (remainder != 0 ? 1 : 0) <= prefix.pageTable.maxPagesPerSeq()
            && prefix.pageTable.numPages() == cache.numPages(),
        "Alpamayo2 prefix page table does not match the LLM KV pool");
    int32_t const* sourcePages = prefix.pageTable.hostRow(0);
    for (int32_t page = 0; page < fullPages + (remainder != 0 ? 1 : 0); ++page)
    {
        ELLM_CHECK(sourcePages[page] >= 0 && sourcePages[page] < cache.numPages(),
            "Alpamayo2 prefix contains an invalid page");
    }
    std::vector<std::vector<int32_t>> rows(static_cast<size_t>(mSequences));
    for (int32_t seq = 0; seq < mSequences; ++seq)
    {
        auto& row = rows[static_cast<size_t>(seq)];
        for (int32_t page = 0; page < fullPages; ++page)
        {
            row.push_back(page);
        }
        for (int32_t page = 0; page < mPrivatePages; ++page)
        {
            row.push_back(computeMaxPagesPerSeq(mCapacity) + seq * mPrivatePages + page);
        }
        mPageTable->setRow(seq, row.data(), static_cast<int32_t>(row.size()));
    }
    mPageTable->upload(stream);

    size_t const pageBytes = static_cast<size_t>(kTOKENS_PER_PAGE) * mHeads * mHeadDim * sizeof(uint16_t);
    for (int32_t layer = 0; layer < mLayers; ++layer)
    {
        auto const& source = cache.getCombinedKVCache(layer);
        auto const geometry = source.getShape();
        ELLM_CHECK(geometry.getNumDims() == 5 && geometry[0] == 2 && geometry[1] == cache.numPages()
                && geometry[2] == kTOKENS_PER_PAGE && geometry[3] == mHeads && geometry[4] == mHeadDim,
            "Alpamayo2 LLM KV pool geometry does not match the action graph");
        auto const* src = static_cast<char const*>(source.rawPointer());
        auto* dst = static_cast<char*>(mPools[static_cast<size_t>(layer)].rawPointer());
        for (int32_t plane = 0; plane < 2; ++plane)
        {
            for (int32_t page = 0; page < fullPages; ++page)
            {
                CUDA_CHECK(cudaMemcpyAsync(dst + (static_cast<size_t>(plane) * mPoolPages + page) * pageBytes,
                    src + (static_cast<size_t>(plane) * cache.numPages() + sourcePages[page]) * pageBytes,
                    pageBytes, cudaMemcpyDeviceToDevice, stream));
            }
            if (remainder != 0)
            {
                for (int32_t seq = 0; seq < mSequences; ++seq)
                {
                    int32_t const target = computeMaxPagesPerSeq(mCapacity) + seq * mPrivatePages;
                    CUDA_CHECK(cudaMemcpyAsync(dst + (static_cast<size_t>(plane) * mPoolPages + target) * pageBytes,
                        src + (static_cast<size_t>(plane) * cache.numPages() + sourcePages[fullPages]) * pageBytes,
                        static_cast<size_t>(remainder) * mHeads * mHeadDim * sizeof(uint16_t),
                        cudaMemcpyDeviceToDevice, stream));
                }
            }
        }
    }

    std::memcpy(mNoiseHost.rawPointer(), noise.data(), noise.size() * sizeof(float));
    for (int32_t seq = 0; seq < mSequences; ++seq)
    {
        mLengthsHost.dataPointer<int32_t>()[seq] = length;
        mAttentionLengthsHost.dataPointer<int32_t>()[seq] = length + mBlock;
        for (int32_t i = 0; i < mBlock; ++i)
        {
            mPositionsHost.dataPointer<int32_t>()[seq * mBlock + i] = length + i;
        }
    }
    for (int32_t dim = 0; dim < 3; ++dim)
    {
        for (int32_t i = 0; i < mTokens; ++i)
        {
            mRopePositionsHost.dataPointer<int64_t>()[dim * mTokens + i] = prefix.mropeDeltas[0] + length + i;
        }
    }
    auto upload = [&](Tensor& device, Tensor const& host) {
        CUDA_CHECK(cudaMemcpyAsync(device.rawPointer(), host.rawPointer(), host.getMemoryCapacity(),
            cudaMemcpyHostToDevice, stream));
    };
    upload(mNoiseDevice, mNoiseHost);
    upload(mLengthsDevice, mLengthsHost);
    upload(mAttentionLengthsDevice, mAttentionLengthsHost);
    upload(mPositionsDevice, mPositionsHost);
    upload(mRopePositionsDevice, mRopePositionsHost);
    kernel::initializeMRopeCosSin(mRopeDevice.dataPointer<float>(), mRopePositionsDevice.dataPointer<int64_t>(),
        mTheta, mHeadDim, mTokens, 1, true, mSectionH, mSectionW, stream);

    auto bind = [&](std::string const& name, void* ptr) {
        ELLM_CHECK(mContext->setTensorAddress(name.c_str(), ptr), "Cannot bind Alpamayo2 tensor " + name);
    };
    bind("noise", mNoiseDevice.rawPointer());
    bind("rope", mRopeDevice.rawPointer());
    bind("past_lengths", mLengthsDevice.rawPointer());
    bind("kv_page_table", mPageTable->kernelView().rawPointer());
    bind("attention_pos_id", mPositionsDevice.rawPointer());
    bind("attention_sequence_lengths", mAttentionLengthsDevice.rawPointer());
    bind("execution_phase_marker", mPhaseDevice.rawPointer());
    bind("context_sequence_count_carrier", mContextCountDevice.rawPointer());
    bind("actions", mActionsDevice.rawPointer());
    for (int32_t layer = 0; layer < mLayers; ++layer)
    {
        void* pool = mPools[static_cast<size_t>(layer)].rawPointer();
        bind("pool_" + std::to_string(layer), pool);
        for (int32_t step = 0; step < mSteps; ++step)
        {
            bind("present_" + std::to_string(step) + "_" + std::to_string(layer), pool);
        }
    }
    ELLM_CHECK(mContext->enqueueV3(stream), "Alpamayo2 action enqueue failed");
    CUDA_CHECK(cudaMemcpyAsync(mActionsHost.rawPointer(), mActionsDevice.rawPointer(), mActionsHost.getMemoryCapacity(),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::vector<FutureTrajectoryPoint>> result(static_cast<size_t>(mCandidates));
    auto const* values = mActionsHost.dataPointer<float>();
    for (int32_t candidate = 0; candidate < mCandidates; ++candidate)
    {
        auto& trajectory = result[static_cast<size_t>(candidate)];
        trajectory.reserve(static_cast<size_t>(mTokens));
        for (int32_t token = 0; token < mTokens; ++token)
        {
            size_t const offset = (static_cast<size_t>(candidate) * mTokens + token) * 2;
            ELLM_CHECK(std::isfinite(values[offset]) && std::isfinite(values[offset + 1]),
                "Alpamayo2 action engine returned non-finite values");
            trajectory.emplace_back(values[offset], values[offset + 1]);
        }
    }
    return result;
}

} // namespace trt_edgellm::rt
