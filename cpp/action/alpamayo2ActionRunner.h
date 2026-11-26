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

#pragma once

#include "action/actionKvBatch.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/kvCacheManager.h"
#include "runtime/llmRuntimeUtils.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm::rt
{

//! Runs one Alpamayo2 tree-action graph against an isolated copy of one LLM prefix.
//! This object is not safe for concurrent calls.
class Alpamayo2ActionRunner
{
public:
    Alpamayo2ActionRunner(std::string const& engineDir, cudaStream_t stream);

    //! Noise has shape [candidates, tokens, 2]; the result contains raw acceleration/curvature pairs.
    //! The caller must retain the prefix pages until this synchronous call returns.
    std::vector<std::vector<FutureTrajectoryPoint>> sampleTrajectory(cudaStream_t stream,
        KVCacheManager const& cache, ActionKvBatchView const& prefix, std::vector<float> const& noise);

private:
    int32_t mLayers{};
    int32_t mHeads{};
    int32_t mHeadDim{};
    int32_t mTokens{};
    int32_t mSteps{};
    int32_t mCandidates{};
    int32_t mSequences{};
    int32_t mBlock{};
    int32_t mCapacity{};
    int32_t mPrivatePages{};
    int32_t mPoolPages{};
    int32_t mTablePages{};
    int32_t mSectionH{};
    int32_t mSectionW{};
    float mTheta{};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    AuxStreamSet mAuxStreams;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::unique_ptr<KVPageTable> mPageTable;
    std::vector<Tensor> mPools;
    Tensor mNoiseHost, mNoiseDevice, mActionsHost, mActionsDevice;
    Tensor mLengthsHost, mLengthsDevice, mAttentionLengthsHost, mAttentionLengthsDevice;
    Tensor mPositionsHost, mPositionsDevice, mRopePositionsHost, mRopePositionsDevice, mRopeDevice;
    Tensor mPhaseDevice, mContextCountDevice;
};

} // namespace trt_edgellm::rt
