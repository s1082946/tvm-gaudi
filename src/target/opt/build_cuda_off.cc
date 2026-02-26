/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Optional module when build cuda is switched to off
 */
#include "../../runtime/cuda/cuda_module.h"
#include "../source/codegen_gaudi.h"
#include "../source/codegen_source_base.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace runtime {

ffi::Module CUDAModuleCreate(std::string data, std::string fmt,
                             std::unordered_map<std::string, FunctionInfo> fmap,
                             std::string cuda_source) {
  LOG(FATAL) << "CUDA is not enabled";
  TVM_FFI_UNREACHABLE();
}
}  // namespace runtime

namespace codegen {

ffi::Module BuildGaudi(IRModule mod, Target target) {
  (void)target;
  CodeGenGaudi cg;
  cg.Init(/*output_ssa=*/false);
  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) continue;
    cg.DeclareFunction(kv.first, Downcast<PrimFunc>(kv.second));
  }
  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) continue;
    cg.AddFunction(kv.first, Downcast<PrimFunc>(kv.second));
  }
  std::string src = cg.Finish();
  return CSourceModuleCreate(src, "tpc-c", {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.gaudi", BuildGaudi);
}

}  // namespace codegen
}  // namespace tvm
