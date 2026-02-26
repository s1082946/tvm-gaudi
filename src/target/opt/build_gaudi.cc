/*!
 * \file build_gaudi.cc
 * \brief Build function for Gaudi/TPC-C target.
 *
 * Registers target.build.gaudi which lowers a TIR IRModule to a
 * CSourceModule containing TPC-C-style pseudo-C code.
 */

#include "../source/codegen_gaudi.h"
#include "../source/codegen_source_base.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace codegen {

ffi::Module BuildGaudi(IRModule mod, Target target) {
  (void)target;

  CodeGenGaudi cg;
  cg.Init(/*output_ssa=*/false);

  for (auto& kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) continue;
    cg.DeclareFunction(kv.first, Downcast<PrimFunc>(kv.second));
  }
  for (auto& kv : mod->functions) {
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
