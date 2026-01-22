/*!
 *  Optional module when build cuda is switched to off
 */

#include "../../runtime/cuda/cuda_module.h"
#include "../source/codegen_source_base.h"
#include "../source/codegen_gaudi.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt.h>

#include <string>
#include <vector>
#include <cstdlib>   // getenv, atoi
namespace tvm {
namespace codegen {

using namespace tvm::ffi;
[[maybe_unused]] static tvm::tir::Stmt FindKernelLikeStmtByFirstStore(const tvm::tir::Stmt& body) {
  using namespace tvm::tir;

  struct Frame {
    Stmt cur;
    Stmt nearest_loop_or_block;  
  };

  std::vector<Frame> st;
  st.push_back(Frame{body, Stmt()});

  while (!st.empty()) {
    Frame fr = st.back();
    st.pop_back();
    Stmt cur = fr.cur;
    if (!cur.defined()) continue;
    if (cur.as<BufferStoreNode>()) {
      if (fr.nearest_loop_or_block.defined()) return fr.nearest_loop_or_block;
      return cur;
    }

    Stmt next_loop_or_block = fr.nearest_loop_or_block;
    if (cur.as<ForNode>() || cur.as<BlockRealizeNode>() || cur.as<BlockNode>()) {
      next_loop_or_block = cur;  
    }
    if (const auto* seq = cur.as<SeqStmtNode>()) {
      for (int i = static_cast<int>(seq->seq.size()) - 1; i >= 0; --i) {
        st.push_back(Frame{seq->seq[i], next_loop_or_block});
      }
      continue;
    }

    if (const auto* let = cur.as<LetStmtNode>()) {
      st.push_back(Frame{let->body, next_loop_or_block});
      continue;
    }

    if (const auto* asrt = cur.as<AssertStmtNode>()) {
      st.push_back(Frame{asrt->body, next_loop_or_block});
      continue;
    }

    if (const auto* attr = cur.as<AttrStmtNode>()) {
      st.push_back(Frame{attr->body, next_loop_or_block});
      continue;
    }

    if (const auto* ifs = cur.as<IfThenElseNode>()) {
      if (ifs->else_case.defined()) {
        st.push_back(Frame{ifs->else_case.value(), next_loop_or_block});
      }
      if (ifs->then_case.defined()) {
        st.push_back(Frame{ifs->then_case, next_loop_or_block});
      }
      continue;
    }

    if (const auto* alloc = cur.as<AllocateNode>()) {
      st.push_back(Frame{alloc->body, next_loop_or_block});
      continue;
    }

    if (const auto* loop = cur.as<ForNode>()) {
      st.push_back(Frame{loop->body, next_loop_or_block});
      continue;
    }

    if (const auto* blk = cur.as<BlockNode>()) {
      st.push_back(Frame{blk->body, next_loop_or_block});
      continue;
    }

    if (const auto* br = cur.as<BlockRealizeNode>()) {
      st.push_back(Frame{br->block->body, next_loop_or_block});
      continue;
    }

    if (cur.as<EvaluateNode>()) {
      continue;
    }
  }

  return Stmt();  // undefined
}



ffi::Module BuildGaudi(IRModule mod, Target target) {
  (void)target;

  IRModule mod_orig = mod;

  Map<GlobalVar, PrimFunc> funcs;
  for (auto kv : mod_orig->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) continue;
    funcs.Set(kv.first, Downcast<PrimFunc>(kv.second));
  }

  // -------- wrapper-only codegen (stable, but includes TVM runtime wrapper noise) --------
  CodeGenGaudi cg_wrap;
  cg_wrap.Init(/*output_ssa=*/false);

  for (auto kv : funcs) cg_wrap.DeclareFunction(kv.first, kv.second);
  for (auto kv : funcs) cg_wrap.AddFunction(kv.first, kv.second);

  std::string src = cg_wrap.Finish();
  return CSourceModuleCreate(src, "tpc-c", {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.gaudi", BuildGaudi);
}

}  // namespace codegen
}  // namespace tvm
