/*!
 * \file codegen_gaudi.h
 * \brief CodeGenGaudi: TIR -> TPC-C kernel code for Habana Gaudi.
 *
 * Produces real TPC-C code using native types (int5, float64) and
 * tensor intrinsics (v_f32_ld_tnsr_b, v_f32_st_tnsr, v_f32_add_b, etc.).
 * Targets tpc-clang for compilation on Gaudi2.
 */

#ifndef TVM_TARGET_SOURCE_CODEGEN_GAUDI_H_
#define TVM_TARGET_SOURCE_CODEGEN_GAUDI_H_

#include <ostream>
#include <string>
#include <unordered_set>

#include "codegen_c.h"

#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace codegen {

class CodeGenGaudi : public CodeGenC {
 public:
  CodeGenGaudi();

  // Emit "void main(tensor A, tensor B, ...)" — always "void main" for TPC-C
  void PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                              std::ostream& os) override;

  // Inject index space preamble + coords variable at function entry
  void PreFunctionBody(const PrimFunc& f) override;

  // TPC-C native types: float64 (float32x64), int5, etc.
  void PrintType(DataType t, std::ostream& os) override;

  // Thread extents → for loops over TPC index space dimensions
  void VisitStmt_(const tir::AttrStmtNode* op) override;

  // TPC tensor intrinsics: v_f32_ld_tnsr_b / v_f32_st_tnsr
  void VisitExpr_(const tir::BufferLoadNode* op, std::ostream& os) override;
  void VisitStmt_(const tir::BufferStoreNode* op) override;

  // Vector ops → TPC intrinsics (v_f32_add_b, v_f32_max_b, v_f32_mov_b, ...)
  void VisitExpr_(const tir::BroadcastNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::AddNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::SubNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::MulNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::DivNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::MaxNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::MinNode* op, std::ostream& os) override;

  // Skip assert statements (not supported in TPC-C)
  void VisitStmt_(const tir::AssertStmtNode* op) override;

 private:
  // Var nodes corresponding to TPC tensor parameters (handle type)
  std::unordered_set<const tir::VarNode*> tensor_buffers_;

  static constexpr int kVecLanes = 64;

  // Maps a thread tag (e.g. "threadIdx.x") to a TPC dimension index (0-4).
  // Returns -1 for unknown tags.
  int TagToDim(const std::string& tag) const;

  // Returns the loop stride for a dimension (64 for dim 0 / depth, 1 otherwise).
  int DimStride(int dim) const;

  // Returns a human-readable name for a dimension ("depth", "width", etc.).
  std::string DimName(int dim) const;

  // Emit a binary vector intrinsic: intrin(a, b)
  void EmitBinaryVec(const std::string& intrin, const PrimExpr& a, const PrimExpr& b,
                     std::ostream& os);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_GAUDI_H_
