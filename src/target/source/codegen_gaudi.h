/*!
 * \file codegen_gaudi.h
 * \brief CodeGenGaudi: TIR -> TPC-C like C code (stub version)
 */

#ifndef TVM_TARGET_SOURCE_CODEGEN_GAUDI_H_
#define TVM_TARGET_SOURCE_CODEGEN_GAUDI_H_

#include <ostream>
#include <string>
#include <vector>

#include "codegen_c.h"

#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace codegen {

class CodeGenGaudi : public CodeGenC {
 public:
  CodeGenGaudi();
  
  void PrintFuncPrefix(std::ostream& os) override;
  void PrintType(DataType t, std::ostream& os) override;

  void VisitStmt_(const tir::ForNode* op) override;
  void VisitStmt_(const tir::BufferStoreNode* op) override;

  void VisitExpr_(const tir::BufferLoadNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::BroadcastNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::AddNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::SubNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::MulNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::DivNode* op, std::ostream& os) override;
  //void VisitExpr_(const tir::NegNode* op, std::ostream& os) override;

  // 可選：周報更好看（Clamp/Relu 常用）
  void VisitExpr_(const tir::MinNode* op, std::ostream& os) override;
  void VisitExpr_(const tir::MaxNode* op, std::ostream& os) override;

  void VisitExpr_(const tir::CastNode* op, std::ostream& os) override;
  void VisitStmt_(const tir::AssertStmtNode* op) override;


  void PreFunctionBody(const PrimFunc& f) override;
  void AddKernel(const std::string& kernel_name,
                 const std::vector<tvm::tir::Buffer>& arg_buffers,
                 const tvm::tir::Stmt& body);
  void EnableKernelDebug(bool v) { kernel_debug_ = v; }
  // Current tile loop iv name, empty if not in tile loop.
std::string tile0_var_;
// Optional: debug print
bool emit_index_space_ = true;
private:
  bool gaudi_intrin_emitted_{false};
  void EmitGaudiIntrinsicsOnce();
  static constexpr int kVecLanes = 64;
  bool MatchRampIndexF32x64(const PrimExpr& index, PrimExpr* out_base) const;
  bool kernel_debug_{false};
  bool in_kernel_emit_{false};

};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_GAUDI_H_
