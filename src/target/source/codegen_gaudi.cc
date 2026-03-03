/*!
 * \file codegen_gaudi.cc
 * \brief TPC-C code generator for Habana Gaudi accelerators.
 *
 * Generates TPC-C kernel source code from TVM TIR.
 * Uses native TPC-C types (int5, float64) and intrinsics
 * (v_f32_ld_tnsr_b, v_f32_st_tnsr, v_f32_add_b, v_f32_mov_b, etc.)
 * for compilation with tpc-clang on Gaudi2.
 */

#include "codegen_gaudi.h"

#include <tvm/tir/stmt_functor.h>

#include <string>

namespace tvm {
namespace codegen {

using namespace tir;

CodeGenGaudi::CodeGenGaudi() {
  // TPC-C does not use the C "restrict" keyword
  restrict_keyword_ = "";
}

// ---------------------------------------------------------------------------
// Function signature: void main(tensor A, tensor B, tensor C)
// ---------------------------------------------------------------------------

void CodeGenGaudi::PrintFunctionSignature(const ffi::String& /*function_name*/,
                                          const PrimFunc& func, std::ostream& os) {
  // TPC-C kernel entry point is always "void main"
  os << "void main(";

  // Reset tensor tracking for this function
  tensor_buffers_.clear();

  for (size_t i = 0; i < func->params.size(); ++i) {
    tir::Var v = func->params[i];
    if (i > 0) os << ", ";

    if (v.dtype().is_handle()) {
      // Handle params are TPC tensor descriptors
      os << "tensor " << AllocVarID(v.get());
      tensor_buffers_.insert(v.get());
    } else {
      PrintType(GetType(v), os);
      os << " " << AllocVarID(v.get());
    }
  }

  os << ")";

  // Register handle element types so the base class can resolve them in
  // GetType() calls during body emission.
  for (const auto& param : func->params) {
    if (auto* ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(param.get(), prim->dtype);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// PreFunctionBody: inject TPC index space initialization + coords variable
// ---------------------------------------------------------------------------

void CodeGenGaudi::PreFunctionBody(const PrimFunc& /*f*/) {
  PrintIndent();
  stream << "const int5 index_space_start = get_index_space_offset();\n";
  PrintIndent();
  stream << "const int5 index_space_end = get_index_space_size() + index_space_start;\n";
  PrintIndent();
  stream << "int5 coords = {0, 0, 0, 0, 0};\n";
}

// ---------------------------------------------------------------------------
// Type printing: TPC-C native SIMD types
// ---------------------------------------------------------------------------

void CodeGenGaudi::PrintType(DataType t, std::ostream& os) {
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "TPC: no vector of handles";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }

  int lanes = t.lanes();

  if (t.is_float()) {
    switch (t.bits()) {
      case 32:
        // 64 lanes → float64 (TPC native SIMD width for float32)
        os << (lanes == 1 ? "float" : "float64");
        return;
      case 16:
        os << (lanes == 1 ? "half" : "bfloat128");
        return;
      default:
        LOG(FATAL) << "TPC: unsupported float width " << t.bits();
    }
  }

  if (t.is_int()) {
    switch (t.bits()) {
      case 32:
        os << (lanes == 1 ? "int" : "int64");
        return;
      case 16:
        os << "short";
        return;
      case 8:
        os << "char";
        return;
      case 1:
        os << "bool";
        return;
      default:
        LOG(FATAL) << "TPC: unsupported int width " << t.bits();
    }
  }

  if (t.is_uint()) {
    switch (t.bits()) {
      case 32:
        os << (lanes == 1 ? "unsigned int" : "uint64");
        return;
      case 16:
        os << "unsigned short";
        return;
      case 8:
        os << "unsigned char";
        return;
      case 1:
        os << "bool";
        return;
      default:
        LOG(FATAL) << "TPC: unsupported uint width " << t.bits();
    }
  }

  LOG(FATAL) << "TPC: unknown type " << t;
}

// ---------------------------------------------------------------------------
// Dimension helpers
// ---------------------------------------------------------------------------

int CodeGenGaudi::TagToDim(const std::string& tag) const {
  if (tag == "threadIdx.x" || tag == "tpc.index_space.0") return 0;
  if (tag == "threadIdx.y" || tag == "tpc.index_space.1") return 1;
  if (tag == "threadIdx.z" || tag == "tpc.index_space.2") return 2;
  if (tag == "blockIdx.x" || tag == "tpc.index_space.3") return 3;
  if (tag == "blockIdx.y" || tag == "tpc.index_space.4") return 4;
  return -1;
}

int CodeGenGaudi::DimStride(int dim) const {
  // Dim 0 ("depth") is the vectorized dimension: each index-space unit = 64 elements
  return (dim == 0) ? kVecLanes : 1;
}

std::string CodeGenGaudi::DimName(int dim) const {
  switch (dim) {
    case 0:
      return "depth";
    case 1:
      return "width";
    case 2:
      return "height";
    case 3:
      return "batch";
    default:
      return "dim" + std::to_string(dim);
  }
}

// ---------------------------------------------------------------------------
// Thread extent → TPC index-space for loop
//
// threadIdx.y extent 1024 →
//   const int widthStart = index_space_start[1];
//   const int widthEnd   = index_space_end[1];
//   for (int i = widthStart; i < widthEnd; i += 1) {
//     coords[1] = i;
//     <body>
//   }
//
// threadIdx.x extent 16 (16 tiles × 64 = 1024 elements) →
//   const int depthStart = index_space_start[0] * 64;
//   const int depthEnd   = index_space_end[0]   * 64;
//   for (int j = depthStart; j < depthEnd; j += 64) {
//     coords[0] = j;
//     <body>
//   }
// ---------------------------------------------------------------------------

void CodeGenGaudi::VisitStmt_(const tir::AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    std::string tag = iv->thread_tag;
    int dim = TagToDim(tag);
    ICHECK_GE(dim, 0) << "TPC: unsupported thread tag '" << tag << "'";

    int stride = DimStride(dim);
    std::string name = DimName(dim);
    std::string loop_var = AllocVarID(iv->var.get());
    std::string start_var = name + "Start";
    std::string end_var = name + "End";

    // Emit start/end constants
    PrintIndent();
    stream << "const int " << start_var << " = index_space_start[" << dim << "]";
    if (stride > 1) stream << " * " << stride;
    stream << ";\n";

    PrintIndent();
    stream << "const int " << end_var << " = index_space_end[" << dim << "]";
    if (stride > 1) stream << " * " << stride;
    stream << ";\n";

    // Emit for loop
    PrintIndent();
    stream << "for (int " << loop_var << " = " << start_var << "; " << loop_var << " < "
           << end_var << "; " << loop_var << " += " << stride << ") {\n";

    int scope = this->BeginScope();

    // Update coords for this dimension
    PrintIndent();
    stream << "coords[" << dim << "] = " << loop_var << ";\n";

    this->PrintStmt(op->body);

    this->EndScope(scope);
    PrintIndent();
    stream << "}\n";
    return;
  }

  // For all other attribute statements, use the base class handler
  CodeGenC::VisitStmt_(op);
}

// ---------------------------------------------------------------------------
// Buffer load: v_f32_ld_tnsr_b(coords, tensor_name)
// ---------------------------------------------------------------------------

void CodeGenGaudi::VisitExpr_(const tir::BufferLoadNode* op, std::ostream& os) {
  Var buffer_var = op->buffer->data;
  if (tensor_buffers_.count(buffer_var.get())) {
    os << "v_f32_ld_tnsr_b(coords, " << GetVarID(buffer_var.get()) << ")";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

// ---------------------------------------------------------------------------
// Buffer store: v_f32_st_tnsr(coords, tensor_name, value)
// ---------------------------------------------------------------------------

void CodeGenGaudi::VisitStmt_(const tir::BufferStoreNode* op) {
  Var buffer_var = op->buffer->data;
  if (tensor_buffers_.count(buffer_var.get())) {
    std::string value = PrintExpr(op->value);
    PrintIndent();
    stream << "v_f32_st_tnsr(coords, " << GetVarID(buffer_var.get()) << ", " << value << ");\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

// ---------------------------------------------------------------------------
// Vector arithmetic → TPC intrinsics
// ---------------------------------------------------------------------------

void CodeGenGaudi::EmitBinaryVec(const std::string& intrin, const PrimExpr& a, const PrimExpr& b,
                                  std::ostream& os) {
  os << intrin << "(";
  this->PrintExpr(a, os);
  os << ", ";
  this->PrintExpr(b, os);
  os << ")";
}

void CodeGenGaudi::VisitExpr_(const tir::BroadcastNode* op, std::ostream& os) {
  // Broadcast scalar → v_f32_mov_b(scalar)
  os << "v_f32_mov_b(";
  this->PrintExpr(op->value, os);
  os << ")";
}

void CodeGenGaudi::VisitExpr_(const tir::AddNode* op, std::ostream& os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  EmitBinaryVec("v_f32_add_b", op->a, op->b, os);
}

void CodeGenGaudi::VisitExpr_(const tir::SubNode* op, std::ostream& os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  EmitBinaryVec("v_f32_sub_b", op->a, op->b, os);
}

void CodeGenGaudi::VisitExpr_(const tir::MulNode* op, std::ostream& os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  EmitBinaryVec("v_f32_mul_b", op->a, op->b, os);
}

void CodeGenGaudi::VisitExpr_(const tir::DivNode* op, std::ostream& os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  EmitBinaryVec("v_f32_div_b", op->a, op->b, os);
}

void CodeGenGaudi::VisitExpr_(const tir::MaxNode* op, std::ostream& os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  EmitBinaryVec("v_f32_max_b", op->a, op->b, os);
}

void CodeGenGaudi::VisitExpr_(const tir::MinNode* op, std::ostream& os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  EmitBinaryVec("v_f32_min_b", op->a, op->b, os);
}

// ---------------------------------------------------------------------------
// Skip assert statements (TPC-C does not support them)
// ---------------------------------------------------------------------------

void CodeGenGaudi::VisitStmt_(const tir::AssertStmtNode* op) {
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
