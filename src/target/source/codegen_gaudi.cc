#include "codegen_gaudi.h"

#include <sstream>

namespace tvm {
namespace codegen {

CodeGenGaudi::CodeGenGaudi() { restrict_keyword_ = "restrict"; }

void CodeGenGaudi::PrintFuncPrefix(std::ostream& os) { (void)os; }

void CodeGenGaudi::VisitStmt_(const tir::ForNode* op) {
  this->PrintIndent();
  this->stream << "// [Gaudi] For begin\n";
  CodeGenC::VisitStmt_(op);
  this->PrintIndent();
  this->stream << "// [Gaudi] For end\n";
}

void CodeGenGaudi::PrintType(DataType t, std::ostream& os) {
  if (t.lanes() == 0) {
    DataType scalar(t.code(), t.bits(), 1);
    CodeGenC::PrintType(scalar, os);
    return;
  }
  if (t.lanes() == 1) {
    CodeGenC::PrintType(t, os);
    return;
  }

  if (t.is_float() && t.bits() == 32 && t.lanes() == kVecLanes) {
    os << "gaudi_vec_f32x64";
    return;
  }

  DataType elem = t.element_of();
  CodeGenC::PrintType(elem, os);
  os << " /* unsupported vector x" << t.lanes() << " */";
}

void CodeGenGaudi::VisitExpr_(const tir::BroadcastNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();
  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support Broadcast float32x64, got " << op->dtype;

  os << "gaudi_vsplat_f32x64(";
  this->PrintExpr(op->value, os);  // scalar
  os << ")";
}

void CodeGenGaudi::VisitStmt_(const tir::BufferStoreNode* op) {
  this->PrintIndent();
  this->stream << "// [Gaudi] BufferStore to " << op->buffer->name << "\n";
  ICHECK_EQ(op->indices.size(), 1);

  const DataType t = op->value.dtype();

  // Vector store: float32x64 with Ramp(base, 1, 64)
  if (t.is_float() && t.bits() == 32 && t.lanes() == kVecLanes) {
    PrimExpr base;
    ICHECK(MatchRampIndexF32x64(op->indices[0], &base))
        << "Gaudi MVP: vector BufferStore must use Ramp(base, 1, 64) index";

    // Compute effective offset = tile0*64 + base
    // If not in tile loop, fallback to base.
    this->PrintIndent();
    this->stream << "gaudi_vstore_f32x64(" << op->buffer->name << ", ";

    if (!tile0_var_.empty()) {
      this->stream << "(" << tile0_var_ << " * " << kVecLanes << " + ";
      this->PrintExpr(base, this->stream);
      this->stream << ")";
    } else {
      this->PrintExpr(base, this->stream);
    }

    this->stream << ", ";
    this->PrintExpr(op->value, this->stream);
    this->stream << ");\n";
    return;
  }

  // Scalar store: float32
  if (t.is_float() && t.bits() == 32 && t.lanes() == 1) {
    std::ostringstream rhs;
    this->PrintExpr(op->value, rhs);

    this->PrintIndent();
    this->stream << "gaudi_store_f32(" << op->buffer->name << ", ";
    this->PrintExpr(op->indices[0], this->stream);
    this->stream << ", " << rhs.str() << ");\n";
    return;
  }

  ICHECK(false) << "Gaudi MVP: unsupported BufferStore dtype " << t;
}


void CodeGenGaudi::VisitExpr_(const tir::BufferLoadNode* op, std::ostream& os) {
  ICHECK_EQ(op->indices.size(), 1);

  const DataType t = op->dtype;

  // Vector load: float32x64 with Ramp(base, 1, 64)
  if (t.is_float() && t.bits() == 32 && t.lanes() == kVecLanes) {
    PrimExpr base;
    ICHECK(MatchRampIndexF32x64(op->indices[0], &base))
        << "Gaudi MVP: vector BufferLoad must use Ramp(base, 1, 64) index";

    os << "gaudi_vload_f32x64(" << op->buffer->name << ", ";

    if (!tile0_var_.empty()) {
      os << "(" << tile0_var_ << " * " << kVecLanes << " + ";
      this->PrintExpr(base, os);
      os << ")";
    } else {
      this->PrintExpr(base, os);
    }

    os << ")";
    return;
  }

  // Scalar load: float32
  if (t.is_float() && t.bits() == 32 && t.lanes() == 1) {
    os << "gaudi_load_f32(" << op->buffer->name << ", ";
    this->PrintExpr(op->indices[0], os);
    os << ")";
    return;
  }

  CodeGenC::VisitExpr_(op, os);
}


void CodeGenGaudi::AddKernel(const std::string& kernel_name,
                             const std::vector<tvm::tir::Buffer>& arg_buffers,
                             const tvm::tir::Stmt& body) {
  EmitGaudiIntrinsicsOnce();

  if (kernel_debug_) {
    this->PrintIndent();
    this->stream << "\n// [Gaudi][debug] AddKernel begin: " << kernel_name << "\n";
    this->PrintIndent();
    this->stream << "// [Gaudi][debug] arg_buffers = " << arg_buffers.size() << "\n";
  }

  // Signature
  this->PrintIndent();
  this->stream << "void " << kernel_name << "(";
  for (size_t i = 0; i < arg_buffers.size(); ++i) {
    if (i != 0) this->stream << ", ";

    const tvm::tir::Buffer& buf = arg_buffers[i];
    ICHECK(buf->dtype.is_float() && buf->dtype.bits() == 32 && buf->dtype.lanes() == 1)
        << "Gaudi MVP: kernel arg buffer must be float32 scalar, but got " << buf->dtype;
    this->stream << "float* " << buf->name;
  }
  this->stream << ") {\n";

  bool prev = in_kernel_emit_;
  in_kernel_emit_ = true;

  // === Index-space skeleton (SPMD tile loop) ===
  // We mimic Habana samples:
  //   start = get_index_space_offset()
  //   end   = get_index_space_size() + start
  //
  // MVP: only use dim0 as "tile id", each tile handles 64 elements along flattened axis.
  if (emit_index_space_) {
    int scope = this->BeginScope();

    this->PrintIndent();
    this->stream << "gaudi_int5 indexSpaceStart = get_index_space_offset();\n";
    this->PrintIndent();
    this->stream << "gaudi_int5 indexSpaceSize  = get_index_space_size();\n";
    this->PrintIndent();
    this->stream << "int indexSpaceStart0 = indexSpaceStart.v[0];\n";
    this->PrintIndent();
    this->stream << "int indexSpaceEnd0   = indexSpaceStart0 + indexSpaceSize.v[0];\n";

    // tile loop
    this->PrintIndent();
    this->stream << "for (int tile0 = indexSpaceStart0; tile0 < indexSpaceEnd0; ++tile0) {\n";
    int tile_scope = this->BeginScope();

    // set tile0_var_ so BufferLoad/Store can use it
    std::string prev_tile = tile0_var_;
    tile0_var_ = "tile0";

    // Body
    this->VisitStmt(body);

    // restore
    tile0_var_ = prev_tile;

    this->EndScope(tile_scope);
    this->PrintIndent();
    this->stream << "}\n";

    this->EndScope(scope);
  } else {
    // no index-space, directly emit body
    this->VisitStmt(body);
  }

  in_kernel_emit_ = prev;

  this->PrintIndent();
  this->stream << "}\n\n";

  if (kernel_debug_) {
    this->PrintIndent();
    this->stream << "// [Gaudi][debug] AddKernel end: " << kernel_name << "\n";
  }
}

void CodeGenGaudi::VisitExpr_(const tir::AddNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();

  if (lanes == 1) {
    os << "gaudi_add(";
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ")";
    return;
  }
  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support vector add float32x64, got " << op->dtype;

  os << "gaudi_vadd_f32x64(";
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ")";
}
void CodeGenGaudi::VisitExpr_(const tir::SubNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();

  if (lanes == 1) {
    os << "gaudi_sub(";
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ")";
    return;
  }
  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support vector sub float32x64, got " << op->dtype;

  os << "gaudi_vsub_f32x64(";
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ")";
}
void CodeGenGaudi::VisitExpr_(const tir::MulNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();

  if (lanes == 1) {
    os << "gaudi_mul(";
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ")";
    return;
  }

  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support vector mul float32x64, got " << op->dtype;

  os << "gaudi_vmul_f32x64(";
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ")";
}

void CodeGenGaudi::VisitExpr_(const tir::DivNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();

  if (lanes == 1) {
    os << "gaudi_div(";
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ")";
    return;
  }

  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support vector div float32x64, got " << op->dtype;
  os << "gaudi_vdiv_f32x64(";
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ")";
}

void CodeGenGaudi::VisitExpr_(const tir::MinNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();
  if (lanes == 1) {
    os << "gaudi_min(";
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ")";
    return;
  }

  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support vector min float32x64, got " << op->dtype;

  os << "gaudi_vmin_f32x64(";
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ")";
}

void CodeGenGaudi::VisitExpr_(const tir::MaxNode* op, std::ostream& os) {
  const int lanes = op->dtype.lanes();

  if (lanes == 1) {
    os << "gaudi_max(";
    this->PrintExpr(op->a, os);
    os << ", ";
    this->PrintExpr(op->b, os);
    os << ")";
    return;
  }
  ICHECK(op->dtype.is_float() && op->dtype.bits() == 32 && lanes == kVecLanes)
      << "Gaudi MVP: only support vector max float32x64, got " << op->dtype;

  os << "gaudi_vmax_f32x64(";
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ")";
}

void CodeGenGaudi::VisitStmt_(const tir::AssertStmtNode* op) {
  if (in_kernel_emit_) {
    this->VisitStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenGaudi::VisitExpr_(const tir::CastNode* op, std::ostream& os) {
  os << "/* [Gaudi] Cast to ";
  this->PrintType(op->dtype, os);
  os << " */ ";
  os << "gaudi_cast(";
  this->PrintExpr(op->value, os);
  os << ", \"" << op->dtype << "\")";
}

void CodeGenGaudi::EmitGaudiIntrinsicsOnce() {
  if (gaudi_intrin_emitted_) return;
  gaudi_intrin_emitted_ = true;

  this->stream << "\n// === Gaudi/TPC-ish pseudo intrinsics (module-level) ===\n";
  this->stream << "#ifndef GAUDI_STDINT_INCLUDED\n";
  this->stream << "#define GAUDI_STDINT_INCLUDED\n";
  this->stream << "#include <stdint.h>\n";
  this->stream << "#endif\n\n";

  // ---- int5 + index-space API (stub) ----
  this->stream << "#ifndef GAUDI_INT5_DEFINED\n";
  this->stream << "#define GAUDI_INT5_DEFINED\n";
  this->stream << "typedef struct { int32_t v[5]; } gaudi_int5;\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef get_index_space_offset\n";
  this->stream << "static inline gaudi_int5 get_index_space_offset() {\n";
  this->stream << "  gaudi_int5 r; r.v[0]=0; r.v[1]=0; r.v[2]=0; r.v[3]=0; r.v[4]=0; return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef get_index_space_size\n";
  this->stream << "static inline gaudi_int5 get_index_space_size() {\n";
  this->stream << "  // MVP stub: one tile by default.\n";
  this->stream << "  gaudi_int5 r; r.v[0]=1; r.v[1]=1; r.v[2]=1; r.v[3]=1; r.v[4]=1; return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  // ---- vector type ----
  this->stream << "#ifndef GAUDI_VEC_F32X64_DEFINED\n";
  this->stream << "#define GAUDI_VEC_F32X64_DEFINED\n";
  this->stream << "typedef struct { float v[" << kVecLanes << "]; } gaudi_vec_f32x64;\n";
  this->stream << "#endif\n\n";

  // ---- scalar ops ----
  this->stream << "#ifndef gaudi_add\n";
  this->stream << "#define gaudi_add(a, b) ((a) + (b))\n";
  this->stream << "#endif\n";
  this->stream << "#ifndef gaudi_sub\n";
  this->stream << "#define gaudi_sub(a, b) ((a) - (b))\n";
  this->stream << "#endif\n";
  this->stream << "#ifndef gaudi_mul\n";
  this->stream << "#define gaudi_mul(a, b) ((a) * (b))\n";
  this->stream << "#endif\n";
  this->stream << "#ifndef gaudi_div\n";
  this->stream << "#define gaudi_div(a, b) ((a) / (b))\n";
  this->stream << "#endif\n";
  this->stream << "#ifndef gaudi_cast\n";
  this->stream << "#define gaudi_cast(x, dtype_str) (x)\n";
  this->stream << "#endif\n";
  this->stream << "#ifndef gaudi_load_f32\n";
  this->stream << "#define gaudi_load_f32(base, idx) (((float*)(base))[(idx)])\n";
  this->stream << "#endif\n";
  this->stream << "#ifndef gaudi_store_f32\n";
  this->stream << "#define gaudi_store_f32(base, idx, value) (((float*)(base))[(idx)] = (value))\n";
  this->stream << "#endif\n\n";

  // ---- vector load/store ----
  this->stream << "#ifndef gaudi_vload_f32x64\n";
  this->stream << "static inline gaudi_vec_f32x64 gaudi_vload_f32x64(float* base, int32_t off) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = base[off + i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vstore_f32x64\n";
  this->stream
      << "static inline void gaudi_vstore_f32x64(float* base, int32_t off, gaudi_vec_f32x64 v) {\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { base[off + i] = v.v[i]; }\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  // ---- vector splat + arith ----
  this->stream << "#ifndef gaudi_vsplat_f32x64\n";
  this->stream << "static inline gaudi_vec_f32x64 gaudi_vsplat_f32x64(float x) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = x; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vadd_f32x64\n";
  this->stream
      << "static inline gaudi_vec_f32x64 gaudi_vadd_f32x64(gaudi_vec_f32x64 a, gaudi_vec_f32x64 b) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = a.v[i] + b.v[i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vsub_f32x64\n";
  this->stream
      << "static inline gaudi_vec_f32x64 gaudi_vsub_f32x64(gaudi_vec_f32x64 a, gaudi_vec_f32x64 b) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = a.v[i] - b.v[i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vmul_f32x64\n";
  this->stream
      << "static inline gaudi_vec_f32x64 gaudi_vmul_f32x64(gaudi_vec_f32x64 a, gaudi_vec_f32x64 b) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = a.v[i] * b.v[i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vdiv_f32x64\n";
  this->stream
      << "static inline gaudi_vec_f32x64 gaudi_vdiv_f32x64(gaudi_vec_f32x64 a, gaudi_vec_f32x64 b) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = a.v[i] / b.v[i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vmin_f32x64\n";
  this->stream
      << "static inline gaudi_vec_f32x64 gaudi_vmin_f32x64(gaudi_vec_f32x64 a, gaudi_vec_f32x64 b) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream
      << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = (a.v[i] < b.v[i]) ? a.v[i] : b.v[i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";

  this->stream << "#ifndef gaudi_vmax_f32x64\n";
  this->stream
      << "static inline gaudi_vec_f32x64 gaudi_vmax_f32x64(gaudi_vec_f32x64 a, gaudi_vec_f32x64 b) {\n";
  this->stream << "  gaudi_vec_f32x64 r;\n";
  this->stream
      << "  for (int i = 0; i < " << kVecLanes << "; ++i) { r.v[i] = (a.v[i] > b.v[i]) ? a.v[i] : b.v[i]; }\n";
  this->stream << "  return r;\n";
  this->stream << "}\n";
  this->stream << "#endif\n\n";
}


void CodeGenGaudi::PreFunctionBody(const PrimFunc& f) {
  EmitGaudiIntrinsicsOnce();
  CodeGenC::PreFunctionBody(f);
}



bool CodeGenGaudi::MatchRampIndexF32x64(const PrimExpr& index, PrimExpr* out_base) const {
  const auto* r = index.as<tir::RampNode>();
  if (r == nullptr) return false;
  const auto* lanes_imm = r->lanes.as<tir::IntImmNode>();
  if (lanes_imm == nullptr) return false;
  if (static_cast<int>(lanes_imm->value) != kVecLanes) return false;
  if (!is_one(r->stride)) return false;
  *out_base = r->base;
  return true;
}

}  // namespace codegen
}  // namespace tvm
