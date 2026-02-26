"""
Phase 1: TIR -> TPC-C kernel codegen for Gaudi.

Usage:
    python tests/python/test_gaudi_phase1.py

The script:
  1. Defines a simple computation using TE (element-wise add, size=128).
  2. Schedules with vectorize(64) so TIR has Ramp-style vector indices.
  3. Builds with target="gaudi".
  4. Prints the generated TPC-C kernel source to stdout.
  5. Optionally saves it to a .c file.

The output kernel uses stubs (#define / static inline) for all Gaudi
intrinsics so it also compiles as-is with plain gcc for local testing:

    gcc -O2 -o kernel_test kernel_vadd.c host_vadd.c && ./kernel_test
"""

import sys
import os

import tvm
from tvm import te


# ---------------------------------------------------------------------------
# Helper: build and return the kernel source string.
# ---------------------------------------------------------------------------

def build_vadd_kernel(n: int = 128) -> str:
    """Build an element-wise float32 add kernel for Gaudi, return TPC-C source."""
    assert n % 64 == 0, "n must be a multiple of 64 (vector width)"

    A = te.placeholder((n,), dtype="float32", name="A")
    B = te.placeholder((n,), dtype="float32", name="B")
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")

    s = te.create_schedule(C.op)
    # Split outer tile / inner vector (64 lanes = Gaudi vector width)
    xo, xi = s[C].split(s[C].op.axis[0], factor=64)
    s[C].vectorize(xi)

    target = tvm.target.Target("gaudi")
    mod = tvm.build(s, [A, B, C], target=target, name="vadd")
    return mod.get_source()


def build_vmul_kernel(n: int = 128) -> str:
    """Build an element-wise float32 multiply kernel."""
    assert n % 64 == 0

    A = te.placeholder((n,), dtype="float32", name="A")
    B = te.placeholder((n,), dtype="float32", name="B")
    C = te.compute((n,), lambda i: A[i] * B[i], name="C")

    s = te.create_schedule(C.op)
    xo, xi = s[C].split(s[C].op.axis[0], factor=64)
    s[C].vectorize(xi)

    target = tvm.target.Target("gaudi")
    mod = tvm.build(s, [A, B, C], target=target, name="vmul")
    return mod.get_source()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "gaudi_kernels")
    os.makedirs(out_dir, exist_ok=True)

    # --- vadd ---
    print("=" * 60)
    print("Building vadd kernel (n=128)...")
    src_vadd = build_vadd_kernel(n=128)

    print(src_vadd)
    print("=" * 60)

    path_vadd = os.path.join(out_dir, "kernel_vadd.c")
    with open(path_vadd, "w") as f:
        f.write(src_vadd)
    print(f"[OK] Saved to {path_vadd}")

    # --- vmul ---
    print("=" * 60)
    print("Building vmul kernel (n=128)...")
    src_vmul = build_vmul_kernel(n=128)

    print(src_vmul)
    print("=" * 60)

    path_vmul = os.path.join(out_dir, "kernel_vmul.c")
    with open(path_vmul, "w") as f:
        f.write(src_vmul)
    print(f"[OK] Saved to {path_vmul}")

    # --- Host code template (for manual copy to Gaudi machine) ---
    host_template = """\
/*
 * host_vadd.c  --  Phase 1 host code template for Gaudi vadd kernel.
 *
 * On a real Gaudi machine, replace the stub implementations below with
 * the actual Habana SynapseAI API calls.
 *
 * To test locally with gcc stubs:
 *   gcc -O2 -o test_vadd kernel_vadd.c host_vadd.c && ./test_vadd
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Pull in the kernel function declaration. */
void vadd(float* A, float* B, float* C);

#define N 128

int main(void) {
    float A[N], B[N], C[N], ref[N];

    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
        ref[i] = A[i] + B[i];
    }

    vadd(A, B, C);

    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(C[i] - ref[i]) > 1e-5f) {
            printf("MISMATCH at i=%d: got %f, expected %f\\n", i, C[i], ref[i]);
            ok = 0;
        }
    }
    printf(ok ? "PASS\\n" : "FAIL\\n");
    return ok ? 0 : 1;
}
"""
    path_host = os.path.join(out_dir, "host_vadd.c")
    with open(path_host, "w") as f:
        f.write(host_template)
    print(f"[OK] Host template saved to {path_host}")
    print()
    print("To test locally (no Gaudi required):")
    print(f"  gcc -O2 -o test_vadd {path_vadd} {path_host} -lm && ./test_vadd")
