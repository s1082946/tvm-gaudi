import os
import tvm
from tvm.script import tir as T
@tvm.script.ir_module
class ModClampVec:
    @T.prim_func
    def main(a: T.Buffer((64,), "float32"),
             b: T.Buffer((64,), "float32")):
        idx = T.Ramp(0, 1, 64)
        x = a[idx] * T.Broadcast(T.float32(2.0), 64) + T.Broadcast(T.float32(1.0), 64)
        y = T.min(x, T.Broadcast(T.float32(1.0), 64))
        z = T.max(y, T.Broadcast(T.float32(0.0), 64))
        b[idx] = z


mod = ModClampVec
print("========== NORMAL BUILD (wrapper + kernel section) ==========")
rt_mod = tvm.build(mod, target="gaudi")
dev_mod = rt_mod.imports[0] if rt_mod.imports else rt_mod
full_src = dev_mod.inspect_source("tpc-c")
print(full_src)
print("\n========== KERNEL-ONLY BUILD ==========")
os.environ["TVM_GAUDI_KERNEL_ONLY"] = "1"
rt_mod_kernel = tvm.build(mod, target="gaudi")
dev_mod_kernel = rt_mod_kernel.imports[0] if rt_mod_kernel.imports else rt_mod_kernel
kernel_only_src = dev_mod_kernel.inspect_source("tpc-c")
print(kernel_only_src)
