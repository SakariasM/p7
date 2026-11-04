#!/bin/bash
# Sourced inside the Singularity container by serve_llm.slurm.
#
# Root cause: Triton's libcuda_dirs() trusts `ldconfig -p` first and only
# falls back to LD_LIBRARY_PATH if ldconfig finds NOTHING matching libcuda.
# This image's baked-in ldconfig cache already lists
# /usr/local/cuda/compat/lib/libcuda.so.1 (a build-time compat stub that
# doesn't actually exist/work at runtime under --nv), so that fallback never
# triggers - no LD_LIBRARY_PATH change can fix this on its own.
#
# Fix: copy the real, host-mounted driver library into a job-local dir, and
# shadow `ldconfig` on PATH so Triton's `ldconfig -p` reports our copy.

CUDA_COMPAT_DIR="/tmp/cuda_compat_${SLURM_JOB_ID}"
mkdir -p "${CUDA_COMPAT_DIR}"

REAL_LIBCUDA=$(find /.singularity.d -name 'libcuda.so.1' 2>/dev/null | head -1)
if [ -z "$REAL_LIBCUDA" ]; then
    REAL_LIBCUDA=$(find /usr/lib/x86_64-linux-gnu -name 'libcuda.so.1' 2>/dev/null | head -1)
fi
if [ -z "$REAL_LIBCUDA" ]; then
    echo "WARNING: no host-mounted libcuda.so.1 found - falling back to the compat shim"
    REAL_LIBCUDA=$(find /usr/local/cuda/compat -name 'libcuda.so.1' 2>/dev/null | head -1)
fi

if [ -z "$REAL_LIBCUDA" ]; then
    echo "WARNING: could not locate libcuda.so.1 anywhere - vLLM will likely fail to start"
    return 0 2>/dev/null || exit 0
fi

RESOLVED_LIBCUDA=$(readlink -f "$REAL_LIBCUDA")
cp "$RESOLVED_LIBCUDA" "${CUDA_COMPAT_DIR}/libcuda.so.1"
ln -sf "${CUDA_COMPAT_DIR}/libcuda.so.1" "${CUDA_COMPAT_DIR}/libcuda.so"
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"

REAL_LDCONFIG="$(command -v ldconfig)"
cat > "${CUDA_COMPAT_DIR}/ldconfig" <<EOF
#!/bin/bash
if [ "\$1" = "-p" ]; then
    echo "1 libs found in cache"
    echo -e "\tlibcuda.so.1 (libc6,x86-64) => ${CUDA_COMPAT_DIR}/libcuda.so.1"
else
    exec ${REAL_LDCONFIG} "\$@"
fi
EOF
chmod +x "${CUDA_COMPAT_DIR}/ldconfig"
export PATH="${CUDA_COMPAT_DIR}:${PATH}"

echo "libcuda.so.1 resolved from ${REAL_LIBCUDA} -> ${RESOLVED_LIBCUDA}, copied into ${CUDA_COMPAT_DIR}, ldconfig shadowed to report it"
