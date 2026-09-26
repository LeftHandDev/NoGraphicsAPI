# Reproducible software-Vulkan test environment.
#
# Two things make the golden-image tests reproducible and both are pinned here:
#   * the Vulkan implementation that executes the shaders (Mesa lavapipe), and
#   * the slang compiler that generates the SPIR-V.
#
# Mesa is pinned by pointing pacman at the Arch Linux Archive snapshot below;
# slang is pinned to a fixed GitHub release. Bumping either may change a few
# least-significant bits in the output, so regenerate the goldens (ci-run.sh
# --generate) in the same image whenever you change these versions.
#
# Build:  docker build -t ngapi-ci ci
# Run:    docker run --rm -v "$PWD":/src -w /src ngapi-ci tests/ci-run.sh
FROM archlinux:base-devel

# --- Mesa / toolchain, pinned to a single Arch Linux Archive snapshot --------
# Snapshot 2026-06-09 -> Mesa 26.1.2 (the first lavapipe with the
# descriptor-buffer fix this project's bindless path depends on). The tests
# fail without the validation layer; the X11/Wayland headers are for GLFW, so
# the (windowed) samples compile here too even though only the tests run.
ARG ALA_SNAPSHOT=2026/06/09
RUN printf 'Server=https://archive.archlinux.org/repos/%s/$repo/os/$arch\n' "$ALA_SNAPSHOT" \
        > /etc/pacman.d/mirrorlist \
 && sed -i 's/^ *SigLevel.*/SigLevel = Never/' /etc/pacman.conf \
 && pacman -Sy --noconfirm --needed \
        cmake ninja git curl \
        vulkan-headers vulkan-icd-loader vulkan-swrast vulkan-validation-layers \
        libx11 libxrandr libxinerama libxcursor libxi libxkbcommon wayland \
 && pacman -Scc --noconfirm

# --- slang, pinned to a fixed release for deterministic SPIR-V ----------------
ARG SLANG_VERSION=2026.10.2
RUN curl -fsSL -o /tmp/slang.tar.gz \
      "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.tar.gz" \
 && mkdir -p /opt/slang \
 && tar -xzf /tmp/slang.tar.gz -C /opt/slang \
 && rm /tmp/slang.tar.gz \
 && test -x /opt/slang/bin/slangc
ENV PATH="/opt/slang/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/slang/lib"

# --- Force the software rasteriser, single-threaded for deterministic FP ------
# lavapipe is the only ICD installed, but pin it explicitly so a stray GPU on a
# self-hosted runner can never be picked instead. LP_NUM_THREADS=1 removes the
# parallel-raster floating-point nondeterminism (gives byte-identical frames).
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    LP_NUM_THREADS=1

# --- Validation: synchronization validation on top of the core checks --------
# The tests request VK_LAYER_KHRONOS_validation themselves; this adds the
# hazard checks (transfers, blits, layout transitions). Syncval does not see
# shader accesses made through descriptor buffers or buffer device addresses.
ENV VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true

CMD ["bash"]
