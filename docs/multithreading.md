# Multithreading

The first half of this document is the audit that drove the work; the
findings below it are now **implemented** on this branch. Status of each:

| Finding | Fix |
| --- | --- |
| §1 single `VkCommandPool` | Each `GpuCommandBuffer` owns a transient pool, created in `gpuStartCommandRecording` and destroyed when its submission retires. Recording takes no locks. The device-level pool remains for swapchain present transitions only. |
| §2 `allocations` vector | `std::shared_mutex`: shared lock for `findAllocation`, exclusive for create/free. |
| §3 `currentPipeline` map | Moved into `GpuCommandBuffer_T`. |
| §4 queue submission/retirement | `submitMutex` serializes `vkQueueSubmit`, the present transition + present, and the retirement map (now pools, not buffers). `gpuWaitSemaphore` collects retired pools under the lock and destroys them outside it. |
| §5 lazy init | Everything is created eagerly in `gpuCreateDevice` (`initDeviceResources`); the descriptor-view counters are atomics. |
| §6 static-sampler dedup | `samplerMutex` around slot lookup/creation. |
| §7 `acquireFence` | Moved into the swapchain. Instance/device lifecycle and per-swapchain use are documented as externally synchronized. |

The contract (also in `NoGraphicsAPI_Impl.h`): creation/destruction and queue
operations are thread-safe; recording is parallel across command buffers; a
single command buffer, each swapchain, and instance/device lifecycle are
externally synchronized.

**Verification** — `samples/multithreading` (headless, self-checking):
8 workers, each with its own allocator, queue, semaphore and concurrently
created pipeline, recording/submitting/verifying in a loop. 30+ consecutive
runs clean, 15 of them with the validation layer's `thread_safety` checking
enabled — zero validation messages. The lavapipe golden tests cover the
single-threaded paths (including the eagerly-created patching machinery)
unchanged.

## Validation is opt-in (`NGAPI_VALIDATION=1`)

The implementation used to call `request_validation_layers(true)`
unconditionally — every build, every user. The validation layer wraps every
recorded command with a global lock and ~10µs of checking, which both
dominated single-threaded recording (≈ 20× overhead) and serialized
multithreaded recording into a lock convoy (the investigation initially
looked like a driver-side futex storm; it was the layer). Validation is now
requested only when `NGAPI_VALIDATION` is set; the test harness sets it
automatically, since "no validation messages" is part of every test's pass
criteria.

Measured on an RX 9070 XT (RADV, 9800X3D, 8 cores), recording streams of
8192 dispatch+barrier pairs per command buffer: single-threaded recording
~0.5µs/command; 8 threads sustain **3.2–3.7×** single-threaded throughput,
stable across runs (with validation enabled the same measurement was
12µs/command and 0.2–1.9× run-to-run).

Command pools are also recycled through a per-device free-list (reset, not
destroyed, at retirement) so steady-state recording allocates nothing through
the driver.

## Known limits

- **Descriptor patching devices** (descriptor size ≠ 32, e.g. lavapipe): the
  patch *destination* heaps are device-global, so two patching submissions
  executing concurrently race on the GPU. Recording them is safe; their
  execution must not overlap. Fix would be per-submission patch buffers —
  deferred until a patching device matters for multithreaded use.
- **Remaining recording sub-linearity** (3.4× of 8 ideal) is at sane
  magnitudes (≈ 60M commands/s aggregate into write-combined memory) and sits
  below the API; one genuine NGAPI contributor — a heap-allocated vector per
  `gpuBarrier` — was found and fixed along the way.
- One unexplained phase-1 failure was observed once under validation timing
  (a CPU re-read of an input buffer returned a value never written, while
  the GPU output was correct). All subsequent runs are clean; the sample's
  verification now distinguishes input trampling from output errors so a
  recurrence will be classified precisely.

---

# Original audit (pre-implementation)

State of CPU-side thread safety in the NGAPI Vulkan implementation, as of the
fork of `multithreading_support` (post static-samplers). Every public entry
point in `NoGraphicsAPI_Impl.h` was walked against the shared state it touches
in `NoGraphicsAPI_Impl.cpp`.

**Verdict: the implementation is single-threaded today.** Nothing is locked,
several pieces of shared mutable state sit on the hottest paths, and the one
Vulkan object that structurally forbids parallel recording (a single
`VkCommandPool`) is shared by every command buffer. None of this is
criticism — the blog post's API is silent on threading, and the samples are
single-threaded — but it means "multithreading support" is real work, not a
sprinkle of mutexes.

## Target contract (proposed)

The blog says nothing about threading; its one relevant choice — transient
command buffers (`gpuStartCommandRecording` → `gpuSubmit`, no reuse) — happens
to map cleanly onto parallel recording. The contract worth implementing, in
line with what Vulkan/Metal users expect:

1. **Resource creation/destruction is thread-safe**: `gpuMalloc`/`gpuFree`,
   `gpuCreateTexture`, `gpu*ViewDescriptor`, `gpuCreate*Pipeline`,
   `gpuCreateSemaphore`, acceleration structures — callable from any thread
   concurrently.
2. **Command recording is parallel across command buffers**: one thread per
   `GpuCommandBuffer`; a single command buffer is externally synchronized
   (never two threads in one `cb`, which nobody expects anyway).
3. **Queues are internally synchronized**: `gpuSubmit`, `gpuWaitSemaphore`
   and `gpuPresent` may be called from any thread.
4. **Externally synchronized (documented, not locked)**: instance/device
   create/destroy, and each swapchain (`gpuSwapchainImage`/`gpuPresent` pairs
   belong to one thread at a time).

## Shared-state inventory and findings

Ordered by severity.

### 1. Single `VkCommandPool` per device — the structural blocker

`VulkanDevice::commandPool` backs every command buffer.
`gpuStartCommandRecording` allocates from it, `gpuWaitSemaphore` frees into
it, and — the part that blocks the headline feature — *recording* into two
command buffers from the same pool concurrently is itself invalid: every
`vkCmd*` allocates from the pool's allocator, and pools are externally
synchronized. Even with all the maps fixed, two threads recording two
separate command buffers would still be a spec violation.

Affected: `gpuStartCommandRecording`, every `gpuCmd`-style call (`gpuDispatch`,
`gpuDraw*`, `gpuBarrier`, `gpuMemCpy`, copies/blits, render passes,
`gpuBuildAccelerationStructures`), `gpuWaitSemaphore` (frees).

Fix direction: a pool per `GpuCommandBuffer` (transient model: create pool +
buffer in `gpuStartCommandRecording`, destroy the pool at retirement —
simplest, correct, slightly wasteful) or a per-thread pool cache (faster,
more machinery). The retirement bookkeeping (`submittedCommandBuffers`)
then tracks pools instead of buffers.

### 2. `allocations` vector — unsynchronized reads and writes on hot paths

`std::vector<Allocation>` mutated by `createAllocation` (`push_back`,
invalidates everything) and `freeAllocation` (`erase`), linearly scanned by
`findAllocation`. The scan is not just in allocation calls: it runs inside
**`gpuSetActiveTextureHeapPtr`** (every dispatch/draw setup),
`gpuHostToDevicePointer`, `gpuCreateTexture`, and the lazy heap inits. A
`gpuMalloc` on one thread while another thread records is undefined behavior
today.

Fix direction: `std::shared_mutex` (reads dominate; writers are
malloc/free), or a stable container + read-copy-update if the scan ever
shows up in profiles. Independently worthwhile: the linear scan is O(n) per
lookup and `gpuFree` is O(n) — an interval map would fix both.

### 3. `currentPipeline` — a device-global map for per-command-buffer state

`std::map<GpuCommandBuffer, GpuPipeline>` written by `gpuSetPipeline` (every
recording thread), erased in `gpuSubmit`, read by the descriptor-patching
path. This is per-`cb` state living on the device for no structural reason.

Fix direction: move the field into `GpuCommandBuffer_T`. Removes the race
*and* a map lookup per bind. No locking needed afterward.

### 4. Queue submission and retirement

- `gpuSubmit` calls `vkQueueSubmit` — `VkQueue` is externally synchronized,
  and `gpuCreateQueue` always returns the *same* underlying graphics queue,
  so two "different" `GpuQueue`s submitting concurrently still race.
- `gpuPresent` fetches that same graphics queue and submits the layout
  transition on it — racing any concurrent `gpuSubmit` — then presents on
  `presentQueue`.
- `gpuSubmit` writes and `gpuWaitSemaphore` reads/erases
  `submittedCommandBuffers` (and the wait-side retire loop also frees into
  the shared command pool, see §1).

Fix direction: one mutex per underlying `VkQueue` (submit + present
transition take the graphics-queue mutex; present takes the present-queue
mutex), and the retirement map either under the same lock or replaced by the
per-`cb`-pool scheme from §1.

### 5. Lazy initialization on first use

First-use creation happens in awkward places with no synchronization:

- `gpuSetActiveTextureHeapPtr` (i.e., *inside command recording*) lazily
  creates the sampler descriptor heap, the patched-descriptor heaps, the
  `PatchDescriptorsData` block, and — via `gpuCreateComputePipeline` — the
  embedded patch pipeline, which itself walks the static-sampler machinery.
- `gpuTextureViewDescriptor` / `gpuRWTextureViewDescriptor` lazily allocate
  `descriptorDataCpu`/`rwDescriptorDataCpu` and bump the unsynchronized
  `descriptorsUsed`/`rwDescriptorsUsed` counters (descriptor-patching
  devices only, e.g. lavapipe — RADV takes the raw-blob path, which is
  thread-clean).

Fix direction: eager-create everything fixed-size at device creation
(sampler heap, patch pipeline, patch heaps — they are all bounded by
`descriptorCount`), make the counters atomic. Eliminates the whole class.

### 6. Static-sampler machinery at pipeline creation

`staticSamplerSlot` reads/writes `staticSamplerSlots`, `staticSamplers`,
`nextSamplerSlot`, and writes a descriptor into the sampler heap. Two
threads creating pipelines concurrently race the dedup map and can double-
allocate slots.

Fix direction: a small mutex around slot lookup/creation. Pipeline creation
is otherwise thread-safe (`vkCreateShaderModule`/`vkCreate*Pipelines` are
internally synchronized at the device level).

### 7. Instance-level odds and ends (minor, init-time)

- `gpuCreateInstance` / `gpuDestroyInstance` mutate the global
  `vulkanInstance` unsynchronized (double-create TOCTOU).
- `gpuDeviceCount`/`gpuDeviceDesc` lazily populate `deviceDescs`
  (concurrent first calls duplicate entries).
- `gpuSwapchainImage` uses the device-wide `acquireFence` — two swapchains
  (or threads) acquiring concurrently race it; it belongs on the swapchain.

Fix direction: document instance/device lifecycle as externally
synchronized (standard), move `acquireFence` into the swapchain, and treat
each swapchain as externally synchronized.

## What is already fine

- All pure `vkCmd*` recording calls touch only their command buffer — once
  §1 and §3 are fixed, recording parallelizes with no locks.
- Creation functions (`vkCreate*`) are internally synchronized by Vulkan at
  the device level; only our bookkeeping around them races.
- `gpuWaitSemaphore`'s actual wait (`vkWaitSemaphores`) is thread-safe; only
  the retirement bookkeeping isn't.
- Handle wrappers (`GpuTexture_T`, `GpuPipeline_T`, ...) are immutable after
  creation.
- On non-patching devices the descriptor-blob path (`gpu*ViewDescriptor`)
  is already thread-clean.

## Suggested implementation order

1. **Per-`cb` command pools + `currentPipeline` into `GpuCommandBuffer_T`**
   (§1, §3) — unlocks parallel recording, no locks on the record path.
2. **`allocations` under a `shared_mutex`** (§2) — makes malloc/free and
   pointer translation safe.
3. **Queue mutexes + retirement rework** (§4).
4. **Eager init + atomic descriptor counters** (§5), **sampler mutex** (§6).
5. **Documentation of the contract** (§7 + the API header).

Verification: a headless stress test (N threads × record/submit/wait loops +
concurrent malloc/free and pipeline creation) run under the validation
layers' thread-safety checks (`VK_LAYER_KHRONOS_validation` reports
externally-synchronized-object races precisely), plus the existing goldens
to prove the single-threaded paths didn't move.
