#include "Common.h"
#include "Tensor.h"
#include "Utilities.h"
#include <cstring>
#include <map>
#include <algorithm>
#include <random>
#include <iostream>
#include <set>
#include <chrono>
#include <fstream>

const uint64_t FRAMES_IN_FLIGHT = 2;
std::vector<Device*> devices;
std::map<STAGE, std::set<Tensor>> tensors_pending_writes;

std::string to_string(Shape shape)
{
    if (shape.empty())
    {
        return "()";
    }

    std::string result = "(";
    for (auto& n : shape)
    {
        result += std::to_string(n) + ", ";
    };
    result.pop_back();
    result.pop_back();
    return result + ")";
}

uint64_t flatten(Shape shape)
{
    uint64_t size = 1;
    for (auto& x : shape)
    {
        size *= x;
    }
    return size;
}

Shape append(Shape base, Shape ext)
{
    base.insert(base.end(), ext.begin(), ext.end());
    return base;
}

// Fold a 1D workgroup count into a 2D grid so no single dimension exceeds the limit
static const unsigned int kMaxGroups = 65535u;
inline uint3 grid1d(uint64_t elements, unsigned int local = 64u)
{
    uint64_t groups = (elements + local - 1) / local;
    unsigned int gx = static_cast<unsigned int>(groups < kMaxGroups ? groups : kMaxGroups);
    unsigned int gy = static_cast<unsigned int>((groups + kMaxGroups - 1) / kMaxGroups);
    return { gx, gy, 1 };
}

using TensorAllocator =
    FallbackAllocator<
        FreeListAllocator<1024 * 1024 * 1024, MEMORY_DEFAULT>,
        GpuMallocator<MEMORY_DEFAULT>>;

using StructAllocator =
    FallbackAllocator<
        StackAllocator<64 * 1024 * 1024, MEMORY_DEFAULT>,
        GpuMallocator<MEMORY_DEFAULT>>;

using ReadbackAllocator =
    FallbackAllocator<
        StackAllocator<64 * 1024 * 1024, MEMORY_READBACK>,
        GpuMallocator<MEMORY_DEFAULT>>;

class Device_impl : public Device
{
public:
    Device_impl(int index)
    {
        device = gpuCreateDevice(index);
        queue = gpuCreateQueue(device);

        tensor_allocator = new TensorAllocator(device);

        for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
        {
            struct_allocator[i] = new StructAllocator(device);
            readback_allocator[i] = new ReadbackAllocator(device);
        }

        auto tensorIR = loadIR("shaders/learning/Tensor.spv");
        for (auto op : { "add", "sub", "mul", "div", "dot", "permute", "unfold", "unfold_backward", "reduce_sum", "reduce_block", "reduce_max", "reduce_max_backward", "concat_scatter", "concat_gather", "expand", "mT", "matmul", "matmul_splitk",
                         "pow", "exp", "log", "sin", "cos", "tan", "cosh", "tanh",
                         "relu", "relu_backward", "gelu", "gelu_backward",
                         "adam", "rand" })
        {
            std::string name = op;
            pipelines[name][Tensor::Type::float32] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), ("_" + name + "_f32").c_str());
            pipelines[name][Tensor::Type::float16] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), ("_" + name + "_f16").c_str());
        }
        // Cooperative-matrix matmul is fp16 input / fp32 accumulate only.
        // The CoopMat ops use Subgroup memory scope and assume a 32-wide
        // subgroup, so pin the pipeline's required subgroup size to 32.
        pipelines["matmul_wmma"][Tensor::Type::float16] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_matmul_wmma", 32);
        // Fused affine (matmul + bias) via cooperative matrices; same 32-wide
        // subgroup requirement as the wmma matmul.
        pipelines["affine"][Tensor::Type::float16] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_affine", 32);
        // Type-conversion kernels, keyed by destination type.
        pipelines["cast"][Tensor::Type::float32] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_fp32_fp16");
        pipelines["cast"][Tensor::Type::float16] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_fp16_fp32");

        // Seed the GPU RNG counter non-deterministically so runs differ.
        std::random_device rd;
        rand_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    }

    ~Device_impl()
    {
        submit();
        delete tensor_allocator;
        for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
        {
            delete struct_allocator[i];
            delete readback_allocator[i];
        }

        if (semaphore)
        {
            gpuDestroySemaphore(semaphore);
        }

        gpuDestroyQueue(queue);

        for (auto& [entry, variants] : pipelines)
        {
            for (auto& [type, pipeline] : variants)
            {
                gpuFreePipeline(pipeline);
            }
        }

        auto iter = std::remove(devices.begin(), devices.end(), reinterpret_cast<Device*>(this));
        if (iter == devices.end())
        {
            // device tracking error, should not happen
        }
        gpuDestroyDevice(device);
    }

    virtual Tensor tensor(std::vector<float> data, Shape shape = {}, Tensor::Type type = Tensor::Type::float32) override
    {
        return Tensor(this, data, {}, shape, type);
    }

    virtual Tensor rand(Shape shape, Tensor::Type type = Tensor::Type::float32) override
    {
        auto size = flatten(shape);
        auto allocation = alloc(size, type);

        auto tensor_data = struct_data<TensorData>();
        tensor_data.cpu->n = size;
        tensor_data.cpu->m = rand_seed++;
        tensor_data.cpu->x = nullptr;
        tensor_data.cpu->y = nullptr;
        tensor_data.cpu->z = allocation.gpu;

        auto cmd = record();
        gpuSetPipeline(cmd, pipelines["rand"][type]);
        gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

        auto out = Tensor(this, allocation, {}, shape, type);
        tensors_pending_writes[STAGE_COMPUTE].insert(out);
        return out;
    }

    virtual Tensor zeros(Shape shape, Tensor::Type type = Tensor::Type::float32) override
    {
        std::vector<float> data(flatten(shape), 0.f);
        return Tensor(this, data, {}, shape, type);
    }

    virtual Tensor ones(Shape shape, Tensor::Type type = Tensor::Type::float32) override
    {
        std::vector<float> data(flatten(shape), 1.f);
        return Tensor(this, data, {}, shape, type);
    }

    virtual Tensor repeat(float x, Shape shape) override
    {
        std::vector<float> data(flatten(shape), x);
        return Tensor(this, data, {}, shape);
    }

    virtual Tensor repeat(const Tensor& tensor, Shape shape) override
    {
        auto out = zeros(append(tensor._shape, shape));
        repeat(out, tensor);
        return out;
    }

    void repeat(Tensor dst, const Tensor& src)
    {
        if (dst._shape.empty())
        {
            auto error = "Unable to repeat copy to empty tensor";
            throw std::runtime_error(error);
        }

        if (dst._shape != src._shape)
        {
            for (size_t i = 0; i < dst._shape.front(); i++)
            {
                repeat(dst[i], src);
            }
        }
        else
        {
            dst.copy(src);
        }
    }

    uint64_t ring() const
    {
        return frame % FRAMES_IN_FLIGHT;
    }

    Allocation<float> readback(size_t size)
    {
        return readback_allocator[ring()]->allocate<float>(size);
    }

    Allocation<uint8_t> alloc(size_t size, Tensor::Type type)
    {
        return tensor_allocator->allocate<uint8_t>(size * (type == Tensor::Type::float16 ? sizeof(short) : sizeof(float)));
    }

    template <typename T>
    Allocation<T> struct_data()
    {
        auto alloc = struct_allocator[ring()]->allocate<T>(1);

        // submit command buffer if we run out of struct memory on the stack
        if (struct_allocator[ring()]->fallback_owns<T>(alloc))
        {
            struct_allocator[ring()]->free<T>(alloc);
            submit();
            alloc = struct_allocator[ring()]->allocate<T>(1);
        }

        return alloc;
    }

    template <typename T>
    Allocation<T> struct_data(size_t size)
    {
        auto alloc = struct_allocator[ring()]->allocate<T>(size);

        // submit command buffer if we run out of struct memory on the stack
        if (struct_allocator[ring()]->fallback_owns<T>(alloc))
        {
            struct_allocator[ring()]->free<T>(alloc);
            submit();
            alloc = struct_allocator[ring()]->allocate<T>(1);
        }

        return alloc;
    }

    void barrier(STAGE after, std::vector<Tensor> tensors)
    {
        for (auto iter = tensors_pending_writes.begin(); iter != tensors_pending_writes.end(); iter++)
        {
            auto stage = iter->first;
            for (auto tensor : tensors)
            {
                if (tensors_pending_writes[stage].count(tensor) != 0)
                {
                    gpuBarrier(cmd, stage, after);
                    tensors_pending_writes[stage].clear();
                    break;
                }
            }
        }
    }

    GpuCommandBuffer record()
    {
        if (!cmd)
        {
            cmd = gpuStartCommandRecording(queue);
        }
        return cmd;
    }

    virtual void pushMarker(const char* name) override
    {
        gpuBeginMarker(record(), name);
    }

    virtual void popMarker() override
    {
        if (cmd)
        {
            gpuEndMarker(cmd);
        }
    }

    virtual void submit() override
    {
        if (!cmd)
        {
            return; // no work to submit
        }
        if (!semaphore)
        {
            semaphore = gpuCreateSemaphore(device, 0);
        }
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, frame++);
        cmd = nullptr;

        if (frame > FRAMES_IN_FLIGHT)
        {
            uint64_t wait = frame - FRAMES_IN_FLIGHT;
            // auto stamp = std::chrono::high_resolution_clock::now();
            gpuWaitSemaphore(semaphore, wait);
            // auto delta = std::chrono::high_resolution_clock::now() - stamp;
            // std::cout << "\rWait: " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << "\t" << std::flush;

            for (auto it = cpu_callbacks.begin(); it != cpu_callbacks.end() && it->first <= wait;)
            {
                for (auto& [alloc, cb] : it->second)
                {
                    std::vector<float> data(alloc.size / sizeof(float));
                    memcpy(data.data(), alloc.cpu, alloc.size);
                    cb(data);
                }
                it = cpu_callbacks.erase(it);
            }

            for (auto it = pending_free.begin();
                 it != pending_free.end() && it->first <= wait;)
            {
                for (auto& allocation : it->second)
                {
                    tensor_allocator->free(allocation);
                }
                it = pending_free.erase(it);
            }

            wait = wait % FRAMES_IN_FLIGHT;
            readback_allocator[wait]->reset();
            struct_allocator[wait]->reset();
        }
    }

    void flush()
    {
        if (semaphore)
        {
            gpuWaitSemaphore(semaphore, frame - 1);
        }
    }

    void free(Allocation<uint8_t> allocation)
    {
        pending_free[frame].push_back(allocation);
    }

    Allocation<uint8_t> reduceAxis(uint64_t outer, uint64_t axis, uint64_t inner, Allocation<uint8_t> input, Tensor::Type type)
    {
        const uint64_t BLOCK = 256;
        Allocation<uint8_t> cur = input;
        uint64_t curAxis = axis;
        bool first = true;
        do
        {
            uint64_t groups = (curAxis + BLOCK - 1) / BLOCK;
            auto out = alloc(outer * groups * inner, type);

            auto data = struct_data<TensorReduceBlockData>();
            data.cpu->outer = outer;
            data.cpu->axis = curAxis;
            data.cpu->inner = inner;
            data.cpu->groups = groups;
            data.cpu->x = cur.gpu;
            data.cpu->y = out.gpu;

            auto c = record();
            if (!first)
            {
                // make the previous pass's partials visible to this one
                gpuBarrier(c, STAGE_COMPUTE, STAGE_COMPUTE);
                free(cur); // previous intermediate; freed deferred (after this frame)
            }
            gpuSetPipeline(c, pipelines["reduce_block"][type]);
            gpuDispatch(c, data.gpu, { static_cast<unsigned int>(groups), static_cast<unsigned int>(outer * inner), 1 });

            cur = out;
            curAxis = groups;
            first = false;
        } while (curAxis > 1);

        return cur;
    }

    GpuDevice device = nullptr;
    GpuQueue queue = nullptr;
    GpuCommandBuffer cmd = nullptr;
    GpuSemaphore semaphore = nullptr;
    uint64_t frame = 1;
    uint64_t rand_seed = 0;
    TensorAllocator* tensor_allocator = {};
    StructAllocator* struct_allocator[FRAMES_IN_FLIGHT] = {};
    ReadbackAllocator* readback_allocator[FRAMES_IN_FLIGHT] = {};
    std::map<uint64_t, std::vector<Allocation<uint8_t>>> pending_free;
    std::map<std::string, std::map<Tensor::Type, GpuPipeline>> pipelines;
    std::map<uint64_t, std::vector<std::pair<Allocation<float>, std::function<void(std::vector<float>)>>>> cpu_callbacks;
};

// Prevents grad tracking in backward pass
static bool g_grad_enabled = true;

NoGrad::NoGrad() : _previous(g_grad_enabled)
{
    g_grad_enabled = false;
}

NoGrad::~NoGrad()
{
    g_grad_enabled = _previous;
}

// Proxy for a tensor's backward closure
class GradFn
{
public:
    bool _enabled = false;

    GradFn& operator=(std::function<void(const Tensor&)> fn)
    {
        if (_enabled)
        {
            _fn = std::move(fn);
        }
        return *this;
    }

    explicit operator bool() const
    {
        return static_cast<bool>(_fn);
    }

    void operator()(const Tensor& grad) const
    {
        _fn(grad);
    }

private:
    std::function<void(const Tensor&)> _fn;
};

class Tensor_impl
{
public:
    ~Tensor_impl()
    {
        if (!_slice)
        {
            _device->free(_allocation);
        }
    }
    Device_impl* _device = nullptr;
    Allocation<uint8_t> _allocation;
    bool _slice = false;
    bool _requires_grad = false;
    Tensor grad;
    std::vector<Tensor> _prev;
    GradFn _backward;
};

Instance::Instance()
{
    gpuCreateInstance();
}

Instance::~Instance()
{
    tensors_pending_writes.clear();
    for (auto device : devices)
    {
        delete device;
    }
    gpuDestroyInstance();
}

Device* Instance::device(int index)
{
    auto dev = new Device_impl(index);
    devices.push_back(dev);
    return dev;
}

Tensor::~Tensor()
{
}

Tensor::Tensor(Tensor&& other) noexcept
{
    _shape = std::move(other._shape);
    _self = std::move(other._self);
    _type = other._type;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    _shape = std::move(other._shape);
    _self = std::move(other._self);
    _type = other._type;
    return *this;
}

Tensor::Tensor(Device_impl* device, std::vector<float> data, std::vector<Tensor> prev, Shape shape, Type type, bool slice)
    : _shape(shape), _type(Type::float32)
{
    if (_shape.empty())
    {
        _shape.push_back(data.size());
    }

    if (flatten(_shape) != data.size())
    {
        auto error = "cannot create tensor of size " + std::to_string(data.size()) + " with shape " + to_string(shape);
        throw std::runtime_error(error);
    }

    _self = std::make_shared<Tensor_impl>();
    _self->_device = device;
    _self->_allocation = _self->_device->alloc(data.size(), _type);
    _self->_slice = slice;

    bool requires_grad = false;
    if (g_grad_enabled)
    {
        for (auto& p : prev)
        {
            if (p._self && p._self->_requires_grad)
            {
                requires_grad = true;
                break;
            }
        }
    }
    _self->_requires_grad = requires_grad;
    _self->_backward._enabled = requires_grad;

    // Views must keep their parent alive so the shared buffer stays valid; other
    // tensors only retain their inputs when a backward pass will actually need them.
    if (requires_grad || slice)
    {
        _self->_prev = std::move(prev);
    }

    memcpy(_self->_allocation.cpu, data.data(), _self->_allocation.size);

    if (type == Type::float16)
    {
        *this = float16();
    }
}

Tensor::Tensor(Device_impl* device, Allocation<uint8_t> allocation, std::vector<Tensor> prev, Shape shape, Type type, bool slice)
    : _shape(shape), _type(type)
{
    _self = std::make_shared<Tensor_impl>(device, allocation, slice);

    bool requires_grad = false;
    if (g_grad_enabled)
    {
        for (auto& p : prev)
        {
            if (p._self && p._self->_requires_grad)
            {
                requires_grad = true;
                break;
            }
        }
    }
    _self->_requires_grad = requires_grad;
    _self->_backward._enabled = requires_grad;

    // Views must keep their parent alive so the shared buffer stays valid; other
    // tensors only retain their inputs when a backward pass will actually need them.
    if (requires_grad || slice)
    {
        _self->_prev = std::move(prev);
    }
}

bool Tensor::null() const
{
    return _shape.empty();
}

bool Tensor::requires_grad() const
{
    return _self && _self->_requires_grad;
}

Tensor& Tensor::requires_grad(bool value)
{
    _self->_requires_grad = value;
    _self->_backward._enabled = value;
    return *this;
}

void Tensor::zero() const
{
    _self->grad = {};
}

Shape Tensor::shape() const
{
    return _shape;
}

Tensor::Type Tensor::type() const
{
    return _type;
}

uint64_t Tensor::numel() const
{
    return flatten(_shape);
}

Tensor Tensor::grad() const
{
    return _self->grad;
}

std::string to_string(Allocation<float> readback, Shape shape, uint64_t offset = 0)
{
    std::string result = "[";
    auto front = shape.front();
    shape.erase(shape.begin());
    if (shape.empty())
    {
        for (size_t i = 0; i < front; i++)
        {
            result += std::to_string(readback.cpu[offset + i]) + ", ";
        }
        result.pop_back();
    }
    else
    {

        size_t stride = flatten(shape);
        for (size_t i = 0; i < front; i++)
        {
            result += to_string(readback, shape, offset + stride * i) + "\n";
        }
    }
    result.pop_back();

    return result + "]";
}

Tensor::operator std::string() const
{
    if (_type == Type::float16)
    {
        return std::string(float32());
    }

    auto size = flatten(_shape);
    auto readback = _self->_device->readback(size);
    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuMemCpy(cmd, readback.gpu, _self->_allocation.gpu, readback.size);
    _self->_device->submit();
    _self->_device->flush();

    return to_string(readback, _shape);
}

Device* Tensor::device()
{
    return _self->_device;
}

std::vector<float> Tensor::cpu()
{
    if (_type == Type::float16)
    {
        return float32().cpu();
    }

    auto size = flatten(_shape);
    auto readback = _self->_device->readback(size);
    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuMemCpy(cmd, readback.gpu, _self->_allocation.gpu, readback.size);
    _self->_device->submit();
    _self->_device->flush();

    std::vector<float> result(size);
    memcpy(result.data(), readback.cpu, readback.size);
    return result;
}

void Tensor::cpu(std::function<void(std::vector<float>)> cb)
{
    if (_type == Type::float16)
    {
        float32().cpu(cb);
        return;
    }

    auto size = flatten(_shape);
    auto readback = _self->_device->readback(size);
    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuMemCpy(cmd, readback.gpu, _self->_allocation.gpu, readback.size);

    _self->_device->cpu_callbacks[_self->_device->frame].push_back(std::make_pair(readback, cb));
}

Tensor Tensor::operator+(const Tensor& other) const
{
    if (null())
    {
        return other;
    }

    if (other.null())
    {
        return *this;
    }

    uint broadcast = 1;
    if (_shape != other._shape)
    {
        if (_shape.back() == other._shape.back() || other._shape == unit)
        {
            if (_shape.size() < other._shape.size())
            {
                throw std::runtime_error("cannot broadcast tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
            }
            broadcast = flatten(_shape) / flatten(other._shape);
        }
        else
        {
            throw std::runtime_error("cannot add tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
        }
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["add"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, other });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, broadcast](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad;

        if (broadcast > 1)
        {
            auto inner = static_cast<unsigned int>(flatten(other._shape));
            auto rows = grad.reshape({ broadcast, inner });
            auto reduced = rows.sum(0); // sum over the broadcast (leading) axis
            other._self->grad = other._self->grad + reduced.reshape(other._shape);
        }
        else
        {
            other._self->grad = other._self->grad + grad;
        }
    };
    return out;
}

Tensor Tensor::operator-() const
{
    return *this * -1;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    if (null())
    {
        return -other;
    }

    if (other.null())
    {
        return *this;
    }

    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    uint broadcast = 1;
    if (_shape != other._shape)
    {
        if (_shape.back() == other._shape.back() || other._shape == unit)
        {
            if (_shape.size() < other._shape.size())
            {
                throw std::runtime_error("cannot broadcast tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
            }
            broadcast = flatten(_shape) / flatten(other._shape);
        }
        else
        {
            throw std::runtime_error("cannot subtract tensor of shape " + to_string(other._shape) + " with tensor of shape " + to_string(_shape));
        }
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["sub"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, other });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, broadcast](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad;

        if (broadcast > 1)
        {
            auto inner = static_cast<unsigned int>(flatten(other._shape));
            auto rows = grad.reshape({ broadcast, inner });
            auto reduced = rows.sum(0); // sum over the broadcast (leading) axis
            other._self->grad = other._self->grad - reduced.reshape(other._shape);
        }
        else
        {
            other._self->grad = other._self->grad - grad;
        }
    };
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    uint broadcast = 1;
    if (_shape != other._shape)
    {
        if (_shape.back() == other._shape.back() || other._shape == unit)
        {
            if (_shape.size() < other._shape.size())
            {
                throw std::runtime_error("cannot broadcast tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
            }
            broadcast = flatten(_shape) / flatten(other._shape);
        }
        else
        {
            throw std::runtime_error("cannot multiply tensor of shape " + to_string(other._shape) + " with tensor of shape " + to_string(_shape));
        }
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["mul"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, other });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self._self->grad = self._self->grad + (grad * other);
        other._self->grad = other._self->grad + (grad * self);
    };
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    uint broadcast = 1;
    if (_shape != other._shape)
    {
        if (_shape.back() == other._shape.back() || other._shape == unit)
        {
            if (_shape.size() < other._shape.size())
            {
                throw std::runtime_error("cannot broadcast tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
            }
            broadcast = flatten(_shape) / flatten(other._shape);
        }
        else
        {
            throw std::runtime_error("cannot divide tensor of shape " + to_string(other._shape) + " by tensor of shape " + to_string(_shape));
        }
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["div"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, other });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, broadcast](const Tensor& grad)
    {
        self._self->grad = self._self->grad + (grad / other);

        if (broadcast > 1)
        {
            auto inner = static_cast<unsigned int>(flatten(other._shape));
            auto term = grad * (self / (other * other));
            auto rows = term.reshape({ broadcast, inner });
            auto reduced = rows.sum(0); // sum over the broadcast (leading) axis
            other._self->grad = other._self->grad - reduced.reshape(other._shape);
        }
        else
        {
            other._self->grad = other._self->grad - grad * (self / (other * other));
        }
    };
    return out;
}

Tensor Tensor::operator+(float x) const
{
    return *this + _self->_device->tensor({ x }, {}, _type);
}

Tensor Tensor::operator-(float x) const
{
    return *this - _self->_device->tensor({ x }, {}, _type);
}

Tensor Tensor::operator*(float x) const
{
    return *this * _self->_device->tensor({ x }, {}, _type);
}

Tensor Tensor::operator/(float x) const
{
    return *this / _self->_device->tensor({ x }, {}, _type);
}

Tensor Tensor::operator[](unsigned int i) const
{
    Shape res_shape = _shape;
    res_shape.erase(res_shape.begin());
    if (res_shape.empty())
    {
        res_shape = { 1 };
    }

    auto size = flatten(res_shape);

    Allocation<uint8_t> allocation = _self->_allocation;
    allocation.cpu += (size * i * (_type == Type::float16 ? sizeof(short) : sizeof(float)));
    allocation.gpu += (size * i * (_type == Type::float16 ? sizeof(short) : sizeof(float)));

    auto out = Tensor(_self->_device, allocation, { *this }, res_shape, _type, true);

    Tensor self = *this;
    out._self->_backward = [self, i](const Tensor& grad)
    {
        if (self._self->grad.null())
        {
            self._self->grad = self._self->_device->zeros(self._shape, self._type);
        }

        auto row = self._self->grad[i];
        row.copy(row + grad);

        // TODO: instead of slice bool, we should have a linked list in either Tensor or Tensor_Impl
        // that let's us walk up the ownership chain of slices/views and insert pending writes for everything in the chain
        tensors_pending_writes[STAGE_TRANSFER].insert(self._self->grad);
    };

    return out;
}

Tensor Tensor::repeat(const Tensor& tensor, Shape shape) const
{
    return _self->_device->repeat(tensor, shape);
}

Tensor Tensor::unfold(unsigned int k, unsigned int pad, Pad mode, unsigned int stride) const
{
    if (_shape.size() != 3)
    {
        throw std::runtime_error("unfold expects a (H, W, C) tensor, got shape " + to_string(_shape));
    }
    if (stride == 0)
    {
        throw std::runtime_error("unfold stride must be >= 1");
    }

    auto H = _shape[0];
    auto W = _shape[1];
    auto C = _shape[2];

    if (H + 2 * pad < k || W + 2 * pad < k)
    {
        throw std::runtime_error("unfold kernel larger than padded input for shape " + to_string(_shape));
    }

    unsigned int OH = (H + 2 * pad - k) / stride + 1;
    unsigned int OW = (W + 2 * pad - k) / stride + 1;

    Shape out_shape = { OH, OW, C, k, k };
    auto size = flatten(out_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorUnfoldData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->h = H;
    tensor_data.cpu->w = W;
    tensor_data.cpu->c = C;
    tensor_data.cpu->k = k;
    tensor_data.cpu->pad = pad;
    tensor_data.cpu->mode = static_cast<unsigned int>(mode);
    tensor_data.cpu->stride = stride;
    tensor_data.cpu->oh = OH;
    tensor_data.cpu->ow = OW;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["unfold"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, out_shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, k, pad, mode, stride, OH, OW](const Tensor& grad)
    {
        auto H = self._shape[0];
        auto W = self._shape[1];
        auto C = self._shape[2];
        auto in_size = flatten(self._shape);
        auto allocation = self._self->_device->alloc(in_size, self._type);

        auto tensor_data = self._self->_device->struct_data<TensorUnfoldData>();
        tensor_data.cpu->n = in_size;
        tensor_data.cpu->h = H;
        tensor_data.cpu->w = W;
        tensor_data.cpu->c = C;
        tensor_data.cpu->k = k;
        tensor_data.cpu->pad = pad;
        tensor_data.cpu->mode = static_cast<unsigned int>(mode);
        tensor_data.cpu->stride = stride;
        tensor_data.cpu->oh = OH;
        tensor_data.cpu->ow = OW;
        tensor_data.cpu->x = grad._self->_allocation.gpu; // grad_out (OH,OW,C,k,k)
        tensor_data.cpu->y = allocation.gpu;              // grad_in  (H,W,C)

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["unfold_backward"][self._type]);
        self._self->_device->barrier(STAGE_COMPUTE, { self, grad });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(in_size));

        auto term = Tensor(self._self->_device, allocation, {}, self._shape, self._type);

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = self._self->grad + term;
    };

    return out;
}

Tensor Tensor::sum(int dim, bool keepdim) const
{
    int rank = static_cast<int>(_shape.size());
    if (dim < 0)
    {
        dim += rank;
    }
    if (dim < 0 || dim >= rank)
    {
        throw std::runtime_error("sum dim out of range for shape " + to_string(_shape));
    }

    uint64_t outer = 1;
    for (int i = 0; i < dim; i++)
    {
        outer *= _shape[i];
    }
    uint64_t axis = _shape[dim];
    uint64_t inner = 1;
    for (int i = dim + 1; i < rank; i++)
    {
        inner *= _shape[i];
    }

    Shape out_shape;
    for (int i = 0; i < rank; i++)
    {
        if (i == dim)
        {
            if (keepdim)
            {
                out_shape.push_back(1);
            }
        }
        else
        {
            out_shape.push_back(_shape[i]);
        }
    }
    if (out_shape.empty())
    {
        out_shape.push_back(1);
    }

    _self->_device->record();
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    auto allocation = _self->_device->reduceAxis(outer, axis, inner, _self->_allocation, _type);

    auto out = Tensor(_self->_device, allocation, { *this }, out_shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, outer, axis, inner](const Tensor& grad)
    {
        // backward of sum is expand: broadcast grad back over the reduced axis
        auto in_size = outer * axis * inner;
        auto allocation = self._self->_device->alloc(in_size, self._type);

        auto tensor_data = self._self->_device->struct_data<TensorReduceData>();
        tensor_data.cpu->n = in_size;
        tensor_data.cpu->outer = outer;
        tensor_data.cpu->axis = axis;
        tensor_data.cpu->inner = inner;
        tensor_data.cpu->x = grad._self->_allocation.gpu;
        tensor_data.cpu->y = allocation.gpu;

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["expand"][self._type]);
        self._self->_device->barrier(STAGE_COMPUTE, { grad });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(in_size));

        auto term = Tensor(self._self->_device, allocation, {}, self._shape, self._type);

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = self._self->grad + term;
    };

    return out;
}

Tensor Tensor::max(int dim, bool keepdim) const
{
    int rank = static_cast<int>(_shape.size());
    if (dim < 0)
    {
        dim += rank;
    }
    if (dim < 0 || dim >= rank)
    {
        throw std::runtime_error("max dim out of range for shape " + to_string(_shape));
    }

    uint64_t outer = 1;
    for (int i = 0; i < dim; i++)
    {
        outer *= _shape[i];
    }
    uint64_t axis = _shape[dim];
    uint64_t inner = 1;
    for (int i = dim + 1; i < rank; i++)
    {
        inner *= _shape[i];
    }

    Shape out_shape;
    for (int i = 0; i < rank; i++)
    {
        if (i == dim)
        {
            if (keepdim)
            {
                out_shape.push_back(1);
            }
        }
        else
        {
            out_shape.push_back(_shape[i]);
        }
    }
    if (out_shape.empty())
    {
        out_shape.push_back(1);
    }

    auto size = outer * inner; // == flatten(out_shape)
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorReduceData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->outer = outer;
    tensor_data.cpu->axis = axis;
    tensor_data.cpu->inner = inner;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["reduce_max"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, out_shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, outer, axis, inner](const Tensor& grad)
    {
        // backward of max routes grad only to the (first) argmax element of each group
        auto in_size = outer * axis * inner;
        auto allocation = self._self->_device->alloc(in_size, self._type);

        auto tensor_data = self._self->_device->struct_data<TensorReduceData>();
        tensor_data.cpu->n = in_size;
        tensor_data.cpu->outer = outer;
        tensor_data.cpu->axis = axis;
        tensor_data.cpu->inner = inner;
        tensor_data.cpu->x = self._self->_allocation.gpu; // original input
        tensor_data.cpu->y = allocation.gpu;              // grad_in
        tensor_data.cpu->z = grad._self->_allocation.gpu; // grad_out

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["reduce_max_backward"][self._type]);
        self._self->_device->barrier(STAGE_COMPUTE, { self, grad });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(in_size));

        auto term = Tensor(self._self->_device, allocation, {}, self._shape, self._type);

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = self._self->grad + term;
    };

    return out;
}

Tensor Tensor::broadcast(int dim, unsigned int size) const
{
    int rank = static_cast<int>(_shape.size());
    if (dim < 0)
    {
        dim += rank;
    }
    if (dim < 0 || dim >= rank)
    {
        throw std::runtime_error("broadcast dim out of range for shape " + to_string(_shape));
    }
    if (_shape[dim] != 1)
    {
        throw std::runtime_error("broadcast expects size 1 along dim " + std::to_string(dim) + ", got shape " + to_string(_shape));
    }

    uint64_t outer = 1;
    for (int i = 0; i < dim; i++)
    {
        outer *= _shape[i];
    }
    uint64_t inner = 1;
    for (int i = dim + 1; i < rank; i++)
    {
        inner *= _shape[i];
    }
    uint64_t axis = size;

    Shape out_shape = _shape;
    out_shape[dim] = size;

    auto out_size = outer * axis * inner;
    auto allocation = _self->_device->alloc(out_size, _type);

    auto tensor_data = _self->_device->struct_data<TensorReduceData>();
    tensor_data.cpu->n = out_size;
    tensor_data.cpu->outer = outer;
    tensor_data.cpu->axis = axis;
    tensor_data.cpu->inner = inner;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["expand"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(out_size));

    auto out = Tensor(_self->_device, allocation, { *this }, out_shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, outer, axis, inner](const Tensor& grad)
    {
        // backward of expand is sum: reduce grad over the broadcast axis
        auto in_size = outer * inner;
        auto allocation = self._self->_device->alloc(in_size, self._type);

        auto tensor_data = self._self->_device->struct_data<TensorReduceData>();
        tensor_data.cpu->n = in_size;
        tensor_data.cpu->outer = outer;
        tensor_data.cpu->axis = axis;
        tensor_data.cpu->inner = inner;
        tensor_data.cpu->x = grad._self->_allocation.gpu;
        tensor_data.cpu->y = allocation.gpu;

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["reduce_sum"][self._type]);
        self._self->_device->barrier(STAGE_COMPUTE, { grad });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(in_size));

        auto term = Tensor(self._self->_device, allocation, {}, self._shape, self._type);

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = self._self->grad + term;
    };

    return out;
}

Tensor Tensor::cat(const Tensor& other, int dim) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("cat: type mismatch");
    }
    if (_shape.size() != other._shape.size())
    {
        throw std::runtime_error("cat: rank mismatch between " + to_string(_shape) + " and " + to_string(other._shape));
    }

    int rank = static_cast<int>(_shape.size());
    if (dim < 0)
    {
        dim += rank;
    }
    if (dim < 0 || dim >= rank)
    {
        throw std::runtime_error("cat: dim out of range for shape " + to_string(_shape));
    }
    for (int i = 0; i < rank; i++)
    {
        if (i != dim && _shape[i] != other._shape[i])
        {
            throw std::runtime_error("cat: shapes must match except along dim, got " + to_string(_shape) + " and " + to_string(other._shape));
        }
    }

    uint64_t outer = 1;
    for (int i = 0; i < dim; i++)
    {
        outer *= _shape[i];
    }
    uint64_t inner = 1;
    for (int i = dim + 1; i < rank; i++)
    {
        inner *= _shape[i];
    }
    uint64_t a_axis = _shape[dim];
    uint64_t b_axis = other._shape[dim];
    uint64_t total = a_axis + b_axis;

    Shape out_shape = _shape;
    out_shape[dim] = static_cast<unsigned int>(total);

    auto size = flatten(out_shape);
    auto allocation = _self->_device->alloc(size, _type);

    // scatter this tensor into [0, a_axis) and `other` into [a_axis, total).
    auto scatter = [&](const Tensor& src, uint64_t axis, uint64_t offset)
    {
        auto slab = outer * axis * inner;
        auto tensor_data = _self->_device->struct_data<TensorConcatData>();
        tensor_data.cpu->n = slab;
        tensor_data.cpu->outer = outer;
        tensor_data.cpu->inner = inner;
        tensor_data.cpu->axis = axis;
        tensor_data.cpu->total = total;
        tensor_data.cpu->offset = offset;
        tensor_data.cpu->x = src._self->_allocation.gpu;
        tensor_data.cpu->y = allocation.gpu;

        auto cmd = _self->_device->record();
        gpuSetPipeline(cmd, _self->_device->pipelines["concat_scatter"][_type]);
        _self->_device->barrier(STAGE_COMPUTE, { src });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(slab));
    };

    scatter(*this, a_axis, 0);
    scatter(other, b_axis, a_axis);

    auto out = Tensor(_self->_device, allocation, { *this, other }, out_shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, outer, inner, a_axis, b_axis, total](const Tensor& grad)
    {
        // backward of concat is a slice: gather each slab of grad_out back to its input.
        auto gather = [&](const Tensor& input, uint64_t axis, uint64_t offset)
        {
            auto slab = outer * axis * inner;
            auto allocation = input._self->_device->alloc(slab, input._type);

            auto tensor_data = input._self->_device->struct_data<TensorConcatData>();
            tensor_data.cpu->n = slab;
            tensor_data.cpu->outer = outer;
            tensor_data.cpu->inner = inner;
            tensor_data.cpu->axis = axis;
            tensor_data.cpu->total = total;
            tensor_data.cpu->offset = offset;
            tensor_data.cpu->x = grad._self->_allocation.gpu; // grad_out (wide)
            tensor_data.cpu->y = allocation.gpu;              // grad_in  (slab)

            auto cmd = input._self->_device->record();
            gpuSetPipeline(cmd, input._self->_device->pipelines["concat_gather"][input._type]);
            input._self->_device->barrier(STAGE_COMPUTE, { grad });
            gpuDispatch(cmd, tensor_data.gpu, grid1d(slab));

            auto term = Tensor(input._self->_device, allocation, {}, input._shape, input._type);

            tensors_pending_writes[STAGE_COMPUTE].insert(term);

            input._self->grad = input._self->grad + term;
        };

        gather(self, a_axis, 0);
        gather(other, b_axis, a_axis);
    };

    return out;
}

Tensor Tensor::mT() const
{
    if (_shape.size() != 2)
    {
        auto error = "cannot transpose a tensor that isn't a matrix";
        throw std::runtime_error(error);
    }

    Shape res_shape = { _shape.back(), _shape.front() };
    auto size = flatten(res_shape);

    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorTransposeData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->r = _shape.front();
    tensor_data.cpu->c = _shape.back();
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["mT"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, res_shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    return out;
}

Tensor Tensor::dot(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    if (_shape != other._shape)
    {
        auto error = "cannot compute the dot product of tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(1, _type);
    memset(allocation.cpu, 0, _type == Type::float16 ? sizeof(short) : sizeof(float));

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["dot"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, other });
    gpuDispatch(cmd, tensor_data.gpu, { 1, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, { 1 }, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self._self->grad = self._self->grad + other * grad;
        other._self->grad = other._self->grad + self * grad;
    };
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    if (_shape.size() != 2 || other._shape.size() != 2)
    {
        std::string error = "batched matmul not yet implemented";
        throw std::runtime_error(error);
    }

    // (a,b) mat and (c,d) mat results in (a,d) mat, and b must equal c
    if (_shape.back() != other._shape.front())
    {
        auto error = "cannot matmul tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    Shape res_shape = { _shape.front(), other._shape.back() };
    auto size = flatten(res_shape);

    const uint a = _shape.front();
    const uint b = _shape.back();
    const uint c = other._shape.back();

    // Split-K: when the output is too small to fill the GPU (few 16x16 tiles) but
    // the contraction is huge, partition the contraction across grid.z workgroups
    // into partial products, then sum the partials with the parallel reduction.
    // This is the weight-gradient case (cols^T @ grad): the contraction M = H*W
    // dwarfs the tiny K x N output, so the plain tiled matmul used only a handful
    // of workgroups each looping the entire contraction serially.
    const uint outTiles = ((a + 15) / 16) * ((c + 15) / 16);
    uint splits = 1;
    if (outTiles < 256 && b > 8192)
    {
        splits = (b + 8191) / 8192;
        if (splits > 64)
        {
            splits = 64;
        }
    }

    Allocation<uint8_t> allocation;
    if (splits > 1)
    {
        auto partials = _self->_device->alloc(static_cast<uint64_t>(splits) * a * c, _type);

        auto md = _self->_device->struct_data<TensorMatMulData>();
        md.cpu->n = static_cast<uint64_t>(splits) * a * c;
        md.cpu->a = a;
        md.cpu->b = b;
        md.cpu->c = c;
        md.cpu->splits = splits;
        md.cpu->transposeA = 0;
        md.cpu->x = _self->_allocation.gpu;
        md.cpu->y = other._self->_allocation.gpu;
        md.cpu->z = partials.gpu;

        auto cmd = _self->_device->record();
        gpuSetPipeline(cmd, _self->_device->pipelines["matmul_splitk"][_type]);
        _self->_device->barrier(STAGE_COMPUTE, { *this, other });
        gpuDispatch(cmd, md.gpu, { (c + 15) / 16, (a + 15) / 16, splits });

        // sum the `splits` partials (splits, a*c) down to the (a, c) result.
        gpuBarrier(_self->_device->record(), STAGE_COMPUTE, STAGE_COMPUTE);
        allocation = _self->_device->reduceAxis(1, splits, static_cast<uint64_t>(a) * c, partials, _type);
        _self->_device->free(partials);
    }
    else
    {
        allocation = _self->_device->alloc(size, _type);

        auto tensor_data = _self->_device->struct_data<TensorMatMulData>();
        tensor_data.cpu->n = size;
        tensor_data.cpu->a = a;
        tensor_data.cpu->b = b;
        tensor_data.cpu->c = c;
        tensor_data.cpu->splits = 1;
        tensor_data.cpu->x = _self->_allocation.gpu;
        tensor_data.cpu->y = other._self->_allocation.gpu;
        tensor_data.cpu->z = allocation.gpu;

        auto cmd = _self->_device->record();
        gpuSetPipeline(cmd, _self->_device->pipelines["matmul"][_type]);
        _self->_device->barrier(STAGE_COMPUTE, { *this, other });
        unsigned int rowTiles = (a + 15) / 16;
        gpuDispatch(cmd, tensor_data.gpu, { (c + 15) / 16, rowTiles < kMaxGroups ? rowTiles : kMaxGroups, 1 });
    }

    auto out = Tensor(_self->_device, allocation, { *this, other }, res_shape, _type);
    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad.matmul(other.mT());
        // weight gradient cols^T @ grad, fused so cols^T is never materialized.
        other._self->grad = other._self->grad + self.tmatmul(grad);
    };
    return out;
}

Tensor Tensor::tmatmul(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }
    if (_shape.size() != 2 || other._shape.size() != 2)
    {
        throw std::runtime_error("tmatmul expects 2D tensors");
    }
    if (_shape.front() != other._shape.front())
    {
        throw std::runtime_error("tmatmul contraction mismatch between " + to_string(_shape) + " and " + to_string(other._shape));
    }

    const uint a = _shape.back();       // K (rows of result)
    const uint b = _shape.front();      // M (contraction)
    const uint c = other._shape.back(); // N (cols of result)

    uint splits = (b + 8191) / 8192;
    if (splits < 1)
    {
        splits = 1;
    }
    if (splits > 64)
    {
        splits = 64;
    }

    Shape res_shape = { a, c };

    Allocation<uint8_t> allocation;
    if (splits > 1)
    {
        auto partials = _self->_device->alloc(static_cast<uint64_t>(splits) * a * c, _type);

        auto md = _self->_device->struct_data<TensorMatMulData>();
        md.cpu->n = static_cast<uint64_t>(splits) * a * c;
        md.cpu->a = a;
        md.cpu->b = b;
        md.cpu->c = c;
        md.cpu->splits = splits;
        md.cpu->transposeA = 1;
        md.cpu->x = _self->_allocation.gpu;
        md.cpu->y = other._self->_allocation.gpu;
        md.cpu->z = partials.gpu;

        auto cmd = _self->_device->record();
        gpuSetPipeline(cmd, _self->_device->pipelines["matmul_splitk"][_type]);
        _self->_device->barrier(STAGE_COMPUTE, { *this, other });
        gpuDispatch(cmd, md.gpu, { (c + 15) / 16, (a + 15) / 16, splits });

        gpuBarrier(_self->_device->record(), STAGE_COMPUTE, STAGE_COMPUTE);
        allocation = _self->_device->reduceAxis(1, splits, static_cast<uint64_t>(a) * c, partials, _type);
        _self->_device->free(partials);
    }
    else
    {
        allocation = _self->_device->alloc(static_cast<uint64_t>(a) * c, _type);

        auto md = _self->_device->struct_data<TensorMatMulData>();
        md.cpu->n = static_cast<uint64_t>(a) * c;
        md.cpu->a = a;
        md.cpu->b = b;
        md.cpu->c = c;
        md.cpu->splits = 1;
        md.cpu->transposeA = 1;
        md.cpu->x = _self->_allocation.gpu;
        md.cpu->y = other._self->_allocation.gpu;
        md.cpu->z = allocation.gpu;

        auto cmd = _self->_device->record();
        gpuSetPipeline(cmd, _self->_device->pipelines["matmul_splitk"][_type]);
        _self->_device->barrier(STAGE_COMPUTE, { *this, other });
        gpuDispatch(cmd, md.gpu, { (c + 15) / 16, (a + 15) / 16, 1 });
    }

    auto out = Tensor(_self->_device, allocation, {}, res_shape, _type);
    tensors_pending_writes[STAGE_COMPUTE].insert(out);
    return out;
}

Tensor Tensor::affine(const Tensor& weights, const Tensor& biases) const
{
    if (_type != Type::float16 || weights._type != Type::float16 || biases._type != Type::float32)
    {
        throw std::runtime_error("affine requires fp16 activations and weights, and fp32 biases");
    }

    if (_shape.size() != 2 || weights._shape.size() != 2 || biases._shape.size() != 2)
    {
        throw std::runtime_error("batched affine not yet implemented");
    }

    // (a,b) @ (b,c) -> (a,c), with biases broadcast as (1,c).
    if (_shape.back() != weights._shape.front() || weights._shape.back() != biases._shape.back())
    {
        auto error = "cannot apply affine transformation of weights of shape " + to_string(weights._shape) + " to activations of shape " + to_string(_shape) + " with biases of shape " + to_string(biases._shape);
        throw std::runtime_error(error);
    }

    bool wmma = _shape.front() % 16 == 0 && _shape.back() % 16 == 0 && weights._shape.back() % 16 == 0;

    // No cooperative-matrix path for these shapes: fall back to a plain matmul
    // followed by a (broadcast) bias add.
    if (!wmma)
    {
        return matmul(weights) + biases.float16();
    }

    Shape res_shape = { _shape.front(), weights._shape.back() };
    auto size = flatten(res_shape);

    auto allocation = _self->_device->alloc(size, Type::float16);

    auto tensor_data = _self->_device->struct_data<TensorAffineData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->a = _shape.front();
    tensor_data.cpu->b = _shape.back();
    tensor_data.cpu->c = weights._shape.back();
    tensor_data.cpu->x = _self->_allocation.gpu;         // activations (fp16)
    tensor_data.cpu->y = weights._self->_allocation.gpu; // weights (fp16)
    tensor_data.cpu->z = biases._self->_allocation.gpu;  // biases (fp32)
    tensor_data.cpu->w = allocation.gpu;                 // output (fp16)

    auto cmd = _self->_device->record();
    uint tiles_row = res_shape.front() / 16;
    uint tiles_col = res_shape.back() / 16;
    gpuSetPipeline(cmd, _self->_device->pipelines["affine"][Type::float16]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, weights, biases });
    gpuDispatch(cmd, tensor_data.gpu, { tiles_row * tiles_col, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, weights, biases }, res_shape, Type::float16);
    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    Tensor w = weights;
    Tensor b = biases;
    uint rows = res_shape.front();
    out._self->_backward = [self, w, b, rows](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad.matmul(w.mT());
        w._self->grad = w._self->grad + self.mT().matmul(grad);

        auto inner = static_cast<uint>(flatten(b._shape));
        auto reduced = grad.reshape({ rows, inner }).sum(0).reshape(b._shape);
        reduced = (b._type == Type::float16) ? reduced.float16() : reduced.float32();
        b._self->grad = b._self->grad + reduced;
    };
    return out;
}

Tensor Tensor::pow(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    uint broadcast = 1;
    if (_shape != other._shape)
    {
        if (_shape.back() == other._shape.back() || other._shape == unit)
        {
            if (_shape.size() < other._shape.size())
            {
                throw std::runtime_error("cannot broadcast tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
            }
            broadcast = flatten(_shape) / flatten(other._shape);
        }
        else
        {
            throw std::runtime_error("cannot pow tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape));
        }
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["pow"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, other });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    Tensor result = out.detach();
    out._self->_backward = [self, other, result](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad * other * self.pow(other - 1.f);
        other._self->grad = other._self->grad + grad * result * self.log();
    };
    return out;
}

Tensor Tensor::pow(float x) const
{
    return pow(_self->_device->tensor({ x }));
}

Tensor Tensor::mse(const Tensor& other) const
{
    auto dif = *this - other;
    auto squared = dif * dif;
    if (squared._type == Type::float16)
    {
        squared = squared.float32();
    }
    return squared.sum() / static_cast<float>(flatten(dif._shape));
}

Tensor Tensor::sum() const
{
    auto size = flatten(_shape);

    _self->_device->record();
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    auto allocation = _self->_device->reduceAxis(1, size, 1, _self->_allocation, _type);

    auto out = Tensor(_self->_device, allocation, { *this }, { 1 }, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        auto ones = self._self->_device->ones(self._shape, self._type);
        self._self->grad = self._self->grad + ones * grad;
    };
    return out;
}

Tensor Tensor::sqrt() const
{
    return pow(0.5f);
}

Tensor Tensor::rcp() const
{
    return _self->_device->repeat({ 1.f }, _shape) / *this;
}

Tensor Tensor::exp() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["exp"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad * self.exp();
    };
    return out;
}

Tensor Tensor::expm1() const
{
    return exp() - 1.f;
}

Tensor Tensor::log() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["log"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad / self;
    };
    return out;
}

Tensor Tensor::log1p() const
{
    return (*this + 1).log();
}

Tensor Tensor::sin() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["sin"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad * self.cos();
    };
    return out;
}

Tensor Tensor::cos() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["cos"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad * -self.sin();
    };
    return out;
}

Tensor Tensor::tan() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["tan"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad / self.cos().pow(2.f);
    };
    return out;
}

Tensor Tensor::cosh() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["cosh"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    return out;
}

Tensor Tensor::tanh() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["tanh"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad / self.cosh().pow(2.f);
    };
    return out;
}

Tensor Tensor::sech() const
{
    return cosh().rcp();
}

Tensor Tensor::relu(float alpha) const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->a = alpha;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["relu"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, alpha](const Tensor& grad)
    {
        auto size = flatten(self.shape());
        auto allocation = self._self->_device->alloc(size, self._type);
        auto tensor_data = self._self->_device->struct_data<TensorData>();
        tensor_data.cpu->n = size;
        tensor_data.cpu->a = alpha;
        tensor_data.cpu->x = self._self->_allocation.gpu;
        tensor_data.cpu->y = grad._self->_allocation.gpu;
        tensor_data.cpu->z = allocation.gpu;

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["relu_backward"][self._type]);
        self._self->_device->barrier(STAGE_COMPUTE, { self, grad });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

        auto term = Tensor(self._self->_device, allocation, {}, self.shape(), self._type);

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = self._self->grad + term;
    };
    return out;
}

Tensor Tensor::gelu() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["gelu"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        auto size = flatten(self.shape());
        auto allocation = self._self->_device->alloc(size, self._type);
        auto tensor_data = self._self->_device->struct_data<TensorData>();
        tensor_data.cpu->n = size;
        tensor_data.cpu->x = self._self->_allocation.gpu;
        tensor_data.cpu->y = grad._self->_allocation.gpu;
        tensor_data.cpu->z = allocation.gpu;

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["gelu_backward"][self._type]);
        self._self->_device->barrier(STAGE_COMPUTE, { self, grad });
        gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

        auto term = Tensor(self._self->_device, allocation, {}, self.shape(), self._type);

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = self._self->grad + term;
    };
    return out;
}

Tensor Tensor::softmax() const
{
    auto ex = exp();
    return ex / ex.sum();
}

Tensor Tensor::float16() const
{
    if (_type == Type::float16)
    {
        return *this;
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, Type::float16);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["cast"][Type::float16]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, Type::float16);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad.float32();
    };
    return out;
}

Tensor Tensor::float32() const
{
    if (_type == Type::float32)
    {
        return *this;
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, Type::float32);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["cast"][Type::float32]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, _shape, Type::float32);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad.float16();
    };
    return out;
}

Tensor Tensor::adam(Tensor& mean, Tensor& variance, uint64_t steps, float b1, float b2)
{
    if (_type != mean._type || _type != variance._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto tensor_data = _self->_device->struct_data<TensorAdamData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->b1 = b1;
    tensor_data.cpu->b2 = b2;
    tensor_data.cpu->b1t = ::pow(b1, steps);
    tensor_data.cpu->b2t = ::pow(b2, steps);
    tensor_data.cpu->grad = _self->_allocation.gpu;
    tensor_data.cpu->mean = mean._self->_allocation.gpu;
    tensor_data.cpu->variance = variance._self->_allocation.gpu;
    tensor_data.cpu->adjustment = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["adam"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this, mean, variance });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this, mean, variance }, _shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert({ mean, variance, out });

    return out;
}

Tensor Tensor::reshape(Shape shape) const
{
    if (flatten(_shape) != flatten(shape))
    {
        auto error = "cannot reshape tensor of shape " + to_string(_shape) + " as shape " + to_string(shape);
        throw std::runtime_error(error);
    }

    auto out = Tensor(_self->_device, _self->_allocation, { *this }, shape, _type, true);

    for (auto& [stage, tensors] : tensors_pending_writes)
    {
        if (tensors.count(*this) != 0)
        {
            tensors.insert(out);
        }
    }

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad.reshape(self._shape);
    };

    return out;
}

Tensor Tensor::permute(Shape permutation) const
{
    if (_shape.size() != permutation.size())
    {
        auto error = "cannot permute tensor of shape " + to_string(_shape) + " with permutation " + to_string(permutation);
        throw std::runtime_error(error);
    }

    Shape shape;
    std::vector<bool> seen(_shape.size(), false);
    for (size_t i = 0; i < permutation.size(); i++)
    {
        auto axis = permutation[i];
        if (axis >= _shape.size() || seen[axis])
        {
            auto error = "cannot permute tensor of shape " + to_string(_shape) + " with permutation " + to_string(permutation);
            throw std::runtime_error(error);
        }
        seen[axis] = true;
        shape.push_back(_shape[axis]);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->alloc(size, _type);

    auto shape_data = _self->_device->struct_data<uint32_t>(_shape.size());
    memcpy(shape_data.cpu, _shape.data(), _shape.size() * sizeof(uint32_t));

    auto permutation_data = _self->_device->struct_data<uint32_t>(permutation.size());
    memcpy(permutation_data.cpu, permutation.data(), permutation.size() * sizeof(uint32_t));

    auto tensor_data = _self->_device->struct_data<TensorPermuteData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = _shape.size();
    tensor_data.cpu->s = shape_data.gpu;
    tensor_data.cpu->t = permutation_data.gpu;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["permute"][_type]);
    _self->_device->barrier(STAGE_COMPUTE, { *this });
    gpuDispatch(cmd, tensor_data.gpu, grid1d(size));

    auto out = Tensor(_self->_device, allocation, { *this }, shape, _type);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Shape inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); i++)
    {
        inverse[permutation[i]] = static_cast<unsigned int>(i);
    }

    Tensor self = *this;
    out._self->_backward = [self, inverse](const Tensor& grad)
    {
        self._self->grad = self._self->grad + grad.permute(inverse);
    };

    return out;
}

Tensor Tensor::detach() const
{
    auto size = flatten(_shape);
    auto byte_size = size * (_type == Type::float16 ? sizeof(short) : sizeof(float));
    auto allocation = _self->_device->alloc(size, _type);

    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_TRANSFER, { *this });
    gpuMemCpy(cmd, allocation.gpu, _self->_allocation.gpu, byte_size);

    auto out = Tensor(_self->_device, allocation, {}, _shape, _type);

    tensors_pending_writes[STAGE_TRANSFER].insert(out);

    return out;
}

void Tensor::copy(const Tensor& other) const
{
    if (_type != other._type)
    {
        throw std::runtime_error("Implicit type conversion not yet supported");
    }

    if (flatten(_shape) != flatten(other._shape))
    {
        auto error = "cannot copy tensor of shape " + to_string(_shape) + " from shape " + to_string(other._shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto byte_size = size * (_type == Type::float16 ? sizeof(short) : sizeof(float));

    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_TRANSFER, { other });
    gpuMemCpy(cmd, _self->_allocation.gpu, other._self->_allocation.gpu, byte_size);

    tensors_pending_writes[STAGE_TRANSFER].insert(*this);
}

void Tensor::build(Tensor tensor, std::set<Tensor>& visited, std::vector<Tensor>& graph)
{
    if (visited.count(tensor) == 0)
    {
        visited.insert(tensor);
        for (auto& prev : tensor._self->_prev)
        {
            build(prev, visited, graph);
        }
        graph.push_back(tensor);
    }
}

void Tensor::backward()
{
    std::set<Tensor> visited;
    std::vector<Tensor> graph;
    build(*this, visited, graph);

    _self->grad = _self->_device->ones(_shape, _type);

    // No grad scope lock to prevent grad accumulation in backward
    NoGrad no_grad;
    for (auto iter = graph.rbegin(); iter != graph.rend(); iter++)
    {
        if (iter->_self->_backward)
        {
            iter->_self->_backward(iter->_self->grad);
        }
    }
}

void Module::save(std::filesystem::path path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to open model file for writing: " + path.string());
    }
    auto model = parameters();
    for (auto tensor : model)
    {
        auto data = tensor.cpu();
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }
    if (!file)
    {
        throw std::runtime_error("failed while writing model file: " + path.string());
    }
}

void Module::load(std::filesystem::path path)
{
    std::ifstream file(path, std::ios::binary);
    auto model = parameters();
    for (auto tensor : model)
    {
        std::vector<float> data(tensor.numel());
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        auto temp = tensor.device()->tensor(data, tensor.shape());
        tensor.copy(tensor.type() == Tensor::Type::float32 ? temp : temp.float16());
    }
}

void Adam::save(std::filesystem::path path)
{
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&_steps), sizeof(uint64_t));

    for (auto tensor : _mean)
    {
        auto data = tensor.cpu();
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }

    for (auto tensor : _variance)
    {
        auto data = tensor.cpu();
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }
}

void Adam::load(std::filesystem::path path)
{
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(&_steps), sizeof(uint64_t));

    for (auto tensor : _mean)
    {
        std::vector<float> data(tensor.numel());
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        auto temp = tensor.device()->tensor(data, tensor.shape());
        tensor.copy(tensor.type() == Tensor::Type::float32 ? temp : temp.float16());
    }

    for (auto tensor : _variance)
    {
        std::vector<float> data(tensor.numel());
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        auto temp = tensor.device()->tensor(data, tensor.shape());
        tensor.copy(tensor.type() == Tensor::Type::float32 ? temp : temp.float16());
    }
}
