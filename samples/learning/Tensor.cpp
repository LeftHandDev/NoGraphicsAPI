#include "Common.h"
#include "Tensor.h"
#include "Utilities.h"
#include <map>
#include <algorithm>
#include <random>
#include <iostream>
#include <set>
#include <chrono>

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

void broadcast(Tensor tensor, const Tensor& rec, const Tensor& send, std ::function<Tensor(const Tensor& a, const Tensor& b)> op)
{
    if (rec.shape().empty())
    {
        auto error = "Unable to broadcast to empty tensor";
        throw std::runtime_error(error);
    }

    if (rec.shape() != send.shape())
    {
        for (size_t i = 0; i < rec.shape().front(); i++)
        {
            broadcast(tensor[i], rec[i], send, op);
        }
    }
    else
    {
        return tensor.copy(op(rec, send));
    }
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
        pipelines["add"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_add");
        pipelines["sub"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_sub");
        pipelines["mul"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_mul");
        pipelines["div"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_div");
        pipelines["dot"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_dot");
        pipelines["mT"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_mT");
        pipelines["matmul"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_matmul");
        pipelines["pow"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_pow");
        pipelines["cosh"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_cosh");
        pipelines["tanh"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_tanh");
        pipelines["relu"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_relu");
        pipelines["relu_backward"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_relu_backward");
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

        for (auto [entry, pipeline] : pipelines)
        {
            gpuFreePipeline(pipeline);
        }

        auto iter = std::remove(devices.begin(), devices.end(), reinterpret_cast<Device*>(this));
        if (iter == devices.end())
        {
            // device tracking error, should not happen
        }
        gpuDestroyDevice(device);
    }

    virtual Tensor tensor(std::vector<float> data, Shape shape = {}) override
    {
        return Tensor(this, data, {}, shape);
    }

    virtual Tensor rand(Shape shape) override
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.f, 1.f);

        std::vector<float> data(flatten(shape));
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i] = dis(gen);
        }

        return Tensor(this, data, {}, shape);
    }

    virtual Tensor zeros(Shape shape) override
    {
        std::vector<float> data(flatten(shape), 0.f);
        return Tensor(this, data, {}, shape);
    }

    virtual Tensor ones(Shape shape) override
    {
        std::vector<float> data(flatten(shape), 1.f);
        return Tensor(this, data, {}, shape);
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
        return readback_allocator[ring()]->allocate<float>(size * sizeof(float));
    }

    Allocation<float> floats(size_t size)
    {
        return tensor_allocator->allocate<float>(size);
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

    void barrier(STAGE after, Tensor a, Tensor b)
    {
        for (auto iter = tensors_pending_writes.begin(); iter != tensors_pending_writes.end(); iter++)
        {
            auto stage = iter->first;
            if (tensors_pending_writes[stage].count(a) != 0 || tensors_pending_writes[stage].count(b) != 0)
            {
                gpuBarrier(cmd, stage, after);
                tensors_pending_writes[stage].clear();
            }
        }
    }

    void barrier(STAGE after, Tensor a)
    {
        for (auto iter = tensors_pending_writes.begin(); iter != tensors_pending_writes.end(); iter++)
        {
            auto stage = iter->first;
            if (tensors_pending_writes[stage].count(a) != 0)
            {
                gpuBarrier(cmd, stage, after);
                tensors_pending_writes[stage].clear();
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

        gpuSubmit(queue, Span(&cmd, 1), semaphore, frame++);
        cmd = nullptr;

        if (frame > FRAMES_IN_FLIGHT)
        {
            uint64_t wait = frame - FRAMES_IN_FLIGHT;
            // auto stamp = std::chrono::high_resolution_clock::now();
            gpuWaitSemaphore(semaphore, wait);
            // auto delta = std::chrono::high_resolution_clock::now() - stamp;
            // std::cout << "\rWait: " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << "\t" << std::flush;

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

    void free(Allocation<float> allocation)
    {
        pending_free[frame].push_back(allocation);
    }

    GpuDevice device = nullptr;
    GpuQueue queue = nullptr;
    GpuCommandBuffer cmd = nullptr;
    GpuSemaphore semaphore = nullptr;
    uint64_t frame = 1;
    TensorAllocator* tensor_allocator = {};
    StructAllocator* struct_allocator[FRAMES_IN_FLIGHT] = {};
    ReadbackAllocator* readback_allocator[FRAMES_IN_FLIGHT] = {};
    std::map<uint64_t, std::vector<Allocation<float>>> pending_free;
    std::map<std::string, GpuPipeline> pipelines;
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
    Allocation<float> _allocation;
    bool _slice = false;
    Tensor grad;
    std::vector<Tensor> _prev;
    std::function<void(const Tensor&)> _backward;
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
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    _shape = std::move(other._shape);
    _self = std::move(other._self);
    return *this;
}

Tensor::Tensor(Device_impl* device, std::vector<float> data, std::vector<Tensor> prev, Shape shape, bool slice)
    : _shape(shape)
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
    _self->_allocation = _self->_device->floats(data.size());
    _self->_slice = slice;
    _self->_prev = std::move(prev);
    memcpy(_self->_allocation.cpu, data.data(), data.size() * sizeof(float));
}

Tensor::Tensor(Device_impl* device, Allocation<float> allocation, std::vector<Tensor> prev, Shape shape, bool slice)
    : _shape(shape)
{
    _self = std::make_shared<Tensor_impl>(device, allocation, slice);
    _self->_prev = std::move(prev);
}

bool Tensor::null() const
{
    return _shape.empty();
}

Shape Tensor::shape() const
{
    return _shape;
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
    auto size = flatten(_shape);
    auto byte_size = sizeof(float) * size;
    auto readback = _self->_device->readback(size);
    auto cmd = _self->_device->record();
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
    gpuMemCpy(cmd, readback.gpu, _self->_allocation.gpu, byte_size);
    _self->_device->submit();
    _self->_device->flush();

    return to_string(readback, _shape);
}

std::vector<float> Tensor::cpu()
{
    auto size = flatten(_shape);
    auto byte_size = sizeof(float) * size;
    auto readback = _self->_device->readback(size);
    auto cmd = _self->_device->record();
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
    gpuMemCpy(cmd, readback.gpu, _self->_allocation.gpu, byte_size);
    _self->_device->submit();
    _self->_device->flush();

    std::vector<float> result(size, 0.f);
    memcpy(result.data(), readback.cpu, byte_size);
    return result;
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
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["add"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, broadcast](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + grad).detach();

        if (broadcast > 1)
        {
            auto inner = static_cast<unsigned int>(flatten(other._shape));
            auto rows = grad.reshape({ broadcast, inner });
            auto reduced = self._self->_device->ones({ 1, broadcast }).matmul(rows);
            other._self->grad = (other._self->grad + reduced.reshape(other._shape)).detach();
        }
        else
        {
            other._self->grad = (other._self->grad + grad).detach();
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
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["sub"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, broadcast](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + grad).detach();

        if (broadcast > 1)
        {
            auto inner = static_cast<unsigned int>(flatten(other._shape));
            auto rows = grad.reshape({ broadcast, inner });
            auto reduced = self._self->_device->ones({ 1, broadcast }).matmul(rows);
            other._self->grad = (other._self->grad - reduced.reshape(other._shape)).detach();
        }
        else
        {
            other._self->grad = (other._self->grad - grad).detach();
        }
    };
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const
{
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
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["mul"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + (grad * other)).detach();
        other._self->grad = (other._self->grad + (grad * self)).detach();
    };
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const
{
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
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["div"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other, broadcast](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + (grad / other)).detach();

        if (broadcast > 1)
        {
            auto inner = static_cast<unsigned int>(flatten(other._shape));
            auto term = grad * (self / (other * other));
            auto rows = term.reshape({ broadcast, inner });
            auto reduced = self._self->_device->ones({ 1, broadcast }).matmul(rows);
            other._self->grad = (other._self->grad - reduced.reshape(other._shape)).detach();
        }
        else
        {
            other._self->grad = (other._self->grad - grad * (self / (other * other))).detach();
        }
    };
    return out;
}

Tensor Tensor::operator+(float x) const
{
    return *this + _self->_device->tensor({ x });
}

Tensor Tensor::operator-(float x) const
{
    return *this - _self->_device->tensor({ x });
}

Tensor Tensor::operator*(float x) const
{
    return *this * _self->_device->tensor({ x });
}

Tensor Tensor::operator/(float x) const
{
    return *this / _self->_device->tensor({ x });
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

    Allocation<float> allocation = _self->_allocation;
    allocation.cpu += (size * i);
    allocation.gpu += (size * i);

    return Tensor(_self->_device, allocation, { *this }, res_shape, true);
}

Tensor Tensor::repeat(const Tensor& tensor, Shape shape) const
{
    return _self->_device->repeat(tensor, shape);
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

    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorTransposeData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->r = _shape.front();
    tensor_data.cpu->c = _shape.back();
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["mT"]);
    _self->_device->barrier(STAGE_COMPUTE, *this);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this }, res_shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    return out;
}

Tensor Tensor::dot(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot compute the dot product of tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->floats(1);
    memset(allocation.cpu, 0, sizeof(float));

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["dot"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { 1, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, { 1 });

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + other * grad).detach();
        other._self->grad = (other._self->grad + self * grad).detach();
    };
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const
{
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

    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorMatMulData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->a = _shape.front();
    tensor_data.cpu->b = _shape.back();
    tensor_data.cpu->c = other._shape.back();
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["matmul"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, res_shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + grad.matmul(other.mT())).detach();
        other._self->grad = (other._self->grad + self.mT().matmul(grad)).detach();
    };
    return out;
}

Tensor Tensor::pow(const Tensor& other) const
{
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
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->m = flatten(other._shape);
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["pow"]);
    _self->_device->barrier(STAGE_COMPUTE, *this, other);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    return out;
}

Tensor Tensor::pow(float x) const
{
    return pow(_self->_device->tensor({ x }));
}

Tensor Tensor::sum() const
{
    return dot(_self->_device->ones(_shape));
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
    return _self->_device->repeat(e, _shape).pow(*this);
}

Tensor Tensor::cosh() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["cosh"]);
    _self->_device->barrier(STAGE_COMPUTE, *this);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    return out;
}

Tensor Tensor::tanh() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["tanh"]);
    _self->_device->barrier(STAGE_COMPUTE, *this);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self._self->grad = (self._self->grad + grad / self.cosh().pow(2.f)).detach();
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
    auto allocation = _self->_device->floats(size);

    auto tensor_data = _self->_device->struct_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->a = alpha;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["relu"]);
    _self->_device->barrier(STAGE_COMPUTE, *this);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

    auto out = Tensor(_self->_device, allocation, { *this }, _shape);

    tensors_pending_writes[STAGE_COMPUTE].insert(out);

    Tensor self = *this;
    out._self->_backward = [self, alpha](const Tensor& grad)
    {
        auto size = flatten(self.shape());
        auto allocation = self._self->_device->floats(size);
        auto tensor_data = self._self->_device->struct_data<TensorData>();
        tensor_data.cpu->n = size;
        tensor_data.cpu->a = alpha;
        tensor_data.cpu->x = self._self->_allocation.gpu;
        tensor_data.cpu->y = grad._self->_allocation.gpu;
        tensor_data.cpu->z = allocation.gpu;

        auto cmd = self._self->_device->record();
        gpuSetPipeline(cmd, self._self->_device->pipelines["relu_backward"]);
        self._self->_device->barrier(STAGE_COMPUTE, self, grad);
        gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });

        auto term = Tensor(self._self->_device, allocation, {}, self.shape());

        tensors_pending_writes[STAGE_COMPUTE].insert(term);

        self._self->grad = (self._self->grad + term).detach();
    };
    return out;
}

Tensor Tensor::gelu() const
{
    return *this * 0.5f * (1.f + (::sqrt(2.f / pi) * (*this + (*this * *this * *this) * 0.044715f)).tanh());
}

Tensor Tensor::reshape(Shape shape) const
{
    if (flatten(_shape) != flatten(shape))
    {
        auto error = "cannot reshape tensor of shape " + to_string(_shape) + " into shape " + to_string(shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto byte_size = size * sizeof(float);
    auto allocation = _self->_device->floats(size);

    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_TRANSFER, *this);
    gpuMemCpy(cmd, allocation.gpu, _self->_allocation.gpu, byte_size);

    auto out = Tensor(_self->_device, allocation, { *this }, shape);

    tensors_pending_writes[STAGE_TRANSFER].insert(out);

    return out;
}

Tensor Tensor::detach() const
{
    auto size = flatten(_shape);
    auto byte_size = size * sizeof(float);
    auto allocation = _self->_device->floats(size);

    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_TRANSFER, *this);
    gpuMemCpy(cmd, allocation.gpu, _self->_allocation.gpu, byte_size);

    auto out = Tensor(_self->_device, allocation, {}, _shape);

    tensors_pending_writes[STAGE_TRANSFER].insert(out);

    return out;
}

void Tensor::copy(const Tensor& other) const
{
    if (flatten(_shape) != flatten(other._shape))
    {
        auto error = "cannot copy tensor of shape " + to_string(_shape) + " from shape " + to_string(other._shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto byte_size = size * sizeof(float);

    auto cmd = _self->_device->record();
    _self->_device->barrier(STAGE_TRANSFER, other);
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

    _self->grad = _self->_device->ones({ _shape });
    for (auto iter = graph.rbegin(); iter != graph.rend(); iter++)
    {
        if (iter->_self->_backward)
        {
            iter->_self->_backward(iter->_self->grad);
        }
    }
}
