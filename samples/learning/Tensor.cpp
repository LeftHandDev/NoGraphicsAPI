#include "Common.h"
#include "Tensor.h"
#include "Utilities.h"
#include <cstring>
#include <map>
#include <algorithm>
#include <random>
#include <iostream>

std::vector<Device*> devices;

std::string to_string(Shape shape)
{
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

class Device_impl : public Device
{
public:
    Device_impl(int index)
    {
        device = gpuCreateDevice(index);
        queue = gpuCreateQueue(device);
        allocator = new LinearAllocator<MEMORY_DEFAULT>(device);

        auto tensorIR = loadIR("shaders/learning/Tensor.spv");
        pipelines["add"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_add");
        pipelines["sub"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_sub");
        pipelines["mul"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_mul");
        pipelines["div"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_div");
        pipelines["dot"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_dot");
        pipelines["mT"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_mT");
        pipelines["matmul"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_matmul");
        pipelines["pow"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_pow");
        pipelines["tanh"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_tanh");
    }

    ~Device_impl()
    {
        submit();
        delete readback_ring;
        delete struct_allocator;
        delete allocator;

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
        return Tensor(this, data, shape);
    }

    virtual Tensor rand(Shape shape) override
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.f, 1.f);

        std::vector<float> data(flatten(shape));
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i] = dis(gen);
        }

        return Tensor(this, data, shape);
    }

    virtual Tensor zeros(Shape shape) override
    {
        std::vector<float> data(flatten(shape), 0.f);
        return Tensor(this, data, shape);
    }

    virtual Tensor ones(Shape shape) override
    {
        std::vector<float> data(flatten(shape), 1.f);
        return Tensor(this, data, shape);
    }

    virtual Tensor repeat(float x, Shape shape = {}) override
    {
        std::vector<float> data(flatten(shape), x);
        return Tensor(this, data, shape);
    }

    Allocation<float> readback(size_t size)
    {
        size_t byte_size = size * sizeof(float);
        if (readback_ring == nullptr || byte_size > readback_ring->size())
        {
            delete readback_ring;
            readback_ring = new RingBuffer<MEMORY_READBACK>(device, byte_size);
        }

        return readback_ring->allocate<float>(size);
    }

    template <typename T>
    Allocation<T> tensor_data()
    {
        if (struct_allocator == nullptr)
        {
            delete struct_allocator;
            struct_allocator = new RingBuffer<MEMORY_DEFAULT>(device);
        }

        if (struct_allocator->wrap<T>(1))
        {
            submit();
        }

        return struct_allocator->allocate<T>(1);
    }

    GpuCommandBuffer record()
    {
        if (!cmd)
        {
            cmd = gpuStartCommandRecording(queue);
        }
        return cmd;
    }

    void submit()
    {
        if (!cmd)
        {
            return; // no work to submit
        }

        if (!semaphore)
        {
            semaphore = gpuCreateSemaphore(device, 0);
        }
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, frame);
        cmd = nullptr;
        gpuWaitSemaphore(semaphore, frame++);
        for (auto& allocation : pending_free)
        {
            allocator->free(allocation.cpu);
        }
        pending_free.clear();
    }

    void free(Allocation<float> allocation)
    {
        pending_free.push_back(allocation);
    }

    GpuDevice device = nullptr;
    GpuQueue queue = nullptr;
    GpuCommandBuffer cmd = nullptr;
    GpuSemaphore semaphore = nullptr;
    uint64_t frame = 1;
    LinearAllocator<MEMORY_DEFAULT>* allocator = nullptr;
    RingBuffer<MEMORY_DEFAULT>* struct_allocator = nullptr;
    RingBuffer<MEMORY_READBACK>* readback_ring = nullptr;
    std::vector<Allocation<float>> pending_free;
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
};

Instance::Instance()
{
    gpuCreateInstance();
}

Instance::~Instance()
{
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
    if (_tensor)
    {
        // std::cout << "Destroying tensor of shape " << to_string(_shape) << std::endl;
    }
}

Tensor::Tensor(Tensor&& tensor) noexcept
{
    _shape = tensor._shape;
    _tensor = std::move(tensor._tensor);
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept
{
    _shape = tensor._shape;
    _tensor = std::move(tensor._tensor);
    return *this;
}

Tensor::Tensor(Device_impl* device, std::vector<float> data, Shape shape, bool slice)
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

    _tensor = std::make_unique<Tensor_impl>();
    _tensor->_device = device;
    _tensor->_allocation = _tensor->_device->allocator->allocate<float>(data.size());
    _tensor->_slice = slice;
    memcpy(_tensor->_allocation.cpu, data.data(), data.size() * sizeof(float));
    // std::cout << "Creating tensor of shape " << to_string(_shape) << std::endl;
}

Tensor::Tensor(Device_impl* device, Allocation<float> allocation, Shape shape, bool slice)
    : _shape(shape)
{
    _tensor = std::make_unique<Tensor_impl>(device, allocation, slice);
}

Shape Tensor::shape() const
{
    return _shape;
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
    auto readback = _tensor->_device->readback(size);
    auto cmd = _tensor->_device->record();
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
    gpuMemCpy(cmd, readback.gpu, _tensor->_allocation.gpu, byte_size);
    _tensor->_device->submit();

    return to_string(readback, _shape);
}

std::vector<float> Tensor::cpu()
{
    auto size = flatten(_shape);
    auto byte_size = sizeof(float) * size;
    auto readback = _tensor->_device->readback(size);
    auto cmd = _tensor->_device->record();
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
    gpuMemCpy(cmd, readback.gpu, _tensor->_allocation.gpu, byte_size);
    _tensor->_device->submit();

    std::vector<float> result(size, 0.f);
    memcpy(result.data(), readback.cpu, byte_size);
    return result;
}

Tensor Tensor::operator+(const Tensor& tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot add tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["add"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, _shape);
}

Tensor Tensor::operator-(const Tensor& tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot subtract tensor of shape " + to_string(tensor._shape) + " with tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["sub"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, _shape);
}

Tensor Tensor::operator*(const Tensor& tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot multiply tensor of shape " + to_string(tensor._shape) + " with tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["mul"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, _shape);
}

Tensor Tensor::operator/(const Tensor& tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot divide tensor of shape " + to_string(tensor._shape) + " by tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["div"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, _shape);
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

    Allocation<float> allocation = _tensor->_allocation;
    allocation.cpu += (size * i);
    allocation.gpu += (size * i);

    return Tensor(_tensor->_device, allocation, res_shape, true);
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

    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorTransposeData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->r = _shape.front();
    tensor_data.cpu->c = _shape.back();
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["mT"]);
    gpuDispatch(cmd, tensor_data.gpu, { 1, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, res_shape);
}

Tensor Tensor::dot(const Tensor& tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot compute the dot product of tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(1);
    memset(allocation.cpu, 0, sizeof(float));

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["dot"]);
    gpuDispatch(cmd, tensor_data.gpu, { 1, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, { 1 });
}

Tensor Tensor::matmul(const Tensor& tensor) const
{
    if (_shape.size() != 2 || tensor._shape.size() != 2)
    {
        std::string error = "batched matmul not yet implemented";
        throw std::runtime_error(error);
    }

    // (a,b) mat and (c,d) mat results in (a,d) mat, and b must equal c
    if (_shape.back() != tensor._shape.front())
    {
        auto error = "cannot matmul tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    Shape res_shape = { _shape.front(), tensor._shape.back() };
    auto size = flatten(res_shape);

    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorMatMulData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->a = _shape.front();
    tensor_data.cpu->b = _shape.back();
    tensor_data.cpu->c = tensor._shape.back();
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["matmul"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, res_shape);
}

Tensor Tensor::pow(const Tensor& tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot pow tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["pow"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, _shape);
}

Tensor Tensor::exp() const
{
    return _tensor->_device->repeat(e, _shape).pow(*this);
}

Tensor Tensor::tanh() const
{
    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["tanh"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, _shape);
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
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto cmd = _tensor->_device->record();
    gpuMemCpy(cmd, allocation.gpu, _tensor->_allocation.gpu, byte_size);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);

    return Tensor(_tensor->_device, allocation, shape);
}

void Tensor::copy(const Tensor& tensor) const
{
    if (flatten(_shape) != flatten(tensor._shape))
    {
        auto error = "cannot copy tensor of shape " + to_string(_shape) + " from shape " + to_string(tensor._shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto byte_size = size * sizeof(float);

    auto cmd = _tensor->_device->record();
    gpuMemCpy(cmd, _tensor->_allocation.gpu, tensor._tensor->_allocation.gpu, byte_size);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);
}

void Tensor::backward()
{
}
