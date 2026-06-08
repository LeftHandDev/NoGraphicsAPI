#include "Common.h"
#include "Tensor.h"
#include "../Common/Utilities.h"
#include <map>
#include <algorithm>
#include <random>

std::vector<device *> devices;

std::string to_string(shape shape)
{
    std::string result = "(";
    for (auto &n : shape)
    {
        result += std::to_string(n) + ", ";
    };
    result.pop_back();
    result.pop_back();
    return result + ")";
}

uint64_t flatten(shape shape)
{
    uint64_t size = 1;
    std::for_each(shape.begin(), shape.end(), [&size](const auto &n)
                  { size *= n; });
    return size;
}

class tensor_impl
{
    ~tensor_impl()
    {
        _device->allocator->free(_allocation.cpu);
    }
    device_impl *_device = nullptr;
    Allocation<float> _allocation;
};

class device_impl : public device
{
public:
    device_impl()
    {
        device = gpuCreateDevice(0);
        queue = gpuCreateQueue(device);
        allocator = new LinearAllocator<MEMORY_DEFAULT>(device);

        auto tensorIR = loadIR("../Shaders/Learning/Tensor.spv");
        pipelines["add"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_add");
        pipelines["sub"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_sub");
        pipelines["mul"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_mul");
        pipelines["div"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_div");
        pipelines["dot"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_dot");
        pipelines["mT"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_mT");
        pipelines["pow"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_pow");
        pipelines["tanh"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_tanh");
    }

    ~device_impl()
    {
        delete allocator;
        std::for_each(pipelines.begin(), pipelines.end(), [](const auto &pair)
                      { 
                        auto [entry, pipeline] = pair;
                        gpuFreePipeline(pipeline); });
        auto iter = std::remove(devices.begin(), devices.end(), reinterpret_cast<::device *>(this));
        if (iter == devices.end())
        {
            // device tracking error, should not happen
        }
        gpuDestroyDevice(device);
    }

    virtual ::tensor tensor(std::vector<float> data, shape shape = {}) override
    {
        return ::tensor(this, data, shape);
    }

    virtual ::tensor rand(::shape shape) override
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.f, 1.f);

        std::vector<float> data(flatten(shape));
        for (size_t i = 0; i < data.size(); i++)
        {
            data[i] = dis(gen);
        }

        return ::tensor(this, data, shape);
    }

    virtual ::tensor zeros(::shape shape) override
    {
        std::vector<float> data(flatten(shape), 0.f);
        return ::tensor(this, data, shape);
    }

    virtual ::tensor ones(::shape shape) override
    {
        std::vector<float> data(flatten(shape), 1.f);
        return ::tensor(this, data, shape);
    }

    virtual ::tensor repeat(float x, ::shape shape = {}) override
    {
        std::vector<float> data(flatten(shape), x);
        return ::tensor(this, data, shape);
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

    Allocation<TensorData> tensor_data()
    {
        if (default_ring == nullptr)
        {
            delete default_ring;
            default_ring = new RingBuffer<MEMORY_DEFAULT>(device);
        }

        return default_ring->allocate<TensorData>(1);
    }

    Allocation<TensorTransposeData> tensor_transpose_data()
    {
        if (default_ring == nullptr)
        {
            delete default_ring;
            default_ring = new RingBuffer<MEMORY_DEFAULT>(device);
        }

        return default_ring->allocate<TensorTransposeData>(1);
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
        if (!semaphore)
        {
            semaphore = gpuCreateSemaphore(device, frame);
        }
        frame++;
        gpuSubmit(queue, Span(&cmd, 1), semaphore, frame);
        cmd = nullptr;
        gpuWaitSemaphore(semaphore, frame);
    }

    GpuDevice device = nullptr;
    GpuQueue queue = nullptr;
    GpuCommandBuffer cmd = nullptr;
    GpuSemaphore semaphore = nullptr;
    uint64_t frame = 0;
    LinearAllocator<MEMORY_DEFAULT> *allocator = nullptr;
    RingBuffer<MEMORY_READBACK> *readback_ring = nullptr;
    RingBuffer<MEMORY_DEFAULT> *default_ring = nullptr;
    std::map<std::string, GpuPipeline> pipelines;
};

instance::instance()
{
    gpuCreateInstance();
}

instance::~instance()
{
    for (auto device : devices)
    {
        delete device;
    }
    gpuDestroyInstance();
}

device *instance::device()
{
    auto dev = new device_impl();
    devices.push_back(dev);
    return dev;
}

tensor::~tensor()
{
}

tensor::tensor(tensor &&tensor) noexcept
{
    _shape = tensor._shape;
    _tensor = std::move(tensor._tensor);
}

tensor &tensor::operator=(tensor &&tensor) noexcept
{
    _shape = tensor._shape;
    _tensor = std::move(tensor._tensor);
    return *this;
}

tensor::tensor(device_impl *device, std::vector<float> data, ::shape shape)
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

    _tensor = std::make_unique<tensor_impl>();
    _tensor->_device = device;
    _tensor->_allocation = _tensor->_device->allocator->allocate<float>(data.size());
    memcpy(_tensor->_allocation.cpu, data.data(), data.size() * sizeof(float));
}

tensor::tensor(tensor_impl tensor, ::shape shape)
    : _shape(shape)
{
    _tensor = std::make_unique<tensor_impl>(tensor);
}

shape tensor::shape() const
{
    return _shape;
}

std::string to_string(Allocation<float> readback, shape shape, uint64_t offset = 0)
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

tensor::operator std::string() const
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

std::vector<float> tensor::cpu()
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

tensor tensor::operator+(const tensor &tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot add tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["add"]);
    gpuDispatch(cmd, tensor_data.gpu, {static_cast<unsigned int>(size + 63) / 64, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, _shape);
}

tensor tensor::operator-(const tensor &tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot subtract tensor of shape " + to_string(tensor._shape) + " with tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["sub"]);
    gpuDispatch(cmd, tensor_data.gpu, {static_cast<unsigned int>(size + 63) / 64, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, _shape);
}

tensor tensor::operator*(const tensor &tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot multiply tensor of shape " + to_string(tensor._shape) + " with tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["mul"]);
    gpuDispatch(cmd, tensor_data.gpu, {static_cast<unsigned int>(size + 63) / 64, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, _shape);
}

tensor tensor::operator/(const tensor &tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot divide tensor of shape " + to_string(tensor._shape) + " by tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["div"]);
    gpuDispatch(cmd, tensor_data.gpu, {static_cast<unsigned int>(size + 63) / 64, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, _shape);
}

tensor tensor::operator[](unsigned int i) const
{
    if (_shape.size() != 2)
    {
        auto error = "cannot take a slice of a tensor that isn't a matrix";
        throw std::runtime_error(error);
    }

    ::shape res_shape = {_shape.back()};
    auto size = flatten(res_shape);

    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto cmd = _tensor->_device->record();
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
    gpuMemCpy(_tensor->_device->record(), allocation.gpu, _tensor->_allocation.gpu + (_shape.back() * i), size * sizeof(float));
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, res_shape);
}

tensor tensor::mT() const
{
    if (_shape.size() != 2)
    {
        auto error = "cannot transpose a tensor that isn't a matrix";
        throw std::runtime_error(error);
    }

    ::shape res_shape = {_shape.back(), _shape.front()};
    auto size = flatten(res_shape);

    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_transpose_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->r = _shape.front();
    tensor_data.cpu->c = _shape.back();
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["mT"]);
    gpuDispatch(cmd, tensor_data.gpu, {1, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, res_shape);
}

tensor tensor::dot(const tensor &tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot compute the dot product of tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(1);
    memset(allocation.cpu, 0, sizeof(float));

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["dot"]);
    gpuDispatch(cmd, tensor_data.gpu, {1, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, {1});
}

tensor tensor::matmul(const tensor &tensor) const
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

    ::shape res_shape = {_shape.front(), tensor._shape.back()};
    auto size = flatten(res_shape);

    auto allocation = _tensor->_device->allocator->allocate<float>(size);
    memset(allocation.cpu, 0, sizeof(float) * size);

    auto cmd = _tensor->_device->record();

    // TODO: very inefficient currently
    auto tensor_transposed = tensor.mT();
    for (size_t i = 0; i < _shape.front(); i++)
    {
        auto x = (*this)[i];
        for (size_t j = 0; j < tensor._shape.back(); j++)
        {
            auto y = tensor_transposed[j];
            auto z = x.dot(y);
            gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
            gpuMemCpy(cmd, allocation.gpu + (i * tensor._shape.back() + j), z._tensor->_allocation.gpu, sizeof(float));
        }
    }

    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, res_shape);
}

tensor tensor::pow(const tensor &tensor) const
{
    if (_shape != tensor._shape)
    {
        auto error = "cannot pow tensor of shape " + to_string(tensor._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = tensor._tensor->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["pow"]);
    gpuDispatch(cmd, tensor_data.gpu, {static_cast<unsigned int>(size + 63) / 64, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, _shape);
}

tensor tensor::exp() const
{
    return _tensor->_device->repeat(e, _shape).pow(*this);
}

tensor tensor::tanh() const
{
    auto size = flatten(_shape);
    auto allocation = _tensor->_device->allocator->allocate<float>(size);

    auto tensor_data = _tensor->_device->tensor_data();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _tensor->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _tensor->_device->record();
    gpuSetPipeline(cmd, _tensor->_device->pipelines["tanh"]);
    gpuDispatch(cmd, tensor_data.gpu, {static_cast<unsigned int>(size + 63) / 64, 1, 1});
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return ::tensor({_tensor->_device, allocation}, _shape);
}

tensor tensor::reshape(::shape shape) const
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

    return tensor({_tensor->_device, allocation}, shape);
}

void tensor::backward()
{
}
