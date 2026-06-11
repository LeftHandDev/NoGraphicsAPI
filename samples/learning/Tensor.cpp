#include "Common.h"
#include "Tensor.h"
#include "Utilities.h"
#include <map>
#include <algorithm>
#include <random>
#include <iostream>
#include <set>

std::vector<Device*> devices;

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
        pipelines["cosh"] = gpuCreateComputePipeline(device, ByteSpan(tensorIR), "_cosh");
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
        return Tensor(this, data, {}, shape);
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

    virtual Tensor repeat(float x, Shape shape = {}) override
    {
        std::vector<float> data(flatten(shape), x);
        return Tensor(this, data, {}, shape);
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
        gpuSubmit(queue, Span(&cmd, 1), semaphore, frame);
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
    _self->_allocation = _self->_device->allocator->allocate<float>(data.size());
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

    std::vector<float> result(size, 0.f);
    memcpy(result.data(), readback.cpu, byte_size);
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot add tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["add"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);
    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self.init_grad();
        other.init_grad();

        self._self->grad = self._self->grad + grad;
        other._self->grad = other._self->grad + grad;
    };
    return out;
}

Tensor Tensor::operator-() const
{
    return *this * -1;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot subtract tensor of shape " + to_string(other._shape) + " with tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["sub"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);
    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self.init_grad();
        other.init_grad();

        self._self->grad = self._self->grad + grad;
        other._self->grad = other._self->grad - grad;
    };
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot multiply tensor of shape " + to_string(other._shape) + " with tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["mul"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    auto out = Tensor(_self->_device, allocation, { *this, other }, _shape);
    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self.init_grad();
        other.init_grad();

        self._self->grad = self._self->grad + (other * grad);
        other._self->grad = other._self->grad + (self * grad);
    };
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot divide tensor of shape " + to_string(other._shape) + " by tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["div"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, { *this, other }, _shape);
}

Tensor Tensor::operator+(float x) const
{
    return *this + _self->_device->repeat(x, _shape);
}

Tensor Tensor::operator-(float x) const
{
    return *this - _self->_device->repeat(x, _shape);
}

Tensor Tensor::operator*(float x) const
{
    return *this * _self->_device->repeat(x, _shape);
}

Tensor Tensor::operator/(float x) const
{
    return *this / _self->_device->repeat(x, _shape);
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

Tensor Tensor::mT() const
{
    if (_shape.size() != 2)
    {
        auto error = "cannot transpose a tensor that isn't a matrix";
        throw std::runtime_error(error);
    }

    Shape res_shape = { _shape.back(), _shape.front() };
    auto size = flatten(res_shape);

    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorTransposeData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->r = _shape.front();
    tensor_data.cpu->c = _shape.back();
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["mT"]);
    gpuDispatch(cmd, tensor_data.gpu, { 1, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, { *this }, res_shape);
}

Tensor Tensor::dot(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot compute the dot product of tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(1);
    memset(allocation.cpu, 0, sizeof(float));

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["dot"]);
    gpuDispatch(cmd, tensor_data.gpu, { 1, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, { *this, other }, { 1 });
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

    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorMatMulData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->a = _shape.front();
    tensor_data.cpu->b = _shape.back();
    tensor_data.cpu->c = other._shape.back();
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["matmul"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    auto out = Tensor(_self->_device, allocation, { *this, other }, res_shape);
    Tensor self = *this;
    out._self->_backward = [self, other](const Tensor& grad)
    {
        self.init_grad();
        other.init_grad();

        self._self->grad = self._self->grad + grad.matmul(other.mT());
        other._self->grad = other._self->grad + self.mT().matmul(grad);
    };
    return out;
}

Tensor Tensor::pow(const Tensor& other) const
{
    if (_shape != other._shape)
    {
        auto error = "cannot pow tensor of shape " + to_string(other._shape) + " to tensor of shape " + to_string(_shape);
        throw std::runtime_error(error);
    }

    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = other._self->_allocation.gpu;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["pow"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, { *this, other }, _shape);
}

Tensor Tensor::pow(float x) const
{
    return pow(_self->_device->repeat({ x }, _shape));
}

Tensor Tensor::sqrt() const
{
    auto two = _self->_device->repeat({ 2.f }, _shape);
    return pow(two.rcp());
}

Tensor Tensor::rcp() const
{
    return _self->_device->repeat(1, _shape) / *this;
}

Tensor Tensor::exp() const
{
    return _self->_device->repeat(e, _shape).pow(*this);
}

Tensor Tensor::cosh() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["cosh"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, { *this }, _shape);
}

Tensor Tensor::tanh() const
{
    auto size = flatten(_shape);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto tensor_data = _self->_device->tensor_data<TensorData>();
    tensor_data.cpu->n = size;
    tensor_data.cpu->x = _self->_allocation.gpu;
    tensor_data.cpu->y = nullptr;
    tensor_data.cpu->z = allocation.gpu;

    auto cmd = _self->_device->record();
    gpuSetPipeline(cmd, _self->_device->pipelines["tanh"]);
    gpuDispatch(cmd, tensor_data.gpu, { static_cast<unsigned int>(size + 63) / 64, 1, 1 });
    gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE);

    auto out = Tensor(_self->_device, allocation, { *this }, _shape);
    Tensor self = *this;
    out._self->_backward = [self](const Tensor& grad)
    {
        self.init_grad();

        auto two = self._self->_device->repeat(2.f, self._shape);
        self._self->grad = self._self->grad + self.sech().pow(two) * grad;
    };
    return out;
}

Tensor Tensor::sech() const
{
    return cosh().rcp();
}

Tensor Tensor::gelu() const
{
    auto sqrt_2_pi = (_self->_device->repeat({ 2.f }, _shape) / pi).sqrt();
    return *this * 0.5f * (1.f + (sqrt_2_pi * (*this + pow(3.f) * 0.044715f)).tanh());
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
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto cmd = _self->_device->record();
    gpuMemCpy(cmd, allocation.gpu, _self->_allocation.gpu, byte_size);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, { *this }, shape);
}

Tensor Tensor::detach() const
{
    auto size = flatten(_shape);
    auto byte_size = size * sizeof(float);
    auto allocation = _self->_device->allocator->allocate<float>(size);

    auto cmd = _self->_device->record();
    gpuMemCpy(cmd, allocation.gpu, _self->_allocation.gpu, byte_size);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);

    return Tensor(_self->_device, allocation, {}, _shape);
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
    gpuMemCpy(cmd, _self->_allocation.gpu, other._self->_allocation.gpu, byte_size);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);
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

void Tensor::init_grad() const
{
    if (_self->grad._shape.empty())
    {
        _self->grad = _self->_device->zeros(_shape);
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
